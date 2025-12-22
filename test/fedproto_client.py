import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from . import config
import copy

class FedProtoClient:
    def __init__(self, model, train_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.criterion = nn.NLLLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.scaler = GradScaler() if config.USE_AMP else None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LEARNING_RATE, momentum=0.5)

    def compute_prototypes(self):
        self.model.eval()
        agg_protos_label = {}

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()
                _, protos = self.model(x, return_protos=True)

                for i in range(len(y)):
                    label = y[i].item()
                    if label in agg_protos_label:
                        agg_protos_label[label].append(protos[i, :])
                    else:
                        agg_protos_label[label] = [protos[i, :]]

        prototypes = {}
        for label, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                prototypes[label] = proto / len(proto_list)
            else:
                prototypes[label] = proto_list[0].data

        return prototypes

    def train(self, epochs, server_prototypes=None, round_num=1):
        self.model.train()
        epoch_loss = {'total': [], '1': [], '2': []}

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f} (decay factor: {decay_factor:.2f})")

        for iter in range(epochs):
            batch_loss = {'total': [], '1': [], '2': []}

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()

                self.optimizer.zero_grad()

                with autocast(enabled=config.USE_AMP):
                    log_probs, protos = self.model(images, return_protos=True)
                    loss1 = self.criterion(log_probs, labels)

                    loss2 = 0
                    if server_prototypes:
                        target_protos = protos.detach().clone()
                        unique_labels = labels.unique()
                        for label in unique_labels:
                            l_item = label.item()
                            if l_item in server_prototypes:
                                mask = (labels == label)
                                target_protos[mask] = server_prototypes[l_item].to(protos.device, dtype=protos.dtype)
                        loss2 = self.mse_loss(target_protos, protos)

                    loss = loss1 + loss2 * config.PROTOTYPE_WEIGHT

                if config.USE_AMP:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item() if isinstance(loss2, torch.Tensor) else 0)

            epoch_total = sum(batch_loss['total']) / len(batch_loss['total'])
            epoch_ce = sum(batch_loss['1']) / len(batch_loss['1'])
            epoch_proto = sum(batch_loss['2']) / len(batch_loss['2'])

            epoch_loss['total'].append(epoch_total)
            epoch_loss['1'].append(epoch_ce)
            epoch_loss['2'].append(epoch_proto)

            print(f"Client {self.client_id} Epoch {iter+1}/{epochs}: Loss={epoch_total:.4f} (CE={epoch_ce:.4f}, Proto={epoch_proto:.4f})")

        return {}
