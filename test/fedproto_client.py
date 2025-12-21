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
        # AMP scaler for mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None

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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr, momentum=0.5)

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f} (decay factor: {decay_factor:.2f})")

        for iter in range(epochs):
            batch_loss = {'total': [], '1': [], '2': []}
            agg_protos_label = {}

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()

                self.model.zero_grad()

                # Use autocast for mixed precision training
                with autocast(enabled=config.USE_AMP):
                    log_probs, protos = self.model(images, return_protos=True)
                    loss1 = self.criterion(log_probs, labels)

                    loss_mse = nn.MSELoss()
                    if not server_prototypes:
                        loss2 = 0 * loss1
                    else:
                        proto_new = copy.deepcopy(protos.data)
                        for i, label in enumerate(labels):
                            if label.item() in server_prototypes:
                                proto_new[i, :] = server_prototypes[label.item()].data
                        loss2 = loss_mse(proto_new, protos)

                    loss = loss1 + loss2 * config.PROTOTYPE_WEIGHT

                # Backward pass with gradient scaling if AMP is enabled
                if config.USE_AMP:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [protos[i, :]]

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

        prototypes = {}
        for label, proto_list in agg_protos_label.items():
            if len(proto_list) > 1:
                prototypes[label] = torch.stack(proto_list).mean(dim=0)
            else:
                prototypes[label] = proto_list[0]

        return prototypes
