# client.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import config

class Client:
    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def get_logits_on_public(self):
        self.model.eval()
        logits = []
        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.cuda()
                out = self.model(x)
                logits.append(out.cpu())
        return torch.cat(logits)

    def compute_prototypes(self):
        self.model.eval()
        proto_dict = defaultdict(list)
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()
                feats = self.model.get_features(x)
                for f, label in zip(feats, y):
                    proto_dict[label.item()].append(f.detach().cpu())

        prototypes = {}
        for c in proto_dict:
            prototypes[c] = torch.stack(proto_dict[c]).mean(dim=0)
        return prototypes

    def train(self, epochs, server_prototypes=None, server_logits=None, round_num=1):
        self.model.train()

        # Learning rate with decay based on round number
        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor

        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f} (decay factor: {decay_factor:.2f})")

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()
                out = self.model(x)
                feats = self.model.get_features(x)

                # Supervised loss
                loss = self.ce_loss(out, y)

                # Contrastive loss (MOON-inspired)
                if server_prototypes is not None:
                    proto_align = 0.0
                    valid_count = 0
                    for i in range(len(x)):
                        true_label = y[i].item()
                        if true_label in server_prototypes:
                            client_feat = feats[i]
                            proto = server_prototypes[true_label].cuda()
                            sim = F.cosine_similarity(client_feat, proto, dim=0)
                            proto_align += (1 - sim)  # encourage alignment
                            valid_count += 1
                    if valid_count > 0:
                        proto_align /= valid_count
                        loss += config.PROTOTYPE_WEIGHT * proto_align

                # Logit distillation (FedMD-inspired)
                if server_logits is not None:
                    with torch.no_grad():
                        server_logit = server_logits[:len(x)].cuda()
                    kd_loss = F.mse_loss(out, server_logit)
                    loss += config.DISTILLATION_WEIGHT * kd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
