import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import config


class ULCDClient:
    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_prototypes(self):
        self.model.eval()
        device = next(self.model.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                latents = self.model.get_features(x)

                latent_norms = torch.norm(latents, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)

                for i in range(len(latents)):
                    if outlier_mask[i]:
                        label = y[i].item()
                        class_latents[label].append(latents[i].cpu())
                        class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id] and len(class_latents[class_id]) >= 2:
                class_stack = torch.stack(class_latents[class_id])

                if len(class_latents[class_id]) > 10:
                    norms = torch.norm(class_stack, dim=1)
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms) - 2*k, largest=False)
                    if len(indices) > 2*k:
                        indices = indices[k:-k]
                        class_stack = class_stack[indices]

                proto = class_stack.mean(dim=0)
                class_prototypes[class_id] = proto
            elif class_latents[class_id]:
                class_prototypes[class_id] = class_latents[class_id][0]

        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask

    def train(self, epochs, server_prototypes=None, round_num=1):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        proto_weight = min(config.PROTOTYPE_WARMUP_RATE * round_num, config.PROTOTYPE_WEIGHT)

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f}, Proto Weight = {proto_weight:.2f}")

        public_iter = iter(self.public_loader)

        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_proto_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                out = self.model(x)
                ce_loss = self.ce_loss(out, y)
                loss = ce_loss
                total_ce_loss += ce_loss.item()

                if server_prototypes is not None:
                    try:
                        x_pub, y_pub = next(public_iter)
                    except StopIteration:
                        public_iter = iter(self.public_loader)
                        x_pub, y_pub = next(public_iter)

                    x_pub, y_pub = x_pub.cuda(), y_pub.cuda()
                    feats_pub = self.model.get_features(x_pub)

                    proto_align = 0.0
                    valid_count = 0

                    for i in range(len(x_pub)):
                        true_label = y_pub[i].item()
                        if true_label in server_prototypes:
                            client_feat = feats_pub[i]
                            proto = server_prototypes[true_label].cuda()
                            sim = F.cosine_similarity(client_feat, proto, dim=0)
                            proto_align += (1 - sim)
                            valid_count += 1

                    if valid_count > 0:
                        proto_align /= valid_count
                        loss += proto_weight * proto_align
                        total_proto_loss += proto_align.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.ULCD_GRADIENT_CLIP)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_ce = total_ce_loss / num_batches
            avg_proto = total_proto_loss / num_batches if server_prototypes else 0

            print(f"  Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, Proto={avg_proto:.4f})")
