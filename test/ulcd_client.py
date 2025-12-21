import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from . import config


class ULCDClient:
    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        # AMP scaler for mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None


    def compute_prototypes(self):
        self.model.eval()
        device = next(self.model.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                consensus_latents = self.model.get_consensus_features(x)

                for i in range(len(consensus_latents)):
                    label = y[i].item()
                    class_latents[label].append(consensus_latents[i].cpu())
                    class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id]:
                class_stack = torch.stack(class_latents[class_id])
                proto = class_stack.mean(dim=0)

                proto_norm = torch.norm(proto)
                if proto_norm > 1e-8 and not torch.isnan(proto_norm):
                    proto = proto / (proto_norm + 1e-8)
                    if not torch.isnan(proto).any():
                        class_prototypes[class_id] = proto
                    else:
                        print(f"Client {self.client_id}: WARNING - NaN prototype for class {class_id} after normalization, skipping")
                else:
                    print(f"Client {self.client_id}: WARNING - Zero-norm prototype for class {class_id} (norm={proto_norm:.6f}), skipping")

        # Calculate the relative frequency of each class on this client
        # The mask is sent to the server to allow for a weighted consensus
        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask

    def compute_contrastive_consensus_loss(self, x_align, y_align, server_prototypes):
        if not server_prototypes or len(server_prototypes) < 2:
            return torch.tensor(0.0).cuda()

        consensus_feats = self.model.get_consensus_features(x_align)

        if torch.isnan(consensus_feats).any() or torch.isinf(consensus_feats).any():
            return torch.tensor(0.0).cuda()

        tau = config.CONTRASTIVE_TEMP

        mask = torch.tensor([label.item() in server_prototypes for label in y_align], device='cuda')
        if not mask.any():
            return torch.tensor(0.0).cuda()

        valid_labels = y_align[mask]
        valid_feats = consensus_feats[mask]

        all_proto_labels = sorted(server_prototypes.keys())
        all_protos = torch.stack([server_prototypes[k].cuda() for k in all_proto_labels])

        if torch.isnan(all_protos).any() or torch.isinf(all_protos).any():
            return torch.tensor(0.0).cuda()

        similarities = F.cosine_similarity(valid_feats.unsqueeze(1), all_protos.unsqueeze(0), dim=2) / tau

        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            return torch.tensor(0.0).cuda()

        label_to_idx = {label: idx for idx, label in enumerate(all_proto_labels)}
        pos_indices = torch.tensor([label_to_idx[label.item()] for label in valid_labels], device='cuda')

        pos_sim = similarities[torch.arange(len(valid_labels), device='cuda'), pos_indices]

        pos_exp = torch.exp(pos_sim)
        all_exp = torch.exp(similarities).sum(dim=1)

        loss = -torch.log(pos_exp / all_exp + 1e-8).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0).cuda()

        return loss

    def train(self, epochs, server_prototypes=None, round_num=1):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        # Weight warmup allows the client to learn local features before conforming
        if config.ULCD_USE_WARMUP:
            proto_weight = min(config.PROTOTYPE_WARMUP_RATE * round_num, config.PROTOTYPE_WEIGHT)
        else:
            proto_weight = config.PROTOTYPE_WEIGHT

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"Client {self.client_id}: LR = {current_lr:.6f}, Proto Weight = {proto_weight:.2f}")

        # Public data to better generalize across the dataset (Fedmd's approach)
        if config.ULCD_USE_PUBLIC_ALIGNMENT:
            alignment_iter = iter(self.public_loader)
        else:
            alignment_iter = iter(self.train_loader)

        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_proto_loss = 0
            total_contrastive_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                # Use autocast for mixed precision training
                with autocast(enabled=config.USE_AMP):
                    # Private training
                    out = self.model(x)
                    ce_loss = self.ce_loss(out, y)
                    loss = ce_loss
                    total_ce_loss += ce_loss.item()

                    # Prototype alignment
                    if server_prototypes is not None:
                        try:
                            x_align, y_align = next(alignment_iter)
                        except StopIteration:
                            if config.ULCD_USE_PUBLIC_ALIGNMENT:
                                alignment_iter = iter(self.public_loader)
                            else:
                                alignment_iter = iter(self.train_loader)
                            x_align, y_align = next(alignment_iter)

                        x_align, y_align = x_align.cuda(), y_align.cuda()
                        feats_align = self.model.get_consensus_features(x_align)

                        if torch.isnan(feats_align).any() or torch.isinf(feats_align).any():
                            print(f"Client {self.client_id}: WARNING - Invalid features detected, skipping alignment")
                            continue

                        mask = torch.tensor([label.item() in server_prototypes for label in y_align], device='cuda')
                        if mask.any():
                            valid_labels = y_align[mask]
                            valid_feats = feats_align[mask]

                            proto_stack = torch.stack([server_prototypes[label.item()].cuda() for label in valid_labels])

                            if torch.isnan(proto_stack).any() or torch.isinf(proto_stack).any():
                                print(f"Client {self.client_id}: WARNING - Invalid server prototypes, skipping alignment")
                            else:
                                sims = F.cosine_similarity(valid_feats, proto_stack, dim=1)
                                proto_align = (1 - sims).mean()

                                if not torch.isnan(proto_align) and not torch.isinf(proto_align):
                                    loss += proto_weight * proto_align
                                    total_proto_loss += proto_align.item()
                                else:
                                    print(f"Client {self.client_id}: WARNING - Invalid proto_align loss")

                        # Contrastive consensus loss (enforces semantic structure)
                        if config.ULCD_USE_CONTRASTIVE:
                            contrastive_loss = self.compute_contrastive_consensus_loss(
                                x_align, y_align, server_prototypes
                            )
                            loss += config.CONTRASTIVE_WEIGHT * contrastive_loss
                            total_contrastive_loss += contrastive_loss.item()

                optimizer.zero_grad()

                # Check for invalid loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Client {self.client_id}: WARNING - Invalid loss={loss.item()}, skipping batch")
                    continue

                # Backward pass with gradient scaling if AMP is enabled
                if config.USE_AMP:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Check for NaN in model parameters after update
                params_valid = all(
                    not torch.isnan(p).any() and not torch.isinf(p).any()
                    for p in self.model.parameters()
                )
                if not params_valid:
                    print(f"Client {self.client_id}: ERROR - Model parameters became NaN/Inf after update!")
                    print(f"  Last loss: CE={ce_loss.item():.4f}, Proto={proto_align.item() if 'proto_align' in locals() else 0:.4f}")
                    raise RuntimeError("Model parameters became NaN/Inf")

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_ce = total_ce_loss / num_batches
            avg_proto = total_proto_loss / num_batches if server_prototypes else 0
            avg_contrastive = total_contrastive_loss / num_batches if server_prototypes and config.ULCD_USE_CONTRASTIVE else 0

            print(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, Proto={avg_proto:.4f}, Contr={avg_contrastive:.4f})")
