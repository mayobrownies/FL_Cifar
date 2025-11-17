"""
ULCD Client with prototype-guided training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import config


class ULCDClient:
    """Client for ULCD-based federated learning"""

    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def get_logits_on_public(self):
        """Get model logits on public dataset (for distillation)"""
        self.model.eval()
        logits = []
        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.cuda()
                out = self.model(x)
                logits.append(out.cpu())
        return torch.cat(logits)

    def compute_prototypes(self):
        """
        Compute per-class prototypes from local training data

        Returns:
            dict: {class_id: prototype_tensor}
        """
        return self.model.get_latent_summary(self.train_loader)

    def train(self, epochs, server_prototypes=None, server_logits=None, round_num=1):
        """
        Train with ULCD prototype guidance

        Args:
            epochs: Number of training epochs
            server_prototypes: Global per-class prototypes from server
            server_logits: Server logits on public data (for distillation)
            round_num: Current round number for progressive distillation
        """
        self.model.train()

        # Learning rate with decay based on round number
        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor

        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f} (decay factor: {decay_factor:.2f})")

        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_proto_loss = 0
            total_distill_loss = 0
            num_batches = 0
            loss_explosion_count = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                # Forward pass
                out = self.model(x)
                feats = self.model.get_features(x)

                # 1. Classification loss
                ce_loss = self.ce_loss(out, y)
                loss = ce_loss
                total_ce_loss += ce_loss.item()

                # 2. Prototype alignment loss (ULCD consensus)
                if server_prototypes is not None:
                    proto_align = 0.0
                    valid_count = 0

                    for i in range(len(x)):
                        true_label = y[i].item()
                        if true_label in server_prototypes:
                            client_feat = feats[i]
                            proto = server_prototypes[true_label].cuda()

                            # Cosine similarity alignment
                            sim = F.cosine_similarity(client_feat, proto, dim=0)
                            proto_align += (1 - sim)  # Encourage alignment
                            valid_count += 1

                    if valid_count > 0:
                        proto_align /= valid_count
                        loss += config.PROTOTYPE_WEIGHT * proto_align
                        total_proto_loss += proto_align.item()

                # 3. Knowledge distillation loss with progressive weighting
                if server_logits is not None and config.ULCD_ENABLE_DISTILLATION:
                    with torch.no_grad():
                        server_logit = server_logits[:len(x)].cuda()

                    # Progressive distillation weight (starts small, increases)
                    progressive_weight = min(0.02 * round_num, 0.1)
                    distill_loss = F.mse_loss(out, server_logit)
                    loss += progressive_weight * distill_loss
                    total_distill_loss += distill_loss.item()

                # Loss explosion protection
                if loss.item() > config.ULCD_MAX_LOSS:
                    loss_explosion_count += 1
                    # Fallback to classification only
                    loss = ce_loss
                    print(f"  [WARNING] Loss explosion detected ({loss.item():.2f}), "
                          f"falling back to classification only")

                # Backpropagation with gradient clipping
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=config.ULCD_GRADIENT_CLIP
                )

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Epoch summary
            avg_loss = total_loss / num_batches
            avg_ce = total_ce_loss / num_batches
            avg_proto = total_proto_loss / num_batches if server_prototypes else 0
            avg_distill = total_distill_loss / num_batches if server_logits else 0

            print(f"  Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                  f"Loss={avg_loss:.4f} (CE={avg_ce:.4f}, Proto={avg_proto:.4f}, "
                  f"Distill={avg_distill:.4f}) [Explosions: {loss_explosion_count}]")
