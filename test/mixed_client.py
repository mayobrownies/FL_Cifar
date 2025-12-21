import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from . import config

class MixedClient:
    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        # Use L1 like FedMDClient or KL Div
        self.distill_loss = nn.L1Loss() 

    def get_public_logits(self):
        """From FedMDClient: Get logits on public data for server"""
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.cuda()
                logits = self.model(x)
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)

    def compute_prototypes(self):
        """From ULCDClient: Get prototypes from private data"""
        self.model.eval()
        device = next(self.model.parameters()).device
        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                latents = self.model.get_features(x)
                
                # Simple outlier filtering
                latent_norms = torch.norm(latents, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)

                for i in range(len(latents)):
                    if outlier_mask[i]:
                        label = y[i].item()
                        class_latents[label].append(latents[i].cpu())
                        class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id]:
                class_stack = torch.stack(class_latents[class_id])
                # Robust mean if enough samples
                if len(class_latents[class_id]) > 10:
                    norms = torch.norm(class_stack, dim=1)
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms) - 2*k, largest=False)
                    if len(indices) > 2*k:
                        class_stack = class_stack[indices[k:-k]]
                
                class_prototypes[class_id] = class_stack.mean(dim=0)

        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask

    def train(self, epochs, server_prototypes=None, server_logits=None, round_num=1):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        # Progressive weights
        warmup = min(1.0, round_num / 20.0)
        proto_weight = config.PROTOTYPE_WEIGHT * warmup
        distill_weight = config.MIXED_DISTILL_WEIGHT * warmup

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"Client {self.client_id}: LR={current_lr:.6f}, P-Weight={proto_weight:.2f}, D-Weight={distill_weight:.2f}")

        # iterator for public data (features/images)
        public_iter = iter(self.public_loader)
        # We also need to cycle through server_logits if they exist
        public_idx = 0
        total_public = len(self.public_loader.dataset) if self.public_loader else 0

        for epoch in range(epochs):
            total_loss = 0
            total_ce = 0
            total_proto = 0
            total_distill = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                # 1. CE on Private Data
                out = self.model(x)
                ce_loss = self.ce_loss(out, y)
                loss = ce_loss
                total_ce += ce_loss.item()

                # 2. Alignment on Public Data
                if (server_prototypes is not None) or (server_logits is not None):
                    # Fetch Public Batch
                    try:
                        x_pub, y_pub = next(public_iter)
                    except StopIteration:
                        public_iter = iter(self.public_loader)
                        x_pub, y_pub = next(public_iter)
                        public_idx = 0 # Reset logit index
                    
                    x_pub = x_pub.cuda()
                    current_batch_size = x_pub.size(0)

                    # Forward pass on public
                    # Note: We need logits for distillation AND features for prototypes
                    # models.py forward(..., return_protos=True) gives (log_probs, protos)
                    # but we need raw logits for MAE loss usually, let's just get features and logits separately
                    # or assume forward returns logits, and get_features returns features.
                    
                    pub_logits = self.model(x)
                    pub_logits = self.model(x_pub)
                    pub_feats = self.model.get_features(x_pub)

                    # A. Logit Distillation (FedMD)
                    if server_logits is not None and config.MIXED_USE_LOGITS:
                        # Slice the correct consensus logits corresponding to this batch
                        # Assumes public_loader is Shuffle=False and deterministic
                        if public_idx + current_batch_size <= server_logits.size(0):
                            batch_server_logits = server_logits[public_idx : public_idx + current_batch_size].cuda()
                            d_loss = self.distill_loss(pub_logits, batch_server_logits)
                            loss += distill_weight * d_loss
                            total_distill += d_loss.item()
                        
                        public_idx = (public_idx + current_batch_size) % total_public

                    # B. Prototype Alignment (ULCD)
                    if server_prototypes is not None:
                        p_loss = 0.0
                        valid = 0
                        for i in range(len(x_pub)):
                            label = y_pub[i].item() # Use public labels
                            if label in server_prototypes:
                                proto = server_prototypes[label].cuda()
                                sim = F.cosine_similarity(pub_feats[i], proto, dim=0)
                                p_loss += (1 - sim)
                                valid += 1
                        
                        if valid > 0:
                            p_loss /= valid
                            loss += proto_weight * p_loss
                            total_proto += p_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1