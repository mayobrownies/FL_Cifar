"""
FedMD Client - Pure knowledge distillation via public dataset
No prototype alignment, only logit distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class FedMDClient:
    """Client for FedMD (Federated Model Distillation)"""

    def __init__(self, model, train_loader, public_loader, client_id, num_classes=10):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

        # FedMD uses KL divergence for distillation
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def get_public_logits(self):
        """
        Compute logits on public dataset for server aggregation

        Returns:
            torch.Tensor: Logits on public data [N, num_classes]
        """
        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for x, _ in self.public_loader:
                x = x.cuda()
                logits = self.model(x)
                all_logits.append(logits.cpu())

        return torch.cat(all_logits, dim=0)

    def train(self, epochs, consensus_logits=None, round_num=1):
        """
        Train with FedMD distillation

        Args:
            epochs: Number of training epochs
            consensus_logits: Server consensus logits on public data
            round_num: Current round number for learning rate decay
        """
        self.model.train()

        # Learning rate with decay based on round number
        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor

        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        if round_num % config.LR_DECAY_STEP == 1:
            print(f"  Client {self.client_id}: LR = {current_lr:.6f} (decay factor: {decay_factor:.2f})")

        for epoch in range(epochs):
            # Phase 1: Train on private data (supervised)
            total_ce_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.ce_loss(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_ce_loss += loss.item()
                num_batches += 1

            avg_ce = total_ce_loss / num_batches

            # Phase 2: Distillation on public data (if consensus available)
            total_distill_loss = 0
            distill_batches = 0

            if consensus_logits is not None:
                public_data = list(self.public_loader)
                batch_size = public_data[0][0].size(0)
                start_idx = 0

                for x, _ in public_data:
                    x = x.cuda()
                    end_idx = start_idx + x.size(0)

                    # Get consensus for this batch
                    consensus_batch = consensus_logits[start_idx:end_idx].cuda()

                    # Student predictions
                    student_logits = self.model(x)

                    # KL divergence distillation
                    # Apply temperature scaling
                    T = config.ULCD_TEMPERATURE
                    student_soft = F.log_softmax(student_logits / T, dim=1)
                    teacher_soft = F.softmax(consensus_batch / T, dim=1)

                    distill_loss = self.kl_loss(student_soft, teacher_soft) * (T * T)

                    optimizer.zero_grad()
                    distill_loss.backward()
                    optimizer.step()

                    total_distill_loss += distill_loss.item()
                    distill_batches += 1
                    start_idx = end_idx

            avg_distill = total_distill_loss / distill_batches if distill_batches > 0 else 0

            print(f"  Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                  f"CE Loss={avg_ce:.4f}, Distill Loss={avg_distill:.4f}")
