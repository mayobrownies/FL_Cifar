import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class FedMDClient:
    def __init__(self, model, train_loader, public_loader, client_id):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.client_id = client_id
        self.ce_loss = nn.CrossEntropyLoss()
        self.mae_loss = nn.L1Loss()

    def get_public_logits(self, alignment_data):
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for x in alignment_data:
                x = x.cuda()
                logits = self.model(x)
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)

    def logits_matching(self, alignment_data, consensus_logits, epochs, round_num):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            idx = 0
            for x_batch in alignment_data:
                batch_size = x_batch.size(0)
                x_batch = x_batch.cuda()
                consensus_batch = consensus_logits[idx:idx+batch_size].cuda()

                student_logits = self.model(x_batch)
                loss = self.mae_loss(student_logits, consensus_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                idx += batch_size

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if epoch == epochs - 1:
                print(f"Logits matching loss: {avg_loss:.4f}")

    def private_training(self, epochs, round_num):
        self.model.train()

        decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
        current_lr = config.LEARNING_RATE * decay_factor
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                x, y = x.cuda(), y.cuda()
                out = self.model(x)
                loss = self.ce_loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if epoch == epochs - 1:
                print(f"Private training loss: {avg_loss:.4f}")

    def train(self, logits_matching_epochs, private_training_epochs,
              alignment_data=None, consensus_logits=None, round_num=1):

        if round_num % config.LR_DECAY_STEP == 1:
            decay_factor = config.LR_DECAY_GAMMA ** (round_num // config.LR_DECAY_STEP)
            current_lr = config.LEARNING_RATE * decay_factor
            print(f"Client {self.client_id}: LR = {current_lr:.6f}")

        if alignment_data is not None and consensus_logits is not None:
            print(f"[Phase 1] Logits matching on alignment data")
            self.logits_matching(alignment_data, consensus_logits,
                               logits_matching_epochs, round_num)

        print(f"  [Phase 2] Private data training")
        self.private_training(private_training_epochs, round_num)
