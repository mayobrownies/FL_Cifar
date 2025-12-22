import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class FedMDServer:
    def __init__(self, num_classes=10):
        self.consensus_logits = None
        self.num_classes = num_classes

    def generate_alignment_data(self, public_dataset, N_alignment, batch_size):
        X_public, y_public = public_dataset

        total_samples = len(X_public)
        if N_alignment > total_samples:
            N_alignment = total_samples

        indices = np.random.choice(total_samples, N_alignment, replace=False)
        X_alignment = X_public[indices]
        y_alignment = y_public[indices]

        dataset = TensorDataset(X_alignment, y_alignment)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        alignment_batches = []
        for x_batch, _ in dataloader:
            alignment_batches.append(x_batch)

        return alignment_batches, X_alignment, y_alignment

    def aggregate_logits(self, client_logits_list):
        if len(client_logits_list) == 0:
            return None
        stacked_logits = torch.stack(client_logits_list, dim=0)
        self.consensus_logits = stacked_logits.mean(dim=0)

    def broadcast(self):
        return self.consensus_logits
