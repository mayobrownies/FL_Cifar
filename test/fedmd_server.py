"""
FedMD Server - Averages logits to create consensus
"""
import torch
import numpy as np


class FedMDServer:
    """Server for FedMD (Federated Model Distillation)"""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.consensus_logits = None

    def clear(self):
        """Clear temporary state for new round"""
        pass

    def aggregate_logits(self, client_logits_list):
        """
        Aggregate client logits by averaging

        Args:
            client_logits_list: List of logit tensors [N, num_classes]
        """
        if not client_logits_list:
            return

        # Simple average of all client logits
        stacked_logits = torch.stack(client_logits_list, dim=0)  # [num_clients, N, num_classes]
        self.consensus_logits = stacked_logits.mean(dim=0)  # [N, num_classes]

        # Log statistics
        logit_norms = [torch.norm(logits).item() for logits in client_logits_list]
        consensus_norm = torch.norm(self.consensus_logits).item()

        print(f"[FedMD Server] Aggregated {len(client_logits_list)} clients")
        print(f"[FedMD Server] Client logit norms: mean={np.mean(logit_norms):.4f}, "
              f"std={np.std(logit_norms):.4f}")
        print(f"[FedMD Server] Consensus logit norm: {consensus_norm:.4f}")

    def broadcast(self):
        """Broadcast consensus logits to clients"""
        if self.consensus_logits is None:
            return None

        return self.consensus_logits.clone()

    def get_consensus_predictions(self):
        """Get class predictions from consensus logits"""
        if self.consensus_logits is None:
            return None

        return self.consensus_logits.argmax(dim=1)
