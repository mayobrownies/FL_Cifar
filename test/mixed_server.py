import torch
import numpy as np
from . import config

class MixedServer:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.ema_momentum = config.ULCD_EMA_MOMENTUM
        
        self.global_prototypes = {}
        self.consensus_logits = None

    def aggregate(self, client_summaries, client_logits_list):
        """
        client_summaries: list of (prototypes, class_mask) from ULCD
        client_logits_list: list of tensors from FedMD
        """
        # 1. Aggregate Prototypes (ULCD Style)
        if client_summaries:
            class_collections = {i: [] for i in range(self.num_classes)}
            class_weights = {i: [] for i in range(self.num_classes)}

            for client_protos, class_mask in client_summaries:
                for class_id, prototype in client_protos.items():
                    class_collections[class_id].append(prototype)
                    w = class_mask[class_id].item() if class_id < len(class_mask) else 1.0
                    class_weights[class_id].append(w)

            for class_id, prototypes in class_collections.items():
                if not prototypes: continue
                
                weights = torch.tensor(class_weights[class_id])
                weights = weights / weights.sum().clamp(min=1e-8)
                weighted_mean = (torch.stack(prototypes) * weights.unsqueeze(1)).sum(dim=0)

                if class_id not in self.global_prototypes:
                    self.global_prototypes[class_id] = weighted_mean
                else:
                    self.global_prototypes[class_id] = (
                        self.ema_momentum * weighted_mean +
                        (1 - self.ema_momentum) * self.global_prototypes[class_id]
                    )

        # 2. Aggregate Logits (FedMD Style)
        if client_logits_list:
            stacked = torch.stack(client_logits_list)
            self.consensus_logits = stacked.mean(dim=0)

    def broadcast(self):
        protos = self.global_prototypes.copy() if self.global_prototypes else None
        logits = self.consensus_logits.clone() if self.consensus_logits is not None else None
        return protos, logits