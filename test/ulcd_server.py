import torch
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
from . import config


class ULCDServer:
    def __init__(self, latent_dim=None, num_classes=10, ema_momentum=None):
        self.latent_dim = latent_dim or config.FEATURE_DIM
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum or config.ULCD_EMA_MOMENTUM

        self.global_prototypes = {}
        self.initialized = False

    def clear(self):
        pass

    def aggregate_prototypes(self, client_summaries, round_num=1):
        if not client_summaries:
            return

        class_collections = {i: [] for i in range(self.num_classes)}
        class_weights = {i: [] for i in range(self.num_classes)}

        for client_protos, class_mask in client_summaries:
            for class_id, prototype in client_protos.items():
                class_collections[class_id].append(prototype)
                # Weigh classes using the client's class mask to prevent noisy or rare samples from drastically influencing global consensus
                weight = class_mask[class_id].item() if class_id < len(class_mask) else 1.0
                class_weights[class_id].append(weight)

        for class_id, prototypes in class_collections.items():
            if not prototypes:
                continue

            # Weighted average
            weights = torch.tensor(class_weights[class_id])
            weights = weights / weights.sum().clamp(min=1e-8)

            prototype_stack = torch.stack(prototypes)
            # Weighted sum
            weighted_mean = (prototype_stack * weights.unsqueeze(1)).sum(dim=0)

            if torch.isnan(weighted_mean).any() or torch.isinf(weighted_mean).any():
                print(f"Server: WARNING - Invalid weighted_mean for class {class_id}, skipping")
                continue

            # Blend weighted sum with the current momentum for stabilization
            if config.ULCD_USE_EMA:
                if class_id not in self.global_prototypes:
                    self.global_prototypes[class_id] = weighted_mean
                else:
                    alpha = self.ema_momentum
                    new_proto = alpha * weighted_mean + (1 - alpha) * self.global_prototypes[class_id]
                    if torch.isnan(new_proto).any() or torch.isinf(new_proto).any():
                        print(f"Server: WARNING - Invalid EMA update for class {class_id}, keeping old prototype")
                        continue
                    self.global_prototypes[class_id] = new_proto
            else:
                self.global_prototypes[class_id] = weighted_mean

        self.initialized = True

        prototype_norms = [torch.norm(p).item() for p in self.global_prototypes.values()]
        print(f"Server: Aggregated {len(client_summaries)} clients, "
              f"{len(self.global_prototypes)} classes with prototypes")
        print(f"Server: Prototype norms: mean={np.mean(prototype_norms):.4f}, "
              f"std={np.std(prototype_norms):.4f}")

    def broadcast(self):
        if not self.global_prototypes:
            return None, None
        return self.global_prototypes.copy(), None

    # def visualize_prototypes(self, round_num, output_dir="vis"):
    #     if not self.global_prototypes or len(self.global_prototypes) < 2:
    #         return

    #     os.makedirs(output_dir, exist_ok=True)

    #     class_ids = sorted(self.global_prototypes.keys())
    #     prototypes = torch.stack([self.global_prototypes[c] for c in class_ids])
    #     prototypes_np = prototypes.detach().cpu().numpy()

    #     if prototypes_np.shape[1] > 2:
    #         pca = PCA(n_components=2)
    #         protos_2d = pca.fit_transform(prototypes_np)
    #         var_explained = pca.explained_variance_ratio_
    #     else:
    #         protos_2d = prototypes_np
    #         var_explained = [1.0, 0.0]

    #     plt.figure(figsize=(10, 8))
    #     scatter = plt.scatter(protos_2d[:, 0], protos_2d[:, 1],
    #                          c=class_ids, cmap='tab10', s=200, alpha=0.7,
    #                          edgecolors='black', linewidth=2)

    #     for i, class_id in enumerate(class_ids):
    #         plt.annotate(f'C{class_id}', (protos_2d[i, 0], protos_2d[i, 1]),
    #                     fontsize=12, fontweight='bold',
    #                     ha='center', va='center')

    #     plt.colorbar(scatter, label='Class ID')
    #     plt.xlabel(f'PC1 ({var_explained[0]:.1%} var)')
    #     plt.ylabel(f'PC2 ({var_explained[1]:.1%} var)')
    #     plt.title(f'ULCD Global Prototypes - Round {round_num}')
    #     plt.grid(True, alpha=0.3)

    #     filename = f'{output_dir}/proto_pca_round{round_num}.png'
    #     plt.savefig(filename, dpi=150, bbox_inches='tight')
    #     plt.close()

    #     print(f"Server: Saved visualization: {filename}")