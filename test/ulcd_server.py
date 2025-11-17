"""
ULCD Server for per-class prototype aggregation with consensus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
import config


class ULCDServer:
    """Server for ULCD consensus-based federated learning"""

    def __init__(self, latent_dim=None, num_classes=10, ema_momentum=None):
        self.latent_dim = latent_dim or config.ULCD_LATENT_DIM
        self.num_classes = num_classes
        self.ema_momentum = ema_momentum or config.ULCD_EMA_MOMENTUM

        # Global per-class prototypes
        self.global_prototypes = {}

        # Client trust scores
        self.client_trust_scores = {}

        # Distillation network (teacher model)
        self.teacher_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )

        self.initialized = False

    def clear(self):
        """Clear temporary state for new round"""
        pass

    def aggregate_prototypes(self, client_summaries, round_num=1):
        """
        Aggregate per-class prototypes with class-aware weighting and diversity regularization

        Args:
            client_summaries: List of tuples (client_protos_dict, class_mask)
            round_num: Current round number for adaptive update
        """
        if not client_summaries:
            return

        # Collect all prototypes by class with weights
        class_collections = {i: [] for i in range(self.num_classes)}
        class_weights = {i: [] for i in range(self.num_classes)}

        for client_protos, class_mask in client_summaries:
            for class_id, prototype in client_protos.items():
                class_collections[class_id].append(prototype)
                # Use class mask as weight (higher weight for classes with more samples)
                weight = class_mask[class_id].item() if class_id < len(class_mask) else 1.0
                class_weights[class_id].append(weight)

        # Aggregate each class
        for class_id, prototypes in class_collections.items():
            if not prototypes:
                continue

            # Weighted mean of client prototypes
            weights = torch.tensor(class_weights[class_id])
            weights = weights / weights.sum().clamp(min=1e-8)

            prototype_stack = torch.stack(prototypes)
            weighted_mean = (prototype_stack * weights.unsqueeze(1)).sum(dim=0)

            # Apply distillation if enabled
            if config.ULCD_ENABLE_DISTILLATION:
                with torch.no_grad():
                    device = next(self.teacher_network.parameters()).device
                    distilled = self.teacher_network(weighted_mean.to(device))
                    distilled = distilled.cpu()
            else:
                distilled = weighted_mean

            # Adaptive update rate based on round
            base_alpha = config.ULCD_BASE_ALPHA
            round_factor = max(0.5, 1.0 - round_num * 0.05)
            alpha = base_alpha * round_factor

            # EMA update or initialize
            if class_id not in self.global_prototypes:
                self.global_prototypes[class_id] = distilled
            else:
                # EMA with adaptive rate
                self.global_prototypes[class_id] = (
                    alpha * distilled +
                    (1 - alpha) * self.global_prototypes[class_id]
                )

                # Diversity regularization - blend with simple mean
                simple_mean = prototype_stack.mean(dim=0)
                diversity_weight = config.ULCD_DIVERSITY_WEIGHT
                self.global_prototypes[class_id] = (
                    (1 - diversity_weight) * self.global_prototypes[class_id] +
                    diversity_weight * simple_mean
                )

        self.initialized = True

        # Log statistics
        prototype_norms = [torch.norm(p).item() for p in self.global_prototypes.values()]
        print(f"[ULCD Server] Aggregated {len(client_summaries)} clients, "
              f"{len(self.global_prototypes)} classes with prototypes")
        print(f"[ULCD Server] Prototype norms: mean={np.mean(prototype_norms):.4f}, "
              f"std={np.std(prototype_norms):.4f}")

    def broadcast(self):
        """Broadcast global prototypes to clients"""
        if not self.global_prototypes:
            return None, None

        return self.global_prototypes.copy(), None

    def detect_anomalies(self, client_summaries, threshold=None):
        """
        Detect anomalous clients using similarity + coverage scores

        Args:
            client_summaries: List of tuples (client_protos_dict, class_mask)
            threshold: Trust score threshold

        Returns:
            trusted_clients: List of client indices
            flagged_clients: List of anomalous client indices
        """
        threshold = threshold or config.ULCD_ANOMALY_THRESHOLD

        if len(client_summaries) <= 1 or not self.global_prototypes:
            return list(range(len(client_summaries))), []

        trust_scores = []

        for client_id, (client_protos, class_mask) in enumerate(client_summaries):
            if not client_protos:
                trust_scores.append(0.0)
                continue

            # Coverage score - how many classes does client have?
            coverage_score = class_mask.sum().item() / self.num_classes

            # Similarity score - how similar to global prototypes?
            similarities = []
            for class_id, proto in client_protos.items():
                if class_id in self.global_prototypes:
                    sim = F.cosine_similarity(
                        proto.unsqueeze(0),
                        self.global_prototypes[class_id].unsqueeze(0)
                    ).item()
                    similarities.append(sim)

            if similarities:
                similarity_score = np.mean(similarities)
            else:
                similarity_score = 0.0

            # Combined trust score: 60% similarity + 40% coverage
            trust_score = 0.6 * similarity_score + 0.4 * coverage_score

            trust_scores.append(trust_score)
            self.client_trust_scores[client_id] = trust_score

        # Identify trusted vs flagged
        trusted = [i for i, score in enumerate(trust_scores) if score >= threshold]
        flagged = [i for i, score in enumerate(trust_scores) if score < threshold]

        # Ensure at least 1 client is trusted
        if not trusted and client_prototypes_list:
            best_client = np.argmax(trust_scores)
            trusted = [best_client]
            flagged = [i for i in range(len(client_prototypes_list)) if i != best_client]

        print(f"[ULCD Server] Anomaly detection: {len(trusted)} trusted, {len(flagged)} flagged")
        if trust_scores:
            print(f"[ULCD Server] Trust scores: mean={np.mean(trust_scores):.3f}, "
                  f"min={np.min(trust_scores):.3f}, max={np.max(trust_scores):.3f}")

        return trusted, flagged

    def visualize_prototypes(self, round_num, output_dir="vis"):
        """Visualize global prototypes using PCA"""
        if not self.global_prototypes or len(self.global_prototypes) < 2:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Stack prototypes
        class_ids = sorted(self.global_prototypes.keys())
        prototypes = torch.stack([self.global_prototypes[c] for c in class_ids])
        prototypes_np = prototypes.detach().cpu().numpy()

        # Apply PCA
        if prototypes_np.shape[1] > 2:
            pca = PCA(n_components=2)
            protos_2d = pca.fit_transform(prototypes_np)
            var_explained = pca.explained_variance_ratio_
        else:
            protos_2d = prototypes_np
            var_explained = [1.0, 0.0]

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(protos_2d[:, 0], protos_2d[:, 1],
                             c=class_ids, cmap='tab10', s=200, alpha=0.7,
                             edgecolors='black', linewidth=2)

        # Add class labels
        for i, class_id in enumerate(class_ids):
            plt.annotate(f'C{class_id}', (protos_2d[i, 0], protos_2d[i, 1]),
                        fontsize=12, fontweight='bold',
                        ha='center', va='center')

        plt.colorbar(scatter, label='Class ID')
        plt.xlabel(f'PC1 ({var_explained[0]:.1%} var)')
        plt.ylabel(f'PC2 ({var_explained[1]:.1%} var)')
        plt.title(f'ULCD Global Prototypes - Round {round_num}')
        plt.grid(True, alpha=0.3)

        filename = f'{output_dir}/proto_pca_round{round_num}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[ULCD Server] Saved visualization: {filename}")
