# ULCD (Unified Latent Consensus Distillation) Components for CIFAR-10
# Adapted for image classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ULCD SERVER FOR CIFAR-10
# ============================================================================
class ULCDServer_CIFAR(nn.Module):
    """ULCD Server for federated learning with CIFAR-10"""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.prototype = nn.Parameter(torch.randn(latent_dim))

    def aggregate_latents(self, latents: List[torch.Tensor]):
        """Aggregate latent representations into prototype"""
        if latents:
            # Stack and compute mean
            latent_stack = torch.stack(latents)
            self.prototype.data = latent_stack.mean(dim=0)
            
            # Add some regularization to prevent collapse
            noise = torch.randn_like(self.prototype) * 0.01
            self.prototype.data += noise
            
            print(f"[ULCD Server] Aggregated {len(latents)} latent summaries")
            print(f"[ULCD Server] Prototype norm: {torch.norm(self.prototype):.4f}")
        else:
            print("[WARNING] No latents to aggregate")

    def detect_anomalies(self, latents: List[torch.Tensor], threshold=0.3):
        """Detect anomalous clients using latent similarity clustering"""
        trusted, flagged = [], []
        
        if len(latents) <= 1:
            # If only one client, trust it
            trusted = [(i, latent) for i, latent in enumerate(latents)]
            return trusted, flagged
        
        # Check if prototype is still in initial random state (first round)
        prototype_norm = torch.norm(self.prototype)
        if prototype_norm < 1e-3:
            # First round - trust all clients
            trusted = [(i, latent) for i, latent in enumerate(latents)]
            print(f"[ULCD] First round detected - trusting all {len(trusted)} clients")
            return trusted, flagged
        
        # Calculate similarities to current prototype
        similarities = []
        for i, latent in enumerate(latents):
            # Cosine similarity
            similarity = F.cosine_similarity(
                latent.unsqueeze(0), 
                self.prototype.unsqueeze(0)
            ).item()
            similarities.append((i, latent, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Adaptive thresholding based on distribution
        if len(similarities) >= 3:
            sim_values = [s[2] for s in similarities]
            mean_sim = np.mean(sim_values)
            std_sim = np.std(sim_values)
            
            # Use mean - std as threshold (more inclusive)
            adaptive_threshold = max(threshold, mean_sim - std_sim)
        else:
            adaptive_threshold = threshold
        
        # Classify clients
        for i, latent, similarity in similarities:
            if similarity >= adaptive_threshold:
                trusted.append((i, latent))
            else:
                flagged.append(i)
        
        # Ensure at least one client is trusted (even if all are flagged)
        if not trusted and similarities:
            best_client = similarities[0]
            trusted.append((best_client[0], best_client[1]))
            if best_client[0] in flagged:
                flagged.remove(best_client[0])
        
        print(f"[ULCD] Anomaly detection: {len(trusted)} trusted, {len(flagged)} flagged")
        print(f"[ULCD] Threshold used: {adaptive_threshold:.4f}")
        
        return trusted, flagged

    def get_prototype_guidance(self):
        """Get current prototype for client guidance"""
        return self.prototype.detach().clone()

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================
def visualize_latents_cifar(latent_summaries: List[Tuple[int, torch.Tensor]], 
                           prototype: torch.Tensor = None, 
                           title: str = "CIFAR-10 Latent Space"):
    """Visualize latent representations using PCA"""
    
    if len(latent_summaries) < 2:
        print("Not enough latent summaries to visualize")
        return
    
    # Extract latents
    client_ids = [client_id for client_id, _ in latent_summaries]
    latents = [latent.detach().cpu().numpy() for _, latent in latent_summaries]
    
    # Stack latents
    latent_matrix = np.stack(latents)  # [num_clients, latent_dim]
    
    # Add prototype if available
    if prototype is not None:
        prototype_np = prototype.detach().cpu().numpy()
        latent_matrix = np.vstack([latent_matrix, prototype_np.reshape(1, -1)])
        client_ids = client_ids + ["Prototype"]
    
    # Apply PCA for visualization
    if latent_matrix.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_matrix)
        explained_var = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    else:
        latent_2d = latent_matrix
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot client latents
    for i, (client_id, (x, y)) in enumerate(zip(client_ids[:-1], latent_2d[:-1])):
        plt.scatter(x, y, label=f'Client {client_id}', s=100, alpha=0.7)
    
    # Plot prototype if available
    if prototype is not None:
        proto_x, proto_y = latent_2d[-1]
        plt.scatter(proto_x, proto_y, label='Prototype', s=200, marker='*', 
                   color='red', edgecolor='black', linewidth=2)
    
    plt.xlabel('PC1' if latent_matrix.shape[1] > 2 else 'Latent Dim 1')
    plt.ylabel('PC2' if latent_matrix.shape[1] > 2 else 'Latent Dim 2')
    plt.title(f'{title}\nCIFAR-10 Federated Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('fl_plots', exist_ok=True)
    filename = f'fl_plots/{title.replace(" ", "_").lower()}_cifar10.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Saved latent visualization: {filename}")

def analyze_latent_statistics(latent_summaries: List[Tuple[int, torch.Tensor]], 
                             prototype: torch.Tensor = None):
    """Analyze statistics of latent representations"""
    
    if not latent_summaries:
        print("No latent summaries to analyze")
        return
    
    print(f"\n{'='*60}")
    print("LATENT SPACE ANALYSIS")
    print(f"{'='*60}")
    
    # Extract latents
    latents = [latent.detach().cpu().numpy() for _, latent in latent_summaries]
    client_ids = [client_id for client_id, _ in latent_summaries]
    
    # Stack latents
    latent_matrix = np.stack(latents)  # [num_clients, latent_dim]
    
    # Calculate statistics
    mean_latent = np.mean(latent_matrix, axis=0)
    std_latent = np.std(latent_matrix, axis=0)
    
    print(f"Number of clients: {len(latents)}")
    print(f"Latent dimension: {latent_matrix.shape[1]}")
    print(f"Mean latent norm: {np.linalg.norm(mean_latent):.4f}")
    print(f"Std latent norm: {np.linalg.norm(std_latent):.4f}")
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            # Cosine similarity
            sim = np.dot(latents[i], latents[j]) / (
                np.linalg.norm(latents[i]) * np.linalg.norm(latents[j]) + 1e-8
            )
            similarities.append(sim)
    
    if similarities:
        print(f"Pairwise cosine similarities:")
        print(f"  Mean: {np.mean(similarities):.4f}")
        print(f"  Std: {np.std(similarities):.4f}")
        print(f"  Min: {np.min(similarities):.4f}")
        print(f"  Max: {np.max(similarities):.4f}")
    
    # Compare with prototype if available
    if prototype is not None:
        prototype_np = prototype.detach().cpu().numpy()
        proto_similarities = []
        
        for i, latent in enumerate(latents):
            sim = np.dot(latent, prototype_np) / (
                np.linalg.norm(latent) * np.linalg.norm(prototype_np) + 1e-8
            )
            proto_similarities.append(sim)
            print(f"  Client {client_ids[i]} -> Prototype: {sim:.4f}")
        
        print(f"Prototype similarities:")
        print(f"  Mean: {np.mean(proto_similarities):.4f}")
        print(f"  Std: {np.std(proto_similarities):.4f}")
    
    print(f"{'='*60}")

# ============================================================================
# ULCD TRAINING UTILITIES FOR CIFAR-10
# ============================================================================
def compute_prototype_alignment_loss(model, dataloader, prototype, device, alpha=0.1):
    """Compute alignment loss between model latents and prototype"""
    
    if not hasattr(model, 'image_encoder'):
        return 0.0
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            
            # Flatten if needed
            if len(features.shape) == 4 and hasattr(model, 'image_encoder'):
                features = features.view(features.size(0), -1)
            
            try:
                # Get latent representation
                latent = model.image_encoder(features)
                batch_mean_latent = latent.mean(dim=0)
                
                # Compute MSE loss with prototype
                loss = F.mse_loss(batch_mean_latent, prototype)
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"[WARNING] Failed to compute prototype loss: {e}")
                break
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        return alpha * avg_loss
    else:
        return 0.0

def extract_client_latent_summary(model, dataloader, device):
    """Extract latent summary from client data for ULCD"""
    
    if hasattr(model, 'get_latent_summary'):
        # Use model's built-in method
        return model.get_latent_summary(dataloader)
    
    # Fallback method
    if not hasattr(model, 'image_encoder'):
        # Non-ULCD model - create dummy summary
        return torch.randn(64)  # Default latent dim
    
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            
            # Flatten if needed
            if len(features.shape) == 4:
                features = features.view(features.size(0), -1)
            
            try:
                latent = model.image_encoder(features)
                batch_latent = latent.mean(dim=0)
                latent_vectors.append(batch_latent)
            except Exception as e:
                print(f"[WARNING] Failed to extract latent: {e}")
                break
    
    if latent_vectors:
        client_latent = torch.stack(latent_vectors).mean(dim=0)
        return client_latent
    else:
        return torch.zeros(64)  # Default latent dim

# ============================================================================
# CIFAR-10 SPECIFIC ANALYSIS
# ============================================================================
def analyze_cifar_class_distribution_in_latents(latent_summaries: List[Tuple[int, torch.Tensor]], 
                                               partitions: Dict, prototype: torch.Tensor = None):
    """Analyze how CIFAR-10 class distributions affect latent representations"""
    
    if not partitions:
        print("No partition data available for analysis")
        return
    
    print(f"\n{'='*60}")
    print("CIFAR-10 CLASS DISTRIBUTION vs LATENT ANALYSIS")
    print(f"{'='*60}")
    
    from .data_utils import CIFAR10_CLASSES
    
    for client_id, latent in latent_summaries:
        if client_id in partitions:
            _, labels = partitions[client_id]
            
            # Calculate class distribution
            unique_classes, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            
            # Calculate diversity metrics
            total_samples = len(labels)
            entropy = -np.sum([(count/total_samples) * np.log2(count/total_samples + 1e-10) 
                              for count in counts])
            
            # Calculate latent norm and prototype similarity
            latent_norm = torch.norm(latent).item()
            
            if prototype is not None:
                proto_sim = F.cosine_similarity(
                    latent.unsqueeze(0), 
                    prototype.unsqueeze(0)
                ).item()
            else:
                proto_sim = 0.0
            
            print(f"\nClient {client_id}:")
            print(f"  Total samples: {total_samples}")
            print(f"  Classes present: {len(unique_classes)}/10")
            print(f"  Entropy: {entropy:.3f} (max: 3.32)")
            print(f"  Latent norm: {latent_norm:.4f}")
            print(f"  Prototype similarity: {proto_sim:.4f}")
            
            # Show class distribution
            class_names = [CIFAR10_CLASSES[i] for i in unique_classes]
            percentages = [(count/total_samples)*100 for count in counts]
            class_info = list(zip(class_names, percentages))
            class_info.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top classes: {class_info[:3]}")
    
    print(f"{'='*60}")

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================
def compare_ulcd_vs_traditional_fl(ulcd_results: Dict, traditional_results: Dict):
    """Compare ULCD FL performance with traditional FL"""
    
    print(f"\n{'='*60}")
    print("ULCD vs TRADITIONAL FL COMPARISON")
    print(f"{'='*60}")
    
    metrics_to_compare = ['accuracy', 'f1', 'top5_accuracy']
    
    for metric in metrics_to_compare:
        if metric in ulcd_results and metric in traditional_results:
            ulcd_value = ulcd_results[metric]
            trad_value = traditional_results[metric]
            improvement = ((ulcd_value - trad_value) / trad_value) * 100 if trad_value > 0 else 0
            
            print(f"{metric.upper()}:")
            print(f"  ULCD FL: {ulcd_value:.4f}")
            print(f"  Traditional FL: {trad_value:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            print()
    
    print(f"{'='*60}")

def save_cifar_fl_results(results: Dict, experiment_name: str = "cifar10_fl"):
    """Save FL results for CIFAR-10 experiments"""
    
    import json
    from datetime import datetime
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    results['experiment'] = experiment_name
    
    # Create results directory
    os.makedirs('fl_results', exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'fl_results/{experiment_name}_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[RESULTS] Saved experiment results: {filename}")
    
    return filename