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
    """Enhanced ULCD Server with knowledge distillation and non-IID handling"""
    
    def __init__(self, latent_dim=32, num_classes=10, num_subspaces=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_subspaces = num_subspaces

        # Simplified global prototype - single vector for all classes
        self.global_prototype = nn.Parameter(
            torch.randn(latent_dim) * 0.01  # [latent_dim] for simplified ULCD
        )
        
        # Knowledge distillation network (teacher model) - simplified
        self.teacher_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
        # Client trust scores
        self.client_trust_scores = {}
        self.initialized = False  # Track if prototype is initialized from real data

    def distill_knowledge(self, client_summaries: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Distill knowledge from client summaries - simplified for single vector"""

        if not client_summaries:
            return None

        distilled_knowledge = []

        for summary, class_mask in client_summaries:
            # Simplified: summary is [latent_dim]
            if summary.dim() == 1 and summary.size(0) == self.latent_dim:
                distilled = self.teacher_network(summary)
                distilled_knowledge.append(distilled)
            else:
                print(f"[WARNING] Unexpected summary size: {summary.size()}, expected ({self.latent_dim},)")
                continue

        return distilled_knowledge
    
    def aggregate_with_distillation(self, client_summaries: List[Tuple[int, torch.Tensor, torch.Tensor]], 
                                   round_num: int):
        """Aggregate using distillation and class-aware weighting"""
        
        if not client_summaries:
            return
        
        # Initialize on first round
        if not self.initialized:
            summaries_only = [summary for _, summary, _ in client_summaries]
            if summaries_only:
                with torch.no_grad():
                    # Stack and mean - should be [latent_dim]
                    mean_summary = torch.stack(summaries_only).mean(0)
                    self.global_prototype.data = mean_summary
                    self.initialized = True
                    print(f"[Enhanced ULCD Server] Prototype initialized from {len(summaries_only)} clients, shape: {self.global_prototype.shape}")
            return
        
        # Separate summaries and class masks
        summaries_with_masks = [(summary, mask) for _, summary, mask in client_summaries]
        
        # Distill knowledge
        distilled = self.distill_knowledge(summaries_with_masks)

        # Simplified aggregation - single global prototype
        if distilled:
            stacked_distilled = torch.stack(distilled)  # [num_clients, latent_dim]
            mean_distilled = stacked_distilled.mean(0)  # [latent_dim]

            # Adaptive update rate
            base_alpha = 0.2
            round_factor = max(0.5, 1.0 - round_num * 0.05)
            alpha = base_alpha * round_factor

            # Update global prototype with EMA
            with torch.no_grad():
                self.global_prototype.data = alpha * mean_distilled + (1 - alpha) * self.global_prototype.data

        # Simple mean aggregation as fallback
        all_summaries = torch.stack([s for _, s, _ in client_summaries])  # [num_clients, latent_dim]
        mean_summary = all_summaries.mean(0)  # [latent_dim]

        # Diversity regularization
        diversity_weight = 0.1
        with torch.no_grad():
            self.global_prototype.data = (1 - diversity_weight) * self.global_prototype.data + diversity_weight * mean_summary

        print(f"[ULCD Server] Aggregated {len(client_summaries)} summaries")
        print(f"[ULCD Server] Global prototype shape: {self.global_prototype.shape}, norm: {torch.norm(self.global_prototype):.4f}")
    
    def detect_anomalies(self, client_summaries, threshold=0.2):
        """Enhanced anomaly detection using distilled representations"""

        trusted = []
        flagged = []
        
        if len(client_summaries) <= 1:
            return [(i, s, m) for i, s, m in client_summaries], []
        
        # Extract distilled components and compute similarities
        for client_id, summary, class_mask in client_summaries:
            # Get class coverage score
            coverage_score = class_mask.sum() / self.num_classes

            # Get similarity to global prototype - both should be [latent_dim]
            similarity = F.cosine_similarity(
                summary.unsqueeze(0),
                self.global_prototype.unsqueeze(0)
            ).item()
            
            # Combined trust score
            trust_score = 0.6 * similarity + 0.4 * coverage_score.item()
            
            # Store trust score
            self.client_trust_scores[client_id] = trust_score
            
            if trust_score >= threshold:
                trusted.append((client_id, summary, class_mask))
            else:
                flagged.append(client_id)

        # Ensure at least 2 clients are trusted for better consensus
        if len(trusted) < 2 and len(client_summaries) >= 2:
            # Trust the top 2 clients by score
            sorted_clients = sorted(client_summaries,
                                  key=lambda x: self.client_trust_scores.get(x[0], 0),
                                  reverse=True)
            trusted = sorted_clients[:2]
            flagged = [c[0] for c in sorted_clients[2:]]
        elif not trusted and client_summaries:
            # Fallback: trust best client
            best_client = max(client_summaries,
                            key=lambda x: self.client_trust_scores.get(x[0], 0))
            trusted.append(best_client)
            if best_client[0] in flagged:
                flagged.remove(best_client[0])
        
        print(f"[Enhanced ULCD] Anomaly detection: {len(trusted)} trusted, {len(flagged)} flagged")
        
        return trusted, flagged
    
    def get_prototype_for_client(self, client_classes: List[int]):
        """Get simplified prototype for client guidance"""

        # Simplified: just return the global prototype (single vector)
        return self.global_prototype.detach().clone()
    
    def get_prototype_guidance(self):
        """Get current prototype for client guidance"""
        return self.global_prototype.detach().clone()

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
        # Use model's built-in method (preferred for ULCD models and compatible CNNs)
        return model.get_latent_summary(dataloader)
    
    # Fallback method for models with image_encoder
    if hasattr(model, 'image_encoder'):
        model.eval()
        latent_vectors = []
        
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(device)
                
                # Determine if we need to flatten based on model type
                from shared.models import get_model_type
                model_type = get_model_type(model)
                
                if model_type in ["ulcd", "logistic", "mlp", "lstm", "moe"]:
                    # Flatten for non-convolutional models
                    if len(features.shape) == 4:
                        features = features.view(features.size(0), -1)
                # CNN-based models (including cnn_ulcd) keep 4D shape
                
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
            # Use the model's latent dimension if available
            latent_dim = getattr(model, 'latent_dim', 16)
            return torch.zeros(latent_dim)
    
    # Non-ULCD model - create dummy summary
    print(f"[WARNING] Model has no latent extraction capability, returning dummy summary")
    # Use a default that matches our current configuration
    return torch.randn(16)

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

def save_metrics_to_txt(results: Dict, experiment_name: str = "cifar10_fl"):
    """Save FL metrics summary to a readable txt file"""
    
    from datetime import datetime
    import pandas as pd
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs('fl_plots', exist_ok=True)
    
    # Create txt filename
    txt_filename = f'fl_plots/{experiment_name}_{timestamp}_metrics.txt'
    
    with open(txt_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIFAR-10 FEDERATED LEARNING RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write("\n")
        
        # Configuration details
        if 'configuration' in results:
            config = results['configuration']
            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of Rounds: {config.get('num_rounds', 'N/A')}\n")
            f.write(f"Local Epochs: {config.get('local_epochs', 'N/A')}\n")
            f.write(f"Number of Clients: {config.get('num_clients', 'N/A')}\n")
            f.write(f"Partition Type: {config.get('partition_type', 'N/A')}\n")
            f.write(f"Models Tested: {', '.join(config.get('models_tested', []))}\n")
            f.write(f"Strategies Tested: {', '.join(config.get('strategies_tested', []))}\n")
            f.write("\n")
        
        # Results summary table
        if 'all_results' in results:
            all_results = results['all_results']
            successful_results = [r for r in all_results if r.get('success', False)]
            
            if successful_results:
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Strategy+Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Top-5 Acc':<10} {'Size (MB)':<10} {'Time (s)':<10}\n")
                f.write("-" * 90 + "\n")
                
                for result in successful_results:
                    experiment_id = f"{result['strategy']}_{result['model_name']}"
                    accuracy = result.get('accuracy', 0) * 100
                    f1 = result.get('f1', 0)
                    top5 = result.get('top5_accuracy', 0) * 100
                    model_size = result.get('model_size_mb', 0.0)
                    time_s = result.get('training_time', 0)
                    
                    f.write(f"{experiment_id:<20} {accuracy:<10.2f} {f1:<10.4f} {top5:<10.2f} {model_size:<10.2f} {time_s:<10.1f}\n")
                
                f.write("-" * 90 + "\n\n")
                
                # Efficiency Analysis Table
                f.write("EFFICIENCY ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Strategy+Model':<20} {'Size (MB)':<10} {'F1/MB':<10} {'Acc/MB':<10} {'Top5/MB':<10}\n")
                f.write("-" * 80 + "\n")
                
                for result in successful_results:
                    experiment_id = f"{result['strategy']}_{result['model_name']}"
                    model_size = result.get('model_size_mb', 0.1)  # Avoid division by zero
                    accuracy = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    top5 = result.get('top5_accuracy', 0)
                    
                    # Calculate efficiency metrics (performance per MB)
                    f1_per_mb = f1 / model_size if model_size > 0 else 0
                    acc_per_mb = accuracy / model_size if model_size > 0 else 0
                    top5_per_mb = top5 / model_size if model_size > 0 else 0
                    
                    f.write(f"{experiment_id:<20} {model_size:<10.2f} {f1_per_mb:<10.4f} {acc_per_mb:<10.4f} {top5_per_mb:<10.4f}\n")
                
                f.write("-" * 80 + "\n\n")
                
                # Per-class accuracy details
                f.write("PER-CLASS ACCURACY DETAILS:\n")
                f.write("-" * 80 + "\n")
                
                from .data_utils import CIFAR10_CLASSES
                for result in successful_results:
                    experiment_id = f"{result['strategy']}_{result['model_name']}"
                    f.write(f"\n{experiment_id}:\n")
                    
                    for class_name in CIFAR10_CLASSES:
                        acc_key = f"{class_name}_accuracy"
                        if acc_key in result:
                            acc_val = result[acc_key] * 100
                            f.write(f"  {class_name:<12}: {acc_val:6.2f}%\n")
                
                f.write("\n")
                
                # Strategy comparison
                strategies = set([r['strategy'] for r in successful_results])
                if len(strategies) > 1:
                    f.write("STRATEGY COMPARISON:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"{'Strategy':<15} {'Avg Accuracy':<15} {'Avg F1':<10} {'Avg Time':<10}\n")
                    f.write("-" * 50 + "\n")
                    
                    for strategy in strategies:
                        strategy_results = [r for r in successful_results if r['strategy'] == strategy]
                        if strategy_results:
                            avg_acc = sum([r.get('accuracy', 0) for r in strategy_results]) / len(strategy_results) * 100
                            avg_f1 = sum([r.get('f1', 0) for r in strategy_results]) / len(strategy_results)
                            avg_time = sum([r.get('training_time', 0) for r in strategy_results]) / len(strategy_results)
                            
                            f.write(f"{strategy:<15} {avg_acc:<15.2f} {avg_f1:<10.4f} {avg_time:<10.1f}\n")
                    
                    f.write("-" * 50 + "\n\n")
                
                # ULCD vs Traditional comparison
                ulcd_results = [r for r in successful_results if r['strategy'] == 'ulcd']
                traditional_results = [r for r in successful_results if r['strategy'] in ['fedavg', 'fedprox']]
                
                if ulcd_results and traditional_results:
                    f.write("ULCD vs TRADITIONAL FL COMPARISON:\n")
                    f.write("-" * 60 + "\n")
                    
                    # Calculate averages
                    avg_ulcd_acc = sum([r.get('accuracy', 0) for r in ulcd_results]) / len(ulcd_results)
                    avg_ulcd_f1 = sum([r.get('f1', 0) for r in ulcd_results]) / len(ulcd_results)
                    avg_ulcd_top5 = sum([r.get('top5_accuracy', 0) for r in ulcd_results]) / len(ulcd_results)
                    
                    avg_trad_acc = sum([r.get('accuracy', 0) for r in traditional_results]) / len(traditional_results)
                    avg_trad_f1 = sum([r.get('f1', 0) for r in traditional_results]) / len(traditional_results)
                    avg_trad_top5 = sum([r.get('top5_accuracy', 0) for r in traditional_results]) / len(traditional_results)
                    
                    # Calculate improvements
                    acc_improvement = ((avg_ulcd_acc - avg_trad_acc) / avg_trad_acc) * 100 if avg_trad_acc > 0 else 0
                    f1_improvement = ((avg_ulcd_f1 - avg_trad_f1) / avg_trad_f1) * 100 if avg_trad_f1 > 0 else 0
                    top5_improvement = ((avg_ulcd_top5 - avg_trad_top5) / avg_trad_top5) * 100 if avg_trad_top5 > 0 else 0
                    
                    f.write(f"ACCURACY:\n")
                    f.write(f"  ULCD FL: {avg_ulcd_acc:.4f}\n")
                    f.write(f"  Traditional FL: {avg_trad_acc:.4f}\n")
                    f.write(f"  Improvement: {acc_improvement:+.2f}%\n\n")
                    
                    f.write(f"F1 SCORE:\n")
                    f.write(f"  ULCD FL: {avg_ulcd_f1:.4f}\n")
                    f.write(f"  Traditional FL: {avg_trad_f1:.4f}\n")
                    f.write(f"  Improvement: {f1_improvement:+.2f}%\n\n")
                    
                    f.write(f"TOP-5 ACCURACY:\n")
                    f.write(f"  ULCD FL: {avg_ulcd_top5:.4f}\n")
                    f.write(f"  Traditional FL: {avg_trad_top5:.4f}\n")
                    f.write(f"  Improvement: {top5_improvement:+.2f}%\n\n")
            
            # Failed experiments
            failed_results = [r for r in all_results if not r.get('success', False)]
            if failed_results:
                f.write("FAILED EXPERIMENTS:\n")
                f.write("-" * 40 + "\n")
                for result in failed_results:
                    experiment_id = f"{result['strategy']}_{result['model_name']}"
                    error = result.get('error', 'Unknown error')
                    f.write(f"{experiment_id}: {error}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"[RESULTS] Saved metrics summary: {txt_filename}")
    return txt_filename

def calculate_model_size_mb(model_params):
    """Calculate model size in MB from parameter list"""
    if not model_params:
        return 0.0
    
    total_params = 0
    for param in model_params:
        if hasattr(param, 'numel'):
            total_params += param.numel()
        elif hasattr(param, 'shape'):
            import numpy as np
            total_params += np.prod(param.shape)
        else:
            # Assume it's a numpy array or similar
            import numpy as np
            if isinstance(param, np.ndarray):
                total_params += param.size
            else:
                try:
                    total_params += len(param)
                except:
                    total_params += 1
    
    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb