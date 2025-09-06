from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch

import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, EvaluateIns, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from .ulcd_components import ULCDServer_CIFAR, visualize_latents_cifar, analyze_latent_statistics

# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================
def weighted_average(metrics):
    """Aggregate evaluation metrics using weighted average"""
    try:
        print(f"Aggregating metrics from {len(metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        if total_examples == 0:
            print("Warning: No examples found in metrics")
            return {}
        
        weighted_metrics = {}
        
        # Core metrics
        for key in ["accuracy", "f1", "precision", "recall", "top5_accuracy", "loss"]:
            if any(key in m for _, m in metrics):
                weighted_sum = sum([num_examples * m.get(key, 0) for num_examples, m in metrics])
                weighted_metrics[key] = weighted_sum / total_examples
                print(f"Weighted {key}: {weighted_metrics[key]:.4f}")
        
        # CIFAR-10 per-class accuracies
        from .data_utils import CIFAR10_CLASSES
        for class_name in CIFAR10_CLASSES:
            acc_key = f"{class_name}_accuracy"
            if any(acc_key in m for _, m in metrics):
                weighted_sum = sum([num_examples * m.get(acc_key, 0) for num_examples, m in metrics])
                weighted_metrics[acc_key] = weighted_sum / total_examples
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average: {e}")
        return {}

def weighted_average_fit(fit_metrics):
    """Aggregate fit metrics using weighted average"""
    try:
        print(f"Aggregating fit metrics from {len(fit_metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in fit_metrics])
        
        if total_examples == 0:
            return {}
        
        weighted_metrics = {}
        
        # Training loss
        if any("train_loss" in m for _, m in fit_metrics):
            weighted_sum = sum([num_examples * m.get("train_loss", 0) for num_examples, m in fit_metrics])
            weighted_metrics["train_loss"] = weighted_sum / total_examples
            print(f"Weighted training loss: {weighted_metrics['train_loss']:.4f}")
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average_fit: {e}")
        return {}

# ============================================================================
# ULCD STRATEGY FOR CIFAR-10
# ============================================================================
class ULCDStrategy_CIFAR(fl.server.strategy.Strategy):
    """ULCD Strategy adapted for CIFAR-10"""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        latent_dim: int = 64,
        anomaly_threshold: float = 0.3,
        enable_visualization: bool = True,
    ):
        super().__init__()
        
        # Flower strategy parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        
        # ULCD-specific parameters
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold
        self.enable_visualization = enable_visualization
        
        # Initialize ULCD server
        self.ulcd_server = ULCDServer_CIFAR(latent_dim=latent_dim)
        self.round_num = 0
        
        print(f"[ULCD Strategy CIFAR] Initialized with latent_dim={latent_dim}, threshold={anomaly_threshold}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize with dummy parameters"""
        from flwr.common import ndarrays_to_parameters
        dummy_weights = [np.array([0.0])]
        return ndarrays_to_parameters(dummy_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure fit round for ULCD"""
        
        self.round_num = server_round
        print(f"\n[ULCD Strategy CIFAR] Configuring fit round {server_round}")
        
        # Sample clients
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_available_clients
        )
        
        # Create fit instructions
        fit_ins = []
        for client in clients:
            # Send current prototype to clients
            if hasattr(self.ulcd_server, 'prototype'):
                prototype_array = self.ulcd_server.prototype.detach().cpu().numpy()
                from flwr.common import ndarrays_to_parameters
                fit_parameters = ndarrays_to_parameters([prototype_array])
            else:
                fit_parameters = parameters
            
            config = {
                "ulcd_mode": True,
                "server_round": server_round,
                "request_latent_summary": True,
                "anomaly_threshold": self.anomaly_threshold
            }
            
            fit_ins.append((client, fl.common.FitIns(fit_parameters, config)))
        
        print(f"[ULCD Strategy CIFAR] Configured {len(fit_ins)} clients for round {server_round}")
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate latent summaries using ULCD consensus"""
        
        print(f"\n[ULCD Strategy CIFAR] Aggregating round {server_round}")
        print(f"Received {len(results)} results, {len(failures)} failures")
        
        if not results:
            return None, {}
        
        # Extract latent summaries and model weights
        latent_summaries = []
        client_weights = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            try:
                from flwr.common import parameters_to_ndarrays
                all_arrays = parameters_to_ndarrays(fit_res.parameters)
                
                # Check if client sent both latent and weights
                if fit_res.metrics and fit_res.metrics.get("has_model_weights", False):
                    num_weight_arrays = fit_res.metrics.get("num_weight_arrays", 0)
                    
                    if len(all_arrays) >= num_weight_arrays + 1:
                        # Extract latent summary (first array)
                        latent_array = all_arrays[0]
                        latent_tensor = torch.from_numpy(latent_array).float()
                        
                        # Extract model weights (remaining arrays)
                        model_weights = all_arrays[1:num_weight_arrays+1]
                        
                        client_id = len(latent_summaries)
                        latent_summaries.append((client_id, latent_tensor))
                        client_weights.append((client_id, model_weights, fit_res.num_examples))
                        
                        print(f"  Client {client_id}: latent shape {latent_tensor.shape}, weights {len(model_weights)} arrays")
                
                # Collect metrics
                if fit_res.metrics:
                    client_metrics.append((fit_res.num_examples, fit_res.metrics))
                
            except Exception as e:
                print(f"[WARNING] Failed to decode parameters from client: {e}")
                continue
        
        if not latent_summaries:
            print("[ULCD Strategy CIFAR] No valid latent summaries received")
            return None, {}
        
        # Visualize latents before aggregation
        if self.enable_visualization:
            visualize_latents_cifar(
                latent_summaries, 
                self.ulcd_server.prototype, 
                f"Round_{server_round}_Before_Aggregation"
            )
            analyze_latent_statistics(latent_summaries, self.ulcd_server.prototype)
        
        # Detect anomalies and filter trusted clients
        trusted_clients, flagged_clients = self.ulcd_server.detect_anomalies(
            [latent for _, latent in latent_summaries], 
            threshold=self.anomaly_threshold
        )
        
        trusted_client_ids = [cid for cid, _ in trusted_clients]
        print(f"  Trusted clients: {trusted_client_ids}")
        print(f"  Flagged clients: {flagged_clients}")
        
        # Aggregate trusted latents
        if trusted_clients:
            trusted_latents = [latent for _, latent in trusted_clients]
            self.ulcd_server.aggregate_latents(trusted_latents)
            
            # Visualize after aggregation
            if self.enable_visualization:
                visualize_latents_cifar(
                    latent_summaries, 
                    self.ulcd_server.prototype, 
                    f"Round_{server_round}_After_Aggregation"
                )
        
        # Aggregate model weights from trusted clients (FedAvg style)
        aggregated_parameters = None
        if client_weights:
            trusted_weights = [(cid, weights, num_examples) for cid, weights, num_examples in client_weights 
                             if cid in trusted_client_ids]
            
            if trusted_weights:
                print(f"  Aggregating model weights from {len(trusted_weights)} trusted clients")
                
                # Weighted average of model weights
                total_examples = sum(num_examples for _, _, num_examples in trusted_weights)
                aggregated_weights = None
                
                for cid, weights, num_examples in trusted_weights:
                    weight_factor = num_examples / total_examples
                    
                    if aggregated_weights is None:
                        aggregated_weights = [w * weight_factor for w in weights]
                    else:
                        for i, w in enumerate(weights):
                            aggregated_weights[i] += w * weight_factor
                
                from flwr.common import ndarrays_to_parameters
                aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
                print(f"  Model weight aggregation completed")
        
        # Aggregate metrics
        aggregated_metrics = {}
        if client_metrics and self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(client_metrics)
        
        # Add ULCD-specific metrics
        aggregated_metrics.update({
            "ulcd_trusted_clients": len(trusted_clients),
            "ulcd_flagged_clients": len(flagged_clients),
            "ulcd_consensus_ratio": len(trusted_clients) / len(latent_summaries),
            "ulcd_prototype_norm": float(torch.norm(self.ulcd_server.prototype).item())
        })
        
        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation round"""
        
        # Sample clients for evaluation
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
        
        print(f"[ULCD Strategy CIFAR] Configuring federated evaluation (round {server_round})")
        
        # Create evaluation instructions
        evaluate_ins = []
        for client in clients:
            config = {
                "ulcd_eval_mode": True,
                "server_round": float(server_round),
                "use_global_test": False  # Use client's own test set
            }
            
            # Send the aggregated model weights for evaluation
            evaluate_ins.append((client, fl.common.EvaluateIns(parameters, config)))
        
        print(f"[ULCD Strategy CIFAR] Configured {len(evaluate_ins)} clients for evaluation")
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results"""
        
        print(f"\n[ULCD Strategy CIFAR] Aggregating evaluation results for round {server_round}")
        
        if not results:
            return None, {}
        
        # Extract evaluation metrics
        eval_metrics = []
        total_examples = 0
        total_loss = 0.0
        
        for _, eval_res in results:
            total_examples += eval_res.num_examples
            total_loss += eval_res.loss * eval_res.num_examples
            
            if eval_res.metrics:
                eval_metrics.append((eval_res.num_examples, eval_res.metrics))
        
        # Calculate weighted average loss
        aggregated_loss = total_loss / total_examples if total_examples > 0 else None
        
        # Aggregate other metrics
        aggregated_metrics = {}
        if eval_metrics and self.evaluate_metrics_aggregation_fn:
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        print(f"  Aggregated loss: {aggregated_loss:.4f}")
        print(f"  Final metrics: {aggregated_metrics}")
        
        return aggregated_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Server-side evaluation (optional)"""
        return None

# ============================================================================
# STRATEGY FACTORY
# ============================================================================
def get_strategy(strategy_name: str, run_config: Dict = None, **kwargs):
    """Factory function to create federated learning strategies for CIFAR-10"""
    
    # Common aggregation functions
    evaluate_metrics_aggregation_fn = weighted_average
    fit_metrics_aggregation_fn = weighted_average_fit
    
    if strategy_name == "ulcd":
        # ULCD strategy parameters
        latent_dim = kwargs.get('latent_dim', 64)
        anomaly_threshold = kwargs.get('anomaly_threshold', 0.3)
        enable_visualization = kwargs.get('enable_visualization', True)
        
        num_clients = run_config.get("num_clients", 10) if run_config else 10
        
        return ULCDStrategy_CIFAR(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            latent_dim=latent_dim,
            anomaly_threshold=anomaly_threshold,
            enable_visualization=enable_visualization,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    
    elif strategy_name == "fedavg":
        num_clients = run_config.get("num_clients", 10) if run_config else 10
        
        return FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    
    elif strategy_name == "fedprox":
        num_clients = run_config.get("num_clients", 10) if run_config else 10
        
        return FedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            proximal_mu=kwargs.get('proximal_mu', 0.1),
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")