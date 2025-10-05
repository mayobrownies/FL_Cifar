from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch

import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, EvaluateIns, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from shared.ulcd_components import ULCDServer_CIFAR, visualize_latents_cifar, analyze_latent_statistics

# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================
def weighted_average(metrics):
    """Aggregate evaluation metrics using weighted average with robust NaN handling"""
    try:
        print(f"Aggregating metrics from {len(metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        if total_examples == 0:
            print("Warning: No examples found in metrics")
            return {}
        
        weighted_metrics = {}
        
        # Core metrics with NaN filtering
        for key in ["accuracy", "f1", "precision", "recall", "top5_accuracy", "loss"]:
            if any(key in m for _, m in metrics):
                # Filter out NaN and infinite values
                valid_metrics = []
                valid_examples = 0
                
                for num_examples, m in metrics:
                    if key in m:
                        metric_value = m[key]
                        if not (np.isnan(metric_value) or np.isinf(metric_value)):
                            valid_metrics.append((num_examples, metric_value))
                            valid_examples += num_examples
                        else:
                            print(f"Warning: Filtering out NaN/Inf value for {key} from client")
                
                if valid_metrics and valid_examples > 0:
                    weighted_sum = sum([num_examples * metric_value for num_examples, metric_value in valid_metrics])
                    weighted_metrics[key] = weighted_sum / valid_examples
                    print(f"Weighted {key}: {weighted_metrics[key]:.4f}")
                else:
                    # Fallback: set to 0 if all values are invalid
                    weighted_metrics[key] = 0.0
                    print(f"WARNING: All {key} values were invalid, setting to 0.0")
                    print(f"  This suggests evaluation is failing completely for all clients")
        
        # CIFAR-10 per-class accuracies with NaN filtering
        from .data_utils import CIFAR10_CLASSES
        for class_name in CIFAR10_CLASSES:
            acc_key = f"{class_name}_accuracy"
            if any(acc_key in m for _, m in metrics):
                # Filter out NaN and infinite values
                valid_metrics = []
                valid_examples = 0
                
                for num_examples, m in metrics:
                    if acc_key in m:
                        metric_value = m[acc_key]
                        if not (np.isnan(metric_value) or np.isinf(metric_value)):
                            valid_metrics.append((num_examples, metric_value))
                            valid_examples += num_examples
                
                if valid_metrics and valid_examples > 0:
                    weighted_sum = sum([num_examples * metric_value for num_examples, metric_value in valid_metrics])
                    weighted_metrics[acc_key] = weighted_sum / valid_examples
                else:
                    weighted_metrics[acc_key] = 0.0
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average: {e}")
        # Return zero metrics on error to prevent NaN propagation
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "top5_accuracy": 0.0,
            "loss": 0.0,
        }

def weighted_average_fit(fit_metrics):
    """Aggregate fit metrics using weighted average with robust NaN handling"""
    try:
        print(f"Aggregating fit metrics from {len(fit_metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in fit_metrics])
        
        if total_examples == 0:
            return {}
        
        weighted_metrics = {}
        
        # Training loss with NaN filtering
        if any("train_loss" in m for _, m in fit_metrics):
            valid_metrics = []
            valid_examples = 0
            
            for num_examples, m in fit_metrics:
                if "train_loss" in m:
                    loss_value = m["train_loss"]
                    if not (np.isnan(loss_value) or np.isinf(loss_value)):
                        valid_metrics.append((num_examples, loss_value))
                        valid_examples += num_examples
                    else:
                        print(f"Warning: Filtering out NaN/Inf training loss from client")
            
            if valid_metrics and valid_examples > 0:
                weighted_sum = sum([num_examples * loss_value for num_examples, loss_value in valid_metrics])
                weighted_metrics["train_loss"] = weighted_sum / valid_examples
                print(f"Weighted training loss: {weighted_metrics['train_loss']:.4f}")
            else:
                weighted_metrics["train_loss"] = 0.0
                print(f"Warning: All training loss values were invalid, setting to 0.0")
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average_fit: {e}")
        return {"train_loss": 0.0}

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
        
        # Enhanced ULCD-specific parameters
        self.latent_dim = latent_dim
        self.num_classes = 10  # CIFAR-10 classes
        self.num_subspaces = 3  # Number of latent subspaces
        self.anomaly_threshold = anomaly_threshold
        self.enable_visualization = enable_visualization
        
        # Initialize enhanced ULCD server
        self.ulcd_server = ULCDServer_CIFAR(latent_dim=latent_dim, num_classes=self.num_classes, num_subspaces=self.num_subspaces)
        self.round_num = 0
        self.client_class_distributions = {}  # Track client class distributions
        self.last_aggregated_parameters = None  # Store aggregated model weights
        
        print(f"[Enhanced ULCD Strategy CIFAR] Initialized with latent_dim={latent_dim}, classes={self.num_classes}, threshold={anomaly_threshold}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize with dummy parameters"""
        from flwr.common import ndarrays_to_parameters
        dummy_weights = [np.array([0.0])]
        return ndarrays_to_parameters(dummy_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure fit round with customized prototypes per client"""
        
        self.round_num = server_round
        print(f"\n[Enhanced ULCD Strategy CIFAR] Configuring fit round {server_round}")
        
        # Sample clients
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_available_clients
        )
        
        # Create fit instructions with customized prototypes
        fit_ins = []
        for client in clients:
            # Get client's class distribution (if known)
            client_id = int(client.cid)
            client_classes = self.client_class_distributions.get(client_id, list(range(self.num_classes)))
            
            # Get customized prototype for this client
            prototype = self.ulcd_server.get_prototype_for_client(client_classes)
            
            # Send prototype as parameters
            from flwr.common import ndarrays_to_parameters
            fit_parameters = ndarrays_to_parameters([prototype.detach().cpu().numpy()])
            
            config = {
                "ulcd_mode": True,
                "distillation_mode": True,
                "server_round": server_round,
                "round_num": server_round,
                "request_latent_summary": True,
                "anomaly_threshold": self.anomaly_threshold
            }
            
            fit_ins.append((client, fl.common.FitIns(fit_parameters, config)))
        
        print(f"[Enhanced ULCD Strategy CIFAR] Configured {len(fit_ins)} clients for round {server_round}")
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
        
        # Extract enhanced latent summaries with class masks
        client_summaries = []
        client_weights = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            try:
                from flwr.common import parameters_to_ndarrays
                all_arrays = parameters_to_ndarrays(fit_res.parameters)
                
                client_id = int(client_proxy.cid)
                
                # Check if client sent both latent summary, class mask, and weights
                if fit_res.metrics and fit_res.metrics.get("has_model_weights", False):
                    num_weight_arrays = fit_res.metrics.get("num_weight_arrays", 0)
                    
                    if len(all_arrays) >= num_weight_arrays + 2:  # latent + class_mask + weights
                        # Extract latent summary (first array)
                        latent_array = all_arrays[0]
                        latent_tensor = torch.from_numpy(latent_array).float()
                        
                        # Extract class mask (second array)
                        class_mask_array = all_arrays[1]
                        class_mask = torch.from_numpy(class_mask_array).float()
                        
                        # Extract model weights (remaining arrays)
                        model_weights = all_arrays[2:num_weight_arrays+2]
                        
                        client_summaries.append((client_id, latent_tensor, class_mask))
                        client_weights.append((client_id, model_weights, fit_res.num_examples))
                        
                        # Update known class distribution
                        self.client_class_distributions[client_id] = [
                            i for i in range(self.num_classes) if class_mask[i] > 0
                        ]
                        
                        print(f"  Client {client_id}: latent shape {latent_tensor.shape}, classes {class_mask.sum().int().item()}, weights {len(model_weights)} arrays")
                
                # Collect metrics
                if fit_res.metrics:
                    client_metrics.append((fit_res.num_examples, fit_res.metrics))
                
            except Exception as e:
                print(f"[WARNING] Failed to decode parameters from client: {e}")
                continue
        
        if not client_summaries:
            print("[Enhanced ULCD Strategy CIFAR] No valid enhanced summaries received")
            return None, {}
        
        # Detect anomalies with distillation (more lenient in early rounds)
        adaptive_threshold = self.anomaly_threshold
        if server_round <= 3:  # First few rounds are more lenient
            adaptive_threshold = max(0.1, self.anomaly_threshold * 0.5)
            print(f"  Early round {server_round}: using adaptive threshold {adaptive_threshold:.3f}")
        
        trusted, flagged = self.ulcd_server.detect_anomalies(
            client_summaries, 
            adaptive_threshold
        )
        
        trusted_client_ids = [client_id for client_id, _, _ in trusted]
        print(f"  Trusted clients: {trusted_client_ids}")
        print(f"  Flagged clients: {flagged}")
        
        # Aggregate with distillation
        if trusted:
            self.ulcd_server.aggregate_with_distillation(trusted, server_round)
            
            # Visualize after aggregation
            if self.enable_visualization:
                simple_summaries = [(cid, summary) for cid, summary, _ in client_summaries]
                visualize_latents_cifar(
                    simple_summaries, 
                    self.ulcd_server.global_prototype, 
                    f"Round_{server_round}_After_Aggregation"
                )
        
        # IMPORTANT: In ULCD, we do NOT aggregate model weights for heterogeneous FL
        # Each client keeps its own model architecture and weights
        # Only latent vectors are aggregated (already done in ulcd_server.aggregate_with_distillation)
        aggregated_parameters = None
        print(f"  [ULCD] Skipping model weight aggregation (heterogeneous FL - clients keep own weights)")
        
        # Aggregate metrics
        aggregated_metrics = {}
        if client_metrics and self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(client_metrics)
        
        # Add Enhanced ULCD-specific metrics
        aggregated_metrics.update({
            "ulcd_trusted_clients": len(trusted),
            "ulcd_flagged_clients": len(flagged),
            "ulcd_consensus_ratio": len(trusted) / len(client_summaries) if client_summaries else 0,
            "ulcd_prototype_norm": float(torch.norm(self.ulcd_server.global_prototype).item())
        })

        # Store aggregated parameters for use in evaluation
        if aggregated_parameters is not None:
            self.last_aggregated_parameters = aggregated_parameters
            print(f"  Stored aggregated parameters for evaluation")

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

        # For heterogeneous FL: clients use their own weights, not aggregated weights
        # We send the provided parameters (which Flower requires), but clients will ignore them
        print(f"  [ULCD Heterogeneous] Clients will use their own model weights for evaluation")

        # Create evaluation instructions
        evaluate_ins = []
        for client in clients:
            config = {
                "ulcd_eval_mode": True,
                "server_round": float(server_round),
                "use_global_test": False,  # Use client's own test set
                "heterogeneous_mode": True  # Flag to tell client to ignore parameters and use own weights
            }

            # Send parameters (required by Flower API, but clients will ignore them in heterogeneous mode)
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
# FEDAVG WITH LATENT AGGREGATION STRATEGY
# ============================================================================
class FedAvgLatentStrategy(fl.server.strategy.Strategy):
    """FedAvg with simple latent aggregation - best performing configuration"""

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

        # Simple latent aggregation parameters
        self.latent_dim = latent_dim
        self.round_num = 0
        self.last_aggregated_parameters = None

        # Simple global latent prototype (EMA)
        self.global_latent_prototype = None
        self.prototype_momentum = 0.5

        print(f"[FedAvg+Latent Strategy] Initialized with latent_dim={latent_dim}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize with dummy parameters"""
        from flwr.common import ndarrays_to_parameters
        dummy_weights = [np.array([0.0])]
        return ndarrays_to_parameters(dummy_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure fit round with latent aggregation mode"""

        self.round_num = server_round
        print(f"\n[FedAvg+Latent Strategy] Configuring fit round {server_round}")

        # Sample clients
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )

        # Create fit instructions with latent aggregation mode
        fit_ins = []
        for client in clients:
            # Send current global prototype (if available)
            if self.global_latent_prototype is not None:
                from flwr.common import ndarrays_to_parameters
                fit_parameters = ndarrays_to_parameters([self.global_latent_prototype.detach().cpu().numpy()])
            else:
                from flwr.common import ndarrays_to_parameters
                fit_parameters = ndarrays_to_parameters([np.zeros(self.latent_dim)])

            config = {
                "latent_aggregation_mode": True,  # Enable simple latent extraction
                "server_round": server_round,
                "request_latent_summary": True,
            }

            fit_ins.append((client, fl.common.FitIns(fit_parameters, config)))

        print(f"[FedAvg+Latent Strategy] Configured {len(fit_ins)} clients for round {server_round}")
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate using FedAvg + simple latent aggregation"""

        print(f"\n[FedAvg+Latent Strategy] Aggregating round {server_round}")
        print(f"Received {len(results)} results, {len(failures)} failures")

        if not results:
            return None, {}

        # Extract latent summaries and model weights
        client_latents = []
        client_weights = []
        client_metrics = []

        for client_proxy, fit_res in results:
            try:
                from flwr.common import parameters_to_ndarrays
                all_arrays = parameters_to_ndarrays(fit_res.parameters)

                client_id = int(client_proxy.cid)

                # Check if client sent both latent summary and weights
                if fit_res.metrics and fit_res.metrics.get("has_model_weights", False):
                    num_weight_arrays = fit_res.metrics.get("num_weight_arrays", 0)

                    if len(all_arrays) >= num_weight_arrays + 1:  # latent + weights
                        # Extract latent summary (first array)
                        latent_array = all_arrays[0]
                        latent_tensor = torch.from_numpy(latent_array).float()

                        # Extract model weights (remaining arrays)
                        model_weights = all_arrays[1:num_weight_arrays+1]

                        client_latents.append((client_id, latent_tensor))
                        client_weights.append((client_id, model_weights, fit_res.num_examples))

                        print(f"  Client {client_id}: latent shape {latent_tensor.shape}, weights {len(model_weights)} arrays")

                # Collect metrics
                if fit_res.metrics:
                    client_metrics.append((fit_res.num_examples, fit_res.metrics))

            except Exception as e:
                print(f"[WARNING] Failed to decode parameters from client: {e}")
                continue

        # Simple latent aggregation (EMA)
        if client_latents:
            # Compute mean latent representation
            all_latents = torch.stack([latent for _, latent in client_latents])
            current_prototype = torch.mean(all_latents, dim=0)

            # Update global prototype with EMA
            if self.global_latent_prototype is None:
                self.global_latent_prototype = current_prototype
            else:
                self.global_latent_prototype = (
                    self.prototype_momentum * self.global_latent_prototype +
                    (1 - self.prototype_momentum) * current_prototype
                )

            print(f"  Updated global latent prototype: norm={torch.norm(self.global_latent_prototype).item():.4f}")

        # IMPORTANT: In heterogeneous FL, we do NOT aggregate model weights
        # Each client keeps its own model architecture and weights
        # Only latent vectors are aggregated (already done above)
        aggregated_parameters = None
        print(f"  [FedAvg+Latent] Skipping model weight aggregation (heterogeneous FL - clients keep own weights)")

        # Aggregate metrics
        aggregated_metrics = {}
        if client_metrics and self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(client_metrics)

        # Add latent-specific metrics
        if self.global_latent_prototype is not None:
            aggregated_metrics.update({
                "latent_prototype_norm": float(torch.norm(self.global_latent_prototype).item()),
                "num_latent_clients": len(client_latents)
            })

        # Store aggregated parameters for evaluation
        if aggregated_parameters is not None:
            self.last_aggregated_parameters = aggregated_parameters
            print(f"  Stored aggregated parameters for evaluation")

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

        print(f"[FedAvg+Latent Strategy] Configuring federated evaluation (round {server_round})")

        # For heterogeneous FL: clients use their own weights, not aggregated weights
        print(f"  [FedAvg+Latent Heterogeneous] Clients will use their own model weights for evaluation")

        # Create evaluation instructions
        evaluate_ins = []
        for client in clients:
            config = {
                "server_round": float(server_round),
                "use_global_test": False,
                "heterogeneous_mode": True  # Flag to tell client to use own weights
            }

            # Send parameters (required by API, but ignored in heterogeneous mode)
            evaluate_ins.append((client, fl.common.EvaluateIns(parameters, config)))

        print(f"[FedAvg+Latent Strategy] Configured {len(evaluate_ins)} clients for evaluation")
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results"""

        print(f"\n[FedAvg+Latent Strategy] Aggregating evaluation results for round {server_round}")

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

    elif strategy_name == "fedavg_latent":
        # FedAvg + Latent Aggregation - best performing configuration
        latent_dim = kwargs.get('latent_dim', 64)
        num_clients = run_config.get("num_clients", 10) if run_config else 10

        return FedAvgLatentStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            latent_dim=latent_dim,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")