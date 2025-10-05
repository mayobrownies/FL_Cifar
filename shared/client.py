import copy
import warnings
from typing import Dict

import flwr as fl
import torch

from shared.task import load_client_data, train, test, get_weights, set_weights
from shared.models import get_model

# Suppress warnings in each Ray actor process
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="This process.*multi-threaded.*fork")
warnings.filterwarnings("ignore", message="'mode' parameter is deprecated")

# ============================================================================
# CIFAR-10 FEDERATED LEARNING CLIENT
# ============================================================================
class FlowerClient_CIFAR(fl.client.NumPyClient):
    """Flower client for CIFAR-10 federated learning"""
    
    def __init__(self, run_config: Dict, client_id: int):
        try:
            self.run_config = run_config
            self.client_id = client_id
            
            if not run_config.get("quiet_mode", False):
                print(f"[CLIENT {client_id}] Initializing CIFAR-10 client...")
            
            # Load client data
            if not run_config.get("quiet_mode", False):
                print(f"[CLIENT {client_id}] Loading client data...")
            self.trainloader, self.testloader, input_dim, output_dim = load_client_data(
                client_id=client_id,
                batch_size=run_config.get("batch_size", 32),
                test_split=run_config.get("test_split", 0.2),
                partitions=run_config.get("partitions")
            )
            
            # Initialize model
            model_name = run_config.get("model_name", "cnn")
            model_params = run_config.get("model_params", {})
            
            self.net = get_model(model_name, input_dim, output_dim, **model_params)
            
            if self.net is None:
                raise ValueError(f"get_model returned None for model {model_name}")
            
            # Setup device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            if not run_config.get("quiet_mode", False):
                print(f"[CLIENT {client_id}] Initialized with {model_name} model on {self.device}")
                print(f"[CLIENT {client_id}] Train: {len(self.trainloader.dataset)}, Test: {len(self.testloader.dataset)} samples")
            
        except Exception as e:
            print(f"[ERROR] Client {client_id} initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def get_learning_rate(self, config, ulcd_mode, latent_aggregation_mode=False):
        """Get appropriate learning rate based on model type and mode"""
        base_lr = self.run_config.get("learning_rate", 0.001)

        if ulcd_mode:
            # Reduce learning rate for ULCD models for better stability
            ulcd_lr = base_lr * 0.1  # 10x smaller for ULCD
            print(f"[CLIENT {self.client_id}] Using reduced ULCD learning rate: {ulcd_lr}")
            return ulcd_lr
        elif latent_aggregation_mode:
            # Slightly reduce learning rate for latent aggregation mode for stability
            latent_lr = base_lr * 0.5  # 2x smaller for latent aggregation
            print(f"[CLIENT {self.client_id}] Using reduced latent aggregation learning rate: {latent_lr}")
            return latent_lr
        else:
            return base_lr

    def get_parameters(self, config):
        """Return model parameters"""
        return get_weights(self.net)

    def fit(self, parameters, config):
        """Train model with given parameters"""
        if not self.run_config.get("quiet_mode", False):
            print(f"[CLIENT {self.client_id}] Starting training...")
        
        # Check for ULCD mode or latent aggregation mode
        ulcd_mode = config.get("ulcd_mode", False)
        latent_aggregation_mode = config.get("latent_aggregation_mode", False)
        prototype_tensor = None

        if ulcd_mode or latent_aggregation_mode:
            mode_name = "ULCD" if ulcd_mode else "Latent Aggregation"
            print(f"[CLIENT {self.client_id}] {mode_name} training mode - using prototype guidance")

            # Check model compatibility for latent aggregation
            model_name = self.run_config.get("model_name", "")
            ulcd_compatible_models = ["cnn_ulcd", "cnn_ulcd_light", "cnn_ulcd_heavy", "ulcd"]
            if latent_aggregation_mode and model_name not in ulcd_compatible_models:
                print(f"[CLIENT {self.client_id}] WARNING: Model {model_name} not compatible with latent aggregation, falling back to standard FedAvg")
                # Fall back to standard FedAvg training
                set_weights(self.net, parameters)
                latent_aggregation_mode = False
                ulcd_mode = False
                prototype_tensor = None
            else:
                try:
                    # Handle prototype parameter
                    if hasattr(parameters, 'tensors'):
                        from flwr.common import parameters_to_ndarrays
                        prototype_array = parameters_to_ndarrays(parameters)[0]
                    elif isinstance(parameters, list) and len(parameters) > 0:
                        import numpy as np
                        prototype_array = parameters[0] if isinstance(parameters[0], np.ndarray) else np.array(parameters[0])
                    else:
                        raise ValueError(f"Unknown parameter format: {type(parameters)}")

                    prototype_tensor = torch.from_numpy(prototype_array).float().to(self.device)
                    print(f"[CLIENT {self.client_id}] Received prototype: shape {prototype_tensor.shape}, norm {torch.norm(prototype_tensor):.3f}")

                except Exception as e:
                    print(f"[CLIENT {self.client_id}] Failed to decode prototype: {e}")
                    prototype_tensor = None
        else:
            # Standard FedAvg: load model weights
            set_weights(self.net, parameters)
        
        # Create global model copy for regularization
        global_net = copy.deepcopy(self.net)
        
        # Train model
        train_loss = train(
            net=self.net,
            global_net=global_net,
            trainloader=self.trainloader,
            epochs=self.run_config.get("local_epochs", 3),
            learning_rate=self.get_learning_rate(config, ulcd_mode, latent_aggregation_mode),
            proximal_mu=self.run_config.get("proximal_mu", 0.0),
            device=self.device,
            use_focal_loss=self.run_config.get("use_focal_loss", False),
            prototype=prototype_tensor,
            round_num=config.get("server_round", 1)
        )
        
        # Calculate training metrics and model size for efficiency analysis
        num_examples = len(self.trainloader.dataset)
        model_weights = get_weights(self.net)
        from shared.ulcd_components import calculate_model_size_mb
        model_size_mb = calculate_model_size_mb(model_weights)
        
        metrics = {
            "train_loss": train_loss,
            "client_id": self.client_id,
            "model_size_mb": model_size_mb,
        }
        
        if not self.run_config.get("quiet_mode", False):
            print(f"[CLIENT {self.client_id}] Training completed - Loss: {train_loss:.4f}")
        
        if ulcd_mode or latent_aggregation_mode:
            # For ULCD or Latent Aggregation: return both latent summary and model weights
            try:
                from shared.ulcd_components import extract_client_latent_summary

                if latent_aggregation_mode:
                    # Simplified latent aggregation - just extract latent without complex mechanisms
                    if hasattr(self.net, 'get_latent_summary'):
                        # Use enhanced model if available
                        latent_summary, _ = self.net.get_latent_summary(self.trainloader)
                    else:
                        # Extract simple latent for standard models
                        latent_summary = extract_client_latent_summary(self.net, self.trainloader, self.device)

                    latent_array = latent_summary.detach().cpu().numpy()

                    # Get model weights
                    model_weights = get_weights(self.net)

                    # Pack latent summary and model weights (no class mask for simple version)
                    combined_params = [latent_array] + model_weights

                    metrics.update({
                        "latent_norm": float(torch.norm(latent_summary).item()),
                        "has_model_weights": True,
                        "num_weight_arrays": len(model_weights),
                    })

                    print(f"[CLIENT {self.client_id}] Returning simple latent summary: shape {latent_array.shape}")
                    return combined_params, num_examples, metrics

                else:
                    # Full ULCD mode - get enhanced latent summary with class mask
                    if hasattr(self.net, 'get_latent_summary'):
                        # Enhanced model returns both summary and class mask
                        latent_summary, class_mask = self.net.get_latent_summary(self.trainloader)
                        latent_array = latent_summary.detach().cpu().numpy()
                        class_mask_array = class_mask.detach().cpu().numpy()
                    else:
                        # Fallback for standard models
                        latent_summary = extract_client_latent_summary(self.net, self.trainloader, self.device)
                        latent_array = latent_summary.detach().cpu().numpy()
                        # Create dummy class mask (all classes present)
                        class_mask_array = np.ones(10)

                    # Get model weights
                    model_weights = get_weights(self.net)

                    # Pack latent summary, class mask, and model weights
                    combined_params = [latent_array, class_mask_array] + model_weights

                    metrics.update({
                        "latent_norm": float(torch.norm(latent_summary).item()),
                        "has_model_weights": True,
                        "num_weight_arrays": len(model_weights),
                        "classes_present": int(np.sum(class_mask_array)) if isinstance(class_mask_array, np.ndarray) else 10
                    })

                    print(f"[CLIENT {self.client_id}] Returning enhanced latent summary: shape {latent_array.shape}, classes {int(np.sum(class_mask_array))}")
                    return combined_params, num_examples, metrics

            except Exception as e:
                print(f"[CLIENT {self.client_id}] Error generating latent response: {e}")
                # Fallback to standard weights
                return get_weights(self.net), num_examples, metrics
        else:
            # Standard FL: return model weights only
            return get_weights(self.net), num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model with given parameters"""
        if not self.run_config.get("quiet_mode", False):
            print(f"[CLIENT {self.client_id}] Starting evaluation...")
        
        # Check for ULCD evaluation mode
        ulcd_eval_mode = config.get("ulcd_eval_mode", False)
        heterogeneous_mode = config.get("heterogeneous_mode", False)

        if ulcd_eval_mode and heterogeneous_mode:
            # Heterogeneous ULCD: use client's own weights (don't load from server)
            print(f"[CLIENT {self.client_id}] Heterogeneous ULCD eval - using own model weights")
        elif ulcd_eval_mode:
            # Homogeneous ULCD: load aggregated weights from server
            if parameters:
                weights_arrays = self._convert_parameters_to_arrays(parameters)
                set_weights(self.net, weights_arrays)
                print(f"[CLIENT {self.client_id}] Using aggregated ULCD model weights for evaluation")
            else:
                print(f"[CLIENT {self.client_id}] No aggregated weights available, using current model")
        else:
            # Standard evaluation
            if heterogeneous_mode:
                # Heterogeneous mode: use own weights
                print(f"[CLIENT {self.client_id}] Heterogeneous eval - using own model weights")
            else:
                # Load model weights from server
                weights_arrays = self._convert_parameters_to_arrays(parameters)
                set_weights(self.net, weights_arrays)
        
        # Evaluate model
        use_global_test = config.get("use_global_test", False)
        loss, num_examples, metrics = test(self.net, self.testloader, self.device, use_global_test)
        
        if not self.run_config.get("quiet_mode", False):
            print(f"[CLIENT {self.client_id}] Evaluation completed - Loss: {loss:.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return loss, num_examples, metrics

    def _convert_parameters_to_arrays(self, parameters):
        """Convert parameters to numpy arrays, handling both Parameters and list types"""
        if parameters is None:
            return []

        # Handle Flower Parameters object
        if hasattr(parameters, 'tensors'):
            from flwr.common import parameters_to_ndarrays
            return parameters_to_ndarrays(parameters)

        # Handle list of numpy arrays
        elif isinstance(parameters, list):
            import numpy as np
            return [np.array(param) if not isinstance(param, np.ndarray) else param for param in parameters]

        # Handle single numpy array
        else:
            import numpy as np
            return [np.array(parameters)]

# ============================================================================
# CLIENT FACTORY
# ============================================================================
def get_client(run_config: Dict, client_id: int) -> FlowerClient_CIFAR:
    """Factory function to create CIFAR-10 federated learning client"""
    return FlowerClient_CIFAR(run_config, client_id)