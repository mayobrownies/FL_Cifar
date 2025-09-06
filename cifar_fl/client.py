import copy
from typing import Dict

import flwr as fl
import torch

from .task import load_client_data, train, test, get_weights, set_weights
from .models import get_model

# ============================================================================
# CIFAR-10 FEDERATED LEARNING CLIENT
# ============================================================================
class FlowerClient_CIFAR(fl.client.NumPyClient):
    """Flower client for CIFAR-10 federated learning"""
    
    def __init__(self, run_config: Dict, client_id: int):
        self.run_config = run_config
        self.client_id = client_id
        
        print(f"[CLIENT {client_id}] Initializing CIFAR-10 client...")
        
        # Load client data
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
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"[CLIENT {client_id}] Initialized with {model_name} model on {self.device}")
        print(f"[CLIENT {client_id}] Train: {len(self.trainloader.dataset)}, Test: {len(self.testloader.dataset)} samples")

    def get_parameters(self, config):
        """Return model parameters"""
        return get_weights(self.net)

    def fit(self, parameters, config):
        """Train model with given parameters"""
        print(f"[CLIENT {self.client_id}] Starting training...")
        
        # Check for ULCD mode
        ulcd_mode = config.get("ulcd_mode", False)
        prototype_tensor = None
        
        if ulcd_mode:
            print(f"[CLIENT {self.client_id}] ULCD training mode - using prototype guidance")
            
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
            learning_rate=self.run_config.get("learning_rate", 0.001),
            proximal_mu=self.run_config.get("proximal_mu", 0.0),
            device=self.device,
            use_focal_loss=self.run_config.get("use_focal_loss", False),
            prototype=prototype_tensor
        )
        
        # Calculate training metrics
        num_examples = len(self.trainloader.dataset)
        metrics = {
            "train_loss": train_loss,
            "client_id": self.client_id
        }
        
        print(f"[CLIENT {self.client_id}] Training completed - Loss: {train_loss:.4f}")
        
        if ulcd_mode:
            # For ULCD: return both latent summary and model weights
            try:
                from .ulcd_components import extract_client_latent_summary
                
                # Get latent summary
                latent_summary = extract_client_latent_summary(self.net, self.trainloader, self.device)
                latent_array = latent_summary.detach().cpu().numpy()
                
                # Get model weights
                model_weights = get_weights(self.net)
                
                # Pack both latent summary and model weights
                combined_params = [latent_array] + model_weights
                
                metrics.update({
                    "latent_norm": float(torch.norm(latent_summary).item()),
                    "has_model_weights": True,
                    "num_weight_arrays": len(model_weights)
                })
                
                print(f"[CLIENT {self.client_id}] Returning latent summary: shape {latent_array.shape}")
                return combined_params, num_examples, metrics
                
            except Exception as e:
                print(f"[CLIENT {self.client_id}] Error generating ULCD response: {e}")
                # Fallback to standard weights
                return get_weights(self.net), num_examples, metrics
        else:
            # Standard FL: return model weights only
            return get_weights(self.net), num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model with given parameters"""
        print(f"[CLIENT {self.client_id}] Starting evaluation...")
        
        # Check for ULCD evaluation mode
        ulcd_eval_mode = config.get("ulcd_eval_mode", False)
        
        if ulcd_eval_mode:
            # For ULCD evaluation, parameters contain the aggregated global model
            if parameters and len(parameters) > 1:
                # Use aggregated model weights (skip latent summary)
                model_weights = parameters[1:] if len(parameters) > 1 else parameters
                set_weights(self.net, model_weights)
                print(f"[CLIENT {self.client_id}] Using aggregated model weights for evaluation")
            else:
                print(f"[CLIENT {self.client_id}] No model weights available, using current model")
        else:
            # Standard evaluation: load model weights
            set_weights(self.net, parameters)
        
        # Evaluate model
        use_global_test = config.get("use_global_test", False)
        loss, num_examples, metrics = test(self.net, self.testloader, self.device, use_global_test)
        
        print(f"[CLIENT {self.client_id}] Evaluation completed - Loss: {loss:.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return loss, num_examples, metrics

# ============================================================================
# CLIENT FACTORY
# ============================================================================
def get_client(run_config: Dict, client_id: int) -> FlowerClient_CIFAR:
    """Factory function to create CIFAR-10 federated learning client"""
    return FlowerClient_CIFAR(run_config, client_id)