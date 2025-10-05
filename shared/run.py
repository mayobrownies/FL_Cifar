import sys
import os
import warnings
import logging
import time
import gc
from typing import List, Dict

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flwr as fl
from flwr.common import Context
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ray
import torch

from shared.client import get_client
from shared.strategies import get_strategy
from shared.task import load_and_partition_data, GLOBAL_PARTITIONS, evaluate_global_model, get_weights
from shared.data_utils import CIFAR10_CLASSES, CLASS_TO_INDEX
from shared.models import get_model, get_model_type, is_sklearn_model
from shared.ulcd_components import compare_ulcd_vs_traditional_fl, save_metrics_to_txt

# ==============================================================================
# CONFIGURATION OPTIONS - CIFAR-10 Federated Learning
# ==============================================================================

# Data and experiment configuration
CIFAR_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cifar-10-batches-py")

# Federated learning configuration
NUM_ROUNDS = 10              # Number of FL rounds (reduced for debugging)
LOCAL_EPOCHS = 5            # Local training epochs per round (reduced for faster convergence)
BATCH_SIZE = 32             # Batch size for training
LEARNING_RATE = 0.005       # Learning rate (increased for ULCD models)
NUM_CLIENTS = 5             # Number of federated clients
TEST_SPLIT = 0.2            # Train/test split for each client

# FL strategies to test
AVAILABLE_STRATEGIES = ["fedavg", "fedprox", "ulcd", "fedavg_latent"]
STRATEGIES_TO_COMPARE = {"ulcd", "fedavg_latent"}  # Compare ULCD with FedAvg+Latent for heterogeneous FL

# Partitioning schemes: ["iid", "non_iid", "dirichlet", "pathological", "overlap_guaranteed"]
PARTITION_TYPE = "non_iid"  # Non-IID partitioning for realistic heterogeneous scenarios
PARTITION_KWARGS = {
    "alpha": 0.5,                    # For Dirichlet partitioning
    "num_classes_per_client": 3,     # For pathological partitioning
    "classes_per_client": 5,         # For non-IID partitioning (5 classes per client)
    "overlap_classes": 3,            # For overlap-guaranteed partitioning (3 shared classes)
    "min_samples_per_client": 100,   # Minimum samples per client
}

# Models to test - Base model for heterogeneous FL
# Note: Each client will automatically get assigned different variants (light/standard/heavy) based on device type
MODELS_TO_TEST = {"cnn_ulcd"}  # Base model - clients get heterogeneous variants

# Heterogeneous model configuration
ENABLE_HETEROGENEOUS = True  # Enable different models per client based on device type
CLIENT_DEVICE_TYPES = {
    0: "edge",       # Mobile/IoT device
    1: "edge",       # Mobile/IoT device
    2: "standard",   # Regular PC
    3: "standard",   # Regular PC
    4: "powerful",   # High-end PC/Server
}

DEVICE_MODEL_MAPPING = {
    "edge": "cnn_ulcd_light",      # Lightweight CNN for edge devices
    "standard": "cnn_ulcd",         # Standard CNN (current implementation)
    "powerful": "cnn_ulcd_heavy",   # Heavyweight CNN for powerful devices
}

# Model-specific parameters
MODEL_PARAMS = {
    "ulcd": {"latent_dim": 64},
    "cnn": {},
    "cnn_ulcd": {"latent_dim": 64},
    "cnn_ulcd_light": {"latent_dim": 64},   # Same latent dim for consensus
    "cnn_ulcd_heavy": {"latent_dim": 64},   # Same latent dim for consensus
    "resnet": {},
    "lstm": {"hidden_dim": 128, "num_layers": 2},
    "moe": {"num_experts": 4, "expert_dim": 128},
    "mlp": {"hidden_dims": [512, 256, 128]},
    "logistic": {},
    "random_forest": {},
}

# ULCD-specific configuration
ULCD_LATENT_DIM = 64  # Increased latent dimension for better feature capacity
ULCD_ANOMALY_THRESHOLD = 0.0   # Disabled for CIFAR (balanced data) - enable for MIMIC with real anomalies
ULCD_ENABLE_VISUALIZATION = False

# FedProx configuration
PROXIMAL_MU = 0.1

# Advanced options
USE_FOCAL_LOSS = False      # Use focal loss for imbalanced classification
USE_LOCAL_MODE = False      # Use Ray local mode vs distributed
QUIET_MODE = True           # Reduce verbose logging

# Performance optimizations
ENABLE_GPU_OPTIMIZATIONS = True
PIN_MEMORY = True
NUM_WORKERS = 2

# Suppress warnings and verbose logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="This process.*multi-threaded.*fork")
warnings.filterwarnings("ignore", message="'mode' parameter is deprecated")
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def cleanup_ray_temp_files():
    """Clean up Ray temporary files to prevent disk space issues"""
    print("[CLEANUP] Cleaning Ray temporary files...")
    
    import tempfile
    import shutil
    
    temp_dirs_to_check = [
        "/tmp/ray",
        "/tmp/ray_fl_temp",
        os.path.expanduser("~/tmp/ray"),
        os.path.join(tempfile.gettempdir(), "ray"),
    ]
    
    total_freed = 0
    
    for temp_dir in temp_dirs_to_check:
        if os.path.exists(temp_dir):
            try:
                size_before = sum(os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(temp_dir)
                                for filename in filenames)
                
                shutil.rmtree(temp_dir)
                total_freed += size_before
                print(f"  [OK] Cleaned: {temp_dir} ({size_before / (1024*1024):.1f} MB)")
                
            except Exception as e:
                print(f"  [WARNING] Could not clean {temp_dir}: {e}")
    
    if total_freed > 0:
        print(f"[SUCCESS] Total space freed: {total_freed / (1024*1024):.1f} MB")
    else:
        print(f"[INFO] No Ray temp files found to clean")

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_federated_learning_experiment(strategy_name: str, model_name: str, partitions: dict) -> dict:
    """Run a single federated learning experiment"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {strategy_name.upper()} + {model_name.upper()}")
    print(f"{'='*80}")
    
    # Create run configuration
    run_config = {
        "strategy": strategy_name,
        "model_name": model_name,
        "model_params": MODEL_PARAMS.get(model_name, {}),
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_split": TEST_SPLIT,
        "proximal_mu": PROXIMAL_MU if strategy_name == "fedprox" else 0.0,
        "use_focal_loss": USE_FOCAL_LOSS,
        "partitions": partitions,
        "partition_type": PARTITION_TYPE,
        "quiet_mode": QUIET_MODE,
        "enable_heterogeneous": ENABLE_HETEROGENEOUS,
        "client_device_types": CLIENT_DEVICE_TYPES,
        "device_model_mapping": DEVICE_MODEL_MAPPING,
    }

    # Add ULCD-specific parameters
    if strategy_name == "ulcd":
        run_config.update({
            "latent_dim": ULCD_LATENT_DIM,
            "anomaly_threshold": ULCD_ANOMALY_THRESHOLD,
            "enable_visualization": ULCD_ENABLE_VISUALIZATION,
        })

    def client_fn(context: Context) -> fl.client.Client:
        try:
            # Map Flower's large random node_id to valid client_id range [0, NUM_CLIENTS-1]
            node_id = int(context.node_id)
            client_id = node_id % NUM_CLIENTS

            # Determine client's model based on device type (heterogeneous FL)
            if ENABLE_HETEROGENEOUS and strategy_name in ["ulcd", "fedavg_latent"]:
                device_type = CLIENT_DEVICE_TYPES.get(client_id, "standard")
                client_model_name = DEVICE_MODEL_MAPPING.get(device_type, model_name)
                print(f"[HETEROGENEOUS] Client {client_id}: Device={device_type}, Model={client_model_name}")
            else:
                client_model_name = model_name

            if not QUIET_MODE:
                print(f"[SIMULATION] Creating client {client_id} (node_id: {node_id})")

            # Ensure partitions are available
            if not GLOBAL_PARTITIONS:
                raise ValueError("GLOBAL_PARTITIONS not initialized")

            if client_id not in GLOBAL_PARTITIONS:
                raise ValueError(f"Client {client_id} not found in partitions. Available: {list(GLOBAL_PARTITIONS.keys())}")

            # Create client-specific run config with assigned model
            client_run_config = run_config.copy()
            client_run_config["model_name"] = client_model_name
            client_run_config["model_params"] = MODEL_PARAMS.get(client_model_name, {})

            client = get_client(client_run_config, client_id)
            if client is None:
                raise ValueError(f"get_client returned None for client {client_id}")

            # Convert NumPyClient to Client to avoid deprecation warning
            flower_client = client.to_client()
            if flower_client is None:
                raise ValueError(f"to_client() returned None for client {client_id}")

            if not QUIET_MODE:
                print(f"[SIMULATION] Successfully created client {client_id}")
            return flower_client
            
        except Exception as e:
            # Make sure we have client_id defined for error reporting
            try:
                client_id = int(context.node_id) % NUM_CLIENTS
            except:
                client_id = "unknown"
            print(f"[ERROR] Failed to create client {context.node_id} (mapped to {client_id}): {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    # Get strategy with correct latent dimension for the model
    model_latent_dim = MODEL_PARAMS.get(model_name, {}).get('latent_dim', ULCD_LATENT_DIM)
    strategy = get_strategy(
        strategy_name, 
        run_config=run_config,
        latent_dim=model_latent_dim,
        anomaly_threshold=ULCD_ANOMALY_THRESHOLD,
        enable_visualization=ULCD_ENABLE_VISUALIZATION,
        proximal_mu=PROXIMAL_MU
    )
    
    # Configure simulation
    simulation_args = {
        "client_fn": client_fn,
        "num_clients": NUM_CLIENTS,
        "config": fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        "strategy": strategy,
    }
    
    # Set client resources with more conservative allocations
    if not USE_LOCAL_MODE:
        simulation_args["client_resources"] = {
            "num_cpus": 1.0,
            "memory": 800000000,  # 800MB per client (reduced from 1.5GB)
            "num_gpus": 0.0 if model_name == "random_forest" else 0.1
        }
        
        # Don't override Ray init args in simulation - let Ray handle initialization
        # simulation_args["ray_init_args"] is removed to avoid conflicts
    else:
        simulation_args["client_resources"] = {"num_cpus": 0.5, "memory": 300000000}
    
    # Run simulation
    start_time = time.time()
    try:
        print(f"Starting FL simulation: {strategy_name} + {model_name}")
        history = fl.simulation.start_simulation(**simulation_args)
        print(f"Simulation completed successfully")
    except Exception as e:
        print(f"Simulation failed: {e}")
        return {
            "model_name": model_name,
            "strategy": strategy_name,
            "success": False,
            "error": str(e),
            "training_time": 0,
            "accuracy": 0.0,
            "f1": 0.0,
        }
    
    training_time = time.time() - start_time
    
    # Extract results
    results = {
        "model_name": model_name,
        "strategy": strategy_name,
        "success": True,
        "training_time": training_time,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "partition_type": PARTITION_TYPE,
    }
    
    # Collect model size from fit metrics with debugging
    model_size_mb = 0.0
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        print(f"[DEBUG] Available fit metrics: {list(history.metrics_distributed_fit.keys())}")
        if "model_size_mb" in history.metrics_distributed_fit:
            size_data = history.metrics_distributed_fit["model_size_mb"]
            if size_data:
                model_size_mb = size_data[-1][1]  # Get the last reported size
                print(f"[DEBUG] Model size from history: {model_size_mb:.2f} MB")
            else:
                print(f"[DEBUG] model_size_mb data is empty")
        else:
            print(f"[DEBUG] model_size_mb not found in fit metrics")
    else:
        print(f"[DEBUG] No metrics_distributed_fit available")
    
    # Fallback: Calculate model size directly if not found
    if model_size_mb == 0.0:
        try:
            # Create a dummy model to get size
            dummy_model = get_model(model_name, 3072, 10, **MODEL_PARAMS.get(model_name, {}))
            model_weights = get_weights(dummy_model)
            from shared.ulcd_components import calculate_model_size_mb
            model_size_mb = calculate_model_size_mb(model_weights)
            print(f"[DEBUG] Calculated model size directly: {model_size_mb:.2f} MB")
        except Exception as e:
            print(f"[DEBUG] Failed to calculate model size: {e}")
    
    results["model_size_mb"] = model_size_mb
    
    # Get final metrics
    if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
        # Core metrics
        for metric in ["accuracy", "f1", "precision", "recall", "top5_accuracy", "loss"]:
            if metric in history.metrics_distributed:
                metric_data = history.metrics_distributed[metric]
                results[metric] = metric_data[-1][1] if metric_data else 0.0
                results[f"{metric}_history"] = [round_data[1] for round_data in metric_data]
        
        # Per-class accuracies
        for class_name in CIFAR10_CLASSES:
            acc_key = f"{class_name}_accuracy"
            if acc_key in history.metrics_distributed:
                results[acc_key] = history.metrics_distributed[acc_key][-1][1]
    
    # Get training loss
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        if "train_loss" in history.metrics_distributed_fit:
            train_losses = history.metrics_distributed_fit["train_loss"]
            results["final_train_loss"] = train_losses[-1][1] if train_losses else 0.0
            results["train_loss_history"] = [round_data[1] for round_data in train_losses]
    
    # ULCD-specific metrics
    if strategy_name == "ulcd":
        for metric in ["ulcd_trusted_clients", "ulcd_flagged_clients", "ulcd_consensus_ratio", "ulcd_prototype_norm"]:
            if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
                if metric in history.metrics_distributed_fit:
                    results[metric] = history.metrics_distributed_fit[metric][-1][1]
    
    print(f"Experiment completed: {strategy_name} + {model_name}")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"  F1 Score: {results.get('f1', 0):.4f}")
    print(f"  Training Time: {training_time:.1f}s")
    
    return results

# ==============================================================================
# ANALYSIS AND VISUALIZATION
# ==============================================================================

def create_comparison_tables(all_results: List[dict]):
    """Create comprehensive comparison tables"""
    
    print(f"\n{'='*80}")
    print("CIFAR-10 FEDERATED LEARNING RESULTS")
    print(f"{'='*80}")
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results to display")
        return
    
    # Table 1: Overall Performance
    print(f"\n[TABLE] OVERALL PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"{'Strategy+Model':<20} {'Accuracy (%)':<12} {'F1 Score':<10} {'Top-5 Acc (%)':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        if row['success']:
            experiment_id = f"{row['strategy']}_{row['model_name']}"
            accuracy = row.get('accuracy', 0) * 100
            f1 = row.get('f1', 0)
            top5 = row.get('top5_accuracy', 0) * 100
            time_s = row.get('training_time', 0)
            
            print(f"{experiment_id:<20} {accuracy:<12.2f} {f1:<10.4f} {top5:<12.2f} {time_s:<10.1f}")
    
    print("-" * 70)
    
    # Table 2: Per-class Performance (if available)
    print(f"\n[TABLE] PER-CLASS ACCURACY (%)")
    print("-" * 100)
    header = f"{'Strategy+Model':<20}"
    for class_name in CIFAR10_CLASSES[:5]:  # Show first 5 classes
        header += f"{class_name[:8]:<12}"
    print(header)
    print("-" * 100)
    
    for _, row in df.iterrows():
        if row['success']:
            experiment_id = f"{row['strategy']}_{row['model_name']}"
            line = f"{experiment_id:<20}"
            
            for class_name in CIFAR10_CLASSES[:5]:
                acc_key = f"{class_name}_accuracy"
                acc_val = row.get(acc_key, 0) * 100
                line += f"{acc_val:<12.2f}"
            
            print(line)
    
    print("-" * 100)
    
    # Table 3: Strategy Comparison
    strategies = df['strategy'].unique()
    if len(strategies) > 1:
        print(f"\n[TABLE] STRATEGY COMPARISON")
        print("-" * 50)
        print(f"{'Strategy':<15} {'Avg Accuracy':<15} {'Avg F1':<10} {'Avg Time':<10}")
        print("-" * 50)
        
        for strategy in strategies:
            strategy_results = df[df['strategy'] == strategy]
            avg_acc = strategy_results['accuracy'].mean() * 100
            avg_f1 = strategy_results['f1'].mean()
            avg_time = strategy_results['training_time'].mean()
            
            print(f"{strategy:<15} {avg_acc:<15.2f} {avg_f1:<10.4f} {avg_time:<10.1f}")
        
        print("-" * 50)

def plot_training_progression(all_results: List[dict]):
    """Plot training progression for all experiments"""
    
    print(f"\n[PLOT] Generating training progression plots...")
    
    # Disable interactive plotting
    plt.ioff()
    
    # Create plots directory
    os.makedirs('fl_plots', exist_ok=True)
    
    # Filter results with history
    results_with_history = [r for r in all_results if r.get('accuracy_history')]
    
    if not results_with_history:
        print("[WARNING] No training history available for plotting")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, result in enumerate(results_with_history):
        experiment_id = f"{result['strategy']}_{result['model_name']}"
        color = colors[i % len(colors)]
        rounds = range(1, len(result['accuracy_history']) + 1)
        
        # Accuracy progression
        if 'accuracy_history' in result:
            acc_history = [acc * 100 for acc in result['accuracy_history']]
            ax1.plot(rounds, acc_history, color=color, linewidth=2, marker='o', 
                    markersize=4, label=experiment_id)
        
        # F1 progression
        if 'f1_history' in result:
            ax2.plot(rounds, result['f1_history'], color=color, linewidth=2, marker='s', 
                    markersize=4, label=experiment_id)
        
        # Training loss progression
        if 'train_loss_history' in result:
            ax3.plot(rounds, result['train_loss_history'], color=color, linewidth=2, marker='^', 
                    markersize=4, label=experiment_id)
        
        # Top-5 accuracy progression
        if 'top5_accuracy_history' in result:
            top5_history = [acc * 100 for acc in result['top5_accuracy_history']]
            ax4.plot(rounds, top5_history, color=color, linewidth=2, marker='d', 
                    markersize=4, label=experiment_id)
    
    # Configure subplots
    ax1.set_title('Accuracy Progression')
    ax1.set_xlabel('FL Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('F1 Score Progression')
    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_title('Training Loss Progression')
    ax3.set_xlabel('FL Round')
    ax3.set_ylabel('Training Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_title('Top-5 Accuracy Progression')
    ax4.set_xlabel('FL Round')
    ax4.set_ylabel('Top-5 Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_filename = f'fl_plots/cifar10_training_progression_{timestamp}.png'
    
    try:
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Training progression plot: {plot_filename}")
    except Exception as e:
        print(f"[WARNING] Could not save plot: {e}")
    
    plt.close()

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Run comprehensive CIFAR-10 federated learning experiments"""
    
    print(f"\n{'='*80}")
    print("CIFAR-10 FEDERATED LEARNING EXPERIMENT SUITE")
    print(f"{'='*80}")
    print(f"Data directory: {CIFAR_DATA_DIR}")
    print(f"Partition type: {PARTITION_TYPE}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Strategies to test: {', '.join(STRATEGIES_TO_COMPARE)}")
    print(f"Models to test: {', '.join(MODELS_TO_TEST)}")
    print(f"FL Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print(f"{'='*80}")
    
    # Load and partition CIFAR-10 data
    print(f"\nLoading CIFAR-10 data and creating federated partitions...")
    partitions = load_and_partition_data(
        data_dir=CIFAR_DATA_DIR,
        partition_type=PARTITION_TYPE,
        num_clients=NUM_CLIENTS,
        **PARTITION_KWARGS
    )
    
    if not partitions:
        print("ERROR: Failed to create data partitions")
        return
    
    # Store partitions globally for client access
    global GLOBAL_PARTITIONS
    GLOBAL_PARTITIONS = partitions
    
    # Initialize Ray
    try:
        ray.shutdown()
        print("[SHUTDOWN] Existing Ray instance shut down")
        cleanup_ray_temp_files()
        force_memory_cleanup()
    except Exception as e:
        print(f"[INFO] No existing Ray instance: {e}")
        cleanup_ray_temp_files()
    
    # Ray initialization with more conservative settings
    if USE_LOCAL_MODE:
        ray_init_args = {
            "ignore_reinit_error": True,
            "log_to_driver": not QUIET_MODE,
            "include_dashboard": False,
            "num_cpus": 4,
            "object_store_memory": 100000000,  # 100MB
            "local_mode": True,
        }
    else:
        ray_init_args = {
            "ignore_reinit_error": True,
            "log_to_driver": not QUIET_MODE,
            "include_dashboard": False,
            "num_cpus": min(16, NUM_CLIENTS * 2),  # Scale with clients
            "object_store_memory": 150000000,  # 150MB (further reduced)
            "num_gpus": 1 if torch.cuda.is_available() else 0,
            "_temp_dir": "/tmp/ray_fl_temp",  # Use custom temp directory
        }
    
    ray.init(**ray_init_args)
    print(f"[RAY] Initialized with {NUM_CLIENTS} clients")
    
    # Run experiments - Clear any cached results
    all_results = []
    total_experiments = len(STRATEGIES_TO_COMPARE) * len(MODELS_TO_TEST)
    experiment_count = 0
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT PLAN: {total_experiments} total experiments")
    print(f"Strategies: {list(STRATEGIES_TO_COMPARE)}")
    print(f"Models: {list(MODELS_TO_TEST)}")
    print(f"{'='*80}")
    
    for strategy_name in STRATEGIES_TO_COMPARE:
        print(f"\n{'='*80}")
        print(f"TESTING STRATEGY: {strategy_name.upper()}")
        print(f"{'='*80}")
        
        for model_name in MODELS_TO_TEST:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] Strategy: {strategy_name.upper()} | Model: {model_name.upper()}")
            
            try:
                # Define ULCD-compatible models
                ulcd_compatible_models = ["ulcd", "cnn_ulcd", "cnn_ulcd_light", "cnn_ulcd_heavy"]

                # Skip incompatible combinations
                if strategy_name == "ulcd" and model_name not in ulcd_compatible_models:
                    print(f"   [SKIP] {model_name} not optimized for ULCD strategy")
                    continue

                # Restrict fedavg_latent to only ULCD-compatible models for proper latent extraction
                if strategy_name == "fedavg_latent" and model_name not in ulcd_compatible_models:
                    print(f"   [SKIP] fedavg_latent strategy requires ULCD-compatible model for proper latent extraction")
                    continue

                # Skip ULCD models with incompatible strategies
                # Allow fedavg_latent to use ULCD models for latent extraction
                if strategy_name not in ["ulcd", "fedavg_latent"] and model_name in ulcd_compatible_models:
                    print(f"   [SKIP] ULCD model {model_name} only works with ULCD or FedAvg+Latent strategies")
                    continue
                
                # Force memory cleanup before each experiment
                force_memory_cleanup()
                
                # Run experiment
                results = run_federated_learning_experiment(strategy_name, model_name, partitions)
                all_results.append(results)
                
                # Force cleanup after experiment
                force_memory_cleanup()
                time.sleep(2)  # Brief pause
                
            except Exception as e:
                print(f"ERROR in experiment {strategy_name} + {model_name}: {e}")
                all_results.append({
                    "model_name": model_name,
                    "strategy": strategy_name,
                    "success": False,
                    "error": str(e),
                    "accuracy": 0.0,
                    "f1": 0.0,
                })
    
    # Generate comprehensive analysis
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    create_comparison_tables(all_results)
    plot_training_progression(all_results)
    
    # Save metrics summary
    experiment_name = f"cifar10_fl_{PARTITION_TYPE}_{NUM_CLIENTS}clients"
    results_data = {
        "all_results": all_results,
        "configuration": {
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "num_clients": NUM_CLIENTS,
            "partition_type": PARTITION_TYPE,
            "models_tested": list(MODELS_TO_TEST),
            "strategies_tested": list(STRATEGIES_TO_COMPARE),
        }
    }
    
    # Save TXT metrics summary
    metrics_txt_file = save_metrics_to_txt(results_data, experiment_name)
    
    # Compare ULCD vs Traditional FL if both were tested
    ulcd_results = [r for r in all_results if r['strategy'] == 'ulcd' and r['success']]
    traditional_results = [r for r in all_results if r['strategy'] in ['fedavg', 'fedprox'] and r['success']]
    
    if ulcd_results and traditional_results:
        # Average results for comparison
        avg_ulcd = {
            'accuracy': np.mean([r.get('accuracy', 0) for r in ulcd_results]),
            'f1': np.mean([r.get('f1', 0) for r in ulcd_results]),
            'top5_accuracy': np.mean([r.get('top5_accuracy', 0) for r in ulcd_results])
        }
        avg_traditional = {
            'accuracy': np.mean([r.get('accuracy', 0) for r in traditional_results]),
            'f1': np.mean([r.get('f1', 0) for r in traditional_results]),
            'top5_accuracy': np.mean([r.get('top5_accuracy', 0) for r in traditional_results])
        }
        compare_ulcd_vs_traditional_fl(avg_ulcd, avg_traditional)
    
    # Final cleanup
    try:
        ray.shutdown()
        print("[SHUTDOWN] Ray instance shut down successfully")
        cleanup_ray_temp_files()
        force_memory_cleanup()
    except Exception as e:
        print(f"[WARNING] Error shutting down Ray: {e}")
        cleanup_ray_temp_files()
    
    print(f"\n{'='*80}")
    print("CIFAR-10 FEDERATED LEARNING EXPERIMENTS COMPLETED")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful experiments: {sum(1 for r in all_results if r.get('success', False))}")
    print(f"Metrics summary saved to: {metrics_txt_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()