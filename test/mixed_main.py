import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .models import CNNModel, MLPModel, ResNetModel
from .mixed_client import MixedClient
from .mixed_server import MixedServer
from .logger import FLLogger
from .data_utils import get_heterogeneous_dataloaders
from . import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Data loading is now handled by data_utils.get_heterogeneous_dataloaders()

def evaluate_ensemble(models, test_loader, device):
    for m in models: m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs = []
            for m in models:
                # Handle tuple return if using return_protos=True, though defaults to False
                out = m(x)
                if isinstance(out, tuple): out = out[0]
                probs.append(torch.softmax(out, dim=1))
            
            avg_prob = torch.stack(probs).mean(0)
            all_preds.extend(avg_prob.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }

def run():
    set_seed(config.SEED)
    device = torch.device("cuda" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")
    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name="FedMixed")

    print("="*80)
    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Feature Dimension: {config.FEATURE_DIM}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    lr_decay_status = "Disabled" if config.LR_DECAY_GAMMA == 1.0 else f"Step={config.LR_DECAY_STEP}, Gamma={config.LR_DECAY_GAMMA}"
    print(f"LR Decay: {lr_decay_status}")
    print(f"Proto Weight: {config.PROTOTYPE_WEIGHT}")
    print(f"Distill Weight: {config.MIXED_DISTILL_WEIGHT}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Epochs per Round: {config.EPOCHS_PER_ROUND}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Label Heterogeneity: {config.LABEL_HETEROGENEITY}")
    print(f"Eval Frequency: Every {config.EVAL_FREQ} rounds")
    print(f"Checkpoint Frequency: Every {config.CHECKPOINT_FREQ} rounds")
    print("="*80)

    client_loaders, public_loaders, test_loader, client_info = get_heterogeneous_dataloaders()

    models = [
        CNNModel(num_classes=10, feature_dim=config.FEATURE_DIM),
        MLPModel(num_classes=10, feature_dim=config.FEATURE_DIM),
        ResNetModel(num_classes=10, feature_dim=config.FEATURE_DIM)
    ]

    clients = [MixedClient(model.to(device), cl, pl, i)
               for i, (model, cl, pl) in enumerate(zip(models, client_loaders, public_loaders))]
    
    server = MixedServer(num_classes=10)

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\nRound {round_num}")
        
        # 1. Broadcast Global Knowledge
        g_protos, g_logits = server.broadcast()
        
        # 2. Train Clients & Collect New Knowledge
        client_summaries = []
        client_logits = []
        
        for client in clients:
            # Train using BOTH protos and logits
            client.train(config.EPOCHS_PER_ROUND, g_protos, g_logits, round_num)
            
            # Compute new Prototypes
            client_summaries.append(client.compute_prototypes())
            
            # Compute new Logits
            client_logits.append(client.get_public_logits())
        
        # 3. Aggregate
        print("Server: Aggregating")
        server.aggregate(client_summaries, client_logits)
        
        # 4. Logging & Eval
        logger.log_round(round_num)
        if round_num % config.EVAL_FREQ == 0:
            clients_metrics = [evaluate_model(c.model, test_loader, device) for c in clients]
            ens_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
            logger.log_evaluation(round_num, clients_metrics, ens_metrics)
            logger.check_convergence(round_num)
            print(f"Ensemble Acc: {ens_metrics['accuracy']:.4f}")

    logger.save_results()
    print("Completed")

if __name__ == "__main__":
    run()