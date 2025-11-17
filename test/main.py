# main.py - FedProto Implementation
# FedProto: Federated Prototype Learning across Heterogeneous Clients
# Method: Per-class prototype aggregation with cosine similarity alignment
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import CNNModel, MLPModel, ResNetModel
from client import Client
from server import Server
from logger import FLLogger
import config

# ---------------------
# Data Setup
# ---------------------
def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True, transform=transform)

    # Create public dataset
    public_idxs = list(range(config.PUBLIC_SIZE))
    public_set = Subset(cifar, public_idxs)
    public_loader = DataLoader(public_set, batch_size=config.BATCH_SIZE, shuffle=False)

    # Test dataset
    test_loader = DataLoader(cifar_test, batch_size=config.BATCH_SIZE, shuffle=False)

    # Non-IID partitioning
    remaining_idxs = list(range(config.PUBLIC_SIZE, len(cifar)))
    random.shuffle(remaining_idxs)
    shard_size = len(remaining_idxs) // (config.NUM_CLIENTS * config.SHARDS_PER_CLIENT)
    client_loaders = []
    for i in range(config.NUM_CLIENTS):
        client_idxs = []
        for _ in range(config.SHARDS_PER_CLIENT):
            shard = remaining_idxs[:shard_size]
            remaining_idxs = remaining_idxs[shard_size:]
            client_idxs.extend(shard)
        client_set = Subset(cifar, client_idxs)
        client_loaders.append(DataLoader(client_set, batch_size=config.BATCH_SIZE, shuffle=True))

    return client_loaders, public_loader, test_loader

# ---------------------
# Evaluation Function
# ---------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

def evaluate_ensemble(models, test_loader, device):
    for model in models:
        model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            ensemble_probs = []
            for model in models:
                logits = model(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                ensemble_probs.append(probs)

            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
            preds = avg_probs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

# ---------------------
# Main training loop
# ---------------------
def run():
    torch.manual_seed(42)
    device = torch.device("cuda" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")

    # Initialize logger
    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name=config.EXPERIMENT_NAME + "_fedproto")

    print("="*80)
    print("FedProto: HETEROGENEOUS FL WITH PER-CLASS PROTOTYPES")
    print("="*80)
    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Method: Per-class prototype aggregation + cosine alignment")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print("="*80)

    client_loaders, public_loader, test_loader = get_dataloaders()
    models = [CNNModel(), MLPModel(), ResNetModel()]
    clients = [Client(model.to(device), cl, public_loader, i) for i, (model, cl) in enumerate(zip(models, client_loaders))]
    server = Server()

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")

        # Step 1: Clients train locally
        server.clear()
        for client in clients:
            prototypes, logits = None, None
            if server.global_prototypes:
                prototypes, logits = server.broadcast()
            client.train(epochs=config.EPOCHS_PER_ROUND, server_prototypes=prototypes,
                        server_logits=logits, round_num=round_num)
            server.collect_logits(client.get_logits_on_public())

        # Step 2: Aggregate prototypes
        client_protos = [client.compute_prototypes() for client in clients]
        server.aggregate_prototypes(client_protos)

        # Step 3: Log round completion
        logger.log_round(round_num)

        # Step 4: Evaluate all clients
        print(f"\n[Evaluation Round {round_num}]")
        clients_metrics = []
        for i, client in enumerate(clients):
            metrics = evaluate_model(client.model, test_loader, device)
            clients_metrics.append(metrics)
            print(f"  Client {i}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

        ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
        print(f"  Ensemble: Acc={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1_macro']:.4f}")

        logger.log_evaluation(round_num, clients_metrics, ensemble_metrics)

        # Step 5: Visualization
        server.visualize_prototypes(round_num)

    # Final Evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS (FedProto)")
    print("="*80)

    for i, (client, model_name) in enumerate(zip(clients, config.MODEL_TYPES)):
        print(f"\nClient {i} ({model_name}):")
        metrics = evaluate_model(client.model, test_loader, device)
        print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 (Macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted):{metrics['f1_weighted']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")

    print(f"\nEnsemble (Global Model):")
    ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
    print(f"  Accuracy:     {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (Macro):   {ensemble_metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted):{ensemble_metrics['f1_weighted']:.4f}")
    print(f"  Precision:    {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:       {ensemble_metrics['recall']:.4f}")

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save results and generate plots
    logger.save_results()
    logger.plot_metrics()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)

if __name__ == "__main__":
    run()
