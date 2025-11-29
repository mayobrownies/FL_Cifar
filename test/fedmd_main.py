import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models import CNNModel, MLPModel, ResNetModel
from fedmd_client import FedMDClient
from fedmd_server import FedMDServer
from logger import FLLogger
import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    cifar = datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True, transform=train_transform)
    cifar_public = datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True, transform=train_transform)
    cifar_test = datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True, transform=test_transform)

    public_idxs = list(range(config.PUBLIC_SIZE))
    public_set = Subset(cifar_public, public_idxs)
    public_loader = DataLoader(public_set, batch_size=config.BATCH_SIZE, shuffle=True)

    X_public_list = []
    y_public_list = []
    for x, y in public_loader:
        X_public_list.append(x)
        y_public_list.append(y)
    X_public = torch.cat(X_public_list, dim=0)
    y_public = torch.cat(y_public_list, dim=0)
    public_dataset = (X_public, y_public)

    test_loader = DataLoader(cifar_test, batch_size=config.BATCH_SIZE, shuffle=False)

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
        client_loaders.append(DataLoader(client_set, batch_size=config.PRIVATE_TRAINING_BATCHSIZE, shuffle=True))

    return client_loaders, public_dataset, test_loader


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

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
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

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
    }


def save_checkpoint(clients, server, round_num, best_acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'round': round_num,
        'best_acc': best_acc,
        'consensus_logits': server.consensus_logits,
        'models': {i: c.model.state_dict() for i, c in enumerate(clients)}
    }
    torch.save(checkpoint, f"{output_dir}/checkpoint_r{round_num}.pt")
    print(f"[Checkpoint] Saved at round {round_num}")


def run():
    set_seed(config.SEED)
    device = torch.device("cuda" if (torch.cuda.is_available() and config.USE_CUDA) else "cpu")

    logger = FLLogger(output_dir=config.OUTPUT_DIR, experiment_name=config.EXPERIMENT_NAME + "_fedmd")

    print("="*80)
    print("FedMD")
    print("="*80)
    print(f"Device: {device}")
    print(f"Models: {', '.join(config.MODEL_TYPES)}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Alignment Size: {config.N_ALIGNMENT}")
    print(f"Eval Frequency: Every {config.EVAL_FREQ} rounds")
    print(f"Checkpoint Frequency: Every {config.CHECKPOINT_FREQ} rounds")
    print("="*80)

    client_loaders, public_dataset, test_loader = get_dataloaders()

    models = [CNNModel(), MLPModel(), ResNetModel()]

    dummy_public_loader = DataLoader([], batch_size=1)

    clients = [
        FedMDClient(model.to(device), cl, dummy_public_loader, i)
        for i, (model, cl) in enumerate(zip(models, client_loaders))
    ]

    server = FedMDServer(num_classes=10)

    best_acc = 0.0
    best_round = 0

    for round_num in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{config.NUM_ROUNDS}")
        print(f"{'='*80}")

        print(f"\n[Step 1] Generate alignment data (N={config.N_ALIGNMENT})")
        alignment_batches, X_alignment, y_alignment = server.generate_alignment_data(
            public_dataset, config.N_ALIGNMENT, batch_size=config.LOGITS_MATCHING_BATCHSIZE
        )

        print(f"\n[Step 2] Collect logits from all clients")
        client_logits = []
        for i, client in enumerate(clients):
            logits = client.get_public_logits(alignment_batches)
            client_logits.append(logits)
            print(f"  Client {i}: Collected {logits.shape[0]} logits")

        print(f"\n[Step 3] Server aggregates logits")
        server.aggregate_logits(client_logits)
        consensus_logits = server.broadcast()

        print(f"\n[Step 4] Clients train with consensus")
        for client in clients:
            print(f"\nClient {client.client_id}:")
            client.train(
                logits_matching_epochs=config.N_LOGITS_MATCHING_ROUND,
                private_training_epochs=config.N_PRIVATE_TRAINING_ROUND,
                alignment_data=alignment_batches,
                consensus_logits=consensus_logits,
                round_num=round_num
            )

        logger.log_round(round_num)

        if round_num % config.EVAL_FREQ == 0 or round_num == config.NUM_ROUNDS:
            print(f"\n[Evaluation Round {round_num}]")
            clients_metrics = []
            for i, client in enumerate(clients):
                metrics = evaluate_model(client.model, test_loader, device)
                clients_metrics.append(metrics)
                print(f"  {config.MODEL_TYPES[i]}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

            ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
            print(f"  Ensemble: Acc={ensemble_metrics['accuracy']:.4f}, F1={ensemble_metrics['f1_macro']:.4f}")

            logger.log_evaluation(round_num, clients_metrics, ensemble_metrics)

            if ensemble_metrics['accuracy'] > best_acc:
                best_acc = ensemble_metrics['accuracy']
                best_round = round_num
                save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/best_fedmd")
                print(f"  [NEW BEST] Accuracy: {best_acc:.4f}")

        if round_num % config.CHECKPOINT_FREQ == 0:
            save_checkpoint(clients, server, round_num, best_acc, config.OUTPUT_DIR + "/checkpoints_fedmd")

        if round_num % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS (FedMD)")
    print("="*80)

    for i, (client, model_name) in enumerate(zip(clients, config.MODEL_TYPES)):
        print(f"\nClient {i} ({model_name}):")
        metrics = evaluate_model(client.model, test_loader, device)
        print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"F1 (Macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 (Weighted):{metrics['f1_weighted']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")

    print(f"\nEnsemble (Global Model):")
    ensemble_metrics = evaluate_ensemble([c.model for c in clients], test_loader, device)
    print(f"Accuracy:     {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):   {ensemble_metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):{ensemble_metrics['f1_weighted']:.4f}")
    print(f"Precision:    {ensemble_metrics['precision']:.4f}")
    print(f"Recall:       {ensemble_metrics['recall']:.4f}")

    print(f"\nBest Ensemble Accuracy: {best_acc:.4f} at Round {best_round}")

    print("\n" + "="*80)
    print("COMMUNICATION COSTS (FedMD)")
    print("="*80)
    upload_bytes = config.N_ALIGNMENT * 10 * 4
    download_bytes = config.N_ALIGNMENT * 10 * 4
    bytes_per_round_per_client = upload_bytes + download_bytes
    total_bytes = bytes_per_round_per_client * config.NUM_CLIENTS * config.NUM_ROUNDS
    total_mb = total_bytes / (1024 * 1024)
    comm_efficiency = ensemble_metrics['accuracy'] / total_mb

    print(f"Bytes per round per client: {bytes_per_round_per_client:,} bytes ({bytes_per_round_per_client/1024:.2f} KB)")
    print(f"Total communication:        {total_bytes:,} bytes ({total_mb:.2f} MB)")
    print(f"Communication efficiency:   {comm_efficiency:.4f} (Accuracy/MB)")

    logger.log_communication(bytes_per_round_per_client, total_bytes, total_mb, comm_efficiency)
    logger.log_best(best_acc, best_round)

    logger.save_results()
    logger.plot_metrics()

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run()
