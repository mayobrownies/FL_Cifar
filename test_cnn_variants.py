import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from shared.models import get_model
from shared.data_utils import load_cifar10_dataset, CIFAR10Dataset, get_cifar10_transforms

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, f1, precision, recall

def train_and_evaluate_model(model_name, model, train_loader, test_loader, device, epochs=50):
    """Train and evaluate a model"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Epochs: {epochs}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc, test_f1, test_precision, test_recall = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Test Acc: {test_acc:6.2f}% | F1: {test_f1:.4f} | Best: {best_acc:6.2f}%")

    final_acc, final_f1, final_precision, final_recall = evaluate(model, test_loader, device)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"Accuracy:  {final_acc:6.2f}%")
    print(f"F1 Score:  {final_f1:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall:    {final_recall:.4f}")
    print(f"Best Acc:  {best_acc:6.2f}%")

    return {
        'accuracy': final_acc,
        'f1': final_f1,
        'precision': final_precision,
        'recall': final_recall,
        'best_acc': best_acc
    }

def main():
    print("=" * 80)
    print("BASELINE CNN VARIANT TESTING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = './cifar-10-batches-py'
    print(f"\nLoading CIFAR-10 from {data_dir}...")
    train_data, train_labels, test_data, test_labels = load_cifar10_dataset(data_dir)
    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=get_cifar10_transforms(train=True))
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=get_cifar10_transforms(train=False))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_configs = [
        ("Basic CNN (FL)", "cnn"),  # This is what FL uses
        ("Lightweight CNN", "cnn_ulcd_light"),
        ("Standard CNN", "cnn_ulcd"),
        ("Heavyweight CNN", "cnn_ulcd_heavy"),
    ]

    results = {}

    for model_name, model_type in model_configs:
        model = get_model(model_type, latent_dim=64)
        results[model_name] = train_and_evaluate_model(
            model_name, model, train_loader, test_loader, device, epochs=50
        )

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':<12} {'Accuracy':<12} {'F1 Score':<12} {'Best Acc':<12}")
    print("-" * 80)

    for model_name, model_type in model_configs:
        model = get_model(model_type, latent_dim=64)
        params = count_parameters(model)
        res = results[model_name]
        print(f"{model_name:<20} {params:<12,} {res['accuracy']:>6.2f}%      {res['f1']:<12.4f} {res['best_acc']:>6.2f}%")

    print("=" * 80)
    print("\nThese results serve as centralized training baselines.")
    print("Compare against heterogeneous federated learning results.")

if __name__ == "__main__":
    main()
