import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, OrderedDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import copy

from .models import get_model, is_sklearn_model, get_model_type, FocalLoss
from .data_utils import (
    load_cifar10_dataset, create_federated_partitions, 
    create_client_dataloaders, get_global_test_loader,
    CIFAR10_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS
)

# Global variables for client data caching
GLOBAL_PARTITIONS = {}
GLOBAL_TEST_DATA = None
GLOBAL_TEST_LABELS = None

# ============================================================================
# WEIGHT UTILITIES
# ============================================================================
def get_weights(model) -> List[np.ndarray]:
    """Extract model weights as numpy arrays"""
    if is_sklearn_model(model):
        # For sklearn models, return dummy weights
        return [np.array([0.0])]
    
    weights = []
    for param in model.parameters():
        weights.append(param.detach().cpu().numpy())
    return weights

def set_weights(model, weights: List[np.ndarray]) -> None:
    """Set model weights from numpy arrays"""
    if is_sklearn_model(model):
        # For sklearn models, weights are not directly settable
        return
    
    params = list(model.parameters())
    for param, weight in zip(params, weights):
        param.data = torch.tensor(weight, dtype=param.dtype, device=param.device)

# ============================================================================
# DATA LOADING FOR FEDERATED LEARNING
# ============================================================================
def load_and_partition_data(data_dir: str, partition_type: str = "iid", num_clients: int = 10,
                          **partition_kwargs) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 and create federated partitions"""
    
    global GLOBAL_TEST_DATA, GLOBAL_TEST_LABELS
    
    # Load CIFAR-10 dataset
    train_data, train_labels, test_data, test_labels = load_cifar10_dataset(data_dir)
    
    # Store global test data
    GLOBAL_TEST_DATA = test_data
    GLOBAL_TEST_LABELS = test_labels
    
    # Create federated partitions
    partitions = create_federated_partitions(
        train_data, train_labels, 
        partition_type=partition_type, 
        num_clients=num_clients,
        **partition_kwargs
    )
    
    return partitions

def load_client_data(client_id: int, batch_size: int = 32, test_split: float = 0.2,
                    partitions: Dict = None) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load data for a specific client"""
    
    global GLOBAL_PARTITIONS
    
    if partitions:
        GLOBAL_PARTITIONS = partitions
    
    if client_id not in GLOBAL_PARTITIONS:
        raise ValueError(f"Client {client_id} not found in partitions")
    
    client_data, client_labels = GLOBAL_PARTITIONS[client_id]
    
    # Create client dataloaders
    train_loader, test_loader = create_client_dataloaders(
        client_data, client_labels, batch_size, test_split
    )
    
    # Return input/output dimensions for CIFAR-10
    input_dim = 3072  # 32 * 32 * 3 (flattened CIFAR-10 image)
    output_dim = 10   # 10 classes
    
    print(f"Client {client_id} loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    
    return train_loader, test_loader, input_dim, output_dim

# ============================================================================
# TRAINING
# ============================================================================
def train_sklearn_model(model, trainloader):
    """Train sklearn model"""
    X_list = []
    y_list = []
    
    for features, labels in trainloader:
        if len(features.shape) == 4:  # [B, C, H, W]
            features = features.view(features.size(0), -1)  # Flatten to [B, 3072]
        X_list.append(features.numpy())
        y_list.append(labels.numpy())
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    model.fit_sklearn(X, y)
    
    # Return dummy loss
    return 0.5

def train(net, global_net, trainloader, epochs, learning_rate, proximal_mu, device, 
         use_focal_loss=False, prototype=None, **kwargs):
    """Train model with various optimizations"""
    
    # Handle sklearn models differently
    if is_sklearn_model(net):
        return train_sklearn_model(net, trainloader)
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1)
        print(f"[INFO] Using Focal Loss (gamma=2.0) for imbalanced classification")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print(f"[INFO] Using CrossEntropy Loss with label smoothing (0.1)")
    
    # Use different optimizers based on model type
    model_type = get_model_type(net)
    if model_type == "ulcd":
        # Use AdamW for ULCD (better for transformers)
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    else:
        # Use SGD with momentum for other models
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.1)
    
    net.to(device)
    global_net.to(device)
    net.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            
            # Flatten features for non-convolutional models
            model_type = get_model_type(net)
            if model_type in ["ulcd", "logistic", "mlp", "lstm", "moe"]:
                if len(features.shape) == 4:  # [B, C, H, W]
                    features = features.view(features.size(0), -1)  # [B, 3072]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(features)
            loss = criterion(outputs, labels)
            
            # ULCD prototype alignment loss
            prototype_loss = 0.0
            if prototype is not None and hasattr(net, 'image_encoder'):
                try:
                    latent = net.image_encoder(features)
                    prototype_loss = 0.1 * F.mse_loss(latent.mean(dim=0), prototype)
                    loss += prototype_loss
                except Exception as e:
                    print(f"[WARNING] Prototype alignment failed: {e}")
            
            # FedProx regularization
            if proximal_mu > 0:
                prox_term = 0.0
                for param, global_param in zip(net.parameters(), global_net.parameters()):
                    prox_term += torch.norm(param - global_param) ** 2
                loss += (proximal_mu / 2) * prox_term
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Update learning rate
        if model_type == "ulcd":
            scheduler.step(epoch_loss / max(epoch_batches, 1))
        else:
            scheduler.step()
        
        avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
        total_loss += avg_epoch_loss
        num_batches += 1
        
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

# ============================================================================
# TESTING
# ============================================================================
def test(net, testloader, device, use_global_test=False):
    """Test model and return comprehensive metrics"""
    
    global GLOBAL_TEST_DATA, GLOBAL_TEST_LABELS
    
    if use_global_test and GLOBAL_TEST_DATA is not None:
        # Use global test set
        global_testloader = get_global_test_loader(GLOBAL_TEST_DATA, GLOBAL_TEST_LABELS, batch_size=128)
        testloader = global_testloader
        print(f"[INFO] Using global test set: {len(GLOBAL_TEST_DATA)} samples")
    
    # Handle sklearn models
    if is_sklearn_model(net):
        return test_sklearn_model(net, testloader, device)
    
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            
            # Flatten features for non-convolutional models
            model_type = get_model_type(net)
            if model_type in ["ulcd", "logistic", "mlp", "lstm", "moe"]:
                if len(features.shape) == 4:  # [B, C, H, W]
                    features = features.view(features.size(0), -1)  # [B, 3072]
            
            outputs = net(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # Calculate top-k accuracies
    top5_accuracy = calculate_topk_accuracy(all_probabilities, all_labels, k=5)
    
    # Calculate per-class accuracy
    per_class_accuracy = calculate_per_class_accuracy(all_labels, all_predictions)
    
    avg_loss = total_loss / len(testloader)
    
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "top5_accuracy": top5_accuracy,
        "loss": avg_loss,
        **per_class_accuracy
    }
    
    print(f"[TEST] Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Top-5: {top5_accuracy:.4f}, Loss: {avg_loss:.4f}")
    
    return avg_loss, len(all_labels), metrics

def test_sklearn_model(model, testloader, device):
    """Test sklearn model"""
    X_list = []
    y_list = []
    
    for features, labels in testloader:
        if len(features.shape) == 4:  # [B, C, H, W]
            features = features.view(features.size(0), -1)
        X_list.append(features.numpy())
        y_list.append(labels.numpy())
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if not model.is_trained:
        # Return random performance if not trained
        accuracy = 0.1  # Random chance for 10 classes
        f1 = 0.1
        return 2.3, len(y), {"accuracy": accuracy, "f1": f1, "precision": f1, "recall": f1}
    
    # Get predictions
    probabilities = model(torch.tensor(X, dtype=torch.float32)).numpy()
    predictions = np.argmax(probabilities, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='macro', zero_division=0)
    precision = precision_score(y, predictions, average='macro', zero_division=0)
    recall = recall_score(y, predictions, average='macro', zero_division=0)
    
    # Dummy loss
    loss = 1.0 - accuracy
    
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "loss": loss
    }
    
    return loss, len(y), metrics

def calculate_topk_accuracy(probabilities, labels, k=5):
    """Calculate top-k accuracy"""
    if k >= probabilities.shape[1]:
        k = probabilities.shape[1]
    
    top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
    correct = 0
    
    for i, label in enumerate(labels):
        if label in top_k_predictions[i]:
            correct += 1
    
    return correct / len(labels)

def calculate_per_class_accuracy(labels, predictions):
    """Calculate per-class accuracy"""
    per_class_acc = {}
    
    for class_idx in range(10):  # CIFAR-10 has 10 classes
        class_mask = labels == class_idx
        if np.sum(class_mask) > 0:
            class_predictions = predictions[class_mask]
            class_accuracy = np.sum(class_predictions == class_idx) / len(class_predictions)
            class_name = CIFAR10_CLASSES[class_idx]
            per_class_acc[f"{class_name}_accuracy"] = class_accuracy
        else:
            class_name = CIFAR10_CLASSES[class_idx]
            per_class_acc[f"{class_name}_accuracy"] = 0.0
    
    return per_class_acc

# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================
def analyze_client_data_distribution(partitions: Dict[int, Tuple[np.ndarray, np.ndarray]]):
    """Analyze and print data distribution across clients"""
    
    print(f"\n{'='*80}")
    print("CLIENT DATA DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    total_samples = 0
    overall_class_counts = np.zeros(10)
    
    for client_id, (data, labels) in partitions.items():
        unique_classes, counts = np.unique(labels, return_counts=True)
        class_dist = {CIFAR10_CLASSES[i]: 0 for i in range(10)}
        
        for class_idx, count in zip(unique_classes, counts):
            class_dist[CIFAR10_CLASSES[class_idx]] = count
            overall_class_counts[class_idx] += count
        
        total_samples += len(labels)
        
        print(f"\nClient {client_id}: {len(labels)} samples")
        print(f"  Class distribution: {class_dist}")
        
        # Calculate entropy (measure of diversity)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        print(f"  Data entropy: {entropy:.3f} (max: {np.log2(len(unique_classes)):.3f})")
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"Average samples per client: {total_samples / len(partitions):.1f}")
    
    print(f"\nGlobal class distribution:")
    for i, count in enumerate(overall_class_counts):
        print(f"  {CIFAR10_CLASSES[i]}: {int(count)} ({count/total_samples*100:.1f}%)")
    
    print(f"{'='*80}")

# ============================================================================
# GLOBAL TEST EVALUATION
# ============================================================================
def evaluate_global_model(model, test_data, test_labels, device, batch_size=128):
    """Evaluate model on global test set"""
    
    # Create global test loader
    global_testloader = get_global_test_loader(test_data, test_labels, batch_size)
    
    print(f"\n{'='*60}")
    print("GLOBAL MODEL EVALUATION")
    print(f"{'='*60}")
    
    loss, num_samples, metrics = test(model, global_testloader, device, use_global_test=False)
    
    print(f"Global test results:")
    print(f"  Samples: {num_samples}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"  Loss: {loss:.4f}")
    
    # Print per-class accuracies
    print(f"\nPer-class accuracies:")
    for class_name in CIFAR10_CLASSES:
        acc_key = f"{class_name}_accuracy"
        if acc_key in metrics:
            print(f"  {class_name}: {metrics[acc_key]:.4f}")
    
    print(f"{'='*60}")
    
    return loss, metrics