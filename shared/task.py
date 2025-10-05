import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, OrderedDict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import copy

from shared.models import get_model, is_sklearn_model, get_model_type, FocalLoss
from shared.data_utils import (
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
    model_type = get_model_type(model)

    for i, param in enumerate(model.parameters()):
        weight_array = param.detach().cpu().numpy()
        weights.append(weight_array)

        # Debug CNN weight shapes during extraction
        if model_type == "cnn_ulcd" and i < 3:  # First few layers are likely conv layers
            print(f"[DEBUG] Extracting weight {i}: shape {weight_array.shape}")
            if len(weight_array.shape) == 4:
                print(f"[DEBUG] Conv2d weight {i}: {weight_array.shape} (good)")
            elif len(param.shape) == 4:
                print(f"[WARNING] Conv2d weight {i} flattened: {weight_array.shape} (should be {param.shape})")

    return weights

def set_weights(model, weights: List[np.ndarray]) -> None:
    """Set model weights from numpy arrays with shape validation"""
    if is_sklearn_model(model):
        # For sklearn models, weights are not directly settable
        return

    params = list(model.parameters())
    model_type = get_model_type(model)

    # Check if we received dummy parameters (common issue with ULCD evaluation)
    if len(weights) == 1 and weights[0].size == 1:
        print(f"[CRITICAL] Received single dummy parameter {weights[0]} - skipping weight update entirely")
        print(f"[CRITICAL] This indicates ULCD strategy evaluation parameter issue")
        return

    for i, (param, weight) in enumerate(zip(params, weights)):
        expected_shape = param.shape
        weight_array = np.array(weight)

        # Debug CNN weight shapes
        if model_type == "cnn_ulcd" and i < 3:  # First few layers are likely conv layers
            print(f"[DEBUG] Setting weight {i}: expected shape {expected_shape}, got shape {weight_array.shape}")

            # Validate conv2d weights (should be 4D)
            if len(expected_shape) == 4 and len(weight_array.shape) != 4:
                print(f"[ERROR] Conv2d weight dimension mismatch! Expected 4D {expected_shape}, got {weight_array.shape}")
                print(f"[ERROR] Weight array: {weight_array.flatten()[:10]}...")

                # Check if this is a dummy parameter (single value like [0.])
                if weight_array.size == 1:
                    print(f"[CRITICAL] Received dummy parameter! Skipping weight update for parameter {i}")
                    print(f"[CRITICAL] This indicates ULCD strategy is sending wrong parameters during evaluation")
                    continue  # Skip this weight update, keep existing model weights

                # Try to reshape if possible
                elif weight_array.size == expected_shape.numel():
                    print(f"[FIX] Attempting to reshape weight from {weight_array.shape} to {expected_shape}")
                    weight_array = weight_array.reshape(expected_shape)
                else:
                    print(f"[CRITICAL] Cannot reshape - size mismatch: {weight_array.size} vs {expected_shape.numel()}")
                    raise ValueError(f"Weight size mismatch for parameter {i}")

        param.data = torch.tensor(weight_array, dtype=param.dtype, device=param.device)

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
    
    # Only print if not in quiet mode
    try:
        from shared.run import QUIET_MODE
        if not QUIET_MODE:
            print(f"Client {client_id} loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    except ImportError:
        pass
    
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
         use_focal_loss=False, prototype=None, round_num=1, **kwargs):
    """Train model with various optimizations"""
    
    # Handle sklearn models differently
    if is_sklearn_model(net):
        return train_sklearn_model(net, trainloader)
    
    # Use distillation training for enhanced ULCD models
    model_type = get_model_type(net)
    if model_type == "ulcd" and hasattr(net, 'encode') and prototype is not None:
        return train_with_distillation(net, trainloader, prototype, epochs, learning_rate, 
                                     device, round_num=round_num, **kwargs)
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1)
        print(f"[INFO] Using Focal Loss (gamma=2.0) for imbalanced classification")
    else:
        class_weights = torch.tensor([1.0] * 10).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Using CrossEntropy Loss")
    
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
            model_type = get_model_type(net)
            if prototype is not None and model_type == "cnn_ulcd":
                try:
                    # For all cnn_ulcd variants: extract latent and align with prototype
                    # Simply do a forward pass - the model already computed latent internally
                    # We just need to access it
                    if hasattr(net, 'feature_extractor') and hasattr(net, 'conv1'):
                        # Extract latent by mimicking each model's forward pass
                        if len(features.shape) == 2:
                            x_input = features.view(-1, 3, 32, 32)
                        else:
                            x_input = features

                        # Different architectures have different forward paths
                        if hasattr(net, 'conv4'):
                            # HeavyweightCNN_ULCD: conv1->relu->conv2->relu->pool, conv3->relu->conv4->relu->pool
                            x = net.pool(F.relu(net.conv2(F.relu(net.conv1(x_input)))))
                            if hasattr(net, 'dropout1'):
                                x = net.dropout1(x)
                            x = net.pool(F.relu(net.conv4(F.relu(net.conv3(x)))))
                            if hasattr(net, 'dropout1'):
                                x = net.dropout1(x)
                        elif hasattr(net, 'conv3'):
                            # ULCDCompatibleCNN: conv1->pool, conv2->pool, conv3->pool
                            x = net.pool(F.relu(net.conv1(x_input)))
                            x = net.pool(F.relu(net.conv2(x)))
                            x = F.relu(net.conv3(x))
                            x = net.pool(x)
                            if hasattr(net, 'dropout1'):
                                x = net.dropout1(x)
                        else:
                            # LightweightCNN_ULCD: conv1->pool, conv2->pool
                            x = net.pool(F.relu(net.conv1(x_input)))
                            x = net.pool(F.relu(net.conv2(x)))
                            if hasattr(net, 'dropout'):
                                x = net.dropout(x)

                        x = x.view(x.size(0), -1)

                        # Now get latent (this will have correct dimensions for each model)
                        latent_features = net.feature_extractor(x)

                        # Align with global prototype
                        batch_latent_mean = latent_features.mean(dim=0)

                        # Progressive weight - stronger guidance in later rounds
                        round_num = kwargs.get('round_num', 1)
                        proto_weight = min(0.01 * round_num, 0.1)  # 0.01 -> 0.1 over rounds

                        prototype_loss = proto_weight * F.mse_loss(batch_latent_mean, prototype.to(device))
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
            
            # Gradient clipping for stability (more aggressive for ULCD)
            model_type = get_model_type(net)
            if model_type == "ulcd":
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)  # Tighter clipping for ULCD
            else:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Standard clipping
            
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

def train_with_distillation(net, trainloader, prototype, epochs, learning_rate,
                           device, round_num=1, **kwargs):
    """Enhanced training with knowledge distillation from prototype"""

    # ULTRA-CONSERVATIVE learning rate for maximum stability
    prototype_quality = torch.norm(prototype).item() if prototype is not None else 1.0
    quality_factor = min(0.5, max(0.1, prototype_quality / 10.0))  # Ultra-conservative scaling

    adaptive_lr = learning_rate * quality_factor
    if round_num <= 5:  # Extended warmup period for better convergence
        adaptive_lr *= (0.1 + 0.4 * round_num / 5.0)  # Start much much lower

    # Use SGD with momentum for more stable convergence than AdamW
    optimizer = torch.optim.SGD(net.parameters(), lr=adaptive_lr, momentum=0.9, weight_decay=1e-4)

    # Cyclic learning rate for better exploration and convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=adaptive_lr * 2.0, epochs=epochs,
        steps_per_epoch=len(trainloader), pct_start=0.3
    )
    
    # Loss functions with stronger class balancing to prevent single-class prediction
    # Calculate class weights for balanced training
    class_weights = torch.ones(10)  # Start with equal weights
    for features, labels in trainloader:
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_weights[label] += count.float()

    # Inverse frequency weighting to balance classes
    class_weights = 1.0 / (class_weights + 1.0)
    class_weights = class_weights / class_weights.mean()  # Normalize
    class_weights = class_weights.to(device)

    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
    distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    net.to(device)
    net.train()
    
    total_loss = 0.0
    
    print(f"[INFO] Using Knowledge Distillation Training")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            
            # Flatten features if needed
            if len(features.shape) == 4:
                features = features.view(features.size(0), -1)
            
            optimizer.zero_grad()
            
            # Forward pass with distillation
            try:
                if hasattr(net, 'forward') and len([p for p in net.forward.__code__.co_varnames if p == 'return_distillation']) > 0:
                    logits, distilled, attention = net(features, return_distillation=True)
                    
                    # Classification loss
                    cls_loss = classification_loss_fn(logits, labels)
                    
                    # Class-aware distillation loss
                    dist_loss = 0.0
                    if prototype is not None and distilled is not None:
                        # Handle class-aware prototype [num_classes, latent_dim]
                        if len(prototype.shape) == 2:  # [num_classes, latent_dim]
                            # Use weighted combination of class prototypes based on batch labels
                            batch_class_weights = torch.zeros(prototype.size(0)).to(prototype.device)  # [num_classes]
                            for label in labels:
                                batch_class_weights[label.item()] += 1
                            batch_class_weights = batch_class_weights / batch_class_weights.sum().clamp(min=1)

                            # Weighted average of class prototypes for this batch
                            proto_distilled = torch.sum(prototype * batch_class_weights.unsqueeze(-1), dim=0)  # [latent_dim]
                            proto_distilled = proto_distilled.unsqueeze(0).expand_as(distilled)
                        else:
                            # Legacy single prototype format
                            proto_size = distilled.size(1)
                            if prototype.size(0) >= proto_size:
                                proto_distilled = prototype[:proto_size].unsqueeze(0).expand_as(distilled)
                            else:
                                proto_distilled = None

                        if proto_distilled is not None:
                            # Adaptive temperature scaling based on round and consensus quality
                            base_temp = getattr(net, 'temperature', 4.0)
                            # Higher temperature early in training for softer targets
                            temp_factor = max(0.5, 1.0 - round_num * 0.1)
                            adaptive_temp = base_temp * temp_factor

                            student_soft = F.log_softmax(distilled / adaptive_temp, dim=1)
                            teacher_soft = F.softmax(proto_distilled / adaptive_temp, dim=1)

                            # Add smoothing to teacher targets for stability
                            smoothing = 0.05
                            teacher_soft = teacher_soft * (1 - smoothing) + smoothing / teacher_soft.size(-1)

                            dist_loss = distillation_loss_fn(student_soft, teacher_soft)
                    
                    # BALANCED distillation to prevent loss explosion while maintaining learning
                    base_weight = 0.01 * round_num  # More reasonable progression
                    quality_bonus = min(0.02, prototype_quality / 20.0)  # Small but meaningful bonus
                    dist_weight = min(0.08, base_weight + quality_bonus)  # Higher max for learning

                    # Gradual reduction instead of complete cutoff
                    if dist_loss > 10.0 or cls_loss > 10.0:  # Only for severe explosions
                        dist_weight = 0.0  # Completely disable distillation
                    elif dist_loss > 5.0 or cls_loss > 5.0:  # Moderate instability
                        dist_weight *= 0.1  # Significant reduction
                    elif dist_loss > 3.0 or cls_loss > 3.0:  # Early warning
                        dist_weight *= 0.5  # Moderate reduction

                    loss = cls_loss + dist_weight * dist_loss
                    
                    # Extremely aggressive loss explosion detection - ULTRA-LOW THRESHOLDS
                    if torch.isnan(loss) or torch.isinf(loss) or loss > 3.0:  # Ultra-low threshold
                        print(f"[WARNING] Loss explosion detected: {loss:.4f}, reverting to classification only")
                        loss = cls_loss
                        # Zero out distillation completely to prevent further explosion
                        dist_weight = 0.0
                    elif loss > 2.0:  # Very low warning threshold
                        print(f"[WARNING] High loss detected: {loss:.4f}, reducing distillation")
                        dist_weight *= 0.1
                    
                    if batch_count == 0:  # Log first batch of epoch
                        print(f"    Batch distillation - cls_loss: {cls_loss:.4f}, dist_loss: {dist_loss:.4f}, weight: {dist_weight:.3f}, temp: {adaptive_temp:.2f}, total: {loss:.4f}")
                    
                else:
                    # Fallback to standard training
                    logits = net(features)
                    loss = classification_loss_fn(logits, labels)
                
            except Exception as e:
                print(f"[WARNING] Distillation forward failed: {e}, using standard forward")
                logits = net(features)
                loss = classification_loss_fn(logits, labels)
            
            # Very aggressive gradient clipping for stability
            loss.backward()

            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))

            # ULTRA-AGGRESSIVE clipping for maximum stability
            if loss.item() > 1.5 or grad_norm > 1.5:
                clip_norm = 0.0005  # Ultra-extremely tight clipping
            elif loss.item() > 0.8 or grad_norm > 0.8:
                clip_norm = 0.005   # Extremely tight clipping
            else:
                clip_norm = 0.05    # Very tight clipping even for stable training

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
            
            optimizer.step()

            # Step scheduler after each batch for OneCycleLR
            scheduler.step()

            epoch_loss += loss.item()
            batch_count += 1
        
        # Step scheduler after each batch for OneCycleLR
        # (OneCycleLR steps are handled automatically during training)

        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        total_loss += avg_epoch_loss

        # ULTRA-CONSERVATIVE early stopping - MUCH LOWER THRESHOLDS
        if avg_epoch_loss > 3.0:  # Ultra-low threshold for early stopping
            print(f"  [WARNING] Loss explosion detected ({avg_epoch_loss:.2f}), stopping early")
            break
        elif avg_epoch_loss > 2.0:  # Very low warning threshold
            print(f"  [WARNING] High loss detected ({avg_epoch_loss:.2f}), monitoring closely")

        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
    
    return total_loss / epochs

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
            
            # Debug: Check feature shape before model forward pass
            if len(features.shape) not in [2, 4]:
                print(f"[DEBUG] Unexpected feature shape: {features.shape} for model type: {model_type}")
            
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