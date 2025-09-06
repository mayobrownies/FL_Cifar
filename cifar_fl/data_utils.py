import os
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random

# ============================================================================
# CIFAR-10 CLASS MAPPINGS
# ============================================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_TO_INDEX = {class_name: idx for idx, class_name in enumerate(CIFAR10_CLASSES)}
INDEX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_INDEX.items()}

# ============================================================================
# DATA LOADING
# ============================================================================
def load_cifar10_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch file"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape data: [N, 3072] -> [N, 3, 32, 32]
    data = data.reshape(-1, 3, 32, 32)
    
    return data, labels

def load_cifar10_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load complete CIFAR-10 dataset"""
    
    # Load training batches
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        train_data.append(data)
        train_labels.extend(labels)
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    
    # Load test batch
    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    test_labels = np.array(test_labels)
    
    print(f"Loaded CIFAR-10: {len(train_data)} train, {len(test_data)} test samples")
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def get_cifar10_transforms(train: bool = True):
    """Get data transforms for CIFAR-10"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

class CIFAR10Dataset(torch.utils.data.Dataset):
    """Custom CIFAR-10 dataset with transforms"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert from [C, H, W] to [H, W, C] for PIL
        if isinstance(image, np.ndarray):
            image = image.transpose(1, 2, 0)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        return image, label

# ============================================================================
# FEDERATED PARTITIONING SCHEMES
# ============================================================================
def create_iid_partitions(data: np.ndarray, labels: np.ndarray, num_clients: int, 
                         min_samples_per_client: int = 100) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create IID (Independent and Identically Distributed) partitions"""
    
    print(f"Creating {num_clients} IID partitions...")
    
    total_samples = len(data)
    indices = np.random.permutation(total_samples)
    
    partitions = {}
    samples_per_client = total_samples // num_clients
    
    for client_id in range(num_clients):
        start_idx = client_id * samples_per_client
        if client_id == num_clients - 1:
            end_idx = total_samples  # Last client gets remaining samples
        else:
            end_idx = start_idx + samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        client_data = data[client_indices]
        client_labels = labels[client_indices]
        
        partitions[client_id] = (client_data, client_labels)
        
        # Print class distribution
        unique_classes, counts = np.unique(client_labels, return_counts=True)
        class_dist = dict(zip(unique_classes, counts))
        print(f"Client {client_id}: {len(client_labels)} samples, classes: {class_dist}")
    
    return partitions

def create_non_iid_partitions(data: np.ndarray, labels: np.ndarray, num_clients: int,
                             num_shards: int = 20) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create non-IID partitions where each client has samples from only 2 classes"""
    
    print(f"Creating {num_clients} non-IID partitions (2 classes per client)...")
    
    # Sort data by label
    sorted_indices = np.argsort(labels)
    sorted_data = data[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Create shards (each class split into num_shards/10 shards)
    shard_size = len(data) // num_shards
    shards = []
    shard_labels = []
    
    for i in range(num_shards):
        start_idx = i * shard_size
        if i == num_shards - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + shard_size
        
        shards.append(sorted_data[start_idx:end_idx])
        shard_labels.append(sorted_labels[start_idx:end_idx])
    
    # Assign 2 shards to each client
    partitions = {}
    shards_per_client = 2
    
    for client_id in range(num_clients):
        client_shards_idx = list(range(client_id * shards_per_client, (client_id + 1) * shards_per_client))
        
        client_data_list = []
        client_labels_list = []
        
        for shard_idx in client_shards_idx:
            if shard_idx < len(shards):
                client_data_list.append(shards[shard_idx])
                client_labels_list.append(shard_labels[shard_idx])
        
        if client_data_list:
            client_data = np.concatenate(client_data_list, axis=0)
            client_labels = np.concatenate(client_labels_list, axis=0)
            
            partitions[client_id] = (client_data, client_labels)
            
            # Print class distribution
            unique_classes, counts = np.unique(client_labels, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            print(f"Client {client_id}: {len(client_labels)} samples, classes: {class_dist}")
    
    return partitions

def create_dirichlet_partitions(data: np.ndarray, labels: np.ndarray, num_clients: int,
                               alpha: float = 0.5) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create non-IID partitions using Dirichlet distribution"""
    
    print(f"Creating {num_clients} Dirichlet partitions (alpha={alpha})...")
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    partitions = {}
    
    for client_id in range(num_clients):
        client_indices = []
        
        for class_idx in range(num_classes):
            # Sample proportion from Dirichlet
            proportions = np.random.dirichlet([alpha] * num_clients)
            class_samples = len(class_indices[class_idx])
            
            # Assign samples to this client based on proportion
            num_samples = int(proportions[client_id] * class_samples)
            
            if num_samples > 0:
                selected_indices = np.random.choice(
                    class_indices[class_idx], 
                    size=min(num_samples, len(class_indices[class_idx])), 
                    replace=False
                )
                client_indices.extend(selected_indices)
                
                # Remove selected indices to avoid overlap
                class_indices[class_idx] = np.setdiff1d(class_indices[class_idx], selected_indices)
        
        if client_indices:
            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)  # Shuffle to mix classes
            
            client_data = data[client_indices]
            client_labels = labels[client_indices]
            
            partitions[client_id] = (client_data, client_labels)
            
            # Print class distribution
            unique_classes, counts = np.unique(client_labels, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            print(f"Client {client_id}: {len(client_labels)} samples, classes: {class_dist}")
    
    return partitions

def create_pathological_partitions(data: np.ndarray, labels: np.ndarray, num_clients: int,
                                  num_classes_per_client: int = 1) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create pathological non-IID partitions where each client only has samples from specific classes"""
    
    print(f"Creating {num_clients} pathological partitions ({num_classes_per_client} classes per client)...")
    
    num_total_classes = len(np.unique(labels))
    
    # Assign classes to clients
    classes_per_client = {}
    
    if num_classes_per_client * num_clients <= num_total_classes:
        # Each client gets unique classes
        all_classes = list(range(num_total_classes))
        random.shuffle(all_classes)
        
        for client_id in range(num_clients):
            start_idx = client_id * num_classes_per_client
            end_idx = start_idx + num_classes_per_client
            classes_per_client[client_id] = all_classes[start_idx:end_idx]
    else:
        # Classes need to be shared among clients
        for client_id in range(num_clients):
            selected_classes = random.sample(range(num_total_classes), num_classes_per_client)
            classes_per_client[client_id] = selected_classes
    
    partitions = {}
    
    for client_id in range(num_clients):
        client_classes = classes_per_client[client_id]
        
        # Get all samples from client's classes
        client_mask = np.isin(labels, client_classes)
        client_data = data[client_mask]
        client_labels = labels[client_mask]
        
        partitions[client_id] = (client_data, client_labels)
        
        # Print class distribution
        unique_classes, counts = np.unique(client_labels, return_counts=True)
        class_dist = dict(zip(unique_classes, counts))
        print(f"Client {client_id}: {len(client_labels)} samples, assigned classes {client_classes}, distribution: {class_dist}")
    
    return partitions

# ============================================================================
# PARTITION FACTORY
# ============================================================================
def create_federated_partitions(data: np.ndarray, labels: np.ndarray, 
                               partition_type: str = "iid", num_clients: int = 10, 
                               **kwargs) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Factory function to create federated data partitions"""
    
    print(f"\n{'='*60}")
    print(f"CREATING FEDERATED PARTITIONS")
    print(f"{'='*60}")
    print(f"Partition type: {partition_type}")
    print(f"Number of clients: {num_clients}")
    print(f"Total samples: {len(data)}")
    print(f"Classes: {list(range(len(np.unique(labels))))}")
    
    if partition_type == "iid":
        partitions = create_iid_partitions(data, labels, num_clients)
    elif partition_type == "non_iid_2class":
        partitions = create_non_iid_partitions(data, labels, num_clients)
    elif partition_type == "dirichlet":
        alpha = kwargs.get('alpha', 0.5)
        partitions = create_dirichlet_partitions(data, labels, num_clients, alpha)
    elif partition_type == "pathological":
        num_classes_per_client = kwargs.get('num_classes_per_client', 1)
        partitions = create_pathological_partitions(data, labels, num_clients, num_classes_per_client)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    print(f"\nCreated {len(partitions)} client partitions")
    print(f"{'='*60}")
    
    return partitions

# ============================================================================
# DATA LOADERS
# ============================================================================
def create_client_dataloaders(client_data: np.ndarray, client_labels: np.ndarray, 
                            batch_size: int = 32, test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for a client"""
    
    # Split client data into train and test
    num_samples = len(client_data)
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_test
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    # Create datasets
    train_dataset = CIFAR10Dataset(
        client_data[train_indices], 
        client_labels[train_indices], 
        transform=get_cifar10_transforms(train=True)
    )
    
    test_dataset = CIFAR10Dataset(
        client_data[test_indices], 
        client_labels[test_indices], 
        transform=get_cifar10_transforms(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_global_test_loader(test_data: np.ndarray, test_labels: np.ndarray, 
                          batch_size: int = 32) -> DataLoader:
    """Create global test dataloader for evaluation"""
    
    test_dataset = CIFAR10Dataset(
        test_data, 
        test_labels, 
        transform=get_cifar10_transforms(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return test_loader