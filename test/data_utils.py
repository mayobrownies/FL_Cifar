import torch
import numpy as np
import random
import os
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import cv2
from PIL import Image
from . import config


class CachedTransformDataset(Dataset):
    """Pre-computed cached transform dataset for faster data loading."""

    def __init__(self, cached_tensors, labels):
        """
        Args:
            cached_tensors: Tensor of pre-transformed images [N, C, H, W]
            labels: Tensor of labels [N]
        """
        self.cached_tensors = cached_tensors
        self.labels = labels

    def __len__(self):
        return len(self.cached_tensors)

    def __getitem__(self, idx):
        return self.cached_tensors[idx], self.labels[idx]


class TransformDataset(Dataset):
    """Wrapper dataset that applies custom transforms to existing dataset."""

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class GrayscaleTransform:
    """Convert RGB to grayscale (3 channels)."""
    def __call__(self, img):
        # img is already a tensor from base transform
        if isinstance(img, torch.Tensor):
            # Convert to PIL for grayscale, then back to tensor
            img_pil = transforms.ToPILImage()(img)
            gray = img_pil.convert('L').convert('RGB')
            return transforms.ToTensor()(gray)
        return img


class EdgeDetectionTransform:
    """Apply Canny edge detection."""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert to numpy
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Edge detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # Convert back to 3-channel
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            # Back to tensor
            return torch.from_numpy(edges_rgb.transpose(2, 0, 1)).float() / 255.0
        return img


class BlurTransform:
    """Apply Gaussian blur."""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_pil = transforms.ToPILImage()(img)
            blurred = img_pil.filter(Image.FILTER.GaussianBlur(radius=2))
            return transforms.ToTensor()(blurred)
        return img


class NoiseTransform:
    """Add Gaussian noise."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self.std
            return torch.clamp(img + noise, 0, 1)
        return img


def get_transform_by_name(transform_name):
    """Get transform function by name."""
    base_transform = transforms.Compose([transforms.ToTensor()])

    transform_map = {
        'rgb': None,  # No additional transform
        'grayscale': GrayscaleTransform(),
        'edge': EdgeDetectionTransform(),
        'blur': BlurTransform(),
        'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        'noise': NoiseTransform(std=0.1)
    }

    if transform_name not in transform_map:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(transform_map.keys())}")

    return transform_map[transform_name]


def get_transform_cache_path(transform_name, dataset_hash):
    """Get cache file path for a specific transform and dataset."""
    cache_dir = config.TRANSFORM_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"transform_{transform_name}_{dataset_hash}.pt")


def compute_dataset_hash(dataset):
    """Compute a hash of dataset indices for cache validation."""
    # Use indices as hash (simple but effective)
    if isinstance(dataset, Subset):
        indices_hash = hash(tuple(sorted(dataset.indices)))
    else:
        indices_hash = hash(len(dataset))
    return str(abs(indices_hash))[:16]


def cache_transformed_dataset(base_dataset, transform, transform_name):
    """
    Pre-compute and cache transformed dataset to disk.

    Args:
        base_dataset: Original dataset (Subset)
        transform: Transform function to apply
        transform_name: Name of transform (for cache filename)

    Returns:
        CachedTransformDataset with pre-computed transforms
    """
    dataset_hash = compute_dataset_hash(base_dataset)
    cache_path = get_transform_cache_path(transform_name, dataset_hash)

    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"  Loading cached transform: {transform_name} from {cache_path}")
        cached_data = torch.load(cache_path)
        return CachedTransformDataset(cached_data['tensors'], cached_data['labels'])

    # Pre-compute transforms
    print(f"  Pre-computing transform: {transform_name}... (this may take a minute)")
    transformed_tensors = []
    labels = []

    # Use a simple loader for pre-computation
    temp_loader = DataLoader(base_dataset, batch_size=100, shuffle=False, num_workers=4)

    for imgs, lbls in temp_loader:
        # Apply transform to each image
        if transform is not None:
            batch_transformed = []
            for img in imgs:
                transformed_img = transform(img)
                batch_transformed.append(transformed_img)
            imgs = torch.stack(batch_transformed)

        transformed_tensors.append(imgs)
        labels.append(lbls)

    # Concatenate all batches
    all_tensors = torch.cat(transformed_tensors, dim=0)
    all_labels = torch.cat(labels, dim=0)

    # Save to cache
    torch.save({'tensors': all_tensors, 'labels': all_labels}, cache_path)
    print(f"  Cached to: {cache_path}")

    return CachedTransformDataset(all_tensors, all_labels)


def get_label_heterogeneity_assignment(num_clients, num_classes=10):
    """
    Generate class assignment for each client based on heterogeneity type.

    Returns:
        list of lists: client_classes[i] = list of class indices for client i
    """
    if config.LABEL_HETEROGENEITY == "none":
        # All clients see all classes
        return [list(range(num_classes)) for _ in range(num_clients)]

    elif config.LABEL_HETEROGENEITY == "partial_overlap":
        # Each client gets LABEL_CLASSES_PER_CLIENT classes
        # Consecutive clients have LABEL_OVERLAP_SIZE classes in common
        client_classes = []
        stride = config.LABEL_CLASSES_PER_CLIENT - config.LABEL_OVERLAP_SIZE

        for i in range(num_clients):
            start_idx = (i * stride) % num_classes
            classes = []
            for j in range(config.LABEL_CLASSES_PER_CLIENT):
                classes.append((start_idx + j) % num_classes)
            client_classes.append(sorted(classes))

        return client_classes

    elif config.LABEL_HETEROGENEITY == "non_overlapping":
        # Split classes evenly across clients with no overlap
        classes_per_client = num_classes // num_clients
        if classes_per_client == 0:
            raise ValueError(f"Cannot split {num_classes} classes across {num_clients} clients")

        client_classes = []
        class_list = list(range(num_classes))

        for i in range(num_clients):
            start = i * classes_per_client
            end = start + classes_per_client if i < num_clients - 1 else num_classes
            client_classes.append(class_list[start:end])

        return client_classes

    else:
        raise ValueError(f"Unknown LABEL_HETEROGENEITY: {config.LABEL_HETEROGENEITY}")


def get_data_transform_assignment(num_clients):
    """
    Get transform assignment for each client based on heterogeneity type.

    Returns:
        list: transform_names[i] = transform name for client i
    """
    if config.DATA_HETEROGENEITY == "none":
        # All clients use RGB (no transform)
        return ['rgb'] * num_clients

    elif config.DATA_HETEROGENEITY == "transform":
        # Each client gets a different transform in sequence
        transforms = config.DATA_TRANSFORM_TYPES
        return [transforms[i % len(transforms)] for i in range(num_clients)]

    elif config.DATA_HETEROGENEITY == "mixed":
        # Use explicit assignment or auto-assign
        if config.DATA_TRANSFORM_ASSIGNMENT and len(config.DATA_TRANSFORM_ASSIGNMENT) == num_clients:
            return config.DATA_TRANSFORM_ASSIGNMENT
        else:
            # Default mixed: first half RGB, second half varied
            assignment = []
            half = num_clients // 2
            assignment.extend(['rgb'] * half)
            transforms = config.DATA_TRANSFORM_TYPES[1:]  # Exclude RGB
            for i in range(num_clients - half):
                assignment.append(transforms[i % len(transforms)])
            return assignment

    else:
        raise ValueError(f"Unknown DATA_HETEROGENEITY: {config.DATA_HETEROGENEITY}")


def filter_dataset_by_classes(dataset, allowed_classes):
    """
    Filter dataset to only include samples from allowed_classes.

    Args:
        dataset: PyTorch dataset with (image, label) format
        allowed_classes: List of class indices to keep

    Returns:
        List of indices in dataset that match allowed_classes
    """
    allowed_classes_set = set(allowed_classes)
    filtered_indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in allowed_classes_set:
            filtered_indices.append(idx)

    return filtered_indices


def create_client_datasets(base_dataset, num_clients, public_size, num_classes=10):
    """
    Create heterogeneous client datasets with both label and data type heterogeneity.

    Args:
        base_dataset: Base CIFAR-10 dataset
        num_clients: Number of clients
        public_size: Size of public dataset
        num_classes: Number of classes in dataset

    Returns:
        tuple: (client_datasets, public_dataset, label_assignments, transform_assignments)
    """
    # Get heterogeneity assignments
    client_label_classes = get_label_heterogeneity_assignment(num_clients, num_classes)
    client_transforms = get_data_transform_assignment(num_clients)

    print(f"\nHeterogeneity Configuration:")
    print(f"  Label Heterogeneity: {config.LABEL_HETEROGENEITY}")
    print(f"  Data Heterogeneity: {config.DATA_HETEROGENEITY}")
    print(f"\nClient Class Assignments:")
    for i, classes in enumerate(client_label_classes):
        print(f"  Client {i}: Classes {classes} | Transform: {client_transforms[i]}")

    # Create public dataset (first public_size samples, all classes, RGB)
    public_indices = list(range(public_size))
    public_dataset = Subset(base_dataset, public_indices)

    # Remaining data for clients
    remaining_indices = list(range(public_size, len(base_dataset)))

    # Group indices by class label
    indices_by_class = {i: [] for i in range(num_classes)}
    for idx in remaining_indices:
        _, label = base_dataset[idx]
        indices_by_class[label].append(idx)

    # Shuffle indices within each class
    for class_indices in indices_by_class.values():
        random.shuffle(class_indices)

    # Create per-client datasets
    client_datasets = []

    for client_id in range(num_clients):
        allowed_classes = client_label_classes[client_id]
        client_indices = []

        # Collect samples from allowed classes
        for class_label in allowed_classes:
            class_samples = indices_by_class[class_label]

            # If label heterogeneity is "none", split class samples among clients
            # Otherwise, give each client access to all samples of their classes
            if config.LABEL_HETEROGENEITY == "none":
                # Split evenly using shards
                samples_per_client = len(class_samples) // num_clients
                start = client_id * samples_per_client
                end = start + samples_per_client if client_id < num_clients - 1 else len(class_samples)
                client_indices.extend(class_samples[start:end])
            else:
                # For heterogeneous cases, clients can access all samples of their classes
                # But we can limit via SHARDS_PER_CLIENT if needed
                if hasattr(config, 'SHARDS_PER_CLIENT') and config.SHARDS_PER_CLIENT > 0:
                    shard_size = len(class_samples) // config.SHARDS_PER_CLIENT
                    # Take a portion based on shard configuration
                    for shard_id in range(min(config.SHARDS_PER_CLIENT, 2)):  # Limit to avoid too much data
                        start = shard_id * shard_size
                        end = start + shard_size
                        client_indices.extend(class_samples[start:end])
                else:
                    client_indices.extend(class_samples)

        # Create subset with base dataset
        client_subset = Subset(base_dataset, client_indices)

        # Apply data type transform if needed
        transform_name = client_transforms[client_id]
        transform = get_transform_by_name(transform_name)

        if transform is not None:
            # Use caching if enabled and transform is expensive
            if (config.CACHE_TRANSFORMS and
                transform_name in ['edge', 'blur', 'grayscale', 'noise']):
                client_dataset = cache_transformed_dataset(client_subset, transform, transform_name)
            else:
                client_dataset = TransformDataset(client_subset, transform)
        else:
            client_dataset = client_subset

        client_datasets.append(client_dataset)

    return client_datasets, public_dataset, client_label_classes, client_transforms


def get_public_dataset_tensors(public_loader):
    """
    Convert public_loader to (X_public, y_public) tensor tuple format.
    Used by FedMD which expects tensors instead of a DataLoader.

    Args:
        public_loader: DataLoader for public dataset

    Returns:
        tuple: (X_public tensor, y_public tensor)
    """
    X_public_list = []
    y_public_list = []
    for x, y in public_loader:
        X_public_list.append(x)
        y_public_list.append(y)
    X_public = torch.cat(X_public_list, dim=0)
    y_public = torch.cat(y_public_list, dim=0)
    return (X_public, y_public)


def get_heterogeneous_dataloaders():
    """
    Create dataloaders with heterogeneity support.

    Returns:
        tuple: (client_loaders, public_loaders, test_loader, client_info)
    """
    # Base transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    cifar_train = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    cifar_test = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create heterogeneous client datasets
    client_datasets, public_dataset, client_classes, client_transforms = create_client_datasets(
        cifar_train,
        config.NUM_CLIENTS,
        config.PUBLIC_SIZE,
        num_classes=10
    )

    # DataLoader kwargs with performance optimizations
    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.DATALOADER_NUM_WORKERS,
        'pin_memory': config.DATALOADER_PIN_MEMORY,
    }

    # Add persistent_workers only if num_workers > 0
    # NOTE: Disable persistent_workers when using multi-GPU to avoid thread-safety issues
    if (config.DATALOADER_NUM_WORKERS > 0 and
        config.DATALOADER_PERSISTENT_WORKERS and
        not (config.USE_MULTI_GPU and len(config.GPU_IDS) > 1)):
        dataloader_kwargs['persistent_workers'] = True
    elif config.USE_MULTI_GPU and len(config.GPU_IDS) > 1:
        print("Note: persistent_workers disabled for multi-GPU thread safety")

    # Create dataloaders with performance optimizations
    client_loaders = [
        DataLoader(dataset, shuffle=True, **dataloader_kwargs)
        for dataset in client_datasets
    ]

    # Create separate public loader for each client to avoid iterator conflicts in multi-GPU
    public_loaders = [
        DataLoader(public_dataset, shuffle=True, **dataloader_kwargs)
        for _ in range(config.NUM_CLIENTS)
    ]

    test_loader = DataLoader(
        cifar_test,
        shuffle=False,
        **dataloader_kwargs
    )

    # Client info for logging
    client_info = {
        'classes': client_classes,
        'transforms': client_transforms
    }

    return client_loaders, public_loaders, test_loader, client_info
