"""
ULCD-compatible models for heterogeneous federated learning
All models output same latent dimension for consensus
Features: Multi-subspace latents, attention, temperature distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import config


class CNN_ULCD(nn.Module):
    """CNN with ULCD latent extraction - ~200K parameters"""
    def __init__(self, num_classes=10, latent_dim=64, num_subspaces=3, temperature=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces or config.ULCD_NUM_SUBSPACES
        self.temperature = temperature or config.ULCD_TEMPERATURE
        self.num_classes = num_classes

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Multi-subspace latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Attention mechanism for subspaces
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_subspaces),
            nn.Softmax(dim=1)
        )

        # Distillation head (for ULCD knowledge transfer)
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_distillation=False):
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        latent = self.latent_proj(x)

        # Clamp for stability
        latent = torch.clamp(latent, -5.0, 5.0)

        # Classification
        logits = self.classifier(latent)

        if return_distillation:
            # Compute attention weights over subspaces
            attention = self.attention(latent)

            # Distilled representation
            distilled = self.distillation_head(latent)
            distilled = torch.clamp(distilled, -2.0, 2.0)

            # Apply temperature scaling to logits
            scaled_logits = logits / self.temperature

            return scaled_logits, distilled, attention

        return logits

    def get_features(self, x):
        """Extract latent features for prototype computation"""
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        latent = self.latent_proj(x)
        return torch.clamp(latent, -5.0, 5.0)

    def encode(self, x):
        """Encode with attention weights"""
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        latent = self.latent_proj(x)
        latent = torch.clamp(latent, -5.0, 5.0)

        attention = self.attention(latent)
        return latent, attention

    def get_latent_summary(self, dataloader):
        """
        Extract per-class latent prototypes with class mask

        Returns:
            class_prototypes: dict {class_id: prototype_tensor}
            class_mask: tensor indicating which classes are present
        """
        self.eval()
        device = next(self.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                latents, _ = self.encode(x)

                # Outlier detection
                latent_norms = torch.norm(latents, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)

                for i in range(len(latents)):
                    if outlier_mask[i]:
                        label = y[i].item()
                        class_latents[label].append(latents[i].cpu())
                        class_counts[label] += 1

        # Compute per-class prototypes with robust averaging
        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id] and len(class_latents[class_id]) >= 2:
                class_stack = torch.stack(class_latents[class_id])
                class_stack = torch.clamp(class_stack, -1.0, 1.0)

                # Remove extreme outliers if enough samples
                if len(class_latents[class_id]) > 10:
                    norms = torch.norm(class_stack, dim=1)
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms) - 2*k, largest=False)
                    if len(indices) > 2*k:
                        indices = indices[k:-k]
                        class_stack = class_stack[indices]

                proto = class_stack.mean(dim=0)
                proto = torch.clamp(proto, -1.0, 1.0)
                class_prototypes[class_id] = proto
            elif class_latents[class_id]:
                proto = class_latents[class_id][0]
                class_prototypes[class_id] = torch.clamp(proto, -1.0, 1.0)

        # Create class mask (normalized counts)
        class_mask = class_counts / class_counts.sum().clamp(min=1)

        return class_prototypes, class_mask


class MLP_ULCD(nn.Module):
    """MLP with ULCD latent extraction - ~100K parameters"""
    def __init__(self, num_classes=10, latent_dim=64, num_subspaces=3, temperature=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces or config.ULCD_NUM_SUBSPACES
        self.temperature = temperature or config.ULCD_TEMPERATURE
        self.num_classes = num_classes

        # Feature extraction
        self.features = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Latent projection
        self.latent_proj = nn.Linear(256, latent_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_subspaces),
            nn.Softmax(dim=1)
        )

        # Distillation head
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_distillation=False):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        x = self.features(x)
        latent = self.latent_proj(x)
        latent = torch.clamp(latent, -5.0, 5.0)

        logits = self.classifier(latent)

        if return_distillation:
            attention = self.attention(latent)
            distilled = self.distillation_head(latent)
            distilled = torch.clamp(distilled, -2.0, 2.0)
            scaled_logits = logits / self.temperature
            return scaled_logits, distilled, attention

        return logits

    def get_features(self, x):
        """Extract latent features for prototype computation"""
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        x = self.features(x)
        latent = self.latent_proj(x)
        return torch.clamp(latent, -5.0, 5.0)

    def encode(self, x):
        """Encode with attention weights"""
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)

        x = self.features(x)
        latent = self.latent_proj(x)
        latent = torch.clamp(latent, -5.0, 5.0)
        attention = self.attention(latent)
        return latent, attention

    def get_latent_summary(self, dataloader):
        """Extract per-class latent prototypes with class mask"""
        self.eval()
        device = next(self.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                latents, _ = self.encode(x)

                latent_norms = torch.norm(latents, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)

                for i in range(len(latents)):
                    if outlier_mask[i]:
                        label = y[i].item()
                        class_latents[label].append(latents[i].cpu())
                        class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id] and len(class_latents[class_id]) >= 2:
                class_stack = torch.stack(class_latents[class_id])
                class_stack = torch.clamp(class_stack, -1.0, 1.0)

                if len(class_latents[class_id]) > 10:
                    norms = torch.norm(class_stack, dim=1)
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms) - 2*k, largest=False)
                    if len(indices) > 2*k:
                        indices = indices[k:-k]
                        class_stack = class_stack[indices]

                proto = class_stack.mean(dim=0)
                proto = torch.clamp(proto, -1.0, 1.0)
                class_prototypes[class_id] = proto
            elif class_latents[class_id]:
                proto = class_latents[class_id][0]
                class_prototypes[class_id] = torch.clamp(proto, -1.0, 1.0)

        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask


class ResNet_ULCD(nn.Module):
    """ResNet with ULCD latent extraction - ~150K parameters"""
    def __init__(self, num_classes=10, latent_dim=64, num_subspaces=3, temperature=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces or config.ULCD_NUM_SUBSPACES
        self.temperature = temperature or config.ULCD_TEMPERATURE
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Simplified ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Latent projection
        self.latent_proj = nn.Linear(128, latent_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_subspaces),
            nn.Softmax(dim=1)
        )

        # Distillation head
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x, return_distillation=False):
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        latent = self.latent_proj(x)
        latent = torch.clamp(latent, -5.0, 5.0)

        logits = self.classifier(latent)

        if return_distillation:
            attention = self.attention(latent)
            distilled = self.distillation_head(latent)
            distilled = torch.clamp(distilled, -2.0, 2.0)
            scaled_logits = logits / self.temperature
            return scaled_logits, distilled, attention

        return logits

    def get_features(self, x):
        """Extract latent features for prototype computation"""
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        latent = self.latent_proj(x)
        return torch.clamp(latent, -5.0, 5.0)

    def encode(self, x):
        """Encode with attention weights"""
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        latent = self.latent_proj(x)
        latent = torch.clamp(latent, -5.0, 5.0)
        attention = self.attention(latent)
        return latent, attention

    def get_latent_summary(self, dataloader):
        """Extract per-class latent prototypes with class mask"""
        self.eval()
        device = next(self.parameters()).device

        class_latents = defaultdict(list)
        class_counts = torch.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                latents, _ = self.encode(x)

                latent_norms = torch.norm(latents, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)

                for i in range(len(latents)):
                    if outlier_mask[i]:
                        label = y[i].item()
                        class_latents[label].append(latents[i].cpu())
                        class_counts[label] += 1

        class_prototypes = {}
        for class_id in range(self.num_classes):
            if class_latents[class_id] and len(class_latents[class_id]) >= 2:
                class_stack = torch.stack(class_latents[class_id])
                class_stack = torch.clamp(class_stack, -1.0, 1.0)

                if len(class_latents[class_id]) > 10:
                    norms = torch.norm(class_stack, dim=1)
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms) - 2*k, largest=False)
                    if len(indices) > 2*k:
                        indices = indices[k:-k]
                        class_stack = class_stack[indices]

                proto = class_stack.mean(dim=0)
                proto = torch.clamp(proto, -1.0, 1.0)
                class_prototypes[class_id] = proto
            elif class_latents[class_id]:
                proto = class_latents[class_id][0]
                class_prototypes[class_id] = torch.clamp(proto, -1.0, 1.0)

        class_mask = class_counts / class_counts.sum().clamp(min=1)
        return class_prototypes, class_mask


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
