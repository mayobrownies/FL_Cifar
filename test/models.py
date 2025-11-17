# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Simple CNN
# ----------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.projection = nn.Linear(64 * 8 * 8, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return self.classifier(x)

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)

# ----------------------
# MLP for flattened images
# ----------------------
class MLPModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
            nn.Linear(512, feature_dim), nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def get_features(self, x):
        return self.features(x)

# ----------------------
# ResNet18 from torchvision
# ----------------------
from torchvision.models import resnet18

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        base = resnet18(pretrained=False)
        base.fc = nn.Identity()
        self.features = base
        self.projection = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = self.projection(feat)
        return self.classifier(feat)

    def get_features(self, x):
        feat = self.features(x)
        return self.projection(feat)
