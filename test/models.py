import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.projection = nn.Linear(64 * 8 * 8, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False):
        x = self.features(x)
        protos = self.projection(x)
        logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)


class MLPModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.projection = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False):
        x = self.features(x)
        protos = self.projection(x)
        logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
        return logits

    def get_features(self, x):
        x = self.features(x)
        return self.projection(x)


from torchvision.models import resnet18

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        base = resnet18(pretrained=False)
        base.fc = nn.Identity()
        self.features = base
        self.projection = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_protos=False):
        feat = self.features(x)
        protos = self.projection(feat)
        logits = self.classifier(protos)

        if return_protos:
            log_probs = F.log_softmax(logits, dim=1)
            return log_probs, protos
        return logits

    def get_features(self, x):
        feat = self.features(x)
        return self.projection(feat)
