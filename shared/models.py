import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================================
# DATASETS
# ============================================================================
class CifarDataset(Dataset):
    """PyTorch dataset wrapper for CIFAR-10 features and labels"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# ULCD MODELS FOR CIFAR-10
# ============================================================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation for ULCD"""
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.down = nn.Linear(in_features, r, bias=False)
        self.up = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))

class ImageTransformer(nn.Module):
    """Vision Transformer module for CIFAR-10 images"""
    def __init__(self, image_dim=3072, d_model=256, num_heads=8, num_layers=3, use_lora=True):
        super().__init__()
        
        # Multi-stage projection for better feature extraction
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=min(num_heads, d_model // 32),  # Ensure reasonable head dimension
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced LoRA adaptation
        self.use_lora = use_lora
        if use_lora:
            self.lora = LoRALinear(d_model, d_model, r=16)  # Increased rank
            
        # Feature refinement layer
        self.feature_refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Handle input shape normalization  
        original_shape = x.shape
        
        # Normalize input to [B, 3072]
        if len(x.shape) == 4:  # [B, 3, 32, 32]
            x = x.view(x.size(0), -1)
        elif len(x.shape) == 1:  # [3072]
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:  # [3, 32, 32]
            x = x.view(1, -1)
            
        # Ensure 2D shape [B, 3072]
        if len(x.shape) != 2:
            batch_size = x.numel() // 3072 if x.numel() >= 3072 else 1
            x = x.view(batch_size, -1)[:, :3072]
            if x.shape[1] < 3072:
                pad_size = 3072 - x.shape[1]
                x = torch.cat([x, torch.zeros(x.shape[0], pad_size, device=x.device)], dim=1)
        
        # Multi-stage projection with enhanced features
        x = self.image_proj(x)  # [B, d_model]
        
        # Add positional encoding
        x = x.unsqueeze(1)  # [B, 1, d_model]
        x = x + self.pos_encoding
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Transformer encoding
        x = self.encoder(x).squeeze(1)  # [B, d_model]
        
        # LoRA adaptation
        if self.use_lora:
            lora_out = self.lora(x)
            x = x + 0.1 * lora_out  # Scale instead of clamp
            
        # Feature refinement
        x = self.feature_refine(x)
        
        # Final stability clamp
        x = torch.clamp(x, -5, 5)
        return x

class SimpleULCD_CIFAR(nn.Module):
    """ULTRA-SIMPLIFIED ULCD for maximum stability on CIFAR-10"""

    def __init__(self, input_dim=3072, output_dim=10, latent_dim=16, num_subspaces=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces

        # ULTRA-SIMPLE feature extractor - minimal layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced from 256
            nn.ReLU(),                   # No BatchNorm for simplicity
            nn.Linear(128, 64),         # Reduced from 128
            nn.ReLU(),
        )

        # MINIMAL latent encoder - single layer
        self.latent_encoder = nn.Linear(64, latent_dim)

        # MINIMAL distillation head - single layer
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()  # Bounded output for stability
        )

        # ULTRA-SIMPLE classifier - minimal complexity
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),  # Small hidden layer
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        # Conservative weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Very small weights
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # High temperature for stable distillation
        self.temperature = 10.0  # Higher for more stability

        # For ULCD compatibility
        self.image_encoder = lambda x: self.latent_encoder(self.feature_extractor(x))
    
    def forward(self, x, return_distillation=False):
        """ULTRA-SIMPLE forward pass for maximum stability"""

        if len(x.shape) == 4:  # [B, C, H, W]
            x = x.view(x.size(0), -1)  # Flatten to [B, 3072]

        # Simple feature extraction
        features = self.feature_extractor(x)

        # Aggressive clamping at every step for stability
        features = torch.clamp(features, -5.0, 5.0)

        # Simple latent encoding
        latent = self.latent_encoder(features)

        # Aggressive clamping and normalization
        latent = torch.clamp(latent, -2.0, 2.0)
        latent = F.normalize(latent, p=2, dim=1)

        # Simple classification
        logits = self.classifier(latent)

        # Aggressive output clamping to prevent explosion
        logits = torch.clamp(logits, -10.0, 10.0)

        if return_distillation:
            # Simple distillation output
            distilled = self.distillation_head(latent)
            distilled = torch.clamp(distilled, -2.0, 2.0)  # Clamp distilled output
            # Simple attention (just ones for compatibility)
            attention = torch.ones(x.size(0), self.num_subspaces).to(x.device) / self.num_subspaces
            return logits, distilled, attention

        return logits
    
    def encode(self, x):
        """Encode input to latent representation"""
        if len(x.shape) == 4:  # [B, C, H, W]
            x = x.view(x.size(0), -1)  # Flatten
        
        features = self.feature_extractor(x)
        latent = self.latent_encoder(features)
        
        # Simple attention for compatibility
        attention = torch.ones(x.size(0), self.num_subspaces).to(x.device) / self.num_subspaces
        return latent, attention
    
    def get_latent_summary(self, dataloader):
        """Get class-aware latent summary for ULCD aggregation with improved stability"""
        self.eval()
        device = next(self.parameters()).device

        # Collect latents per class with outlier detection
        class_latents = {i: [] for i in range(self.output_dim)}
        class_counts = torch.zeros(self.output_dim)

        with torch.no_grad():
            for batch_images, batch_labels in dataloader:
                if len(batch_images.shape) == 4:
                    batch_images = batch_images.view(batch_images.size(0), -1)

                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                # Get latent representation
                latent, _ = self.encode(batch_images)

                # Check for outliers (latents with extreme norms)
                latent_norms = torch.norm(latent, dim=1)
                outlier_mask = (latent_norms > 0.1) & (latent_norms < 50.0)  # Filter extreme values

                # Group latents by class (excluding outliers)
                for i in range(latent.size(0)):
                    if outlier_mask[i]:  # Only include non-outliers
                        class_id = batch_labels[i].item()
                        class_latents[class_id].append(latent[i])
                        class_counts[class_id] += 1

        # Create per-class summaries with robust averaging
        class_summaries = []
        for class_id in range(self.output_dim):
            if class_latents[class_id] and len(class_latents[class_id]) >= 2:
                # Conservative averaging with aggressive clamping
                class_stack = torch.stack(class_latents[class_id])
                # Apply aggressive clamping before any processing
                class_stack = torch.clamp(class_stack, -1.0, 1.0)

                if len(class_latents[class_id]) > 10:
                    # Remove extreme outliers more aggressively
                    norms = torch.norm(class_stack, dim=1)
                    # Only keep middle 80% of samples by norm
                    k = max(1, len(class_latents[class_id]) // 10)
                    _, indices = torch.topk(norms, k=len(norms)-2*k, largest=False, sorted=False)
                    if len(indices) > 2*k:
                        indices = indices[k:-k]  # Remove both extremes
                        class_stack = class_stack[indices]

                class_summary = class_stack.mean(dim=0)
                # Very aggressive clamping to prevent any explosion
                class_summary = torch.clamp(class_summary, -1.0, 1.0)
                class_summaries.append(class_summary)
            elif class_latents[class_id]:  # Single sample
                class_summary = class_latents[class_id][0]
                class_summary = torch.clamp(class_summary, -10, 10)
                class_summaries.append(class_summary)
            else:
                # No samples for this class - use small random vector
                class_summaries.append(torch.randn(self.latent_dim).to(device) * 0.01)

        # Stack all class summaries [num_classes, latent_dim]
        full_summary = torch.stack(class_summaries)

        # Create weighted class mask (use counts as weights)
        class_mask = class_counts / class_counts.sum().clamp(min=1)

        print(f"Enhanced ULCD summary: shape {full_summary.shape}, classes {(class_counts > 0).sum().int().item()}, total samples {class_counts.sum().int().item()}")
        return full_summary, class_mask

class ULCDNet_CIFAR(nn.Module):
    """ULCD Network adapted for CIFAR-10"""
    def __init__(self, input_dim: int, output_dim: int = 10, latent_dim: int = 64):
        super(ULCDNet_CIFAR, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim  # 3072 for CIFAR-10 (32x32x3)
        self.output_dim = output_dim  # 10 for CIFAR-10
        
        # Temperature parameter for calibrating predictions
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        # Enhanced image encoder with transformer architecture
        self.image_encoder = ImageTransformer(
            image_dim=input_dim, 
            d_model=latent_dim,
            num_heads=min(8, latent_dim // 32),
            num_layers=4,  # Increased layers for better feature extraction
            use_lora=True
        )
        
        # Enhanced consensus mechanism for diverse latent spaces
        self.consensus_weights = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.Sigmoid()  # Use sigmoid instead of softmax for element-wise weighting
        )
        
        # Task-specific prediction head - OPTIMIZED for CIFAR-10
        self.task_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),  # Expand before classification
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, output_dim)
        )
        
        # Initialize output layer with smaller weights for better calibration
        nn.init.xavier_normal_(self.task_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.task_head[-1].bias)

    def forward(self, x):
        # Encode to latent space
        latent = self.image_encoder(x)
        
        # Check latent for stability
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            print(f"CRITICAL: NaN/Inf in latent - model instability!")
            latent = torch.zeros_like(latent)
        
        # Apply consensus weighting
        consensus = self.consensus_weights(latent)
        consensus = torch.clamp(consensus, dim=-1)
        weighted_latent = latent * consensus
        
        # Task-specific prediction
        main_output = self.task_head(weighted_latent)
        
        # Apply temperature scaling
        # main_output = main_output / self.temperature
        
        # Check outputs for stability
        if torch.isnan(main_output).any() or torch.isinf(main_output).any():
            main_output = torch.zeros_like(main_output)
            
        return main_output

    def get_latent_summary(self, dataloader):
        """Extract latent summary from training data for ULCD consensus"""
        self.eval()
        latent_vectors = []
        device = next(self.parameters()).device
        
        with torch.no_grad():
            batch_count = 0
            for batch in dataloader:
                if len(batch) == 2:
                    features, _ = batch
                else:
                    features = batch[0]
                
                features = features.to(device)
                
                # Check for corrupted features
                if torch.isnan(features).any() or torch.isinf(features).any():
                    print(f"WARNING: NaN/Inf detected in input features!")
                    continue
                
                latent = self.image_encoder(features)
                
                # Check latent validity
                if torch.isnan(latent).any() or torch.isinf(latent).any():
                    print(f"WARNING: NaN/Inf detected in latent representation!")
                    continue
                
                # Use both mean and std for richer representation
                batch_mean = latent.mean(dim=0)
                batch_std = latent.std(dim=0)
                
                # Combine mean and std (clamped to prevent explosion)
                batch_summary = torch.cat([batch_mean, torch.clamp(batch_std, 0, 10)])
                latent_vectors.append(batch_summary)
                batch_count += 1
                
                # Limit to reasonable number of batches for efficiency
                if batch_count >= 50:
                    break
        
        if latent_vectors:
            client_latent = torch.stack(latent_vectors).mean(dim=0)
            # Normalize to prevent scale issues
            client_latent = torch.clamp(client_latent, -10, 10)
            print(f"Generated latent summary: shape {client_latent.shape}, norm {torch.norm(client_latent):.4f}")
            return client_latent
        else:
            print("WARNING: No valid latent vectors generated!")
            return torch.zeros(self.latent_dim * 2)  # Doubled for mean+std

# ============================================================================
# CNN MODELS FOR CIFAR-10
# ============================================================================
class CNNet(nn.Module):
    """Convolutional Neural Network for CIFAR-10"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10):
        super(CNNet, self).__init__()
        
        # Reshape input to [B, 3, 32, 32] if flattened
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate flattened size: 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Reshape if input is flattened
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class ULCDCompatibleCNN(nn.Module):
    """CNN with ULCD compatibility for federated learning"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, latent_dim: int = 64):
        super(ULCDCompatibleCNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main CNN backbone
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
        
        # Image encoder for ULCD compatibility (maps to latent space)
        self.image_encoder = ImageEncoder(latent_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # Reshape if input is flattened
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
        elif len(x.shape) == 1:
            x = x.view(1, 3, 32, 32)
            
        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        
        # Extract latent features
        latent_features = self.feature_extractor(x)
        
        # Classification
        output = self.classifier(latent_features)
        
        return output

    def get_latent_summary(self, dataloader):
        """Extract latent summary from training data for ULCD consensus"""
        self.eval()
        latent_vectors = []
        device = next(self.parameters()).device
        
        with torch.no_grad():
            batch_count = 0
            for batch in dataloader:
                if len(batch) == 2:
                    features, _ = batch
                else:
                    features = batch[0]
                
                features = features.to(device)
                
                # Reshape if needed
                if len(features.shape) == 2:
                    features = features.view(-1, 3, 32, 32)
                
                # Extract CNN features through the network
                x = self.pool(F.relu(self.conv1(features)))
                x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                
                # Get latent representation
                latent = self.feature_extractor(x)

                # Use just mean for consistency with other models
                batch_mean = latent.mean(dim=0)
                latent_vectors.append(batch_mean)
                batch_count += 1

                # Limit to reasonable number of batches for efficiency
                if batch_count >= 50:
                    break

        if latent_vectors:
            client_latent = torch.stack(latent_vectors).mean(dim=0)
            # Normalize to prevent scale issues
            client_latent = torch.clamp(client_latent, -10, 10)
            print(f"CNN Generated latent summary: shape {client_latent.shape}, norm {torch.norm(client_latent):.4f}")
            return client_latent, torch.ones(10)  # Return tuple for consistency
        else:
            print("WARNING: No valid CNN latent vectors generated!")
            return torch.zeros(self.latent_dim), torch.ones(10)

class LightweightCNN_ULCD(nn.Module):
    """Lightweight CNN for mobile/edge devices - ~50K parameters"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, latent_dim: int = 64):
        super(LightweightCNN_ULCD, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Minimal CNN backbone - fewer filters
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Lightweight feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Simple classifier
        self.classifier = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
        elif len(x.shape) == 1:
            x = x.view(1, 3, 32, 32)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        latent_features = self.feature_extractor(x)
        output = self.classifier(latent_features)

        return output

    def get_latent_summary(self, dataloader):
        """Extract 64-dim latent summary for ULCD consensus"""
        self.eval()
        latents = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    features, _ = batch
                else:
                    features = batch[0]

                features = features.to(device)
                if len(features.shape) == 2:
                    features = features.view(-1, 3, 32, 32)

                x = self.pool(F.relu(self.conv1(features)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)

                latent = self.feature_extractor(x)
                latents.append(latent.mean(dim=0))

        summary = torch.mean(torch.stack(latents), dim=0)
        return summary, torch.ones(10)


class HeavyweightCNN_ULCD(nn.Module):
    """Heavyweight CNN for powerful devices - ~500K parameters"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, latent_dim: int = 64):
        super(HeavyweightCNN_ULCD, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Large CNN backbone - more filters and layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Sophisticated feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )

        # Complex classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
        elif len(x.shape) == 1:
            x = x.view(1, 3, 32, 32)

        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)

        latent_features = self.feature_extractor(x)
        output = self.classifier(latent_features)

        return output

    def get_latent_summary(self, dataloader):
        """Extract 64-dim latent summary for ULCD consensus"""
        self.eval()
        latents = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    features, _ = batch
                else:
                    features = batch[0]

                features = features.to(device)
                if len(features.shape) == 2:
                    features = features.view(-1, 3, 32, 32)

                x = self.pool(F.relu(self.conv2(F.relu(self.conv1(features)))))
                x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
                x = x.view(x.size(0), -1)

                latent = self.feature_extractor(x)
                latents.append(latent.mean(dim=0))

        summary = torch.mean(torch.stack(latents), dim=0)
        return summary, torch.ones(10)


class ImageEncoder(nn.Module):
    """Enhanced image encoder for CNN ULCD compatibility"""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        """Forward pass through enhanced encoder"""
        return self.projection(x)

class ResNet_CIFAR(nn.Module):
    """Simplified ResNet for CIFAR-10"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10):
        super(ResNet_CIFAR, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape if input is flattened
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
            
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
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

# ============================================================================
# ADAPTED MODELS FROM MIMIC FL
# ============================================================================
class LSTMNet_CIFAR(nn.Module):
    """LSTM adapted for image sequences"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, hidden_dim: int = 128, num_layers: int = 2):
        super(LSTMNet_CIFAR, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Convert image to sequence: treat each row as a timestep
        self.seq_len = 32  # 32 rows in CIFAR-10 image
        self.input_size = 96  # 32 * 3 (width * channels)
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(0.3),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for LSTM: [B, 3072] -> [B, 3, 32, 32] -> [B, 32, 96]
        if len(x.shape) == 2:
            x = x.view(-1, 3, 32, 32)
        x = x.permute(0, 2, 1, 3)  # [B, 32, 3, 32]
        x = x.contiguous().view(batch_size, 32, 96)  # [B, 32, 96]
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        output = lstm_out[:, -1, :]
        
        return self.classifier(output)

class MixtureOfExperts_CIFAR(nn.Module):
    """Mixture of Experts adapted for CIFAR-10"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, num_experts: int = 4, expert_dim: int = 128):
        super(MixtureOfExperts_CIFAR, self).__init__()
        
        self.num_experts = num_experts
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(expert_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Expert networks - each is a small CNN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(expert_dim * 2, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(expert_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
            
        gate_weights = self.gate(x)  # [batch_size, num_experts]
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        gate_weights = gate_weights.unsqueeze(2)  # [batch_size, num_experts, 1]
        
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # [batch_size, output_dim]
        return output

class LogisticRegressionNet_CIFAR(nn.Module):
    """Logistic Regression for CIFAR-10"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10):
        super(LogisticRegressionNet_CIFAR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        return self.linear(x)

class MLPNet_CIFAR(nn.Module):
    """MLP adapted for CIFAR-10"""
    def __init__(self, input_dim: int = 3072, output_dim: int = 10, hidden_dims: list = [512, 256, 128]):
        super(MLPNet_CIFAR, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Input normalization
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        return self.network(x)

# ============================================================================
# SKLEARN WRAPPER FOR FEDERATED LEARNING
# ============================================================================
class SklearnWrapper_CIFAR(nn.Module):
    """Sklearn wrapper adapted for CIFAR-10"""
    def __init__(self, sklearn_model, input_dim: int = 3072, output_dim: int = 10):
        super(SklearnWrapper_CIFAR, self).__init__()
        self.sklearn_model = sklearn_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.is_trained = False

    def forward(self, x):
        # Flatten if needed
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
            
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        if not self.is_trained:
            batch_size = x_np.shape[0]
            random_probs = np.random.rand(batch_size, self.output_dim)
            random_probs = random_probs / random_probs.sum(axis=1, keepdims=True)
            return torch.tensor(random_probs, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
        
        x_scaled = self.scaler.transform(x_np)
        
        if hasattr(self.sklearn_model, 'predict_proba'):
            probabilities = self.sklearn_model.predict_proba(x_scaled)
        else:
            predictions = self.sklearn_model.predict(x_scaled)
            probabilities = np.eye(self.output_dim)[predictions]
        
        return torch.tensor(probabilities, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else 'cpu')

    def fit_sklearn(self, X, y):
        """Fit the sklearn model"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], -1)
            
        X_scaled = self.scaler.fit_transform(X)
        self.sklearn_model.fit(X_scaled, y)
        self.is_trained = True

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -torch.sum(smooth_targets * log_probs, dim=1)
            
            with torch.no_grad():
                pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
        
        alpha_t = 1.0
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# MODEL FACTORY
# ============================================================================
def get_model(model_name: str, input_dim: int = 3072, output_dim: int = 10, **kwargs):
    """Factory function to create models for CIFAR-10"""

    if model_name == "ulcd":
        latent_dim = kwargs.get('latent_dim', 32)
        return SimpleULCD_CIFAR(input_dim, output_dim, latent_dim=latent_dim)
    elif model_name == "cnn":
        return CNNet(input_dim, output_dim)
    elif model_name == "cnn_ulcd":
        latent_dim = kwargs.get('latent_dim', 64)
        return ULCDCompatibleCNN(input_dim, output_dim, latent_dim=latent_dim)
    elif model_name == "cnn_ulcd_light":
        latent_dim = kwargs.get('latent_dim', 64)
        return LightweightCNN_ULCD(input_dim, output_dim, latent_dim=latent_dim)
    elif model_name == "cnn_ulcd_heavy":
        latent_dim = kwargs.get('latent_dim', 64)
        return HeavyweightCNN_ULCD(input_dim, output_dim, latent_dim=latent_dim)
    elif model_name == "resnet":
        return ResNet_CIFAR(input_dim, output_dim)
    elif model_name == "lstm":
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        return LSTMNet_CIFAR(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "moe":
        num_experts = kwargs.get('num_experts', 4)
        expert_dim = kwargs.get('expert_dim', 128)
        return MixtureOfExperts_CIFAR(input_dim, output_dim, num_experts=num_experts, expert_dim=expert_dim)
    elif model_name == "mlp":
        hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        return MLPNet_CIFAR(input_dim, output_dim, hidden_dims=hidden_dims)
    elif model_name == "logistic":
        return LogisticRegressionNet_CIFAR(input_dim, output_dim)
    elif model_name == "random_forest":
        sklearn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        return SklearnWrapper_CIFAR(sklearn_model, input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ============================================================================
# MODEL UTILITIES
# ============================================================================
def is_sklearn_model(model):
    """Check if a model is a sklearn wrapper"""
    return isinstance(model, SklearnWrapper_CIFAR)

def get_model_type(model):
    """Get the type of model for special handling"""
    if isinstance(model, (ULCDNet_CIFAR, SimpleULCD_CIFAR)):
        return "ulcd"
    elif isinstance(model, CNNet):
        return "cnn"
    elif isinstance(model, (ULCDCompatibleCNN, LightweightCNN_ULCD, HeavyweightCNN_ULCD)):
        return "cnn_ulcd"
    elif isinstance(model, ResNet_CIFAR):
        return "resnet"
    elif isinstance(model, LSTMNet_CIFAR):
        return "lstm"
    elif isinstance(model, MixtureOfExperts_CIFAR):
        return "moe"
    elif isinstance(model, MLPNet_CIFAR):
        return "mlp"
    elif isinstance(model, LogisticRegressionNet_CIFAR):
        return "logistic"
    elif isinstance(model, SklearnWrapper_CIFAR):
        return "sklearn"
    else:
        return "neural"