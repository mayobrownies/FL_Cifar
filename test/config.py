"""
Configuration for Heterogeneous Federated Learning
"""

# Data Configuration
DATA_DIR = "./data"
NUM_CLIENTS = 3
PUBLIC_SIZE = 2000
SHARDS_PER_CLIENT = 2
BATCH_SIZE = 64

# Training Configuration
NUM_ROUNDS = 5
EPOCHS_PER_ROUND = 3
LEARNING_RATE = 0.001

# Learning Rate Decay
LR_DECAY_STEP = 2  # Decay every N rounds
LR_DECAY_GAMMA = 0.5  # Multiply LR by this factor

# Model Configuration
MODEL_TYPES = ["CNN", "MLP", "ResNet"]

# Prototype Alignment
PROTOTYPE_WEIGHT = 0.3
DISTILLATION_WEIGHT = 0.5

# ULCD-specific Configuration
ULCD_LATENT_DIM = 64  # Latent dimension for consensus (all models must match)
ULCD_NUM_SUBSPACES = 3  # Number of latent subspaces
ULCD_EMA_MOMENTUM = 0.5  # EMA momentum for prototype aggregation
ULCD_ANOMALY_THRESHOLD = 0.3  # Trust score threshold for anomaly detection
ULCD_ENABLE_DISTILLATION = True  # Enable teacher network distillation
ULCD_TEMPERATURE = 4.0  # Temperature for knowledge distillation
ULCD_DIVERSITY_WEIGHT = 0.1  # Diversity regularization weight
ULCD_BASE_ALPHA = 0.2  # Base learning rate for prototype updates
ULCD_GRADIENT_CLIP = 1.0  # Gradient clipping max norm
ULCD_MAX_LOSS = 100.0  # Maximum loss before fallback

# Logging Configuration
OUTPUT_DIR = "./fl_plots"
EXPERIMENT_NAME = "hetero_proto_fl"

# Device Configuration
USE_CUDA = True  # Set to False to force CPU
