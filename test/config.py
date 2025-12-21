# Data Configuration
DATA_DIR = "./data"
NUM_CLIENTS = 3
PUBLIC_SIZE = 2000
SHARDS_PER_CLIENT = 2
BATCH_SIZE = 256

# Training Configuration
NUM_ROUNDS = 100
EPOCHS_PER_ROUND = 3
LEARNING_RATE = 0.01

# Learning Rate Decay
LR_DECAY_STEP = 50
LR_DECAY_GAMMA = 0.5

# Model Configuration
MODEL_TYPES = ["CNN", "MLP", "ResNet"]
FEATURE_DIM = 128

# Prototype Alignment
PROTOTYPE_WEIGHT = 1.0  # For FedProto and ULCD
PROTOTYPE_WARMUP_RATE = 0.01

# # FedTGP-specific Configuration
# FEDTGP_PROTOTYPE_WEIGHT = 10.0  # Original FedTGP paper uses lambda=10.0 (not 1.0)
# FEDTGP_SERVER_EPOCHS = 100  # Match original paper (TGP convergence doesn't scale with client count)
# FEDTGP_SERVER_BATCH_SIZE = 10  # Match original paper (same as client batch size)
# FEDTGP_MARGIN_THRESHOLD = 100.0  # Adaptive margin threshold

# Contrastive Learning (ULCD only)
CONTRASTIVE_TEMP = 0.07
CONTRASTIVE_WEIGHT = 0.5

# FedMD-specific Configuration
N_ALIGNMENT = 1000
N_LOGITS_MATCHING_ROUND = 1
N_PRIVATE_TRAINING_ROUND = 2
PRIVATE_TRAINING_BATCHSIZE = 128  # Smaller batch size for private training

# ULCD-specific Configuration
ULCD_EMA_MOMENTUM = 0.9

ULCD_USE_PUBLIC_ALIGNMENT = False
ULCD_USE_EMA = True
ULCD_USE_WARMUP = False
ULCD_USE_CONTRASTIVE = True

# Mixed-specific Configuration
MIXED_DISTILL_WEIGHT = 0.1
MIXED_USE_LOGITS = True

# Long Run Configuration
CHECKPOINT_FREQ = 20
EVAL_FREQ = 5
SEED = 42

# Performance Reporting
FINAL_AVERAGE_WINDOW = 10  # Average metrics over last N evaluations for steady-state performance

# Convergence Detection
CONVERGENCE_WINDOW = 10  # Number of rounds to check for stability
CONVERGENCE_THRESHOLD = 0.01  # Accuracy must stay within this percentage (1%) to be considered converged

# Logging Configuration
OUTPUT_DIR = "./fl_plots"
EXPERIMENT_NAME = "hetero_fl"

# Device Configuration
USE_CUDA = True

# DataLoader Performance
DATALOADER_NUM_WORKERS = 8  # Multi-threaded data loading (set to 0 to disable)
DATALOADER_PIN_MEMORY = True  # Fast GPU transfer
DATALOADER_PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Mixed Precision Training
USE_AMP = True  # Automatic Mixed Precision (faster training, less memory)

# Multi-GPU Configuration
USE_MULTI_GPU = True  # Train clients in parallel on different GPUs
GPU_IDS = [0, 1]  # List of GPU IDs to use (e.g., [0, 1] for 2 GPUs)

# Transform Caching
CACHE_TRANSFORMS = True  # Pre-compute and cache heterogeneity transforms
TRANSFORM_CACHE_DIR = "./data/transform_cache"  # Directory for cached transforms

# Controls how classes are distributed across clients
LABEL_HETEROGENEITY = "partial_overlap"  # "none", "partial_overlap", "non_overlapping"

# For "partial_overlap": Each client gets LABEL_CLASSES_PER_CLIENT classes,
# with consecutive clients having LABEL_OVERLAP_SIZE classes in common
LABEL_CLASSES_PER_CLIENT = 5  # How many classes each client has
LABEL_OVERLAP_SIZE = 2  # How many classes overlap between consecutive clients

# For "non_overlapping": Classes are split evenly across clients with no overlap
# Example with 10 classes, 5 clients: each gets 2 unique classes

# Controls whether clients see different "types" of data (visual variations)
DATA_HETEROGENEITY = "none"  # "none", "transform", "mixed"

# For "transform": Each client gets a different visual transformation
# Available transforms: "rgb", "grayscale", "edge", "blur", "color_jitter", "noise"
# Clients will be assigned transforms in order from this list
DATA_TRANSFORM_TYPES = ["rgb", "grayscale", "edge", "blur", "color_jitter"]

# For "mixed": Some clients share transforms, some have unique
# Format: List of transform names, one per client (can repeat)
# Example: ["rgb", "rgb", "grayscale", "edge", "edge"] means clients 0,1 share RGB, 3,4 share edge
DATA_TRANSFORM_ASSIGNMENT = None  # If None, auto-assign from DATA_TRANSFORM_TYPES
