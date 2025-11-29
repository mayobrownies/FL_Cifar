# Data Configuration
DATA_DIR = "./data"
NUM_CLIENTS = 3
PUBLIC_SIZE = 2000
SHARDS_PER_CLIENT = 2
BATCH_SIZE = 64

# Training Configuration
NUM_ROUNDS = 100
EPOCHS_PER_ROUND = 3
LEARNING_RATE = 0.01

# Learning Rate Decay
LR_DECAY_STEP = 20
LR_DECAY_GAMMA = 0.5

# Model Configuration
MODEL_TYPES = ["CNN", "MLP", "ResNet"]
FEATURE_DIM = 128

# Prototype Alignment (generic)
PROTOTYPE_WEIGHT = 0.3
PROTOTYPE_WARMUP_RATE = 0.01

# FedProto-specific Configuration
FEDPROTO_LD = 1.0
FEDPROTO_LOCAL_EP = 1
FEDPROTO_LOCAL_BS = 64

# FedMD-specific Configuration
N_ALIGNMENT = 1000
N_LOGITS_MATCHING_ROUND = 1
LOGITS_MATCHING_BATCHSIZE = 128
N_PRIVATE_TRAINING_ROUND = 3
PRIVATE_TRAINING_BATCHSIZE = 32

# ULCD-specific Configuration
ULCD_EMA_MOMENTUM = 0.2
ULCD_GRADIENT_CLIP = 1.0

# Long Run Configuration
CHECKPOINT_FREQ = 20
EVAL_FREQ = 5
SEED = 42

# Logging Configuration
OUTPUT_DIR = "./fl_plots"
EXPERIMENT_NAME = "hetero_proto_fl"

# Device Configuration
USE_CUDA = True
