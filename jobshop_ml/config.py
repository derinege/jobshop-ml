"""
Configuration file for Job Shop Scheduling ML Model
Contains all hyperparameters, weights, and paths.
"""

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_PATH_ISLEM_TAM = "islem_tam_tablo.xlsx"
DATA_PATH_BOLD_SURE = "bold_islem_sure_tablosu.xlsx"

# ============================================================================
# MULTI-OBJECTIVE WEIGHTS (Matching your MIP model - 3 COMPONENTS)
# ============================================================================
# These weights define the relative importance of each objective component
# Modify these to adjust scheduling priorities
# 
# Objective = w_MS * MS + w_W * sum(Wrow) + w_G * sum(Gap)
# where:
#   MS = Makespan (total completion time)
#   Wrow = Row waiting / tardiness (waiting between operations)
#   Gap = Idle gaps on machines
WEIGHTS = {
    'w_MS': 0.3,    # Makespan weight
    'w_W': 0.5,     # Row waiting / tardiness weight
    'w_G': 0.2      # Machine idle gap weight
}

# ============================================================================
# MACHINE MAPPING (Same as your MIP code)
# ============================================================================
# Maps operation names to machine classes
MACHINE_MAP = {
    'KAYNAK': 'WLD',
    'ÖN HAZIRLAMA': 'PRE',
    'KUMLAMA': 'FIN',
    'YIKAMA': 'CLN',
    'MONTAJ': 'ASM',
    'PAKETLEME': 'PKG',
    'BÜKÜM': 'BNG',
    'PLAZMA': 'PLS',
    'LAZER': 'LSR',
    'BANT': 'BND',
    'GENEL': 'GEN'
}

MACHINE_CLASSES = list(set(MACHINE_MAP.values()))

# ============================================================================
# SCHEDULING PARAMETERS
# ============================================================================
H_SHIFT = 540  # Shift duration in minutes (9 hours)
BIG_M = 100000  # Big-M constant for MIP disjunctions

# ============================================================================
# DATASET GENERATION PARAMETERS
# ============================================================================
N_TRAIN_INSTANCES = 100      # Number of training instances to generate
N_VAL_INSTANCES = 20         # Number of validation instances
N_TEST_INSTANCES = 20        # Number of test instances
MIN_JOBS_PER_INSTANCE = 3    # Minimum jobs per training instance
MAX_JOBS_PER_INSTANCE = 8    # Maximum jobs per training instance
LARGER_TEST_JOBS = 15        # Jobs in larger test instances (beyond MIP scale)

# MIP solver parameters
MIP_TIME_LIMIT = 300         # Time limit in seconds for Gurobi
MIP_GAP_TOLERANCE = 0.05     # MIP gap tolerance (5%)

# ============================================================================
# GNN MODEL HYPERPARAMETERS
# ============================================================================
GNN_CONFIG = {
    # Node feature dimensions
    'node_feature_dim': 64,       # Dimension of processed node features
    'edge_feature_dim': 16,       # Dimension of edge features
    
    # GNN architecture
    'hidden_dim': 128,            # Hidden layer dimension
    'num_gnn_layers': 3,          # Number of graph conv layers
    'num_mlp_layers': 2,          # Number of MLP layers after GNN
    'dropout': 0.1,               # Dropout rate
    'use_attention': True,        # Use GAT-style attention
    'num_heads': 4,               # Number of attention heads (if using GAT)
    
    # Output
    'output_dim': 1,              # Score per operation (for ranking)
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    # Optimization
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 8,
    'num_epochs': 100,
    
    # Loss weights
    'action_loss_weight': 1.0,     # Weight for imitation learning loss
    'value_loss_weight': 0.1,      # Weight for value prediction loss (optional)
    
    # Learning rate schedule
    'lr_schedule': 'cosine',       # 'constant', 'step', or 'cosine'
    'lr_decay_factor': 0.5,
    'lr_decay_epochs': 30,
    
    # Early stopping
    'patience': 15,                # Epochs to wait for improvement
    'min_delta': 0.001,            # Minimum change to qualify as improvement
    
    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'save_best_only': True,
}

# ============================================================================
# RL HYPERPARAMETERS (For future RL training)
# ============================================================================
RL_CONFIG = {
    'gamma': 0.99,                 # Discount factor
    'gae_lambda': 0.95,            # GAE lambda for advantage estimation
    'clip_epsilon': 0.2,           # PPO clip parameter
    'value_loss_coef': 0.5,        # Value loss coefficient
    'entropy_coef': 0.01,          # Entropy bonus coefficient
    'max_grad_norm': 0.5,          # Gradient clipping
    'ppo_epochs': 4,               # PPO optimization epochs per batch
    'num_episodes': 1000,          # Number of training episodes
    'episode_batch_size': 16,      # Episodes per training batch
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
EVAL_CONFIG = {
    'compare_with_mip': True,      # Compare ML policy with MIP on small instances
    'use_heuristic_baseline': True, # Compare with dispatch rules
    'metrics_to_track': [
        'makespan',
        'total_tardiness', 
        'total_waiting',
        'total_idle',
        'weighted_objective',
        'feasibility'
    ]
}

# ============================================================================
# HEURISTIC BASELINES (For comparison)
# ============================================================================
HEURISTIC_RULES = [
    'SPT',      # Shortest Processing Time
    'LPT',      # Longest Processing Time
    'FIFO',     # First In First Out
    'EDD',      # Earliest Due Date
    'CR',       # Critical Ratio
]

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_dir': 'logs',
    'tensorboard': False,          # Use TensorBoard logging
    'print_frequency': 10,         # Print every N epochs
}
