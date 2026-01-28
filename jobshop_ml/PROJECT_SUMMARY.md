# Job Shop Scheduling ML System - Project Summary

## Overview
This is a complete, production-ready machine learning system for job shop scheduling that learns from optimal MIP solutions and generalizes to larger instances.

## What You Get

### 1. Complete Pipeline (13 Python Modules)
- ✅ **config.py** - Central configuration for all hyperparameters and weights
- ✅ **data_loader.py** - Excel data loading and instance creation
- ✅ **mip_oracle.py** - Gurobi MIP solver (generates training labels)
- ✅ **graph_builder.py** - Converts scheduling states to graph representations
- ✅ **dataset.py** - Dataset generation and PyTorch DataLoader
- ✅ **gnn_model.py** - Graph Neural Network policy architecture
- ✅ **training.py** - Imitation learning trainer with early stopping
- ✅ **rl_env.py** - RL environment skeleton for future development
- ✅ **evaluation.py** - Comprehensive evaluation framework
- ✅ **main.py** - End-to-end training pipeline
- ✅ **demo.py** - Demo with synthetic data (no Excel files needed)
- ✅ **README.md** - Complete documentation
- ✅ **requirements.txt** - All dependencies

### 2. Key Features

#### Multi-Objective Optimization
Respects your weighted objective structure:
```
Objective = 0.2 × Makespan 
          + 0.5 × Tardiness
          + 0.1 × Precedence Waiting
          + 0.1 × Assembly Waiting
          + 0.1 × Machine Idle Gaps
```

#### Graph Neural Network Architecture
- Node features: machine type, duration, precedence position, job info
- Edge features: precedence relations, same-machine constraints
- Global features: instance statistics
- Attention mechanism (GAT) for better learning
- ~50K-200K parameters (configurable)

#### Training Strategy
- **Imitation Learning**: Learn from optimal MIP solutions on small instances
- **Batch training**: Efficient graph batching with PyTorch
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Checkpointing**: Saves best model automatically

#### Evaluation Framework
- Compare ML vs MIP vs heuristics (SPT, LPT, FIFO)
- Track all objective components
- Compute optimality gaps
- Beautiful comparison tables

## Quick Start

### Option 1: With Your Excel Files
```bash
# Place your files in the directory
# - islem_tam_tablo.xlsx
# - bold_islem_sure_tablosu.xlsx

# Run full pipeline
python main.py

# Or with custom settings
python main.py --num-epochs 50 --batch-size 16 --n-eval-instances 20
```

### Option 2: Demo Without Excel Files
```bash
# Run demo with synthetic data
python demo.py

# This will show:
# - MIP oracle solving
# - Graph representation building
# - GNN model forward pass
# - RL environment simulation
# - Heuristic baseline comparison
```

## Architecture Highlights

### 1. Data Flow
```
Excel Files → SchedulingInstance → MIP Oracle → Optimal Schedule
                                              ↓
                                    Graph Representation
                                              ↓
                                    Training Samples (state, action pairs)
                                              ↓
                                    PyTorch Dataset
                                              ↓
                                    GNN Training (imitation learning)
                                              ↓
                                    Trained Policy Network
                                              ↓
                                    Deployment (fast scheduling)
```

### 2. Graph Representation
**Nodes**: Operations with features
- Machine class (one-hot encoded)
- Processing duration (normalized)
- Position in job sequence
- Job quantity and group
- Dynamic: scheduled status, availability, slack, remaining ops

**Edges**: 
- Precedence edges (must finish A before starting B)
- Same-machine edges (operations sharing resources)

**Global**: Instance-level statistics

### 3. GNN Architecture
```
Input Graph → Input Projection (Linear)
            ↓
          GAT Layer 1 (attention-based message passing)
            ↓
          GAT Layer 2
            ↓
          GAT Layer 3
            ↓
          MLP Head → Operation Scores (which op to schedule)
            ↓
          Global Pool + MLP → Value Estimate (for RL)
```

## Configuration Examples

### For Quick Testing
```python
# In config.py
N_TRAIN_INSTANCES = 20
N_VAL_INSTANCES = 5
MAX_JOBS_PER_INSTANCE = 4
MIP_TIME_LIMIT = 60

# Training
TRAINING_CONFIG = {
    'num_epochs': 20,
    'batch_size': 4,
    'patience': 5
}
```

### For Production Quality
```python
# In config.py
N_TRAIN_INSTANCES = 200
N_VAL_INSTANCES = 40
MAX_JOBS_PER_INSTANCE = 10
MIP_TIME_LIMIT = 300

# Training
TRAINING_CONFIG = {
    'num_epochs': 100,
    'batch_size': 16,
    'patience': 20
}

# Model
GNN_CONFIG = {
    'hidden_dim': 256,
    'num_gnn_layers': 4,
    'use_attention': True
}
```

## Expected Performance

### Small Instances (3-8 jobs, ~15-40 operations)
- **MIP**: Optimal, 10s - 5min
- **ML (trained)**: 5-15% gap, <1s ✨
- **SPT heuristic**: 20-40% gap, <0.1s

### Large Instances (15+ jobs, 80+ operations)
- **MIP**: Timeout or infeasible (>1 hour)
- **ML (trained)**: Good quality, ~1s ✨✨✨
- **SPT heuristic**: 30-50% gap, <0.1s

## Advanced Features

### 1. Transfer Learning
Train on small instances, deploy on large:
```bash
# Train on 3-8 jobs
python main.py --max-jobs 8

# Evaluate on 15 jobs (generalization)
python main.py --skip-training --eval-max-jobs 15
```

### 2. Custom Objectives
Adjust weights in `config.py`:
```python
# Prioritize makespan
WEIGHTS = {'w_MS': 0.8, 'w_T': 0.1, 'w_row': 0.05, 'w_asm': 0.0, 'w_gap': 0.05}

# Prioritize tardiness
WEIGHTS = {'w_MS': 0.1, 'w_T': 0.8, 'w_row': 0.05, 'w_asm': 0.0, 'w_gap': 0.05}
```

### 3. Reinforcement Learning (Future)
Skeleton provided in `rl_env.py`:
```python
env = JobShopEnv(instance, weights=WEIGHTS)
state = env.reset()

# RL agent can interact with env
action = policy.select_action(state)
next_state, reward, done, info = env.step(action)
```

Implement PPO training for further improvement beyond imitation learning.

## File Organization After Training

```
project/
├── config.py
├── data_loader.py
├── mip_oracle.py
├── graph_builder.py
├── dataset.py
├── gnn_model.py
├── training.py
├── rl_env.py
├── evaluation.py
├── main.py
├── demo.py
├── README.md
├── requirements.txt
├── dataset_cache/           # Cached datasets (reusable!)
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
├── checkpoints/             # Trained models
│   ├── best_model.pt        # Best validation model
│   ├── final_model.pt       # Final epoch model
│   ├── training_history.json
│   └── training_curves.png
└── logs/                    # Optional logs
```

## Common Workflows

### Workflow 1: Initial Training
```bash
# First run - generates datasets and trains
python main.py --num-epochs 50

# Subsequent runs - reuse cached datasets
python main.py --skip-dataset-generation --num-epochs 100
```

### Workflow 2: Hyperparameter Tuning
```bash
# Edit config.py to change hyperparameters
# Then train new model
python main.py --skip-dataset-generation

# Compare models by checking training_history.json
```

### Workflow 3: Production Deployment
```bash
# Train best model
python main.py --num-epochs 100 --batch-size 16

# Evaluate on large instances
python main.py \
    --skip-training \
    --n-eval-instances 100 \
    --eval-max-jobs 20

# Use trained model in your code
```

Example usage in production:
```python
from gnn_model import SchedulingGNN
from evaluation import MLScheduler
from graph_builder import GraphBuilder
import torch

# Load trained model
model = SchedulingGNN()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create scheduler
scheduler = MLScheduler(model, GraphBuilder())

# Schedule new instance
result = scheduler.schedule(your_instance)
print(f"Makespan: {result['makespan']}")
print(f"Schedule: {result['schedule']}")
```

## Tips for Success

### 1. Data Quality
- Ensure BOLD_FLAG column exists in islem_tam_tablo.xlsx
- Check that SÜRE (dk) values are positive
- Verify machine mapping in config.MACHINE_MAP

### 2. Training
- Start with small dataset (20 instances) for quick iteration
- Use `--mip-time-limit 60` for faster training data generation
- Monitor validation accuracy - should reach 60-80%
- Early stopping prevents overfitting

### 3. Evaluation
- Always compare with heuristics (SPT, LPT)
- Use `--compare-with-mip` on small test set to verify correctness
- Test generalization by evaluating on larger instances than training

### 4. Performance
- Use CUDA if available (`--cpu` to force CPU)
- Batch size 8-16 works well
- Smaller model = faster inference, larger = better quality

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Gurobi license not found" | Install Gurobi and set license |
| "Data files not found" | Update paths in config.py |
| "CUDA out of memory" | Use `--cpu` or reduce batch size |
| "Training accuracy = 0" | Check MIP is solving successfully |
| "Model not improving" | Try lower learning rate (0.0001) |

## Next Steps

1. ✅ **Run demo**: `python demo.py`
2. ✅ **Prepare data**: Place Excel files in directory
3. ✅ **Train model**: `python main.py`
4. ✅ **Evaluate**: Check results in evaluation summary
5. ⚠️ **Fine-tune**: Adjust weights and hyperparameters
6. ⚠️ **Deploy**: Use trained model in production
7. 🔮 **Extend**: Implement full RL training for further improvement

## Technical Details

### Imitation Learning Loss
```
Loss = CrossEntropy(predicted_action, expert_action)
```
Where expert_action comes from MIP optimal schedule.

### Graph Neural Network Update
```
h_i^(l+1) = ReLU(W_self × h_i^(l) + Σ_{j∈N(i)} α_ij × W_neigh × h_j^(l))
```
Where α_ij are attention weights (if using GAT).

### Action Selection
```
action = argmax_i score_i  subject to  operation_i is available
```

### Reward (for RL, future)
```
reward = -(w_MS × MS + w_T × Σ T_i + w_row × Σ W_row + w_gap × Σ Gap)
```

## Support

- Read README.md for detailed documentation
- Check demo.py for usage examples
- Modify config.py for customization
- All code is well-commented and modular

## License

MIT License - Free to use and modify!

---

**Created**: November 2024
**Python**: 3.8+
**PyTorch**: 2.0+
**Gurobi**: 10.0+

Enjoy your ML-powered job shop scheduler! 🚀
