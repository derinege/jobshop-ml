# Job Shop Scheduling ML Approximation

A machine learning-based approach to job shop scheduling that learns from optimal MIP solutions and generalizes to larger instances.

## Overview

This project implements a **Graph Neural Network (GNN)** policy that approximates optimal job shop schedules. The system:

1. Uses your existing **Gurobi MIP model** as an oracle to generate optimal schedules on small instances
2. Trains a **GNN** via **imitation learning** to mimic these optimal decisions
3. Applies the trained policy to larger instances where MIP is too slow
4. Respects precedence constraints, machine capacity, and your weighted multi-objective structure

## Architecture

```
┌─────────────────┐
│  Excel Data     │  islem_tam_tablo.xlsx, bold_islem_sure_tablosu.xlsx
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Data Loader    │  Extracts jobs, operations, precedence, durations
└────────┬────────┘
         │
         v
┌─────────────────┐
│  MIP Oracle     │  Solves small instances with Gurobi (ground truth)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Graph Builder   │  Converts schedules to graph representations
└────────┬────────┘
         │
         v
┌─────────────────┐
│  GNN Model      │  Policy network (GraphConv/GAT layers + MLP)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Training       │  Imitation learning from MIP solutions
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Evaluation     │  Compare ML vs MIP vs heuristics
└─────────────────┘
```

## Project Structure

```
.
├── config.py              # All hyperparameters and weights
├── data_loader.py         # Load Excel data and create instances
├── mip_oracle.py          # Gurobi MIP solver (oracle for training)
├── graph_builder.py       # Convert scheduling states to graphs
├── dataset.py             # Dataset generation and management
├── gnn_model.py           # GNN policy architecture
├── training.py            # Imitation learning trainer
├── rl_env.py              # RL environment (skeleton for future work)
├── evaluation.py          # Evaluation and comparison tools
├── main.py                # Main training pipeline
└── README.md              # This file
```

## Requirements

```bash
pip install torch pandas numpy openpyxl gurobipy tqdm tabulate matplotlib --break-system-packages
```

**Note**: You need a valid Gurobi license. Academic licenses are free.

## Quick Start

### 1. Prepare Your Data

Ensure you have two Excel files:
- `islem_tam_tablo.xlsx`: Full BOM with TITLE, BOLD_FLAG, GRUP, QTY columns
- `bold_islem_sure_tablosu.xlsx`: Operations with TITLE, İŞLEM_ADI, SIRA_NO, SÜRE (dk)

Place them in the working directory or update paths in `config.py`.

### 2. Run the Full Pipeline

```bash
# Train from scratch (generates datasets, trains model, evaluates)
python main.py

# With custom settings
python main.py \
    --num-epochs 50 \
    --batch-size 16 \
    --n-eval-instances 20 \
    --compare-with-mip
```

### 3. Training Only

```bash
# If you already have cached datasets
python main.py --skip-dataset-generation

# Or skip everything except training
python main.py --skip-data-loading --skip-dataset-generation
```

### 4. Evaluation Only

```bash
# Load pre-trained model and evaluate
python main.py \
    --skip-data-loading \
    --skip-dataset-generation \
    --skip-training \
    --n-eval-instances 30 \
    --compare-with-mip
```

## Configuration

All hyperparameters are in `config.py`:

### Objective Weights

```python
WEIGHTS = {
    'w_MS': 0.2,    # Makespan
    'w_T': 0.5,     # Tardiness
    'w_row': 0.1,   # Precedence waiting
    'w_asm': 0.1,   # Assembly waiting
    'w_gap': 0.1    # Machine idle gaps
}
```

Adjust these to match your priorities.

### Training Parameters

```python
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 8,
    'num_epochs': 100,
    'patience': 15,  # Early stopping
    ...
}
```

### GNN Architecture

```python
GNN_CONFIG = {
    'hidden_dim': 128,
    'num_gnn_layers': 3,
    'use_attention': True,
    'num_heads': 4,
    ...
}
```

## Modules

### data_loader.py

Loads Excel files and creates scheduling instances.

```python
from data_loader import DataLoader

loader = DataLoader()
loader.load_data()

# Full instance with all BOLD products
full_instance = loader.create_full_instance()

# Random subset for training
train_instances = loader.generate_random_instances(
    n_instances=100,
    min_jobs=3,
    max_jobs=8
)
```

### mip_oracle.py

Solves instances with Gurobi MIP.

```python
from mip_oracle import MIPOracle

oracle = MIPOracle(weights=WEIGHTS, time_limit=300)
solution = oracle.solve(instance)

print(solution.makespan)
print(solution.objective_components)
```

### graph_builder.py

Converts scheduling states to graph representations.

```python
from graph_builder import GraphBuilder

graph_builder = GraphBuilder()
graph = graph_builder.build_static_graph(instance)

# Graph has node features, edge indices, global features
print(graph.node_features.shape)
print(graph.edge_index.shape)
```

### gnn_model.py

GNN policy network.

```python
from gnn_model import SchedulingGNN

model = SchedulingGNN(config.GNN_CONFIG)

# Forward pass
outputs = model(batch)
scores = outputs['scores']  # Score for each operation
value = outputs['value']    # Value estimate
```

### training.py

Imitation learning trainer.

```python
from training import ImitationTrainer

trainer = ImitationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

history = trainer.train(num_epochs=100)
```

### evaluation.py

Compare different scheduling methods.

```python
from evaluation import Evaluator

evaluator = Evaluator(ml_model=trained_model)

# Evaluate on test instances
summary = evaluator.evaluate_dataset(
    test_instances,
    methods=['mip', 'ml', 'spt', 'lpt', 'fifo']
)

evaluator.print_comparison_table(summary)
```

## Workflow Examples

### Example 1: Quick Demo

```bash
# Use small instances for quick testing
python main.py \
    --num-epochs 20 \
    --n-eval-instances 5 \
    --eval-max-jobs 4
```

### Example 2: Full Training

```bash
# Generate large dataset and train thoroughly
python main.py \
    --num-epochs 100 \
    --batch-size 16 \
    --mip-time-limit 600 \
    --n-eval-instances 50 \
    --compare-with-mip
```

### Example 3: Evaluate Pre-trained Model

```bash
# Skip training, just evaluate
python main.py \
    --skip-data-loading \
    --skip-dataset-generation \
    --skip-training \
    --n-eval-instances 100 \
    --eval-max-jobs 15
```

## Advanced: Reinforcement Learning

The `rl_env.py` module provides a skeleton for RL training:

```python
from rl_env import JobShopEnv

env = JobShopEnv(instance, weights=WEIGHTS)
state = env.reset()

while not env.done:
    available_ops = env.get_available_operations()
    action = policy.select_action(available_ops)
    next_state, reward, done, info = env.step(action)
```

**TODO for full RL**:
1. Implement experience collection (rollouts)
2. Compute advantages with GAE
3. Update policy with PPO objective
4. Update value function
5. Track metrics and convergence

## Multi-Objective Structure

The system respects your weighted multi-objective formulation:

```
Objective = w_MS * Makespan 
          + w_T * Σ Tardiness
          + w_row * Σ Precedence_Waiting
          + w_asm * Σ Assembly_Waiting  
          + w_gap * Σ Machine_Idle_Gaps
```

All components are tracked during training and evaluation.

## Tips

### Faster Training

1. **Reduce dataset size**: Lower `N_TRAIN_INSTANCES` in `config.py`
2. **Smaller instances**: Reduce `MAX_JOBS_PER_INSTANCE`
3. **Shorter MIP time**: Set `--mip-time-limit 60` for faster (less optimal) solutions
4. **Use cache**: Generate datasets once, then use `--skip-dataset-generation`

### Better Performance

1. **More data**: Increase `N_TRAIN_INSTANCES` (100+ recommended)
2. **Larger model**: Increase `hidden_dim` and `num_gnn_layers` in `GNN_CONFIG`
3. **More epochs**: Train for 100+ epochs with patience=20
4. **Tune weights**: Adjust objective weights in `config.py` to match your priorities

### Debugging

1. **Check data**: Run `python data_loader.py` to verify data loading
2. **Test MIP**: Run `python mip_oracle.py` to test Gurobi
3. **Inspect graphs**: Run `python graph_builder.py` to check graph construction
4. **Dry run**: Use `--num-epochs 2 --n-eval-instances 2` for quick testing

## Output Files

After training:
- `checkpoints/best_model.pt`: Best model checkpoint
- `checkpoints/final_model.pt`: Final model checkpoint
- `checkpoints/training_history.json`: Training metrics
- `checkpoints/training_curves.png`: Loss and accuracy plots
- `dataset_cache/*.pkl`: Cached datasets (reusable)

## Performance Expectations

On small instances (3-8 jobs):
- **MIP**: Optimal, but slow (10s - 5min per instance)
- **ML**: Near-optimal (5-15% gap), fast (<1s per instance)
- **Heuristics**: Fast but poor quality (20-50% gap)

On large instances (15+ jobs):
- **MIP**: Too slow (>1 hour) or infeasible
- **ML**: Good quality, scalable
- **Heuristics**: Fast but suboptimal

## Troubleshooting

### "Gurobi license not found"
- Install Gurobi and get a license (free for academics)
- Set `GRB_LICENSE_FILE` environment variable

### "Data files not found"
- Update paths in `config.py` or use `--islem-tam-path` and `--bold-sure-path`
- Ensure BOLD_FLAG column exists in islem_tam_tablo.xlsx

### "CUDA out of memory"
- Use `--cpu` to train on CPU
- Reduce `batch_size` in config
- Use smaller model (reduce `hidden_dim`)

### "Training accuracy stuck at 0"
- Check that dataset generation succeeded
- Verify MIP solver is finding solutions
- Try lower learning rate (0.0001)

## Citation

If you use this code, please cite:

```bibtex
@misc{jobshop_ml_2024,
  title={Job Shop Scheduling ML Approximation},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - feel free to use and modify.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
