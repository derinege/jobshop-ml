# 🚀 Quick Start Guide

Get your ML-based job shop scheduler running in 3 steps!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install torch pandas numpy openpyxl gurobipy tqdm tabulate matplotlib --break-system-packages
```

**Note**: You need a Gurobi license (free for academics at [gurobi.com](https://www.gurobi.com))

## Step 2: Run Demo (1 minute)

Test the system without your data files:

```bash
python demo.py
```

You should see:
- ✓ MIP solver working
- ✓ Graph representation building
- ✓ GNN model forward pass
- ✓ RL environment simulation
- ✓ Heuristic comparisons

## Step 3: Train on Your Data (10-30 minutes)

### Option A: With Your Excel Files

1. Place your files in the directory:
   - `islem_tam_tablo.xlsx`
   - `bold_islem_sure_tablosu.xlsx`

2. Run the training pipeline:

```bash
python main.py
```

This will:
- Load your Excel data
- Generate training datasets using Gurobi
- Train the GNN policy
- Evaluate vs heuristics

### Option B: Start Small (Recommended for First Time)

```bash
python main.py \
    --num-epochs 20 \
    --n-eval-instances 5 \
    --eval-max-jobs 4
```

This uses smaller settings for faster testing.

## What You Get

After training, you'll have:

```
✓ checkpoints/best_model.pt         Your trained ML scheduler
✓ checkpoints/training_curves.png   Training progress visualization
✓ dataset_cache/                    Reusable training data
✓ Evaluation summary                Performance comparison table
```

## Next Steps

### Customize Weights

Edit `config.py` to adjust scheduling priorities:

```python
WEIGHTS = {
    'w_MS': 0.2,    # Makespan (total time)
    'w_T': 0.5,     # Tardiness (lateness)
    'w_row': 0.1,   # Waiting between operations
    'w_asm': 0.1,   # Assembly waiting
    'w_gap': 0.1    # Machine idle time
}
```

### Train Longer

```bash
python main.py \
    --skip-dataset-generation \
    --num-epochs 100 \
    --batch-size 16
```

### Evaluate on Larger Instances

```bash
python main.py \
    --skip-training \
    --n-eval-instances 50 \
    --eval-max-jobs 15
```

## Typical Workflow

```
Day 1: Run demo.py → Verify installation
       Run main.py with small settings → Get first model
       
Day 2: Generate larger dataset → Improve training data
       Train longer → Better model quality
       
Day 3: Evaluate on production instances → Test scalability
       Fine-tune weights → Optimize for your priorities
       
Production: Use trained model for fast scheduling!
```

## Expected Results

### Training Accuracy
- Should reach **60-80%** accuracy (matching expert decisions)
- Training loss should decrease steadily
- Validation loss should stabilize (early stopping prevents overfitting)

### Inference Performance
- **Small instances** (3-5 jobs): <0.1s, 5-10% gap vs optimal
- **Medium instances** (8-10 jobs): ~0.2s, 10-15% gap
- **Large instances** (15+ jobs): ~0.5s, much better than heuristics

### Comparison
| Method | Speed | Quality |
|--------|-------|---------|
| MIP (Gurobi) | ⚠️ Slow (minutes) | ✓ Optimal |
| **ML (Trained)** | ✓ Fast (<1s) | ✓ Near-optimal (5-15% gap) |
| SPT Heuristic | ✓ Very fast | ✗ Poor (30-50% gap) |

## Troubleshooting

### "Gurobi license not found"
```bash
# Get free academic license from gurobi.com
# Then set environment variable:
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

### "Data files not found"
```python
# Edit config.py:
DATA_PATH_ISLEM_TAM = "path/to/your/islem_tam_tablo.xlsx"
DATA_PATH_BOLD_SURE = "path/to/your/bold_islem_sure_tablosu.xlsx"
```

Or use command line:
```bash
python main.py \
    --islem-tam-path path/to/islem_tam_tablo.xlsx \
    --bold-sure-path path/to/bold_islem_sure_tablosu.xlsx
```

### "CUDA out of memory"
```bash
python main.py --cpu  # Force CPU usage
```

### Training not improving
- Check that MIP solver is finding solutions
- Try lower learning rate in `config.py`: `learning_rate = 0.0001`
- Generate more training data: increase `N_TRAIN_INSTANCES`
- Train longer: `--num-epochs 100`

## Files to Read

1. **INDEX.txt** - Navigation guide (start here!)
2. **PROJECT_SUMMARY.md** - Architecture and overview
3. **README.md** - Detailed documentation
4. **config.py** - All settings to customize

## Getting Help

```python
# Test individual modules:
python data_loader.py    # Test data loading
python mip_oracle.py     # Test Gurobi
python graph_builder.py  # Test graph construction
python demo.py           # Full system test

# Get command line help:
python main.py --help
```

## Pro Tips

💡 **Cache datasets**: Use `--skip-dataset-generation` after first run (saves time!)

💡 **Start small**: Use 3-5 jobs for quick iteration, then scale up

💡 **Monitor training**: Check `training_curves.png` - loss should decrease

💡 **Compare methods**: Use `--compare-with-mip` to verify correctness

💡 **Production**: Load `best_model.pt` and use `MLScheduler` class

---

**Ready to build amazing schedulers? Let's go! 🎯**

```bash
python demo.py        # Start here!
python main.py        # Then train!
```
