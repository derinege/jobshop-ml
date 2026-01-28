# Job Shop Scheduling ML System

Machine learning-based job shop scheduling system that learns from optimal MIP solutions and generalizes to larger instances.

## 🚀 Quick Start

```bash
# Install dependencies
cd jobshop_ml
pip install -r requirements.txt

# Run demo
python demo.py

# Train on your data
python main.py
```

## 📋 Overview

This project implements a **Graph Neural Network (GNN)** policy that approximates optimal job shop schedules. The system:

1. Uses **Gurobi MIP model** as an oracle to generate optimal schedules on small instances
2. Trains a **GNN** via **imitation learning** to mimic these optimal decisions
3. Applies the trained policy to larger instances where MIP is too slow
4. Respects precedence constraints, machine capacity, and weighted multi-objective structure

## 🎯 Objective Function

The system uses **3 weighted objective components**:

```
Objective = w_MS × Makespan + w_W × Total_Waiting + w_G × Total_Idle_Gap
```

Configure weights in `jobshop_ml/config.py`:

```python
WEIGHTS = {
    'w_MS': 0.3,    # Makespan weight
    'w_W': 0.5,     # Waiting weight
    'w_G': 0.2      # Gap weight
}
```

## 📁 Project Structure

```
.
├── jobshop_ml/              # Main project directory
│   ├── core/                # Core modules
│   ├── config.py            # Configuration and hyperparameters
│   ├── main.py              # Main training pipeline
│   ├── demo.py              # Quick demo script
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Detailed documentation
├── README.md                # This file
└── ...
```

## 📊 Data Requirements

Place your Excel files in the `jobshop_ml/` directory:

- `islem_tam_tablo.xlsx`: Full BOM with TITLE, BOLD_FLAG, GRUP, QTY columns
- `bold_islem_sure_tablosu.xlsx`: Operations with TITLE, İŞLEM_ADI, SIRA_NO, SÜRE (dk)

Or update paths in `jobshop_ml/config.py`.

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Gurobi Optimizer (with valid license - free for academics)
- See `jobshop_ml/requirements.txt` for full list

## 📖 Documentation

For detailed documentation, see:

- **Full Guide**: [`jobshop_ml/README.md`](jobshop_ml/README.md) - Comprehensive documentation
- **Updated Guide**: [`jobshop_ml/UPDATED_README.md`](jobshop_ml/UPDATED_README.md) - Latest updates and 3-component objective
- **Quick Start**: [`jobshop_ml/QUICK_START.md`](jobshop_ml/QUICK_START.md)
- **Project Summary**: [`jobshop_ml/PROJECT_SUMMARY.md`](jobshop_ml/PROJECT_SUMMARY.md)

## 🎓 Usage Examples

### Basic Training

```bash
cd jobshop_ml
python main.py --num-epochs 50 --batch-size 16
```

### Evaluation Only

```bash
python main.py --skip-training --n-eval-instances 20 --compare-with-mip
```

### Custom Configuration

Edit `jobshop_ml/config.py` to adjust:
- Objective weights
- Training parameters
- GNN architecture
- Data paths

## 📈 Performance

On small instances (3-8 jobs):
- **MIP**: Optimal, but slow (10s - 5min per instance)
- **ML**: Near-optimal (5-15% gap), fast (<1s per instance)

On large instances (15+ jobs):
- **MIP**: Too slow (>1 hour) or infeasible
- **ML**: Good quality, scalable

## 🔍 Key Features

- ✅ Graph Neural Network architecture
- ✅ Imitation learning from optimal solutions
- ✅ Multi-objective optimization support
- ✅ Precedence constraint handling
- ✅ Machine capacity constraints
- ✅ Evaluation and comparison tools

## 📝 License

MIT License - feel free to use and modify.

## 🤝 Contributing

For questions or issues, please open a GitHub issue.

---

**For detailed information, see [`jobshop_ml/README.md`](jobshop_ml/README.md)**

