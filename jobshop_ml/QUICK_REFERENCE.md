# Quick Reference Guide

## Core Module Imports

```python
from core import (
    SchedulingInstance,      # Instance representation
    ScheduleSolution,        # Solution with save/load
    DataLoader,             # Excel loading
    create_instance,         # Instance creation
    validate_instance,       # Instance validation
    compute_statistics,     # Instance statistics
    ModelBuilder,           # MIP model construction
    Solver,                # Model solving
    Reporter,               # Reporting
    ExcelWriter             # Excel export
)
```

## Common Workflows

### 1. Solve and Save

```python
from core import DataLoader, create_instance, ModelBuilder, Solver

# Load data
loader = DataLoader()
loader.load_data()

# Create instance
instance = create_instance(loader)

# Build and solve
builder = ModelBuilder()
model = builder.build_model(instance)
solver = Solver(time_limit=300, verbose=True)
solution = solver.solve(model, instance)

# Save
solution.save('solution.pkl')
```

### 2. Load and Export (No Re-solving)

```python
from core import ScheduleSolution, ExcelWriter

# Load solution
solution = ScheduleSolution.load('solution.pkl')

# Export to Excel
excel_writer = ExcelWriter()
excel_writer.export_schedule(solution, 'schedule.xlsx')
```

### 3. Command Line: Solve

```bash
python main_scheduling.py --solve --save-sol sol.pkl --export-excel schedule.xlsx
```

### 4. Command Line: Load and Export

```bash
python main_scheduling.py --load-sol sol.pkl --export-excel schedule.xlsx
```

### 5. Command Line: Save LP Only

```bash
python main_scheduling.py --save-lp model.lp
```

## Key Classes

### ScheduleSolution

```python
# Attributes
solution.start_times          # Dict[(job, op)] -> start time
solution.completion_times     # Dict[(job, op)] -> completion time
solution.makespan             # Total makespan
solution.objective_value      # Objective value
solution.is_optimal           # Optimality flag

# Methods
solution.save('file.pkl')                    # Save solution
ScheduleSolution.load('file.pkl')            # Load solution
solution.get_schedule_sequence()             # Get chronological sequence
solution.validate()                          # Validate feasibility
```

### ModelBuilder

```python
builder = ModelBuilder(weights=config.WEIGHTS)
model = builder.build_model(instance)
# Model is ready for solving (not solved yet)
```

### Solver

```python
solver = Solver(time_limit=300, gap_tolerance=0.05, verbose=False)
solution = solver.solve(model, instance)
```

### ExcelWriter

```python
writer = ExcelWriter()
writer.export_schedule(solution, 'file.xlsx')
writer.export_multiple_schedules([('name1', sol1), ('name2', sol2)], 'file.xlsx')
```

### Reporter

```python
reporter = Reporter()
report = reporter.generate_report(solution)  # Get report string
reporter.print_schedule(solution)           # Print to console
metrics = reporter.compute_metrics(solution)  # Get KPIs
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--solve` | Solve the instance |
| `--save-sol FILE` | Save solution to .pkl |
| `--save-lp FILE` | Save model to .lp |
| `--load-sol FILE` | Load solution from .pkl |
| `--load-lp FILE` | Load solution from .lp |
| `--export-excel FILE` | Export to Excel |
| `--print-schedule` | Print schedule |
| `--print-report` | Print detailed report |
| `--time-limit SEC` | Solver time limit |
| `--verbose` | Print solver output |
| `--jobs JOB1 JOB2 ...` | Specific jobs to include |
| `--random-instance` | Generate random subset |

## File Structure

```
core/
├── instance.py          # SchedulingInstance class
├── solution.py          # ScheduleSolution with save/load
├── data_loader.py        # Excel loading
├── preprocessing.py      # Instance creation
├── model_builder.py      # MIP model (optimized)
├── solver.py            # Solver execution
├── reporter.py          # Reporting
└── excel_writer.py      # Excel export
```

## Performance Tips

1. **Use tight Big-M**: Automatically computed (sum of durations)
2. **Reduced binaries**: Only conflicting operations (different jobs)
3. **Solver parameters**: Optimized for scheduling problems
4. **Solution persistence**: Save solutions to avoid re-solving
5. **Excel export**: Can export without re-solving

## Common Issues

### Issue: "Solution file not found"
**Solution**: Check file path, ensure solution was saved first

### Issue: "Instance validation failed"
**Solution**: Check that operations have positive durations, precedence is valid

### Issue: "Failed to solve"
**Solution**: Increase time limit, check instance size, verify Gurobi license

### Issue: "Cannot load from LP"
**Solution**: LP files may not contain solutions. Use .pkl files instead.



