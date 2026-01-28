# Refactoring Plan for Weighted Job Shop Scheduling System

## Current Code Structure Summary

### Current Files:
1. **mip_oracle.py**: Contains MIP model building and solving (mixed responsibilities)
2. **data_loader.py**: Excel loading and instance creation
3. **evaluation.py**: Evaluation and comparison (includes heuristics)
4. **config.py**: Configuration
5. **main.py**: Main training pipeline
6. **Other ML files**: gnn_model.py, training.py, etc. (not part of MIP refactoring)

### Current Issues:
1. **MIP Model Issues:**
   - Big-M = 100,000 (too large, can be reduced)
   - All operation pairs get binary variables (can be optimized)
   - No solution persistence (save/load)
   - No LP file export
   - No Excel export functionality
   - Model building and solving are mixed

2. **Code Organization:**
   - No clear separation of concerns
   - Preprocessing mixed with data loading
   - Reporting/evaluation mixed with solving
   - No Excel writer module

3. **Functionality Gaps:**
   - Cannot load existing solutions
   - Cannot export to Excel without re-solving
   - No flags for solve/save_lp/save_sol
   - Functions may auto-run (not Jupyter-friendly)

## Proposed New Structure

```
jobshop_ml/
├── core/
│   ├── __init__.py
│   ├── data_loader.py          # Excel loading only
│   ├── preprocessing.py        # Data preprocessing and instance creation
│   ├── model_builder.py        # Gurobi model construction (no solving)
│   ├── solver.py               # Solver runner and solution extraction
│   ├── solution.py             # Solution class and save/load
│   ├── reporter.py             # Reporting and statistics
│   └── excel_writer.py         # Excel export functionality
│
├── ml/                         # ML-related files (unchanged)
│   ├── gnn_model.py
│   ├── training.py
│   └── ...
│
├── config.py                   # Configuration
├── main.py                     # Main entry point
└── requirements.txt
```

## Refactoring Tasks

### 1. Data Loader (`core/data_loader.py`)
- **Responsibility**: Load Excel files only
- **Functions**:
  - `load_excel_files()`: Read Excel files
  - `extract_bold_jobs()`: Filter BOLD products
- **No instance creation** (moved to preprocessing)

### 2. Preprocessing (`core/preprocessing.py`)
- **Responsibility**: Create SchedulingInstance from raw data
- **Functions**:
  - `create_instance()`: Build instance from data
  - `validate_instance()`: Validate instance data
  - `compute_statistics()`: Instance statistics

### 3. Model Builder (`core/model_builder.py`)
- **Responsibility**: Build Gurobi model (NO solving)
- **Optimizations**:
  - Reduce Big-M: Use tighter bounds (sum of durations)
  - Remove unnecessary binaries: Only create Z for conflicting operations
  - Tighten constraints: Add valid inequalities
  - Improve variable indexing: Use more efficient data structures
- **Functions**:
  - `build_model()`: Create Gurobi model
  - `compute_big_m()`: Compute tight Big-M value
  - `get_conflicting_operations()`: Only pairs that can conflict
  - `add_valid_inequalities()`: Strengthen formulation

### 4. Solver (`core/solver.py`)
- **Responsibility**: Run solver and extract solutions
- **Functions**:
  - `solve_model()`: Run Gurobi optimizer
  - `extract_solution()`: Extract solution from model
  - `configure_solver()`: Set solver parameters

### 5. Solution (`core/solution.py`)
- **Responsibility**: Solution representation and persistence
- **Functions**:
  - `save_solution()`: Save to .pkl
  - `load_solution()`: Load from .pkl
  - `load_from_lp()`: Load solution from .lp file
  - `validate_solution()`: Check solution feasibility

### 6. Reporter (`core/reporter.py`)
- **Responsibility**: Generate reports and statistics
- **Functions**:
  - `generate_report()`: Text report
  - `print_schedule()`: Print schedule
  - `compute_metrics()`: Calculate KPIs

### 7. Excel Writer (`core/excel_writer.py`)
- **Responsibility**: Export schedules to Excel
- **Functions**:
  - `export_schedule()`: Export single schedule
  - `export_multiple_schedules()`: Batch export
  - `create_schedule_worksheets()`: Format Excel output

## MIP Model Optimizations

### 1. Reduce Big-M
- **Current**: 100,000 (fixed)
- **New**: `sum(all operation durations)` or tighter bound
- **Impact**: Tighter LP relaxation, faster solving

### 2. Remove Unnecessary Binaries
- **Current**: All pairs of operations on same machine
- **New**: Only pairs that can actually conflict (time windows overlap)
- **Impact**: Fewer binary variables, smaller model

### 3. Tighten Constraints
- Add valid inequalities (e.g., precedence-based bounds)
- Add symmetry-breaking constraints
- Strengthen disjunctive constraints

### 4. Improve Variable Indexing
- Use dictionaries with efficient keys
- Cache frequently accessed data
- Optimize constraint generation loops

### 5. Solver Parameters
- Enable presolve
- Set appropriate MIP focus
- Use heuristics
- Set good initial solution if available

## New Features

### 1. Solution Persistence
```python
# Save solution
solution.save('output/solution.pkl')

# Load solution
solution = ScheduleSolution.load('output/solution.pkl')

# Load from LP file
solution = ScheduleSolution.load_from_lp('output/model.lp', instance)
```

### 2. Excel Export
```python
# Export single schedule
excel_writer.export_schedule(solution, 'output/schedule.xlsx')

# Export multiple schedules
excel_writer.export_multiple_schedules(solutions, 'output/all_schedules.xlsx')
```

### 3. Flags and Options
```python
# In main function
solve=True/False          # Whether to solve
save_lp=True/False        # Save LP file
save_sol=True/False       # Save solution
load_sol='path.pkl'       # Load existing solution
```

## Documentation Style

All functions will have academic-style docstrings:
```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description of the function.
    
    This function performs [detailed description]. The implementation
    follows [methodology/reference] and is designed to [purpose].
    
    Parameters:
    -----------
    param1 : Type
        Description of param1, including any constraints or assumptions.
    param2 : Type
        Description of param2.
    
    Returns:
    --------
    ReturnType
        Description of return value, including structure and meaning.
    
    Raises:
    -------
    ValueError
        When invalid input is provided.
    
    Notes:
    ------
    Additional implementation notes, complexity analysis, or references.
    
    Examples:
    --------
    >>> example usage
    """
```

## Implementation Order

1. Create new folder structure
2. Refactor data_loader.py (split into loader + preprocessing)
3. Refactor mip_oracle.py (split into model_builder + solver)
4. Create solution.py (save/load functionality)
5. Create excel_writer.py
6. Create reporter.py
7. Optimize MIP model
8. Update main.py with new flags
9. Add documentation
10. Test all functionality

