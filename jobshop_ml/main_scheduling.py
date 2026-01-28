"""
Main entry point for weighted job shop scheduling optimization.

This script provides a command-line interface for solving scheduling instances
with options to solve, save models, load solutions, and export to Excel.
"""

import argparse
import os
import sys
from typing import Optional

# Import refactored core modules
from core import (
    SchedulingInstance,
    ScheduleSolution,
    DataLoader,
    create_instance,
    validate_instance,
    compute_statistics,
    ModelBuilder,
    Solver,
    Reporter,
    ExcelWriter
)
import config


def main():
    """
    Main function for scheduling optimization pipeline.
    
    This function orchestrates the complete scheduling workflow:
    1. Load data (if needed)
    2. Create or load instance
    3. Build model (if solving)
    4. Solve or load solution
    5. Save solution/LP (if requested)
    6. Export to Excel (if requested)
    7. Generate reports
    """
    parser = argparse.ArgumentParser(
        description="Weighted Job Shop Scheduling Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve and save solution
  python main_scheduling.py --solve --save-sol solution.pkl --export-excel schedule.xlsx
  
  # Load existing solution and export to Excel
  python main_scheduling.py --load-sol solution.pkl --export-excel schedule.xlsx
  
  # Build model, save LP, but don't solve
  python main_scheduling.py --save-lp model.lp
  
  # Solve with custom time limit
  python main_scheduling.py --solve --time-limit 600 --save-sol solution.pkl
        """
    )
    
    # Data loading
    parser.add_argument('--islem-tam-path', type=str, default=None,
                       help='Path to islem_tam_tablo.xlsx (default: config value)')
    parser.add_argument('--bold-sure-path', type=str, default=None,
                       help='Path to bold_islem_sure_tablosu.xlsx (default: config value)')
    
    # Instance creation
    parser.add_argument('--jobs', type=str, nargs='+', default=None,
                       help='Specific jobs to include (default: all BOLD jobs)')
    parser.add_argument('--random-instance', action='store_true',
                       help='Generate random instance subset')
    parser.add_argument('--min-jobs', type=int, default=config.MIN_JOBS_PER_INSTANCE,
                       help='Minimum jobs for random instance')
    parser.add_argument('--max-jobs', type=int, default=config.MAX_JOBS_PER_INSTANCE,
                       help='Maximum jobs for random instance')
    
    # Solving options
    parser.add_argument('--solve', action='store_true',
                       help='Solve the instance (default: False)')
    parser.add_argument('--time-limit', type=int, default=config.MIP_TIME_LIMIT,
                       help='Solver time limit in seconds')
    parser.add_argument('--gap-tolerance', type=float, default=config.MIP_GAP_TOLERANCE,
                       help='MIP gap tolerance (e.g., 0.05 for 5%%)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print solver output')
    
    # Solution loading/saving
    parser.add_argument('--load-sol', type=str, default=None,
                       help='Load solution from .pkl file (skips solving)')
    parser.add_argument('--load-lp', type=str, default=None,
                       help='Load solution from .lp file (skips solving)')
    parser.add_argument('--save-sol', type=str, default=None,
                       help='Save solution to .pkl file')
    parser.add_argument('--save-lp', type=str, default=None,
                       help='Save model to .lp file')
    
    # Export options
    parser.add_argument('--export-excel', type=str, default=None,
                       help='Export schedule to Excel file')
    
    # Reporting
    parser.add_argument('--print-schedule', action='store_true',
                       help='Print schedule to console')
    parser.add_argument('--print-report', action='store_true',
                       help='Print detailed report')
    
    args = parser.parse_args()
    
    # ========================================================================
    # STEP 1: Load Data (if needed)
    # ========================================================================
    
    instance = None
    solution = None
    
    if args.load_sol or args.load_lp:
        # Loading existing solution - may not need data
        print("Loading existing solution...")
        if args.load_sol:
            if not os.path.exists(args.load_sol):
                print(f"Error: Solution file not found: {args.load_sol}")
                sys.exit(1)
            solution = ScheduleSolution.load(args.load_sol)
            instance = solution.instance
            print(f"Loaded solution from {args.load_sol}")
        elif args.load_lp:
            # For LP loading, we need the instance
            print("Warning: LP file loading requires instance. Loading data first...")
            data_loader = DataLoader(
                args.islem_tam_path or config.DATA_PATH_ISLEM_TAM,
                args.bold_sure_path or config.DATA_PATH_BOLD_SURE
            )
            data_loader.load_data()
            if args.jobs:
                instance = create_instance(data_loader, args.jobs)
            else:
                instance = create_instance(data_loader)
            
            solution = ScheduleSolution.load_from_lp(args.load_lp, instance)
            if solution is None:
                print("Error: Could not extract solution from LP file")
                sys.exit(1)
            print(f"Loaded solution from {args.load_lp}")
    else:
        # Need to create instance
        print("Loading data and creating instance...")
        data_loader = DataLoader(
            args.islem_tam_path or config.DATA_PATH_ISLEM_TAM,
            args.bold_sure_path or config.DATA_PATH_BOLD_SURE
        )
        data_loader.load_data()
        
        if args.random_instance:
            from core import generate_random_instances
            instances = generate_random_instances(
                data_loader, n_instances=1,
                min_jobs=args.min_jobs, max_jobs=args.max_jobs
            )
            if not instances:
                print("Error: Failed to generate random instance")
                sys.exit(1)
            instance = instances[0]
        elif args.jobs:
            instance = create_instance(data_loader, args.jobs)
        else:
            instance = create_instance(data_loader)
        
        # Validate instance
        is_valid, errors = validate_instance(instance)
        if not is_valid:
            print("Error: Instance validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        # Print instance statistics
        stats = compute_statistics(instance)
        print(f"Instance: {stats['n_jobs']} jobs, {stats['n_operations']} operations")
    
    # ========================================================================
    # STEP 2: Solve (if requested)
    # ========================================================================
    
    if args.solve and solution is None:
        print("\n" + "="*80)
        print("MIP MODEL OLUŞTURMA VE ÇÖZME")
        print("="*80)
        
        # Build model
        model_builder = ModelBuilder(weights=config.WEIGHTS)
        model = model_builder.build_model(instance)
        
        # Save LP file if requested
        if args.save_lp:
            model.write(args.save_lp)
            print(f"Saved model to {args.save_lp}")
        
        # Solve
        solver = Solver(
            time_limit=args.time_limit,
            gap_tolerance=args.gap_tolerance,
            verbose=args.verbose
        )
        solution = solver.solve(model, instance)
        
        if solution is None:
            print("Error: Failed to solve instance")
            sys.exit(1)
        
        print(f"Solution found: Objective = {solution.objective_value:.2f}, "
              f"Makespan = {solution.makespan:.2f}")
        
        # Save solution if requested
        if args.save_sol:
            solution.save(args.save_sol)
            print(f"Saved solution to {args.save_sol}")
    
    # ========================================================================
    # STEP 3: Export and Report
    # ========================================================================
    
    if solution is None:
        print("Error: No solution available. Use --solve or --load-sol")
        sys.exit(1)
    
    # Export to Excel
    if args.export_excel:
        print(f"\nExporting to Excel: {args.export_excel}")
        excel_writer = ExcelWriter()
        excel_writer.export_schedule(solution, args.export_excel)
    
    # Print reports
    reporter = Reporter()
    
    if args.print_report:
        print("\n" + reporter.generate_report(solution))
    
    if args.print_schedule:
        reporter.print_schedule(solution)
    
    # Always print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Makespan: {solution.makespan:.2f} minutes")
    print(f"Objective: {solution.objective_value:.2f}")
    print(f"Status: {'OPTIMAL' if solution.is_optimal else 'FEASIBLE'}")
    if solution.solve_time > 0:
        print(f"Solve Time: {solution.solve_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()

