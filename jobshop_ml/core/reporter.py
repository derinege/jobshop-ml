"""
Reporting module for generating solution reports and statistics.

This module provides functionality to generate text reports, print schedules,
and compute key performance indicators (KPIs) from scheduling solutions.
"""

from typing import Dict, List, Tuple
from .solution import ScheduleSolution
from .instance import SchedulingInstance
from tabulate import tabulate


class Reporter:
    """
    Generates reports and statistics from scheduling solutions.
    
    This class provides methods to create human-readable reports, print
    schedules in various formats, and compute performance metrics.
    """
    
    def __init__(self):
        """Initialize reporter."""
        pass
    
    def generate_report(self, solution: ScheduleSolution) -> str:
        """
        Generate a comprehensive text report for a solution.
        
        This method creates a formatted text report containing all key
        information about the solution, including objective value, makespan,
        component breakdown, and schedule summary.
        
        Parameters:
        -----------
        solution : ScheduleSolution
            The solution to report on.
        
        Returns:
        --------
        str
            A formatted text report.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SCHEDULING SOLUTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Instance information
        instance = solution.instance
        report_lines.append(f"Instance: {len(instance.jobs)} jobs, "
                          f"{len(instance.operations)} operations")
        report_lines.append("")
        
        # Objective information
        report_lines.append("Objective Value:")
        report_lines.append(f"  Total: {solution.objective_value:.2f}")
        report_lines.append("")
        
        report_lines.append("Objective Components:")
        for component, value in solution.objective_components.items():
            report_lines.append(f"  {component}: {value:.2f}")
        report_lines.append("")
        
        # Makespan
        report_lines.append(f"Makespan: {solution.makespan:.2f} minutes")
        report_lines.append("")
        
        # Solution quality
        report_lines.append(f"Solution Status: {'OPTIMAL' if solution.is_optimal else 'FEASIBLE'}")
        report_lines.append(f"Solve Time: {solution.solve_time:.2f} seconds")
        report_lines.append("")
        
        # Schedule summary
        sequence = solution.get_schedule_sequence()
        report_lines.append(f"Schedule: {len(sequence)} operations")
        report_lines.append("")
        
        report_lines.append("First 10 Operations:")
        for i, (start, job, op, mc) in enumerate(sequence[:10]):
            completion = solution.completion_times.get((job, op), 0)
            report_lines.append(f"  {i+1:2d}. t={start:6.2f}-{completion:6.2f}: "
                              f"{job:15s} - {op:20s} on {mc}")
        
        if len(sequence) > 10:
            report_lines.append(f"  ... and {len(sequence) - 10} more operations")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def print_schedule(self, solution: ScheduleSolution, 
                      max_operations: int = 50) -> None:
        """
        Print the schedule in chronological order.
        
        This method prints operations sorted by start time, showing the
        temporal sequence of the schedule.
        
        Parameters:
        -----------
        solution : ScheduleSolution
            The solution to print.
        max_operations : int, optional
            Maximum number of operations to print. Default is 50.
        """
        sequence = solution.get_schedule_sequence()
        
        print("\n" + "=" * 100)
        print("SCHEDULE (Chronological Order)")
        print("=" * 100)
        print(f"{'#':<4} {'Start':<10} {'End':<10} {'Job':<20} {'Operation':<30} {'Machine':<10}")
        print("-" * 100)
        
        for i, (start, job, op, mc) in enumerate(sequence[:max_operations]):
            completion = solution.completion_times.get((job, op), start)
            print(f"{i+1:<4} {start:<10.2f} {completion:<10.2f} {job:<20s} {op:<30s} {mc:<10s}")
        
        if len(sequence) > max_operations:
            print(f"\n... and {len(sequence) - max_operations} more operations")
        
        print("=" * 100)
    
    def compute_metrics(self, solution: ScheduleSolution) -> Dict:
        """
        Compute key performance indicators (KPIs) from a solution.
        
        This method calculates various metrics that are useful for evaluating
        solution quality, including utilization, waiting times, and idle times.
        
        Parameters:
        -----------
        solution : ScheduleSolution
            The solution to analyze.
        
        Returns:
        --------
        Dict
            Dictionary containing computed metrics:
            - 'makespan': Total makespan
            - 'total_waiting': Total waiting time
            - 'total_idle': Total idle time
            - 'avg_waiting': Average waiting time per precedence
            - 'avg_idle': Average idle time per machine conflict
            - 'machine_utilization': Dict mapping machine class to utilization
            - 'job_completion_times': Dict mapping job to completion time
        """
        instance = solution.instance
        
        # Basic metrics
        metrics = {
            'makespan': solution.makespan,
            'total_waiting': solution.objective_components.get('total_waiting', 0),
            'total_idle': solution.objective_components.get('total_idle', 0),
        }
        
        # Average waiting time
        if len(solution.waiting_times) > 0:
            metrics['avg_waiting'] = (
                sum(solution.waiting_times.values()) / len(solution.waiting_times)
            )
        else:
            metrics['avg_waiting'] = 0.0
        
        # Average idle time
        if len(solution.idle_times) > 0:
            metrics['avg_idle'] = (
                sum(solution.idle_times.values()) / len(solution.idle_times)
            )
        else:
            metrics['avg_idle'] = 0.0
        
        # Machine utilization
        machine_utilization = {}
        for (job, op_name), data in instance.operations.items():
            mc = data['machine_class']
            duration = data['duration']
            
            if mc not in machine_utilization:
                machine_utilization[mc] = {'total_duration': 0, 'makespan': solution.makespan}
            
            machine_utilization[mc]['total_duration'] += duration
        
        for mc in machine_utilization:
            total_dur = machine_utilization[mc]['total_duration']
            makespan = machine_utilization[mc]['makespan']
            machine_utilization[mc]['utilization'] = (
                total_dur / makespan if makespan > 0 else 0.0
            )
        
        metrics['machine_utilization'] = machine_utilization
        
        # Job completion times
        job_completion_times = {}
        for job in instance.jobs:
            job_ops = instance.get_job_operations(job)
            max_completion = 0.0
            for op_name, _ in job_ops:
                completion = solution.completion_times.get((job, op_name), 0)
                max_completion = max(max_completion, completion)
            job_completion_times[job] = max_completion
        
        metrics['job_completion_times'] = job_completion_times
        
        return metrics
    
    def print_comparison_table(self, solutions: List[Tuple[str, ScheduleSolution]]) -> None:
        """
        Print a comparison table for multiple solutions.
        
        This method creates a formatted table comparing multiple solutions,
        which is useful for evaluating different methods or configurations.
        
        Parameters:
        -----------
        solutions : List[Tuple[str, ScheduleSolution]]
            List of (method_name, solution) tuples to compare.
        """
        headers = ['Method', 'Makespan', 'Objective', 'Waiting', 'Idle', 'Time (s)', 'Optimal']
        rows = []
        
        for method_name, solution in solutions:
            rows.append([
                method_name,
                f"{solution.makespan:.2f}",
                f"{solution.objective_value:.2f}",
                f"{solution.objective_components.get('total_waiting', 0):.2f}",
                f"{solution.objective_components.get('total_idle', 0):.2f}",
                f"{solution.solve_time:.2f}",
                "Yes" if solution.is_optimal else "No"
            ])
        
        print("\n" + "=" * 100)
        print("SOLUTION COMPARISON")
        print("=" * 100)
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("=" * 100)



