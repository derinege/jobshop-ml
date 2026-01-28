"""
Excel export module for scheduling solutions.

This module provides functionality to export scheduling solutions to Excel
files in a structured format suitable for analysis and reporting.
"""

import pandas as pd
from typing import List, Optional, Dict, Tuple
from .solution import ScheduleSolution
import os


class ExcelWriter:
    """
    Exports scheduling solutions to Excel files.
    
    This class provides methods to export single or multiple solutions to
    Excel format with multiple worksheets for different views of the data.
    """
    
    def __init__(self):
        """Initialize Excel writer."""
        pass
    
    def export_schedule(self, solution: ScheduleSolution, 
                      filepath: str,
                      include_summary: bool = True,
                      include_sequence: bool = True,
                      include_by_job: bool = True,
                      include_by_machine: bool = True) -> None:
        """
        Export a single scheduling solution to an Excel file.
        
        This method creates an Excel file with multiple worksheets containing
        different views of the schedule: summary statistics, chronological
        sequence, operations grouped by job, and operations grouped by machine.
        
        Parameters:
        -----------
        solution : ScheduleSolution
            The solution to export.
        filepath : str
            Path to the output Excel file (should have .xlsx extension).
        include_summary : bool, optional
            Whether to include summary worksheet. Default is True.
        include_sequence : bool, optional
            Whether to include chronological sequence worksheet. Default is True.
        include_by_job : bool, optional
            Whether to include operations grouped by job. Default is True.
        include_by_machine : bool, optional
            Whether to include operations grouped by machine. Default is True.
        
        Raises:
        -------
        IOError
            If the file cannot be written.
        
        Notes:
        ------
        The Excel file will contain the following worksheets (if enabled):
        - Summary: Objective value, makespan, component breakdown
        - Sequence: All operations in chronological order
        - ByJob: Operations grouped by job
        - ByMachine: Operations grouped by machine class
        """
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                instance = solution.instance
                
                # Summary worksheet
                if include_summary:
                    summary_data = {
                        'Metric': [
                            'Makespan (minutes)',
                            'Objective Value',
                            'Total Waiting Time',
                            'Total Idle Time',
                            'Solve Time (seconds)',
                            'Is Optimal',
                            'Number of Jobs',
                            'Number of Operations'
                        ],
                        'Value': [
                            solution.makespan,
                            solution.objective_value,
                            solution.objective_components.get('total_waiting', 0),
                            solution.objective_components.get('total_idle', 0),
                            solution.solve_time,
                            'Yes' if solution.is_optimal else 'No',
                            len(instance.jobs),
                            len(instance.operations)
                        ]
                    }
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sequence worksheet (chronological)
                if include_sequence:
                    sequence = solution.get_schedule_sequence()
                    sequence_data = []
                    for i, (start, job, op, mc) in enumerate(sequence):
                        completion = solution.completion_times.get((job, op), start)
                        duration = instance.operations.get((job, op), {}).get('duration', 0)
                        sequence_data.append({
                            'Sequence': i + 1,
                            'Start Time': start,
                            'Completion Time': completion,
                            'Duration': duration,
                            'Job': job,
                            'Operation': op,
                            'Machine Class': mc
                        })
                    df_sequence = pd.DataFrame(sequence_data)
                    df_sequence.to_excel(writer, sheet_name='Sequence', index=False)
                
                # By Job worksheet
                if include_by_job:
                    job_data = []
                    for job in instance.jobs:
                        job_ops = instance.get_job_operations(job)
                        for op_name, op_data in job_ops:
                            start = solution.start_times.get((job, op_name), 0)
                            completion = solution.completion_times.get((job, op_name), start)
                            job_data.append({
                                'Job': job,
                                'Operation': op_name,
                                'Sequence': op_data['sira_no'],
                                'Start Time': start,
                                'Completion Time': completion,
                                'Duration': op_data['duration'],
                                'Machine Class': op_data['machine_class']
                            })
                    df_job = pd.DataFrame(job_data)
                    df_job = df_job.sort_values(['Job', 'Sequence'])
                    df_job.to_excel(writer, sheet_name='ByJob', index=False)
                
                # By Machine worksheet
                if include_by_machine:
                    machine_data = []
                    for (job, op_name), data in instance.operations.items():
                        start = solution.start_times.get((job, op_name), 0)
                        completion = solution.completion_times.get((job, op_name), start)
                        machine_data.append({
                            'Machine Class': data['machine_class'],
                            'Start Time': start,
                            'Completion Time': completion,
                            'Duration': data['duration'],
                            'Job': job,
                            'Operation': op_name
                        })
                    df_machine = pd.DataFrame(machine_data)
                    df_machine = df_machine.sort_values(['Machine Class', 'Start Time'])
                    df_machine.to_excel(writer, sheet_name='ByMachine', index=False)
            
            print(f"Solution exported to {filepath}")
            
        except Exception as e:
            raise IOError(f"Failed to export solution to {filepath}: {e}")
    
    def export_multiple_schedules(self, solutions: List[Tuple[str, ScheduleSolution]],
                                  filepath: str) -> None:
        """
        Export multiple solutions to a single Excel file.
        
        This method creates an Excel file with one worksheet per solution,
        plus a comparison worksheet showing key metrics for all solutions.
        This is useful for comparing different methods or configurations.
        
        Parameters:
        -----------
        solutions : List[Tuple[str, ScheduleSolution]]
            List of (solution_name, solution) tuples to export.
        filepath : str
            Path to the output Excel file.
        
        Raises:
        -------
        IOError
            If the file cannot be written.
        
        Notes:
        ------
        The Excel file contains:
        - Comparison: Summary table comparing all solutions
        - One worksheet per solution (named by solution_name) with full details
        """
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Comparison worksheet
                comparison_data = []
                for solution_name, solution in solutions:
                    comparison_data.append({
                        'Solution Name': solution_name,
                        'Makespan': solution.makespan,
                        'Objective Value': solution.objective_value,
                        'Total Waiting': solution.objective_components.get('total_waiting', 0),
                        'Total Idle': solution.objective_components.get('total_idle', 0),
                        'Solve Time (s)': solution.solve_time,
                        'Is Optimal': 'Yes' if solution.is_optimal else 'No'
                    })
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
                
                # Individual solution worksheets
                for solution_name, solution in solutions:
                    # Use simplified name for worksheet (Excel has 31 char limit)
                    sheet_name = solution_name[:31] if len(solution_name) <= 31 else solution_name[:28] + '...'
                    
                    # Export sequence for this solution
                    sequence = solution.get_schedule_sequence()
                    sequence_data = []
                    for i, (start, job, op, mc) in enumerate(sequence):
                        completion = solution.completion_times.get((job, op), start)
                        instance = solution.instance
                        duration = instance.operations.get((job, op), {}).get('duration', 0)
                        sequence_data.append({
                            'Sequence': i + 1,
                            'Start Time': start,
                            'Completion Time': completion,
                            'Duration': duration,
                            'Job': job,
                            'Operation': op,
                            'Machine Class': mc
                        })
                    df_sequence = pd.DataFrame(sequence_data)
                    df_sequence.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"Exported {len(solutions)} solutions to {filepath}")
            
        except Exception as e:
            raise IOError(f"Failed to export solutions to {filepath}: {e}")
    
    def create_schedule_worksheets(self, solution: ScheduleSolution) -> Dict[str, pd.DataFrame]:
        """
        Create pandas DataFrames for each worksheet type.
        
        This method creates the DataFrames that would be written to Excel,
        allowing for custom formatting or additional processing before export.
        
        Parameters:
        -----------
        solution : ScheduleSolution
            The solution to process.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping worksheet names to DataFrames.
        """
        instance = solution.instance
        worksheets = {}
        
        # Summary DataFrame
        summary_data = {
            'Metric': [
                'Makespan (minutes)',
                'Objective Value',
                'Total Waiting Time',
                'Total Idle Time',
                'Solve Time (seconds)',
                'Is Optimal',
                'Number of Jobs',
                'Number of Operations'
            ],
            'Value': [
                solution.makespan,
                solution.objective_value,
                solution.objective_components.get('total_waiting', 0),
                solution.objective_components.get('total_idle', 0),
                solution.solve_time,
                'Yes' if solution.is_optimal else 'No',
                len(instance.jobs),
                len(instance.operations)
            ]
        }
        worksheets['Summary'] = pd.DataFrame(summary_data)
        
        # Sequence DataFrame
        sequence = solution.get_schedule_sequence()
        sequence_data = []
        for i, (start, job, op, mc) in enumerate(sequence):
            completion = solution.completion_times.get((job, op), start)
            duration = instance.operations.get((job, op), {}).get('duration', 0)
            sequence_data.append({
                'Sequence': i + 1,
                'Start Time': start,
                'Completion Time': completion,
                'Duration': duration,
                'Job': job,
                'Operation': op,
                'Machine Class': mc
            })
        worksheets['Sequence'] = pd.DataFrame(sequence_data)
        
        # By Job DataFrame
        job_data = []
        for job in instance.jobs:
            job_ops = instance.get_job_operations(job)
            for op_name, op_data in job_ops:
                start = solution.start_times.get((job, op_name), 0)
                completion = solution.completion_times.get((job, op_name), start)
                job_data.append({
                    'Job': job,
                    'Operation': op_name,
                    'Sequence': op_data['sira_no'],
                    'Start Time': start,
                    'Completion Time': completion,
                    'Duration': op_data['duration'],
                    'Machine Class': op_data['machine_class']
                })
        df_job = pd.DataFrame(job_data)
        worksheets['ByJob'] = df_job.sort_values(['Job', 'Sequence'])
        
        # By Machine DataFrame
        machine_data = []
        for (job, op_name), data in instance.operations.items():
            start = solution.start_times.get((job, op_name), 0)
            completion = solution.completion_times.get((job, op_name), start)
            machine_data.append({
                'Machine Class': data['machine_class'],
                'Start Time': start,
                'Completion Time': completion,
                'Duration': data['duration'],
                'Job': job,
                'Operation': op_name
            })
        df_machine = pd.DataFrame(machine_data)
        worksheets['ByMachine'] = df_machine.sort_values(['Machine Class', 'Start Time'])
        
        return worksheets

