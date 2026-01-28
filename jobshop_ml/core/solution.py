"""
Solution representation and persistence module.

This module provides the ScheduleSolution class for representing scheduling
solutions and methods for saving/loading solutions from disk.
"""

import pickle
import os
from typing import Dict, List, Tuple, Optional
from .instance import SchedulingInstance

# Optional import for LP file loading
try:
    from gurobipy import Model, read
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class ScheduleSolution:
    """
    Represents a solution to a weighted job shop scheduling problem.
    
    This class encapsulates all information about a solved scheduling instance,
    including start times, completion times, machine assignments, and objective
    components. Solutions can be saved to and loaded from disk for later analysis
    or export without re-solving.
    
    Attributes:
    -----------
    instance : SchedulingInstance
        The scheduling instance that was solved.
    start_times : Dict[Tuple[str, str], float]
        Mapping from (job, operation_name) to start time in minutes.
    completion_times : Dict[Tuple[str, str], float]
        Mapping from (job, operation_name) to completion time in minutes.
    machine_assignments : Dict[Tuple[str, str], str]
        Mapping from (job, operation_name) to machine class identifier.
    makespan : float
        Total makespan (maximum completion time) in minutes.
    tardiness : Dict[str, float]
        Mapping from job identifier to tardiness value (currently unused).
    waiting_times : Dict[Tuple[str, str, str], float]
        Mapping from (job, op_a, op_b) to waiting time between precedence operations.
    idle_times : Dict[Tuple[str, str, str, str], float]
        Mapping from (j1, op1, j2, op2) to idle gap on machine between operations.
    objective_value : float
        Final weighted objective value.
    objective_components : Dict[str, float]
        Breakdown of objective components (makespan, total_waiting, total_idle).
    is_optimal : bool
        Whether the solution is proven optimal by the solver.
    solve_time : float
        Time taken to solve in seconds.
    model_status : Optional[int]
        Gurobi model status code (if available).
    """
    
    def __init__(self, instance: SchedulingInstance):
        """
        Initialize an empty solution for a given instance.
        
        Parameters:
        -----------
        instance : SchedulingInstance
            The scheduling instance this solution corresponds to.
        """
        self.instance = instance
        self.start_times = {}
        self.completion_times = {}
        self.machine_assignments = {}
        self.makespan = 0.0
        self.tardiness = {}
        self.waiting_times = {}
        self.idle_times = {}
        self.objective_value = 0.0
        self.objective_components = {}
        self.is_optimal = False
        self.solve_time = 0.0
        self.model_status = None
    
    def get_schedule_sequence(self) -> List[Tuple[float, str, str, str]]:
        """
        Get operations in chronological order by start time.
        
        This method returns all operations sorted by their start time, which
        is useful for visualization and analysis of the schedule.
        
        Returns:
        --------
        List[Tuple[float, str, str, str]]
            List of tuples (start_time, job, op_name, machine_class) sorted
            by start time in ascending order.
        """
        sequence = []
        for (job, op_name), start_time in self.start_times.items():
            machine_class = self.machine_assignments.get((job, op_name), 'UNKNOWN')
            sequence.append((start_time, job, op_name, machine_class))
        
        sequence.sort(key=lambda x: x[0])
        return sequence
    
    def save(self, filepath: str) -> None:
        """
        Save solution to disk as a pickle file.
        
        This method serializes the entire solution object to a pickle file,
        allowing it to be loaded later without re-solving the instance.
        
        Parameters:
        -----------
        filepath : str
            Path to the output pickle file (typically with .pkl extension).
        
        Raises:
        -------
        IOError
            If the file cannot be written.
        """
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise IOError(f"Failed to save solution to {filepath}: {e}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ScheduleSolution':
        """
        Load a solution from a pickle file.
        
        This class method deserializes a previously saved solution from disk.
        The solution can then be used for analysis or export without re-solving.
        
        Parameters:
        -----------
        filepath : str
            Path to the pickle file containing the solution.
        
        Returns:
        --------
        ScheduleSolution
            The loaded solution object.
        
        Raises:
        -------
        FileNotFoundError
            If the file does not exist.
        IOError
            If the file cannot be read or deserialized.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Solution file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                solution = pickle.load(f)
            if not isinstance(solution, ScheduleSolution):
                raise ValueError(f"File {filepath} does not contain a ScheduleSolution object")
            return solution
        except Exception as e:
            raise IOError(f"Failed to load solution from {filepath}: {e}")
    
    @classmethod
    def load_from_lp(cls, lp_filepath: str, instance: SchedulingInstance) -> Optional['ScheduleSolution']:
        """
        Load a solution from a Gurobi LP file.
        
        This method reads a Gurobi model file (.lp) and attempts to extract
        solution information if the model has been solved. Note that LP files
        may not always contain solution values.
        
        Parameters:
        -----------
        lp_filepath : str
            Path to the Gurobi LP file.
        instance : SchedulingInstance
            The scheduling instance corresponding to the model.
        
        Returns:
        --------
        Optional[ScheduleSolution]
            The solution if successfully extracted, None otherwise.
        
        Raises:
        -------
        ImportError
            If gurobipy is not installed.
        
        Notes:
        ------
        LP files primarily contain the model formulation, not necessarily
        solution values. This method will only work if the LP file was saved
        after solving and contains solution information.
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("gurobipy is required for loading LP files. Install with: pip install gurobipy")
        
        if not os.path.exists(lp_filepath):
            raise FileNotFoundError(f"LP file not found: {lp_filepath}")
        
        try:
            model = read(lp_filepath)
            
            # Check if model has been solved
            if model.status not in [2, 3, 5, 9, 10, 11]:  # Optimal, TimeLimit, etc.
                return None
            
            solution = cls(instance)
            
            # Extract variable values
            for var in model.getVars():
                var_name = var.VarName
                var_value = var.X
                
                # Parse variable names to extract solution components
                if var_name.startswith('S_'):
                    # Start time variable: S_job_op
                    parts = var_name.split('_', 2)
                    if len(parts) >= 3:
                        job = parts[1]
                        op_name = '_'.join(parts[2:])
                        solution.start_times[(job, op_name)] = var_value
                
                elif var_name.startswith('C_'):
                    # Completion time variable: C_job_op
                    parts = var_name.split('_', 2)
                    if len(parts) >= 3:
                        job = parts[1]
                        op_name = '_'.join(parts[2:])
                        solution.completion_times[(job, op_name)] = var_value
                        solution.machine_assignments[(job, op_name)] = (
                            instance.operations.get((job, op_name), {}).get('machine_class', 'UNKNOWN')
                        )
                
                elif var_name == 'Makespan':
                    solution.makespan = var_value
            
            # Extract objective value
            solution.objective_value = model.ObjVal
            solution.is_optimal = (model.status == 2)  # GRB.OPTIMAL
            solution.model_status = model.status
            
            return solution
            
        except Exception as e:
            print(f"Warning: Could not load solution from LP file {lp_filepath}: {e}")
            return None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the solution for feasibility and correctness.
        
        This method checks that the solution satisfies all constraints:
        - Processing time constraints (C >= S + duration)
        - Precedence constraints (S_b >= C_a for precedence)
        - Machine disjunction constraints (no overlapping operations on same machine)
        
        Returns:
        --------
        Tuple[bool, List[str]]
            A tuple (is_valid, error_messages) where is_valid is True if the
            solution is feasible, and error_messages contains any violations found.
        """
        errors = []
        
        # Check processing time constraints
        for (job, op_name), data in self.instance.operations.items():
            if (job, op_name) in self.start_times and (job, op_name) in self.completion_times:
                start = self.start_times[(job, op_name)]
                completion = self.completion_times[(job, op_name)]
                duration = data['duration']
                
                if completion < start + duration - 1e-6:  # Small tolerance for floating point
                    errors.append(f"Processing time violation: {job}-{op_name}: "
                                f"C={completion:.2f} < S={start:.2f} + duration={duration:.2f}")
        
        # Check precedence constraints
        for (job, op_a, op_b) in self.instance.precedence:
            if (job, op_a) in self.completion_times and (job, op_b) in self.start_times:
                c_a = self.completion_times[(job, op_a)]
                s_b = self.start_times[(job, op_b)]
                
                if s_b < c_a - 1e-6:
                    errors.append(f"Precedence violation: {job}-{op_a} completes at {c_a:.2f}, "
                                f"but {job}-{op_b} starts at {s_b:.2f}")
        
        # Check machine disjunction (simplified - check for overlaps)
        ops_by_machine = {}
        for (job, op_name), mc in self.machine_assignments.items():
            if mc not in ops_by_machine:
                ops_by_machine[mc] = []
            if (job, op_name) in self.start_times and (job, op_name) in self.completion_times:
                ops_by_machine[mc].append((
                    (job, op_name),
                    self.start_times[(job, op_name)],
                    self.completion_times[(job, op_name)]
                ))
        
        for mc, ops in ops_by_machine.items():
            # Sort by start time
            ops_sorted = sorted(ops, key=lambda x: x[1])
            for i in range(len(ops_sorted) - 1):
                _, start1, comp1 = ops_sorted[i]
                _, start2, comp2 = ops_sorted[i + 1]
                
                if start2 < comp1 - 1e-6:
                    errors.append(f"Machine overlap on {mc}: operations overlap in time")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        """
        String representation of the solution.
        
        Returns:
        --------
        str
            A formatted string describing the solution.
        """
        return (f"ScheduleSolution(makespan={self.makespan:.2f}, "
                f"objective={self.objective_value:.2f}, "
                f"optimal={self.is_optimal})")

