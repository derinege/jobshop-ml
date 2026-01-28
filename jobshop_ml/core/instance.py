"""
Scheduling instance representation module.

This module defines the SchedulingInstance class which represents a single
job shop scheduling problem instance with operations, precedence constraints,
and metadata.
"""

from typing import Dict, List, Tuple


class SchedulingInstance:
    """
    Represents a single job shop scheduling problem instance.
    
    This class encapsulates all information needed to define a scheduling
    problem, including jobs, operations, precedence constraints, and metadata.
    It serves as the input to the MIP model builder and solver.
    
    Attributes:
    -----------
    jobs : List[str]
        List of job identifiers (TITLE values from Excel).
    operations : Dict[Tuple[str, str], Dict]
        Mapping from (job, operation_name) to operation data containing:
        - 'sira_no': int - Sequence number within job
        - 'duration': float - Processing time in minutes
        - 'machine_class': str - Machine class identifier
    precedence : List[Tuple[str, str, str]]
        List of precedence constraints as (job, op_a, op_b) tuples indicating
        that op_a must complete before op_b starts.
    due_dates : Dict[str, float]
        Mapping from job identifier to due date in minutes (currently all 0).
    job_quantities : Dict[str, int]
        Mapping from job identifier to production quantity.
    job_groups : Dict[str, str]
        Mapping from job identifier to group identifier.
    """
    
    def __init__(self, jobs: List[str]):
        """
        Initialize a scheduling instance with a list of jobs.
        
        Parameters:
        -----------
        jobs : List[str]
            List of job identifiers to include in this instance.
        """
        self.jobs = jobs
        self.operations = {}
        self.precedence = []
        self.due_dates = {job: 0.0 for job in jobs}
        self.job_quantities = {}
        self.job_groups = {}
    
    def add_operation(self, job: str, op_name: str, sira_no: int, 
                      duration: float, machine_class: str) -> None:
        """
        Add an operation to the instance.
        
        Parameters:
        -----------
        job : str
            Job identifier.
        op_name : str
            Operation name/identifier.
        sira_no : int
            Sequence number of this operation within the job.
        duration : float
            Processing time in minutes.
        machine_class : str
            Machine class identifier (e.g., 'WLD', 'ASM').
        """
        self.operations[(job, op_name)] = {
            'sira_no': sira_no,
            'duration': duration,
            'machine_class': machine_class
        }
    
    def add_precedence(self, job: str, op_a: str, op_b: str) -> None:
        """
        Add a precedence constraint.
        
        This indicates that operation op_a must complete before operation op_b
        can start. Precedence constraints are typically added sequentially
        based on SIRA_NO values.
        
        Parameters:
        -----------
        job : str
            Job identifier.
        op_a : str
            Predecessor operation name.
        op_b : str
            Successor operation name.
        """
        self.precedence.append((job, op_a, op_b))
    
    def get_job_operations(self, job: str) -> List[Tuple[str, Dict]]:
        """
        Get all operations for a specific job, sorted by sequence number.
        
        Parameters:
        -----------
        job : str
            Job identifier.
        
        Returns:
        --------
        List[Tuple[str, Dict]]
            List of (operation_name, operation_data) tuples sorted by sira_no.
        """
        ops = [(op_name, data) for (j, op_name), data in self.operations.items() if j == job]
        ops.sort(key=lambda x: x[1]['sira_no'])
        return ops
    
    def get_operations_by_machine(self, machine_class: str) -> List[Tuple[str, str]]:
        """
        Get all operations that use a specific machine class.
        
        Parameters:
        -----------
        machine_class : str
            Machine class identifier.
        
        Returns:
        --------
        List[Tuple[str, str]]
            List of (job, operation_name) tuples for operations using this machine.
        """
        return [(job, op_name) for (job, op_name), data in self.operations.items() 
                if data['machine_class'] == machine_class]
    
    def __repr__(self) -> str:
        """
        String representation of the instance.
        
        Returns:
        --------
        str
            A formatted string describing the instance.
        """
        return (f"SchedulingInstance(jobs={len(self.jobs)}, "
                f"operations={len(self.operations)}, "
                f"precedence={len(self.precedence)})")



