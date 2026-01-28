"""
Preprocessing module for creating scheduling instances from raw data.

This module provides functions to transform raw Excel data into SchedulingInstance
objects, including operation mapping, precedence construction, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .instance import SchedulingInstance
from .data_loader import DataLoader
import config


def _map_operation_to_machine(op_name: str) -> str:
    """
    Map operation name to machine class using configuration mapping.
    
    This function uses pattern matching to identify machine classes from
    operation names. It searches for keywords in the operation name and
    returns the corresponding machine class code.
    
    Parameters:
    -----------
    op_name : str
        Operation name (İŞLEM_ADI from Excel).
    
    Returns:
    --------
    str
        Machine class code (e.g., 'WLD', 'ASM', 'GEN').
    
    Notes:
    ------
    The mapping is case-insensitive and uses substring matching. If no
    match is found, returns 'GEN' as default.
    """
    op_upper = str(op_name).upper().strip()
    
    for key, machine_class in config.MACHINE_MAP.items():
        if key in op_upper:
            return machine_class
    
    return 'GEN'


def create_instance(data_loader: DataLoader, job_subset: Optional[List[str]] = None) -> SchedulingInstance:
    """
    Create a scheduling instance from loaded Excel data.
    
    This function transforms raw Excel data into a SchedulingInstance object
    by extracting operations, building precedence constraints, and mapping
    operations to machine classes. If job_subset is provided, only those jobs
    are included; otherwise, all BOLD jobs are included.
    
    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader instance with loaded Excel data.
    job_subset : Optional[List[str]], optional
        List of job TITLEs to include. If None, includes all BOLD jobs.
        Default is None.
    
    Returns:
    --------
    SchedulingInstance
        A fully constructed scheduling instance ready for optimization.
    
    Raises:
    -------
    RuntimeError
        If data_loader has not loaded data yet.
    ValueError
        If job_subset contains no valid BOLD jobs.
    
    Notes:
    ------
    The function performs the following steps:
    1. Filters jobs to BOLD products (or provided subset)
    2. Extracts operations from bold_islem_sure_tablosu
    3. Maps operations to machine classes
    4. Builds precedence constraints based on SIRA_NO
    5. Adds job metadata (quantities, groups)
    """
    if data_loader.df_tam is None or data_loader.df_sure is None:
        raise RuntimeError("DataLoader must load data first. Call load_data() before create_instance()")
    
    # Determine which jobs to include
    if job_subset is None:
        jobs = data_loader.get_bold_jobs()
    else:
        bold_jobs = set(data_loader.get_bold_jobs())
        jobs = [j for j in job_subset if j in bold_jobs]
        
        if not jobs:
            raise ValueError("No valid BOLD jobs in job_subset")
    
    instance = SchedulingInstance(jobs)
    df_tam, df_sure = data_loader.get_dataframes()
    
    # Add job metadata
    for _, row in df_tam[df_tam['TITLE'].isin(jobs)].iterrows():
        job = row['TITLE']
        instance.job_groups[job] = row.get('GRUP', 'UNKNOWN')
        instance.job_quantities[job] = row.get('QTY', 1)
    
    # Add operations
    df_ops = df_sure[df_sure['TITLE'].isin(jobs)].copy()
    
    for _, row in df_ops.iterrows():
        job = row['TITLE']
        op_name = row['İŞLEM_ADI']
        sira_no = row['SIRA_NO']
        duration = row['SÜRE (dk)']
        
        # Skip operations with invalid duration
        if pd.isna(duration) or duration <= 0:
            continue
        
        machine_class = _map_operation_to_machine(op_name)
        instance.add_operation(job, op_name, sira_no, duration, machine_class)
    
    # Build precedence relations (consecutive SIRA_NO within each job)
    for job in instance.jobs:
        job_ops = instance.get_job_operations(job)
        
        for i in range(len(job_ops) - 1):
            op_a_name = job_ops[i][0]
            op_b_name = job_ops[i + 1][0]
            instance.add_precedence(job, op_a_name, op_b_name)
    
    return instance


def validate_instance(instance: SchedulingInstance) -> Tuple[bool, List[str]]:
    """
    Validate a scheduling instance for correctness and completeness.
    
    This function checks that the instance has valid structure:
    - All jobs have at least one operation
    - All operations have positive durations
    - Precedence constraints reference valid operations
    - No duplicate operations
    
    Parameters:
    -----------
    instance : SchedulingInstance
        The instance to validate.
    
    Returns:
    --------
    Tuple[bool, List[str]]
        A tuple (is_valid, error_messages) where is_valid is True if the
        instance is valid, and error_messages contains any issues found.
    """
    errors = []
    
    # Check that all jobs have operations
    for job in instance.jobs:
        job_ops = instance.get_job_operations(job)
        if len(job_ops) == 0:
            errors.append(f"Job {job} has no operations")
    
    # Check operation durations
    for (job, op_name), data in instance.operations.items():
        if data['duration'] <= 0:
            errors.append(f"Operation {job}-{op_name} has non-positive duration: {data['duration']}")
    
    # Check precedence constraints
    for (job, op_a, op_b) in instance.precedence:
        if (job, op_a) not in instance.operations:
            errors.append(f"Precedence references non-existent operation: {job}-{op_a}")
        if (job, op_b) not in instance.operations:
            errors.append(f"Precedence references non-existent operation: {job}-{op_b}")
    
    # Check for duplicate operations
    op_keys = list(instance.operations.keys())
    if len(op_keys) != len(set(op_keys)):
        errors.append("Duplicate operations found in instance")
    
    return len(errors) == 0, errors


def compute_statistics(instance: SchedulingInstance) -> Dict:
    """
    Compute descriptive statistics for a scheduling instance.
    
    This function calculates various metrics about the instance structure,
    including operation counts, machine usage, and duration statistics.
    These statistics are useful for understanding instance complexity and
    for reporting purposes.
    
    Parameters:
    -----------
    instance : SchedulingInstance
        The instance to analyze.
    
    Returns:
    --------
    Dict
        Dictionary containing statistics with keys:
        - 'n_jobs': Number of jobs
        - 'n_operations': Total number of operations
        - 'n_precedence': Number of precedence constraints
        - 'total_duration': Sum of all operation durations
        - 'avg_ops_per_job': Average operations per job
        - 'max_ops_per_job': Maximum operations in any job
        - 'ops_per_machine': Dict mapping machine class to operation count
        - 'avg_duration': Average operation duration
        - 'max_duration': Maximum operation duration
        - 'min_duration': Minimum operation duration
    """
    total_duration = sum(data['duration'] for data in instance.operations.values())
    durations = [data['duration'] for data in instance.operations.values()]
    
    # Operations per machine class
    ops_per_machine = {}
    for (job, op_name), data in instance.operations.items():
        mc = data['machine_class']
        ops_per_machine[mc] = ops_per_machine.get(mc, 0) + 1
    
    # Operations per job
    ops_per_job = {}
    for job in instance.jobs:
        ops_per_job[job] = len(instance.get_job_operations(job))
    
    return {
        'n_jobs': len(instance.jobs),
        'n_operations': len(instance.operations),
        'n_precedence': len(instance.precedence),
        'total_duration': total_duration,
        'avg_ops_per_job': np.mean(list(ops_per_job.values())) if ops_per_job else 0,
        'max_ops_per_job': max(ops_per_job.values()) if ops_per_job else 0,
        'ops_per_machine': ops_per_machine,
        'avg_duration': np.mean(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'min_duration': min(durations) if durations else 0,
    }


def generate_random_instances(data_loader: DataLoader,
                               n_instances: int,
                               min_jobs: int = config.MIN_JOBS_PER_INSTANCE,
                               max_jobs: int = config.MAX_JOBS_PER_INSTANCE,
                               seed: int = config.RANDOM_SEED) -> List[SchedulingInstance]:
    """
    Generate multiple random subset instances for training/validation.
    
    This function creates multiple SchedulingInstance objects by randomly
    sampling subsets of jobs from the full dataset. This is useful for
    generating training data where each instance is a smaller subproblem
    that can be solved efficiently by the MIP solver.
    
    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader instance with loaded data.
    n_instances : int
        Number of instances to generate.
    min_jobs : int, optional
        Minimum number of jobs per instance. Defaults to config value.
    max_jobs : int, optional
        Maximum number of jobs per instance. Defaults to config value.
    seed : int, optional
        Random seed for reproducibility. Defaults to config value.
    
    Returns:
    --------
    List[SchedulingInstance]
        List of randomly generated scheduling instances.
    
    Notes:
    ------
    The function uses numpy random sampling to select job subsets. If an
    instance cannot be created (e.g., no valid operations), it is skipped
    and a warning is printed.
    """
    if data_loader.bold_jobs is None:
        data_loader.load_data()
    
    np.random.seed(seed)
    instances = []
    bold_jobs = data_loader.get_bold_jobs()
    
    for i in range(n_instances):
        n_jobs = np.random.randint(min_jobs, max_jobs + 1)
        job_subset = np.random.choice(bold_jobs, size=min(n_jobs, len(bold_jobs)), replace=False).tolist()
        
        try:
            instance = create_instance(data_loader, job_subset)
            # Validate instance
            is_valid, errors = validate_instance(instance)
            if is_valid:
                instances.append(instance)
            else:
                print(f"Warning: Instance {i} failed validation: {errors[0]}")
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Failed to create instance {i}: {e}")
            continue
    
    print(f"Generated {len(instances)} valid random instances")
    return instances

