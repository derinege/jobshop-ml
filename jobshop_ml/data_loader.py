"""
Data Loading and Preprocessing Module
Reads Excel files and constructs scheduling problem instances.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import config


class SchedulingInstance:
    """
    Represents a single job shop scheduling problem instance.
    
    Attributes:
        jobs: List of job IDs (TITLE values)
        operations: Dict mapping (job, op_name) -> {
            'sira_no': int,
            'duration': float (minutes),
            'machine_class': str
        }
        precedence: List of tuples (job, op_a, op_b) indicating op_a must finish before op_b starts
        due_dates: Dict mapping job -> due date (currently 0 for all)
        job_quantities: Dict mapping job -> quantity
        job_groups: Dict mapping job -> group
    """
    
    def __init__(self, jobs: List[str]):
        self.jobs = jobs
        self.operations = {}
        self.precedence = []
        self.due_dates = {job: 0 for job in jobs}  # Currently all 0
        self.job_quantities = {}
        self.job_groups = {}
        
    def add_operation(self, job: str, op_name: str, sira_no: int, 
                      duration: float, machine_class: str):
        """Add an operation to the instance."""
        self.operations[(job, op_name)] = {
            'sira_no': sira_no,
            'duration': duration,
            'machine_class': machine_class
        }
    
    def add_precedence(self, job: str, op_a: str, op_b: str):
        """Add a precedence constraint: op_a must finish before op_b starts."""
        self.precedence.append((job, op_a, op_b))
    
    def get_job_operations(self, job: str) -> List[Tuple[str, Dict]]:
        """Get all operations for a specific job, sorted by SIRA_NO."""
        ops = [(op_name, data) for (j, op_name), data in self.operations.items() if j == job]
        ops.sort(key=lambda x: x[1]['sira_no'])
        return ops
    
    def get_operations_by_machine(self, machine_class: str) -> List[Tuple[str, str]]:
        """Get all (job, op_name) pairs for a specific machine class."""
        return [(job, op_name) for (job, op_name), data in self.operations.items() 
                if data['machine_class'] == machine_class]
    
    def __repr__(self):
        return (f"SchedulingInstance(jobs={len(self.jobs)}, "
                f"operations={len(self.operations)}, "
                f"precedence={len(self.precedence)})")


class DataLoader:
    """
    Loads and preprocesses Excel data files to create scheduling instances.
    """
    
    def __init__(self, 
                 islem_tam_path: str = config.DATA_PATH_ISLEM_TAM,
                 bold_sure_path: str = config.DATA_PATH_BOLD_SURE):
        """
        Initialize data loader.
        
        Args:
            islem_tam_path: Path to islem_tam_tablo.xlsx
            bold_sure_path: Path to bold_islem_sure_tablosu.xlsx
        """
        self.islem_tam_path = islem_tam_path
        self.bold_sure_path = bold_sure_path
        self.machine_map = config.MACHINE_MAP
        
        # Will be populated by load_data()
        self.df_tam = None
        self.df_sure = None
        self.bold_jobs = None
        
    def load_data(self):
        """Load Excel files and extract BOLD products."""
        print(f"Loading {self.islem_tam_path}...")
        self.df_tam = pd.read_excel(self.islem_tam_path)
        
        print(f"Loading {self.bold_sure_path}...")
        self.df_sure = pd.read_excel(self.bold_sure_path)
        
        # Filter for BOLD products
        bold_mask = self.df_tam['BOLD_FLAG'] == 1
        self.bold_jobs = self.df_tam[bold_mask]['TITLE'].unique().tolist()
        
        print(f"Found {len(self.bold_jobs)} BOLD jobs")
        print(f"Total operations in bold_islem_sure_tablosu: {len(self.df_sure)}")
        
        return self
    
    def _map_operation_to_machine(self, op_name: str) -> str:
        """
        Map operation name to machine class using MACHINE_MAP.
        
        Args:
            op_name: Operation name (İŞLEM_ADI)
            
        Returns:
            Machine class code (e.g., 'WLD', 'ASM', etc.)
        """
        op_upper = str(op_name).upper().strip()
        
        for key, machine_class in self.machine_map.items():
            if key in op_upper:
                return machine_class
        
        # Default to general if no match
        return 'GEN'
    
    def create_full_instance(self) -> SchedulingInstance:
        """
        Create a scheduling instance with ALL BOLD jobs.
        
        Returns:
            SchedulingInstance containing all BOLD products
        """
        if self.bold_jobs is None:
            self.load_data()
        
        instance = SchedulingInstance(self.bold_jobs)
        
        # Add job metadata
        for _, row in self.df_tam[self.df_tam['TITLE'].isin(self.bold_jobs)].iterrows():
            job = row['TITLE']
            instance.job_groups[job] = row.get('GRUP', 'UNKNOWN')
            instance.job_quantities[job] = row.get('QTY', 1)
        
        # Add operations
        df_bold_ops = self.df_sure[self.df_sure['TITLE'].isin(self.bold_jobs)].copy()
        
        for _, row in df_bold_ops.iterrows():
            job = row['TITLE']
            op_name = row['İŞLEM_ADI']
            sira_no = row['SIRA_NO']
            duration = row['SÜRE (dk)']
            
            # Skip operations with 0 or negative duration
            if pd.isna(duration) or duration <= 0:
                continue
            
            machine_class = self._map_operation_to_machine(op_name)
            
            instance.add_operation(job, op_name, sira_no, duration, machine_class)
        
        # Build precedence relations (consecutive SIRA_NO within each job)
        for job in instance.jobs:
            job_ops = instance.get_job_operations(job)
            
            for i in range(len(job_ops) - 1):
                op_a_name = job_ops[i][0]
                op_b_name = job_ops[i + 1][0]
                instance.add_precedence(job, op_a_name, op_b_name)
        
        print(f"Created instance: {instance}")
        return instance
    
    def create_subset_instance(self, job_subset: List[str]) -> SchedulingInstance:
        """
        Create a scheduling instance with a subset of jobs.
        Useful for generating smaller training instances.
        
        Args:
            job_subset: List of job TITLEs to include
            
        Returns:
            SchedulingInstance containing only specified jobs
        """
        if self.bold_jobs is None:
            self.load_data()
        
        # Filter to valid BOLD jobs
        valid_jobs = [j for j in job_subset if j in self.bold_jobs]
        
        if not valid_jobs:
            raise ValueError("No valid BOLD jobs in subset")
        
        instance = SchedulingInstance(valid_jobs)
        
        # Add job metadata
        for _, row in self.df_tam[self.df_tam['TITLE'].isin(valid_jobs)].iterrows():
            job = row['TITLE']
            instance.job_groups[job] = row.get('GRUP', 'UNKNOWN')
            instance.job_quantities[job] = row.get('QTY', 1)
        
        # Add operations for these jobs only
        df_subset_ops = self.df_sure[self.df_sure['TITLE'].isin(valid_jobs)].copy()
        
        for _, row in df_subset_ops.iterrows():
            job = row['TITLE']
            op_name = row['İŞLEM_ADI']
            sira_no = row['SIRA_NO']
            duration = row['SÜRE (dk)']
            
            if pd.isna(duration) or duration <= 0:
                continue
            
            machine_class = self._map_operation_to_machine(op_name)
            instance.add_operation(job, op_name, sira_no, duration, machine_class)
        
        # Build precedence
        for job in instance.jobs:
            job_ops = instance.get_job_operations(job)
            for i in range(len(job_ops) - 1):
                op_a_name = job_ops[i][0]
                op_b_name = job_ops[i + 1][0]
                instance.add_precedence(job, op_a_name, op_b_name)
        
        return instance
    
    def generate_random_instances(self, 
                                   n_instances: int,
                                   min_jobs: int = config.MIN_JOBS_PER_INSTANCE,
                                   max_jobs: int = config.MAX_JOBS_PER_INSTANCE,
                                   seed: int = config.RANDOM_SEED) -> List[SchedulingInstance]:
        """
        Generate multiple random subset instances for training/validation.
        
        Args:
            n_instances: Number of instances to generate
            min_jobs: Minimum number of jobs per instance
            max_jobs: Maximum number of jobs per instance
            seed: Random seed
            
        Returns:
            List of SchedulingInstance objects
        """
        if self.bold_jobs is None:
            self.load_data()
        
        np.random.seed(seed)
        instances = []
        
        for i in range(n_instances):
            n_jobs = np.random.randint(min_jobs, max_jobs + 1)
            job_subset = np.random.choice(self.bold_jobs, size=n_jobs, replace=False).tolist()
            
            try:
                instance = self.create_subset_instance(job_subset)
                instances.append(instance)
            except ValueError as e:
                print(f"Warning: Failed to create instance {i}: {e}")
                continue
        
        print(f"Generated {len(instances)} random instances")
        return instances


def compute_instance_statistics(instance: SchedulingInstance) -> Dict:
    """
    Compute statistics for a scheduling instance.
    
    Args:
        instance: SchedulingInstance object
        
    Returns:
        Dictionary of statistics
    """
    total_duration = sum(data['duration'] for data in instance.operations.values())
    
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
        'avg_ops_per_job': np.mean(list(ops_per_job.values())),
        'max_ops_per_job': max(ops_per_job.values()) if ops_per_job else 0,
        'ops_per_machine': ops_per_machine
    }


if __name__ == "__main__":
    # Example usage
    print("=== Data Loading Module Test ===\n")
    
    loader = DataLoader()
    loader.load_data()
    
    # Create full instance
    print("\n--- Full Instance ---")
    full_instance = loader.create_full_instance()
    stats = compute_instance_statistics(full_instance)
    print(f"Statistics: {stats}")
    
    # Generate random training instances
    print("\n--- Random Training Instances ---")
    train_instances = loader.generate_random_instances(n_instances=5, min_jobs=3, max_jobs=5)
    
    for i, inst in enumerate(train_instances[:3]):
        print(f"\nInstance {i}: {inst}")
        print(f"  Jobs: {inst.jobs[:3]}...")
        print(f"  Sample operations: {list(inst.operations.keys())[:3]}...")
