"""
MIP Oracle Module
Solves scheduling instances using Gurobi to generate optimal/near-optimal schedules.
These solutions serve as training data for the ML model.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_loader import SchedulingInstance
import config


class ScheduleSolution:
    """
    Represents a solution to a scheduling problem.
    
    Attributes:
        instance: The SchedulingInstance that was solved
        start_times: Dict mapping (job, op_name) -> start time
        completion_times: Dict mapping (job, op_name) -> completion time
        machine_assignments: Dict mapping (job, op_name) -> machine_class (redundant but useful)
        makespan: Total makespan
        tardiness: Dict mapping job -> tardiness value
        waiting_times: Dict for precedence waiting times
        idle_times: Dict for machine idle gaps
        objective_value: Final weighted objective value
        objective_components: Breakdown of objective components
        is_optimal: Whether solution is proven optimal
        solve_time: Time taken to solve (seconds)
    """
    
    def __init__(self, instance: SchedulingInstance):
        self.instance = instance
        self.start_times = {}
        self.completion_times = {}
        self.machine_assignments = {}
        self.makespan = 0
        self.tardiness = {}
        self.waiting_times = {}
        self.idle_times = {}
        self.objective_value = 0
        self.objective_components = {}
        self.is_optimal = False
        self.solve_time = 0
    
    def get_schedule_sequence(self) -> List[Tuple[float, str, str, str]]:
        """
        Get operations in chronological order.
        
        Returns:
            List of (start_time, job, op_name, machine_class) sorted by start time
        """
        sequence = []
        for (job, op_name), start_time in self.start_times.items():
            machine_class = self.machine_assignments.get((job, op_name), 'UNKNOWN')
            sequence.append((start_time, job, op_name, machine_class))
        
        sequence.sort(key=lambda x: x[0])
        return sequence
    
    def __repr__(self):
        return (f"ScheduleSolution(makespan={self.makespan:.2f}, "
                f"objective={self.objective_value:.2f}, "
                f"optimal={self.is_optimal})")


class MIPOracle:
    """
    Solves scheduling instances using Gurobi MIP formulation.
    Based on your existing MILP model.
    """
    
    def __init__(self, 
                 weights: Dict[str, float] = None,
                 time_limit: int = config.MIP_TIME_LIMIT,
                 gap_tolerance: float = config.MIP_GAP_TOLERANCE,
                 verbose: bool = False):
        """
        Initialize MIP Oracle.
        
        Args:
            weights: Dictionary of objective weights (w_MS, w_T, w_row, w_asm, w_gap)
            time_limit: Gurobi time limit in seconds
            gap_tolerance: MIP gap tolerance
            verbose: Whether to print Gurobi output
        """
        self.weights = weights if weights else config.WEIGHTS
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose
        self.big_m = config.BIG_M
    
    def solve(self, instance: SchedulingInstance) -> Optional[ScheduleSolution]:
        """
        Solve a scheduling instance with Gurobi.
        
        Args:
            instance: SchedulingInstance to solve
            
        Returns:
            ScheduleSolution if successful, None if infeasible/failed
        """
        try:
            model = gp.Model("JobShopScheduling")
            
            if not self.verbose:
                model.setParam('OutputFlag', 0)
            
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.gap_tolerance)
            
            # Extract weights (3 components)
            w_MS = self.weights.get('w_MS', 0.3)
            w_W = self.weights.get('w_W', 0.5)
            w_G = self.weights.get('w_G', 0.2)
            
            # ================================================================
            # DECISION VARIABLES
            # ================================================================
            
            # Start and completion times
            S = {}  # S[job, op_name] = start time
            C = {}  # C[job, op_name] = completion time
            
            for (job, op_name), data in instance.operations.items():
                S[job, op_name] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                                name=f"S_{job}_{op_name}")
                C[job, op_name] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                name=f"C_{job}_{op_name}")
            
            # Makespan
            MS = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Makespan")
            
            # Disjunctive variables for operations on same machine
            Z = {}  # Z[i1,op1,i2,op2] = 1 if op1 before op2
            pairs_same_machine = self._get_operation_pairs_same_machine(instance)
            
            for (j1, op1, j2, op2) in pairs_same_machine:
                Z[j1, op1, j2, op2] = model.addVar(vtype=GRB.BINARY,
                                                     name=f"Z_{j1}_{op1}_{j2}_{op2}")
            
            # Waiting time variables (for objective)
            W_row = {}  # Waiting due to precedence (row waiting)
            for (job, op_a, op_b) in instance.precedence:
                W_row[job, op_a, op_b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                       name=f"Wrow_{job}_{op_a}_{op_b}")
            
            # Idle gap variables (for objective)
            Gap = {}
            for (j1, op1, j2, op2) in pairs_same_machine:
                Gap[j1, op1, j2, op2] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                      name=f"Gap_{j1}_{op1}_{j2}_{op2}")
            
            model.update()
            
            # ================================================================
            # CONSTRAINTS
            # ================================================================
            
            # 1. Processing time constraints: C = S + duration
            for (job, op_name), data in instance.operations.items():
                duration = data['duration']
                model.addConstr(C[job, op_name] >= S[job, op_name] + duration,
                                name=f"proc_{job}_{op_name}")
            
            # 2. Precedence constraints
            for (job, op_a, op_b) in instance.precedence:
                if (job, op_a) in C and (job, op_b) in S:
                    model.addConstr(S[job, op_b] >= C[job, op_a],
                                    name=f"prec_{job}_{op_a}_{op_b}")
            
            # 3. Machine disjunction constraints
            for (j1, op1, j2, op2) in pairs_same_machine:
                model.addConstr(S[j2, op2] >= C[j1, op1] - self.big_m * (1 - Z[j1, op1, j2, op2]),
                                name=f"disj1_{j1}_{op1}_{j2}_{op2}")
                model.addConstr(S[j1, op1] >= C[j2, op2] - self.big_m * Z[j1, op1, j2, op2],
                                name=f"disj2_{j1}_{op1}_{j2}_{op2}")
            
            # 4. Makespan definition
            for (job, op_name) in instance.operations.keys():
                model.addConstr(MS >= C[job, op_name], name=f"ms_{job}_{op_name}")
            
            # 5. Waiting time definition (precedence)
            for (job, op_a, op_b) in instance.precedence:
                if (job, op_a) in C and (job, op_b) in S:
                    model.addConstr(W_row[job, op_a, op_b] >= S[job, op_b] - C[job, op_a],
                                    name=f"wait_{job}_{op_a}_{op_b}")
            
            # 6. Idle gap definition
            for (j1, op1, j2, op2) in pairs_same_machine:
                model.addConstr(Gap[j1, op1, j2, op2] >= 
                                S[j2, op2] - C[j1, op1] - self.big_m * (1 - Z[j1, op1, j2, op2]),
                                name=f"gap_{j1}_{op1}_{j2}_{op2}")
            
            # ================================================================
            # OBJECTIVE (3 components)
            # ================================================================
            
            total_waiting = gp.quicksum(W_row[job, op_a, op_b] 
                                        for (job, op_a, op_b) in instance.precedence
                                        if (job, op_a) in C)
            total_idle = gp.quicksum(Gap[j1, op1, j2, op2] 
                                     for (j1, op1, j2, op2) in pairs_same_machine)
            
            # Final objective: Minimize weighted sum of MS + Wrow + Gap
            objective = (w_MS * MS + 
                        w_W * total_waiting + 
                        w_G * total_idle)
            
            model.setObjective(objective, GRB.MINIMIZE)
            
            # ================================================================
            # SOLVE
            # ================================================================
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                return self._extract_solution(model, instance, S, C, MS, W_row, Gap)
            else:
                print(f"MIP solve failed with status {model.status}")
                return None
                
        except Exception as e:
            print(f"Error in MIP solve: {e}")
            return None
    
    def _get_operation_pairs_same_machine(self, instance: SchedulingInstance) -> List[Tuple]:
        """
        Get all pairs of operations that use the same machine class.
        Returns list of (job1, op1, job2, op2) tuples.
        """
        pairs = []
        
        # Group operations by machine class
        ops_by_machine = {}
        for (job, op_name), data in instance.operations.items():
            mc = data['machine_class']
            if mc not in ops_by_machine:
                ops_by_machine[mc] = []
            ops_by_machine[mc].append((job, op_name))
        
        # Create pairs for each machine class
        for mc, ops in ops_by_machine.items():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    j1, op1 = ops[i]
                    j2, op2 = ops[j]
                    pairs.append((j1, op1, j2, op2))
        
        return pairs
    
    def _extract_solution(self, model, instance, S, C, MS, W_row, Gap) -> ScheduleSolution:
        """Extract solution from solved Gurobi model."""
        solution = ScheduleSolution(instance)
        
        # Extract start and completion times
        for (job, op_name) in instance.operations.keys():
            solution.start_times[job, op_name] = S[job, op_name].X
            solution.completion_times[job, op_name] = C[job, op_name].X
            solution.machine_assignments[job, op_name] = instance.operations[job, op_name]['machine_class']
        
        # Extract objective components (3 components)
        solution.makespan = MS.X
        
        total_waiting = sum(W_row[job, op_a, op_b].X 
                           for (job, op_a, op_b) in instance.precedence
                           if (job, op_a, op_b) in W_row)
        
        total_idle = sum(Gap[j1, op1, j2, op2].X 
                        for key in Gap.keys() 
                        if (j1 := key[0], op1 := key[1], j2 := key[2], op2 := key[3]))
        
        solution.objective_components = {
            'makespan': solution.makespan,
            'total_waiting': total_waiting,
            'total_idle': total_idle
        }
        
        solution.objective_value = model.ObjVal
        solution.is_optimal = (model.status == GRB.OPTIMAL)
        solution.solve_time = model.Runtime
        
        return solution


if __name__ == "__main__":
    # Example usage
    print("=== MIP Oracle Module Test ===\n")
    
    from data_loader import DataLoader
    
    # Load data and create a small instance
    loader = DataLoader()
    loader.load_data()
    
    print("Creating small test instance...")
    test_instances = loader.generate_random_instances(n_instances=1, min_jobs=3, max_jobs=4)
    
    if test_instances:
        instance = test_instances[0]
        print(f"Test instance: {instance}")
        
        # Solve with MIP
        print("\nSolving with Gurobi...")
        oracle = MIPOracle(verbose=True, time_limit=60)
        solution = oracle.solve(instance)
        
        if solution:
            print(f"\nSolution: {solution}")
            print(f"Objective components: {solution.objective_components}")
            print(f"\nFirst 5 operations in schedule:")
            for i, (start, job, op, mc) in enumerate(solution.get_schedule_sequence()[:5]):
                print(f"  {i+1}. t={start:.1f}: {job} - {op} on {mc}")
        else:
            print("Failed to solve instance")
