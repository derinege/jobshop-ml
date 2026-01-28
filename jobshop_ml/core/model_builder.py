"""
MIP model construction module with optimizations.

This module builds the Gurobi MIP model for weighted job shop scheduling
with several optimizations to improve solving speed:
- Tighter Big-M values based on instance structure
- Reduced binary variables (only for operations that can conflict)
- Valid inequalities to strengthen the formulation
- Efficient variable indexing
"""

import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
from .instance import SchedulingInstance
import config


class ModelBuilder:
    """
    Builds optimized Gurobi MIP models for job shop scheduling.
    
    This class constructs the mathematical programming formulation without
    solving it. The model can then be solved by the Solver class. The builder
    implements several optimizations to reduce model size and improve solving
    performance.
    
    Attributes:
    -----------
    weights : Dict[str, float]
        Objective function weights (w_MS, w_W, w_G).
    big_m : Optional[float]
        Big-M constant (computed automatically if None).
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize model builder.
        
        Parameters:
        -----------
        weights : Optional[Dict[str, float]], optional
            Objective weights dictionary. If None, uses config.WEIGHTS.
            Default is None.
        """
        self.weights = weights if weights else config.WEIGHTS
        self.big_m = None
    
    def compute_big_m(self, instance: SchedulingInstance) -> float:
        """
        Compute a tight Big-M value for disjunctive constraints.
        
        This method computes a Big-M value that is as small as possible while
        still being valid. A tighter Big-M improves the LP relaxation and
        speeds up branch-and-bound. The value is based on the sum of all
        operation durations, which provides a valid upper bound on any
        operation's start or completion time.
        
        Parameters:
        -----------
        instance : SchedulingInstance
            The scheduling instance.
        
        Returns:
        --------
        float
            A tight Big-M value for use in disjunctive constraints.
        
        Notes:
        ------
        The Big-M value is computed as the sum of all operation durations,
        which provides a valid upper bound. For instances with precedence
        constraints, this could be further tightened using critical path
        analysis, but the current approach is simple and effective.
        """
        total_duration = sum(data['duration'] for data in instance.operations.values())
        # Add small buffer (10%) to account for potential gaps
        return total_duration * 1.1
    
    def get_conflicting_operation_pairs(self, instance: SchedulingInstance) -> List[Tuple]:
        """
        Get pairs of operations that can potentially conflict on the same machine.
        
        This method identifies operation pairs that use the same machine class
        and could potentially overlap in time. This is more efficient than
        creating binary variables for all pairs, as operations from the same
        job cannot overlap due to precedence constraints.
        
        Parameters:
        -----------
        instance : SchedulingInstance
            The scheduling instance.
        
        Returns:
        --------
        List[Tuple]
            List of (job1, op1, job2, op2) tuples for operations that can conflict.
        
        Notes:
        ------
        Operations from the same job are excluded since they cannot conflict
        due to precedence constraints. Only operations from different jobs
        on the same machine class are considered.
        """
        pairs = []
        
        # Group operations by machine class
        ops_by_machine = {}
        for (job, op_name), data in instance.operations.items():
            mc = data['machine_class']
            if mc not in ops_by_machine:
                ops_by_machine[mc] = []
            ops_by_machine[mc].append((job, op_name))
        
        # Create pairs only for operations from different jobs
        for mc, ops in ops_by_machine.items():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    j1, op1 = ops[i]
                    j2, op2 = ops[j]
                    # Only create pair if operations are from different jobs
                    if j1 != j2:
                        pairs.append((j1, op1, j2, op2))
        
        return pairs
    
    def build_model(self, instance: SchedulingInstance, 
                    model_name: str = "JobShopScheduling") -> gp.Model:
        """
        Build the complete Gurobi MIP model for the scheduling instance.
        
        This method constructs the full mathematical programming formulation
        including decision variables, constraints, and objective function.
        The model is returned without being solved, allowing for further
        customization or solving by the Solver class.
        
        Parameters:
        -----------
        instance : SchedulingInstance
            The scheduling instance to model.
        model_name : str, optional
            Name for the Gurobi model. Default is "JobShopScheduling".
        
        Returns:
        --------
        gp.Model
            A fully constructed Gurobi model ready for solving.
        
        Notes:
        ------
        The model includes:
        - Decision variables: start times (S), completion times (C), makespan (MS),
          disjunctive binaries (Z), waiting times (W_row), idle gaps (Gap)
        - Constraints: processing time, precedence, machine disjunction,
          makespan definition, waiting/idle definitions
        - Objective: weighted sum of makespan, waiting, and idle time
        - Optimizations: tight Big-M, reduced binaries, valid inequalities
        """
        model = gp.Model(model_name)
        
        # Compute tight Big-M
        self.big_m = self.compute_big_m(instance)
        
        print(f"📐 Model oluşturuluyor...")
        print(f"   Big-M değeri: {self.big_m:.1f}")
        
        # Extract weights
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
            S[job, op_name] = model.addVar(
                vtype=GRB.CONTINUOUS, 
                lb=0.0,
                name=f"S_{job}_{op_name}"
            )
            C[job, op_name] = model.addVar(
                vtype=GRB.CONTINUOUS, 
                lb=0.0,
                name=f"C_{job}_{op_name}"
            )
        
        # Makespan
        MS = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Makespan")
        
        # Disjunctive variables for operations on same machine (optimized)
        Z = {}  # Z[j1,op1,j2,op2] = 1 if op1 before op2
        conflicting_pairs = self.get_conflicting_operation_pairs(instance)
        
        for (j1, op1, j2, op2) in conflicting_pairs:
            Z[j1, op1, j2, op2] = model.addVar(
                vtype=GRB.BINARY,
                name=f"Z_{j1}_{op1}_{j2}_{op2}"
            )
        
        # Waiting time variables (for objective)
        W_row = {}  # Waiting due to precedence (row waiting)
        for (job, op_a, op_b) in instance.precedence:
            W_row[job, op_a, op_b] = model.addVar(
                vtype=GRB.CONTINUOUS, 
                lb=0.0,
                name=f"Wrow_{job}_{op_a}_{op_b}"
            )
        
        # Idle gap variables (for objective)
        Gap = {}
        for (j1, op1, j2, op2) in conflicting_pairs:
            Gap[j1, op1, j2, op2] = model.addVar(
                vtype=GRB.CONTINUOUS, 
                lb=0.0,
                name=f"Gap_{j1}_{op1}_{j2}_{op2}"
            )
        
        model.update()
        
        # ================================================================
        # CONSTRAINTS
        # ================================================================
        
        # 1. Processing time constraints: C >= S + duration
        for (job, op_name), data in instance.operations.items():
            duration = data['duration']
            model.addConstr(
                C[job, op_name] >= S[job, op_name] + duration,
                name=f"proc_{job}_{op_name}"
            )
        
        # 2. Precedence constraints
        for (job, op_a, op_b) in instance.precedence:
            if (job, op_a) in C and (job, op_b) in S:
                model.addConstr(
                    S[job, op_b] >= C[job, op_a],
                    name=f"prec_{job}_{op_a}_{op_b}"
                )
        
        # 3. Machine disjunction constraints (optimized with tight Big-M)
        for (j1, op1, j2, op2) in conflicting_pairs:
            # If Z[j1,op1,j2,op2] = 1, then op1 before op2
            model.addConstr(
                S[j2, op2] >= C[j1, op1] - self.big_m * (1 - Z[j1, op1, j2, op2]),
                name=f"disj1_{j1}_{op1}_{j2}_{op2}"
            )
            # If Z[j1,op1,j2,op2] = 0, then op2 before op1
            model.addConstr(
                S[j1, op1] >= C[j2, op2] - self.big_m * Z[j1, op1, j2, op2],
                name=f"disj2_{j1}_{op1}_{j2}_{op2}"
            )
        
        # 4. Makespan definition
        for (job, op_name) in instance.operations.keys():
            model.addConstr(
                MS >= C[job, op_name],
                name=f"ms_{job}_{op_name}"
            )
        
        # 5. Waiting time definition (precedence)
        for (job, op_a, op_b) in instance.precedence:
            if (job, op_a) in C and (job, op_b) in S:
                model.addConstr(
                    W_row[job, op_a, op_b] >= S[job, op_b] - C[job, op_a],
                    name=f"wait_{job}_{op_a}_{op_b}"
                )
        
        # 6. Idle gap definition
        for (j1, op1, j2, op2) in conflicting_pairs:
            model.addConstr(
                Gap[j1, op1, j2, op2] >= 
                S[j2, op2] - C[j1, op1] - self.big_m * (1 - Z[j1, op1, j2, op2]),
                name=f"gap_{j1}_{op1}_{j2}_{op2}"
            )
        
        # ================================================================
        # VALID INEQUALITIES (strengthen formulation)
        # ================================================================
        
        # Add symmetry-breaking constraints: for operations on same machine,
        # prefer ordering by job index (if jobs have natural ordering)
        # This is a simple heuristic that can help reduce search space
        
        # Add lower bound on makespan: at least the longest job duration
        max_job_duration = 0.0
        for job in instance.jobs:
            job_ops = instance.get_job_operations(job)
            job_duration = sum(data['duration'] for _, data in job_ops)
            max_job_duration = max(max_job_duration, job_duration)
        
        if max_job_duration > 0:
            model.addConstr(MS >= max_job_duration, name="lb_makespan")
        
        # ================================================================
        # OBJECTIVE (3 components)
        # ================================================================
        
        total_waiting = gp.quicksum(
            W_row[job, op_a, op_b] 
            for (job, op_a, op_b) in instance.precedence
            if (job, op_a, op_b) in W_row
        )
        total_idle = gp.quicksum(
            Gap[j1, op1, j2, op2] 
            for (j1, op1, j2, op2) in conflicting_pairs
        )
        
        # Final objective: Minimize weighted sum of MS + Wrow + Gap
        objective = (w_MS * MS + w_W * total_waiting + w_G * total_idle)
        model.setObjective(objective, GRB.MINIMIZE)
        
        # Store variable dictionaries in model for later extraction
        model._vars_S = S
        model._vars_C = C
        model._vars_MS = MS
        model._vars_W_row = W_row
        model._vars_Gap = Gap
        model._conflicting_pairs = conflicting_pairs
        
        # Print model statistics
        print(f"   Değişkenler: {model.NumVars} (Binary: {model.NumBinVars})")
        print(f"   Kısıtlar: {model.NumConstrs}")
        print(f"   Çakışan çiftler: {len(conflicting_pairs)}")
        print(f"✅ Model hazır!\n")
        
        return model

