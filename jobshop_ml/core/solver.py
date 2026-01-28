"""
Solver module for running Gurobi optimization.

This module handles the execution of Gurobi solvers on MIP models,
including parameter configuration, solving, and solution extraction.
"""

import gurobipy as gp
from gurobipy import GRB
from typing import Optional, Dict
from .instance import SchedulingInstance
from .solution import ScheduleSolution
from .model_builder import ModelBuilder
import config


class Solver:
    """
    Solves Gurobi MIP models with optimized parameters.
    
    This class handles the solving process, including solver parameter
    configuration, execution, and solution extraction. It works with
    models built by ModelBuilder.
    
    Attributes:
    -----------
    time_limit : int
        Maximum solving time in seconds.
    gap_tolerance : float
        MIP gap tolerance (relative gap).
    verbose : bool
        Whether to print Gurobi output.
    """
    
    def __init__(self,
                 time_limit: int = config.MIP_TIME_LIMIT,
                 gap_tolerance: float = config.MIP_GAP_TOLERANCE,
                 verbose: bool = False):
        """
        Initialize solver with parameters.
        
        Parameters:
        -----------
        time_limit : int, optional
            Maximum solving time in seconds. Defaults to config value.
        gap_tolerance : float, optional
            MIP gap tolerance (0.05 = 5%). Defaults to config value.
        verbose : bool, optional
            Whether to print Gurobi output. Default is False.
        """
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose
    
    def configure_solver(self, model: gp.Model) -> None:
        """
        Configure Gurobi solver parameters for optimal performance.
        
        This method sets various Gurobi parameters to improve solving speed
        and solution quality. Parameters are chosen based on best practices
        for scheduling problems.
        
        Parameters:
        -----------
        model : gp.Model
            The Gurobi model to configure.
        
        Notes:
        ------
        Configured parameters include:
        - TimeLimit: Maximum solving time
        - MIPGap: Relative optimality gap tolerance
        - OutputFlag: Whether to print solver output
        - Presolve: Aggressive presolving
        - MIPFocus: Balance between finding feasible solutions and proving optimality
        - Heuristics: Enable aggressive heuristics
        - Cuts: Enable cutting planes
        """
        if not self.verbose:
            model.setParam('OutputFlag', 0)
        
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.gap_tolerance)
        
        # Performance tuning parameters
        model.setParam('Presolve', 2)  # Aggressive presolve
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
        model.setParam('Heuristics', 0.1)  # Spend 10% of time on heuristics
        model.setParam('Cuts', 2)  # Aggressive cut generation
        model.setParam('Threads', 0)  # Use all available threads
    
    def solve(self, model: gp.Model, instance: SchedulingInstance) -> Optional[ScheduleSolution]:
        """
        Solve a Gurobi model and extract the solution.
        
        This method configures the solver, runs optimization, and extracts
        the solution if one is found. The solution is returned as a
        ScheduleSolution object.
        
        Parameters:
        -----------
        model : gp.Model
            The Gurobi model to solve (built by ModelBuilder).
        instance : SchedulingInstance
            The scheduling instance corresponding to the model.
        
        Returns:
        --------
        Optional[ScheduleSolution]
            The solution if found, None if solving failed or was infeasible.
        
        Notes:
        ------
        The method handles various solver statuses:
        - OPTIMAL: Proven optimal solution
        - TIME_LIMIT: Solution found within time limit (may not be optimal)
        - Other statuses: Returns None
        """
        try:
            # Configure solver
            self.configure_solver(model)
            
            # Add callback for progress updates
            if not self.verbose:
                model.setParam('OutputFlag', 1)  # Enable output for progress
            
            print("\n🔧 Model çözülüyor...")
            print(f"   Zaman limiti: {self.time_limit} saniye")
            print(f"   MIP gap toleransı: {self.gap_tolerance*100:.1f}%")
            print(f"   İşlem sayısı: {len(instance.operations)}")
            print(f"   İş sayısı: {len(instance.jobs)}")
            print("\n⏳ İlerleme:")
            
            # Track progress using a class to avoid nonlocal issues
            class ProgressTracker:
                def __init__(self):
                    self.last_obj = None
                    self.last_time = 0
                
                def callback(self, model, where):
                    """Callback function to show progress"""
                    if where == GRB.Callback.MIP:
                        # Get current best solution
                        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                        runtime = model.cbGet(GRB.Callback.RUNTIME)
                        
                        # Print progress every 5 seconds or when objective improves
                        if runtime - self.last_time >= 5 or (self.last_obj is None or objbst < self.last_obj):
                            gap = float('inf')
                            if abs(objbnd) > 1e-6:
                                gap = abs(objbst - objbnd) / abs(objbnd) * 100
                            
                            if objbst < GRB.INFINITY:
                                print(f"   ⏱️  {runtime:.1f}s | En iyi: {objbst:.2f} | Gap: {gap:.2f}%", end='\r')
                            else:
                                print(f"   ⏱️  {runtime:.1f}s | Çözüm aranıyor...", end='\r')
                            
                            self.last_obj = objbst
                            self.last_time = runtime
            
            tracker = ProgressTracker()
            
            # Solve with progress callback
            try:
                model.optimize(tracker.callback)
            except:
                # If callback fails, solve normally
                model.optimize()
            
            print()  # New line after progress
            
            # Check status and extract solution
            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                solution = self._extract_solution(model, instance)
                if solution:
                    print(f"✅ Çözüm bulundu!")
                    print(f"   Objective: {solution.objective_value:.2f}")
                    print(f"   Makespan: {solution.makespan:.2f} dakika")
                    print(f"   Çözüm süresi: {solution.solve_time:.2f} saniye")
                    print(f"   Durum: {'OPTIMAL' if solution.is_optimal else 'FEASIBLE'}")
                return solution
            else:
                status_name = model.Status
                print(f"❌ Çözüm bulunamadı. Durum: {status_name}")
                return None
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_solution(self, model: gp.Model, instance: SchedulingInstance) -> ScheduleSolution:
        """
        Extract solution from a solved Gurobi model.
        
        This method reads variable values from the solved model and constructs
        a ScheduleSolution object. It extracts start times, completion times,
        objective components, and other solution information.
        
        Parameters:
        -----------
        model : gp.Model
            The solved Gurobi model.
        instance : SchedulingInstance
            The scheduling instance.
        
        Returns:
        --------
        ScheduleSolution
            A complete solution object with all extracted information.
        
        Notes:
        ------
        The method accesses variables stored in the model by ModelBuilder:
        - _vars_S: Start time variables
        - _vars_C: Completion time variables
        - _vars_MS: Makespan variable
        - _vars_W_row: Waiting time variables
        - _vars_Gap: Idle gap variables
        """
        solution = ScheduleSolution(instance)
        
        # Get variable dictionaries from model
        S = model._vars_S
        C = model._vars_C
        MS = model._vars_MS
        W_row = model._vars_W_row
        Gap = model._vars_Gap
        conflicting_pairs = model._conflicting_pairs
        
        # Extract start and completion times
        for (job, op_name) in instance.operations.keys():
            if (job, op_name) in S and (job, op_name) in C:
                solution.start_times[(job, op_name)] = S[(job, op_name)].X
                solution.completion_times[(job, op_name)] = C[(job, op_name)].X
                solution.machine_assignments[(job, op_name)] = (
                    instance.operations[(job, op_name)]['machine_class']
                )
        
        # Extract makespan
        solution.makespan = MS.X
        
        # Extract waiting times
        total_waiting = 0.0
        for (job, op_a, op_b) in instance.precedence:
            if (job, op_a, op_b) in W_row:
                waiting = W_row[(job, op_a, op_b)].X
                solution.waiting_times[(job, op_a, op_b)] = waiting
                total_waiting += waiting
        
        # Extract idle gaps
        total_idle = 0.0
        for (j1, op1, j2, op2) in conflicting_pairs:
            if (j1, op1, j2, op2) in Gap:
                gap = Gap[(j1, op1, j2, op2)].X
                solution.idle_times[(j1, op1, j2, op2)] = gap
                total_idle += gap
        
        # Store objective components
        solution.objective_components = {
            'makespan': solution.makespan,
            'total_waiting': total_waiting,
            'total_idle': total_idle
        }
        
        # Extract objective value and status
        solution.objective_value = model.ObjVal
        solution.is_optimal = (model.status == GRB.OPTIMAL)
        solution.solve_time = model.Runtime
        solution.model_status = model.status
        
        return solution

