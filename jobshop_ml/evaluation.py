"""
Evaluation Module
Compares ML policy with MIP oracle and heuristic baselines.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from tabulate import tabulate

from data_loader import SchedulingInstance, DataLoader
from mip_oracle import MIPOracle, ScheduleSolution
from gnn_model import SchedulingGNN
from graph_builder import GraphBuilder
from rl_env import JobShopEnv
import config


class HeuristicScheduler:
    """
    Simple dispatch rule heuristics for comparison.
    """
    
    @staticmethod
    def schedule_spt(instance: SchedulingInstance) -> Dict:
        """Shortest Processing Time first."""
        return HeuristicScheduler._dispatch_rule(
            instance, 
            key_fn=lambda job, op, data: data['duration']
        )
    
    @staticmethod
    def schedule_lpt(instance: SchedulingInstance) -> Dict:
        """Longest Processing Time first."""
        return HeuristicScheduler._dispatch_rule(
            instance,
            key_fn=lambda job, op, data: -data['duration']
        )
    
    @staticmethod
    def schedule_fifo(instance: SchedulingInstance) -> Dict:
        """First In First Out (arbitrary order)."""
        return HeuristicScheduler._dispatch_rule(
            instance,
            key_fn=lambda job, op, data: 0  # All equal, order doesn't matter
        )
    
    @staticmethod
    def _dispatch_rule(instance: SchedulingInstance, key_fn) -> Dict:
        """
        Generic dispatch rule scheduler.
        
        Args:
            instance: SchedulingInstance
            key_fn: Function to compute priority (lower = higher priority)
            
        Returns:
            Dictionary with schedule info
        """
        env = JobShopEnv(instance)
        env.reset()
        
        while not env.done:
            available = env.get_available_operations()
            
            if not available:
                break
            
            # Compute priorities
            priorities = []
            for job, op_name in available:
                op_data = instance.operations[(job, op_name)]
                priority = key_fn(job, op_name, op_data)
                priorities.append((priority, (job, op_name)))
            
            # Select operation with highest priority (lowest key value)
            priorities.sort(key=lambda x: x[0])
            action = priorities[0][1]
            
            env.step(action)
        
        # Extract metrics
        makespan = max(env.completion_times.values()) if env.completion_times else 0
        
        tardiness_per_job = {}
        for job in instance.jobs:
            job_completion = max(
                [env.completion_times.get((job, op), 0) 
                 for op, _ in instance.get_job_operations(job)],
                default=0
            )
            due_date = instance.due_dates.get(job, 0)
            tardiness_per_job[job] = max(0, job_completion - due_date)
        
        weights = config.WEIGHTS
        # Use 3-component objective: MS + W + G
        objective = (
            weights['w_MS'] * makespan +
            weights['w_W'] * sum(env.waiting_times) +
            weights['w_G'] * sum(env.idle_times)
        )
        
        return {
            'makespan': makespan,
            'total_waiting': sum(env.waiting_times),
            'total_idle': sum(env.idle_times),
            'objective': objective,
            'schedule': env.scheduled_ops
        }


class MLScheduler:
    """
    Uses trained GNN model to schedule.
    """
    
    def __init__(self, 
                 model: SchedulingGNN,
                 graph_builder: GraphBuilder,
                 device: str = 'cpu'):
        """
        Initialize ML scheduler.
        
        Args:
            model: Trained SchedulingGNN
            graph_builder: GraphBuilder instance
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.graph_builder = graph_builder
        self.device = device
    
    def schedule(self, instance: SchedulingInstance) -> Dict:
        """
        Schedule an instance using the trained GNN policy.
        
        Args:
            instance: SchedulingInstance to schedule
            
        Returns:
            Dictionary with schedule info
        """
        env = JobShopEnv(instance, graph_builder=self.graph_builder)
        state = env.reset()
        
        with torch.no_grad():
            while not env.done:
                available = env.get_available_operations()
                
                if not available:
                    break
                
                # Convert state to batch format
                batch = self._state_to_batch(state)
                
                # Get available mask
                available_mask = torch.zeros(state.num_nodes, dtype=torch.bool)
                for job, op_name in available:
                    if (job, op_name) in state.op_to_node:
                        node_idx = state.op_to_node[(job, op_name)]
                        available_mask[node_idx] = True
                
                # Predict action
                outputs = self.model(batch)
                scores = outputs['scores']
                
                # Apply mask and select
                scores = scores.masked_fill(~available_mask, float('-inf'))
                selected_node = torch.argmax(scores).item()
                
                # Convert to action
                action = state.node_to_op[selected_node]
                
                # Take step
                state, _, _, _ = env.step(action)
        
        # Extract metrics (same as heuristic)
        makespan = max(env.completion_times.values()) if env.completion_times else 0
        
        weights = config.WEIGHTS
        # Use 3-component objective: MS + W + G
        objective = (
            weights['w_MS'] * makespan +
            weights['w_W'] * sum(env.waiting_times) +
            weights['w_G'] * sum(env.idle_times)
        )
        
        return {
            'makespan': makespan,
            'total_waiting': sum(env.waiting_times),
            'total_idle': sum(env.idle_times),
            'objective': objective,
            'schedule': env.scheduled_ops
        }
    
    def _state_to_batch(self, state):
        """Convert single state to batch format."""
        batch = {
            'node_features': state.node_features.to(self.device),
            'edge_index': state.edge_index.to(self.device),
            'edge_features': state.edge_features.to(self.device),
            'global_features': state.global_features.unsqueeze(0).to(self.device),
            'batch': torch.zeros(state.num_nodes, dtype=torch.long, device=self.device),
            'batch_size': 1
        }
        return batch


class Evaluator:
    """
    Comprehensive evaluation comparing different scheduling methods.
    """
    
    def __init__(self,
                 ml_model: Optional[SchedulingGNN] = None,
                 device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            ml_model: Trained ML model (optional)
            device: Device for ML inference
        """
        self.ml_model = ml_model
        self.device = device
        self.graph_builder = GraphBuilder()
        
        if ml_model is not None:
            self.ml_scheduler = MLScheduler(ml_model, self.graph_builder, device)
        else:
            self.ml_scheduler = None
        
        self.mip_oracle = MIPOracle(verbose=False)
    
    def evaluate_instance(self, 
                         instance: SchedulingInstance,
                         methods: List[str] = None) -> Dict[str, Dict]:
        """
        Evaluate a single instance with multiple methods.
        
        Args:
            instance: SchedulingInstance to evaluate
            methods: List of methods to use ['mip', 'ml', 'spt', 'lpt', 'fifo']
            
        Returns:
            Dictionary mapping method name to results
        """
        if methods is None:
            methods = ['mip', 'ml', 'spt', 'lpt', 'fifo']
        
        results = {}
        
        # MIP Oracle
        if 'mip' in methods:
            print("  Solving with MIP...")
            start_time = time.time()
            mip_solution = self.mip_oracle.solve(instance)
            mip_time = time.time() - start_time
            
            if mip_solution:
                results['mip'] = {
                    'makespan': mip_solution.makespan,
                    'total_waiting': mip_solution.objective_components['total_waiting'],
                    'total_idle': mip_solution.objective_components['total_idle'],
                    'objective': mip_solution.objective_value,
                    'solve_time': mip_time,
                    'optimal': mip_solution.is_optimal
                }
            else:
                results['mip'] = {'error': 'Failed to solve'}
        
        # ML Policy
        if 'ml' in methods and self.ml_scheduler is not None:
            print("  Solving with ML...")
            start_time = time.time()
            ml_result = self.ml_scheduler.schedule(instance)
            ml_time = time.time() - start_time
            
            ml_result['solve_time'] = ml_time
            results['ml'] = ml_result
        
        # Heuristics
        if 'spt' in methods:
            print("  Solving with SPT...")
            start_time = time.time()
            spt_result = HeuristicScheduler.schedule_spt(instance)
            spt_result['solve_time'] = time.time() - start_time
            results['spt'] = spt_result
        
        if 'lpt' in methods:
            print("  Solving with LPT...")
            start_time = time.time()
            lpt_result = HeuristicScheduler.schedule_lpt(instance)
            lpt_result['solve_time'] = time.time() - start_time
            results['lpt'] = lpt_result
        
        if 'fifo' in methods:
            print("  Solving with FIFO...")
            start_time = time.time()
            fifo_result = HeuristicScheduler.schedule_fifo(instance)
            fifo_result['solve_time'] = time.time() - start_time
            results['fifo'] = fifo_result
        
        return results
    
    def evaluate_dataset(self,
                        instances: List[SchedulingInstance],
                        methods: List[str] = None) -> Dict:
        """
        Evaluate multiple instances.
        
        Args:
            instances: List of instances to evaluate
            methods: Methods to compare
            
        Returns:
            Aggregated results
        """
        all_results = []
        
        for i, instance in enumerate(instances):
            print(f"\nEvaluating instance {i+1}/{len(instances)}")
            results = self.evaluate_instance(instance, methods)
            all_results.append(results)
        
        # Aggregate results
        return self._aggregate_results(all_results, methods)
    
    def _aggregate_results(self, 
                          all_results: List[Dict[str, Dict]],
                          methods: List[str]) -> Dict:
        """Aggregate results across instances."""
        aggregated = {}
        
        for method in methods:
            if method not in aggregated:
                aggregated[method] = {
                    'makespans': [],
                    'objectives': [],
                    'solve_times': []
                }
            
            for result_dict in all_results:
                if method in result_dict and 'error' not in result_dict[method]:
                    aggregated[method]['makespans'].append(
                        result_dict[method]['makespan']
                    )
                    aggregated[method]['objectives'].append(
                        result_dict[method]['objective']
                    )
                    aggregated[method]['solve_times'].append(
                        result_dict[method]['solve_time']
                    )
        
        # Compute statistics
        summary = {}
        for method, data in aggregated.items():
            if data['objectives']:
                summary[method] = {
                    'avg_makespan': np.mean(data['makespans']),
                    'std_makespan': np.std(data['makespans']),
                    'avg_objective': np.mean(data['objectives']),
                    'std_objective': np.std(data['objectives']),
                    'avg_time': np.mean(data['solve_times']),
                    'n_solved': len(data['objectives'])
                }
        
        return summary
    
    def print_comparison_table(self, summary: Dict):
        """Print a nice comparison table."""
        headers = ['Method', 'Avg Makespan', 'Avg Objective', 'Avg Time (s)', 'N Solved']
        rows = []
        
        for method, stats in summary.items():
            rows.append([
                method.upper(),
                f"{stats['avg_makespan']:.2f} ± {stats['std_makespan']:.2f}",
                f"{stats['avg_objective']:.2f} ± {stats['std_objective']:.2f}",
                f"{stats['avg_time']:.3f}",
                stats['n_solved']
            ])
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # Optimality gap (if MIP available)
        if 'mip' in summary and 'ml' in summary:
            mip_obj = summary['mip']['avg_objective']
            ml_obj = summary['ml']['avg_objective']
            gap = ((ml_obj - mip_obj) / mip_obj) * 100
            print(f"\nML vs MIP gap: {gap:.2f}%")


if __name__ == "__main__":
    print("=== Evaluation Module Test ===\n")
    
    from data_loader import DataLoader
    
    # Load data and create test instances
    loader = DataLoader()
    loader.load_data()
    
    print("Generating test instances...")
    test_instances = loader.generate_random_instances(
        n_instances=2,
        min_jobs=3,
        max_jobs=4,
        seed=999
    )
    
    # Create evaluator (without ML model for now)
    evaluator = Evaluator(ml_model=None)
    
    # Evaluate
    print("\nEvaluating instances...")
    summary = evaluator.evaluate_dataset(
        test_instances,
        methods=['mip', 'spt', 'lpt']  # Skip ML for now
    )
    
    evaluator.print_comparison_table(summary)
