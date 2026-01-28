"""
Demo Script
Quick demonstration of the ML scheduling system with synthetic data.
Run this if you don't have the Excel files yet.
"""

import torch
import numpy as np
from typing import Dict, List

from data_loader import SchedulingInstance
from mip_oracle import MIPOracle
from graph_builder import GraphBuilder, solution_to_training_samples
from gnn_model import SchedulingGNN
from rl_env import JobShopEnv
from evaluation import HeuristicScheduler, MLScheduler
import config


def create_synthetic_instance(n_jobs: int = 5, seed: int = 42) -> SchedulingInstance:
    """
    Create a synthetic scheduling instance for testing.
    
    Args:
        n_jobs: Number of jobs
        seed: Random seed
        
    Returns:
        SchedulingInstance
    """
    np.random.seed(seed)
    
    jobs = [f"JOB_{i+1}" for i in range(n_jobs)]
    instance = SchedulingInstance(jobs)
    
    # Add job metadata
    for job in jobs:
        instance.job_groups[job] = "GROUP_A"
        instance.job_quantities[job] = np.random.randint(1, 10)
        instance.due_dates[job] = 0  # All due dates at 0 for simplicity
    
    # Define operation types
    operation_types = [
        ("KAYNAK", "WLD", 20, 40),
        ("ÖN HAZIRLAMA", "PRE", 10, 30),
        ("KUMLAMA", "FIN", 15, 35),
        ("MONTAJ", "ASM", 25, 50),
        ("PAKETLEME", "PKG", 10, 20)
    ]
    
    # Add operations for each job
    for job in jobs:
        n_ops = np.random.randint(3, 6)  # 3-5 operations per job
        selected_ops = np.random.choice(len(operation_types), size=n_ops, replace=False)
        
        for sira_no, op_idx in enumerate(sorted(selected_ops), start=1):
            op_name, machine_class, min_dur, max_dur = operation_types[op_idx]
            duration = np.random.randint(min_dur, max_dur)
            
            instance.add_operation(job, op_name, sira_no, duration, machine_class)
        
        # Add precedence constraints (sequential operations)
        job_ops = instance.get_job_operations(job)
        for i in range(len(job_ops) - 1):
            op_a = job_ops[i][0]
            op_b = job_ops[i+1][0]
            instance.add_precedence(job, op_a, op_b)
    
    return instance


def demo_mip_oracle():
    """Demonstrate MIP oracle on a small instance."""
    print("\n" + "="*80)
    print("DEMO 1: MIP Oracle")
    print("="*80 + "\n")
    
    # Create synthetic instance
    instance = create_synthetic_instance(n_jobs=3)
    print(f"Created instance: {instance}")
    print(f"Jobs: {instance.jobs}")
    print(f"Operations: {len(instance.operations)}")
    print(f"Precedence constraints: {len(instance.precedence)}\n")
    
    # Solve with MIP
    print("Solving with Gurobi MIP...")
    oracle = MIPOracle(weights=config.WEIGHTS, time_limit=30, verbose=False)
    solution = oracle.solve(instance)
    
    if solution:
        print(f"\n✓ Solution found!")
        print(f"  Makespan: {solution.makespan:.2f} minutes")
        print(f"  Total tardiness: {solution.objective_components['total_tardiness']:.2f}")
        print(f"  Total waiting: {solution.objective_components['total_waiting']:.2f}")
        print(f"  Total idle: {solution.objective_components['total_idle']:.2f}")
        print(f"  Weighted objective: {solution.objective_value:.2f}")
        print(f"  Solve time: {solution.solve_time:.2f} seconds")
        print(f"  Optimal: {solution.is_optimal}")
        
        # Show schedule
        print("\nSchedule (first 5 operations):")
        for i, (start, job, op, mc) in enumerate(solution.get_schedule_sequence()[:5]):
            print(f"  {i+1}. t={start:6.2f}: {job:8s} - {op:15s} on {mc}")
    else:
        print("✗ Failed to solve")
    
    return instance, solution


def demo_graph_representation(instance, solution):
    """Demonstrate graph building and training sample generation."""
    print("\n" + "="*80)
    print("DEMO 2: Graph Representation")
    print("="*80 + "\n")
    
    graph_builder = GraphBuilder()
    
    # Build static graph
    static_graph = graph_builder.build_static_graph(instance)
    print(f"Static graph (initial state):")
    print(f"  Nodes: {static_graph.num_nodes}")
    print(f"  Edges: {static_graph.num_edges}")
    print(f"  Node features shape: {static_graph.node_features.shape}")
    print(f"  Edge features shape: {static_graph.edge_features.shape}")
    print(f"  Global features shape: {static_graph.global_features.shape}")
    
    # Build dynamic graph (mid-schedule)
    scheduled_ops = {}
    for i, (start, job, op, mc) in enumerate(solution.get_schedule_sequence()):
        scheduled_ops[(job, op)] = start
        if i >= len(solution.get_schedule_sequence()) // 2:
            break
    
    dynamic_graph = graph_builder.build_dynamic_graph(
        instance, scheduled_ops, current_time=100
    )
    print(f"\nDynamic graph (mid-schedule):")
    print(f"  Node features shape: {dynamic_graph.node_features.shape}")
    print(f"  (includes dynamic features like availability, slack)")
    
    # Generate training samples
    samples = solution_to_training_samples(instance, solution, graph_builder)
    print(f"\nTraining samples generated: {len(samples)}")
    if samples:
        print(f"  Sample 0:")
        print(f"    Action: schedule {samples[0]['job']} - {samples[0]['op_name']}")
        print(f"    At time: {samples[0]['time']:.2f}")
        print(f"    Graph nodes: {samples[0]['graph'].num_nodes}")
    
    return samples


def demo_gnn_model(samples):
    """Demonstrate GNN model forward pass."""
    print("\n" + "="*80)
    print("DEMO 3: GNN Model")
    print("="*80 + "\n")
    
    # Create model
    model = SchedulingGNN(config.GNN_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}\n")
    
    # Prepare batch from first sample
    if samples:
        sample = samples[0]
        graph = sample['graph']
        
        batch = {
            'node_features': graph.node_features,
            'edge_index': graph.edge_index,
            'edge_features': graph.edge_features,
            'global_features': graph.global_features.unsqueeze(0),
            'batch': torch.zeros(graph.num_nodes, dtype=torch.long),
            'batch_size': 1
        }
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(batch)
        
        print(f"✓ Success!")
        print(f"  Scores shape: {outputs['scores'].shape}")
        print(f"  Value shape: {outputs['value'].shape}")
        print(f"  Top scored operation: node {torch.argmax(outputs['scores']).item()}")
        
        # Predict action
        action_node = model.predict_action(batch)
        print(f"  Predicted action: node {action_node.item()}")


def demo_rl_environment(instance):
    """Demonstrate RL environment simulation."""
    print("\n" + "="*80)
    print("DEMO 4: RL Environment")
    print("="*80 + "\n")
    
    env = JobShopEnv(instance)
    state = env.reset()
    
    print(f"Environment initialized")
    print(f"  Total operations: {len(instance.operations)}")
    print(f"  Initial state nodes: {state.num_nodes}\n")
    
    print("Simulating episode with random actions:")
    step = 0
    
    while not env.done and step < 10:
        available = env.get_available_operations()
        
        if not available:
            break
        
        # Random action
        action = available[np.random.randint(len(available))]
        job, op = action
        
        next_state, reward, done, info = env.step(action)
        
        print(f"  Step {step+1}: Scheduled {job} - {op}")
        print(f"           Progress: {info['scheduled_ops']}/{info['total_ops']}, "
              f"Time: {info['current_time']:.1f}")
        
        step += 1
    
    if env.done:
        print(f"\n✓ Episode completed!")
        print(f"  Final reward: {-env._compute_final_reward():.2f}")
        print(f"  Makespan: {max(env.completion_times.values()):.2f}")


def demo_heuristics(instance):
    """Demonstrate heuristic schedulers."""
    print("\n" + "="*80)
    print("DEMO 5: Heuristic Baselines")
    print("="*80 + "\n")
    
    methods = {
        'SPT': HeuristicScheduler.schedule_spt,
        'LPT': HeuristicScheduler.schedule_lpt,
        'FIFO': HeuristicScheduler.schedule_fifo
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"Running {name}...")
        result = method(instance)
        results[name] = result
        print(f"  Makespan: {result['makespan']:.2f}")
        print(f"  Objective: {result['objective']:.2f}")
        print(f"  Time: {result['solve_time']:.4f}s\n")
    
    # Compare
    best_method = min(results.items(), key=lambda x: x[1]['objective'])
    print(f"Best heuristic: {best_method[0]} (objective: {best_method[1]['objective']:.2f})")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("JOB SHOP SCHEDULING ML APPROXIMATION - DEMO")
    print("="*80)
    print("\nThis demo uses synthetic data to showcase the system.")
    print("For real usage, provide your Excel files.")
    
    try:
        # Demo 1: MIP Oracle
        instance, solution = demo_mip_oracle()
        
        if solution:
            # Demo 2: Graph Representation
            samples = demo_graph_representation(instance, solution)
            
            # Demo 3: GNN Model
            demo_gnn_model(samples)
        
        # Demo 4: RL Environment
        demo_rl_environment(instance)
        
        # Demo 5: Heuristics
        demo_heuristics(instance)
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Prepare your Excel data files")
        print("  2. Run: python main.py")
        print("  3. Check README.md for detailed instructions\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
