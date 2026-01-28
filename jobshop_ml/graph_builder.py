"""
Graph Representation Module
Converts scheduling instances and states into graph representations for GNN processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_loader import SchedulingInstance
from mip_oracle import ScheduleSolution
import config


class SchedulingGraph:
    """
    Graph representation of a scheduling state.
    
    Nodes represent operations, edges represent precedence and machine relationships.
    
    Attributes:
        node_features: Tensor of shape (num_nodes, feature_dim)
        edge_index: Tensor of shape (2, num_edges) - COO format
        edge_features: Tensor of shape (num_edges, edge_feature_dim)
        global_features: Tensor of shape (global_feature_dim,)
        node_to_op: Dict mapping node_idx -> (job, op_name)
        op_to_node: Dict mapping (job, op_name) -> node_idx
        num_nodes: Number of operation nodes
        num_edges: Number of edges
    """
    
    def __init__(self):
        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.global_features = None
        self.node_to_op = {}
        self.op_to_node = {}
        self.num_nodes = 0
        self.num_edges = 0


class GraphBuilder:
    """
    Builds graph representations from scheduling instances and states.
    """
    
    def __init__(self, machine_classes: List[str] = None):
        """
        Initialize graph builder.
        
        Args:
            machine_classes: List of machine class names for one-hot encoding
        """
        self.machine_classes = machine_classes if machine_classes else config.MACHINE_CLASSES
        self.machine_to_idx = {mc: i for i, mc in enumerate(self.machine_classes)}
    
    def build_static_graph(self, instance: SchedulingInstance) -> SchedulingGraph:
        """
        Build a static graph representation of the full instance.
        This represents the problem structure without any scheduling decisions yet.
        
        Args:
            instance: SchedulingInstance
            
        Returns:
            SchedulingGraph
        """
        graph = SchedulingGraph()
        
        # Create node mapping
        operations = list(instance.operations.keys())
        graph.num_nodes = len(operations)
        
        for idx, (job, op_name) in enumerate(operations):
            graph.node_to_op[idx] = (job, op_name)
            graph.op_to_node[(job, op_name)] = idx
        
        # Build node features
        node_features = []
        for idx in range(graph.num_nodes):
            job, op_name = graph.node_to_op[idx]
            op_data = instance.operations[(job, op_name)]
            
            features = self._create_node_features(
                job=job,
                op_name=op_name,
                op_data=op_data,
                instance=instance
            )
            node_features.append(features)
        
        graph.node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edges (precedence + same machine)
        edge_list = []
        edge_features = []
        
        # Precedence edges
        for (job, op_a, op_b) in instance.precedence:
            if (job, op_a) in graph.op_to_node and (job, op_b) in graph.op_to_node:
                src_idx = graph.op_to_node[(job, op_a)]
                dst_idx = graph.op_to_node[(job, op_b)]
                edge_list.append([src_idx, dst_idx])
                edge_features.append([1.0, 0.0])  # [is_precedence, is_same_machine]
        
        # Same machine edges (bidirectional)
        ops_by_machine = {}
        for (job, op_name), data in instance.operations.items():
            mc = data['machine_class']
            if mc not in ops_by_machine:
                ops_by_machine[mc] = []
            ops_by_machine[mc].append((job, op_name))
        
        for mc, ops in ops_by_machine.items():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    op1 = ops[i]
                    op2 = ops[j]
                    idx1 = graph.op_to_node[op1]
                    idx2 = graph.op_to_node[op2]
                    
                    # Bidirectional
                    edge_list.append([idx1, idx2])
                    edge_features.append([0.0, 1.0])
                    edge_list.append([idx2, idx1])
                    edge_features.append([0.0, 1.0])
        
        if edge_list:
            graph.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            graph.edge_features = torch.tensor(edge_features, dtype=torch.float32)
            graph.num_edges = len(edge_list)
        else:
            graph.edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph.edge_features = torch.zeros((0, 2), dtype=torch.float32)
            graph.num_edges = 0
        
        # Global features
        graph.global_features = self._create_global_features(instance)
        
        return graph
    
    def build_dynamic_graph(self, 
                           instance: SchedulingInstance,
                           scheduled_ops: Dict[Tuple[str, str], float],
                           current_time: float) -> SchedulingGraph:
        """
        Build a dynamic graph representing the current scheduling state.
        
        Args:
            instance: SchedulingInstance
            scheduled_ops: Dict mapping (job, op_name) -> start_time for already scheduled ops
            current_time: Current simulation time
            
        Returns:
            SchedulingGraph with dynamic features
        """
        graph = self.build_static_graph(instance)
        
        # Update node features with dynamic information
        dynamic_features = []
        
        for idx in range(graph.num_nodes):
            job, op_name = graph.node_to_op[idx]
            
            # Is this operation already scheduled?
            is_scheduled = (job, op_name) in scheduled_ops
            
            # Is this operation available to schedule?
            # (all predecessors are complete)
            is_available = self._is_operation_available(
                job, op_name, instance, scheduled_ops, current_time
            )
            
            # Time until this operation can start (slack)
            earliest_start = self._compute_earliest_start(
                job, op_name, instance, scheduled_ops, current_time
            )
            slack = max(0, earliest_start - current_time)
            
            # Remaining operations for this job
            remaining_ops = self._count_remaining_operations(
                job, op_name, instance, scheduled_ops
            )
            
            dynamic_features.append([
                1.0 if is_scheduled else 0.0,
                1.0 if is_available else 0.0,
                slack / 1000.0,  # Normalize
                remaining_ops / 10.0  # Normalize
            ])
        
        dynamic_tensor = torch.tensor(dynamic_features, dtype=torch.float32)
        graph.node_features = torch.cat([graph.node_features, dynamic_tensor], dim=1)
        
        # Update global features
        n_scheduled = len(scheduled_ops)
        n_total = len(instance.operations)
        progress = n_scheduled / n_total if n_total > 0 else 0
        
        graph.global_features = torch.cat([
            graph.global_features,
            torch.tensor([current_time / 1000.0, progress], dtype=torch.float32)
        ])
        
        return graph
    
    def _create_node_features(self, 
                             job: str, 
                             op_name: str,
                             op_data: Dict,
                             instance: SchedulingInstance) -> List[float]:
        """Create feature vector for a node (operation)."""
        features = []
        
        # Machine class (one-hot)
        machine_class = op_data['machine_class']
        machine_onehot = [0.0] * len(self.machine_classes)
        if machine_class in self.machine_to_idx:
            machine_onehot[self.machine_to_idx[machine_class]] = 1.0
        features.extend(machine_onehot)
        
        # Processing time (normalized)
        duration = op_data['duration']
        features.append(duration / 100.0)  # Normalize by typical duration
        
        # Operation order within job (normalized)
        sira_no = op_data['sira_no']
        job_ops = instance.get_job_operations(job)
        max_sira = len(job_ops)
        features.append(sira_no / max(max_sira, 1))
        
        # Total operations for this job
        features.append(max_sira / 10.0)  # Normalize
        
        # Job quantity (if available)
        quantity = instance.job_quantities.get(job, 1)
        features.append(quantity / 10.0)  # Normalize
        
        # Due date proxy (currently 0, but placeholder)
        due_date = instance.due_dates.get(job, 0)
        features.append(due_date / 1000.0)
        
        return features
    
    def _create_global_features(self, instance: SchedulingInstance) -> torch.Tensor:
        """Create global features for the instance."""
        features = []
        
        # Number of jobs
        features.append(len(instance.jobs) / 50.0)  # Normalize
        
        # Number of operations
        features.append(len(instance.operations) / 200.0)  # Normalize
        
        # Average processing time
        avg_duration = np.mean([data['duration'] for data in instance.operations.values()])
        features.append(avg_duration / 100.0)
        
        # Operations per machine class
        ops_per_machine = {}
        for (job, op_name), data in instance.operations.items():
            mc = data['machine_class']
            ops_per_machine[mc] = ops_per_machine.get(mc, 0) + 1
        
        for mc in self.machine_classes:
            features.append(ops_per_machine.get(mc, 0) / 50.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _is_operation_available(self, 
                                job: str, 
                                op_name: str,
                                instance: SchedulingInstance,
                                scheduled_ops: Dict,
                                current_time: float) -> bool:
        """Check if an operation can be scheduled (all predecessors complete)."""
        # Find all predecessors
        for (j, op_a, op_b) in instance.precedence:
            if j == job and op_b == op_name:
                # op_a must be scheduled and complete
                if (j, op_a) not in scheduled_ops:
                    return False
                
                start_time = scheduled_ops[(j, op_a)]
                duration = instance.operations[(j, op_a)]['duration']
                completion_time = start_time + duration
                
                if completion_time > current_time:
                    return False
        
        return True
    
    def _compute_earliest_start(self,
                               job: str,
                               op_name: str,
                               instance: SchedulingInstance,
                               scheduled_ops: Dict,
                               current_time: float) -> float:
        """Compute earliest possible start time for an operation."""
        earliest = current_time
        
        # Must wait for all predecessors
        for (j, op_a, op_b) in instance.precedence:
            if j == job and op_b == op_name:
                if (j, op_a) in scheduled_ops:
                    start_time = scheduled_ops[(j, op_a)]
                    duration = instance.operations[(j, op_a)]['duration']
                    completion_time = start_time + duration
                    earliest = max(earliest, completion_time)
                else:
                    # Predecessor not scheduled yet
                    earliest = float('inf')
        
        return earliest
    
    def _count_remaining_operations(self,
                                   job: str,
                                   op_name: str,
                                   instance: SchedulingInstance,
                                   scheduled_ops: Dict) -> int:
        """Count how many operations remain for this job after this operation."""
        job_ops = instance.get_job_operations(job)
        current_sira = instance.operations[(job, op_name)]['sira_no']
        
        remaining = 0
        for op, data in job_ops:
            if data['sira_no'] >= current_sira and (job, op) not in scheduled_ops:
                remaining += 1
        
        return remaining


def solution_to_training_samples(instance: SchedulingInstance,
                                 solution: ScheduleSolution,
                                 graph_builder: GraphBuilder) -> List[Dict]:
    """
    Convert an optimal solution into training samples for imitation learning.
    Each sample represents a decision point: which operation to schedule next.
    
    Args:
        instance: SchedulingInstance
        solution: Optimal solution from MIP
        graph_builder: GraphBuilder instance
        
    Returns:
        List of training samples, each containing:
            - 'graph': SchedulingGraph (state)
            - 'action': node index of operation scheduled at this step
            - 'job': job name
            - 'op_name': operation name
    """
    samples = []
    
    # Get schedule sequence
    sequence = solution.get_schedule_sequence()
    
    # Simulate step-by-step scheduling
    scheduled_ops = {}
    
    for step_idx, (start_time, job, op_name, machine_class) in enumerate(sequence):
        # Create graph representing state before this decision
        if step_idx == 0:
            current_time = 0
        else:
            current_time = sequence[step_idx - 1][0]  # Time of previous operation
        
        graph = graph_builder.build_dynamic_graph(instance, scheduled_ops, current_time)
        
        # The action is to schedule (job, op_name)
        if (job, op_name) in graph.op_to_node:
            action_node_idx = graph.op_to_node[(job, op_name)]
            
            samples.append({
                'graph': graph,
                'action': action_node_idx,
                'job': job,
                'op_name': op_name,
                'step': step_idx,
                'time': start_time
            })
        
        # Update scheduled operations
        scheduled_ops[(job, op_name)] = start_time
    
    return samples


if __name__ == "__main__":
    # Example usage
    print("=== Graph Representation Module Test ===\n")
    
    from data_loader import DataLoader
    from mip_oracle import MIPOracle
    
    # Create a small instance
    loader = DataLoader()
    loader.load_data()
    instances = loader.generate_random_instances(n_instances=1, min_jobs=3, max_jobs=3)
    
    if instances:
        instance = instances[0]
        print(f"Instance: {instance}")
        
        # Build static graph
        graph_builder = GraphBuilder()
        static_graph = graph_builder.build_static_graph(instance)
        
        print(f"\nStatic graph:")
        print(f"  Nodes: {static_graph.num_nodes}")
        print(f"  Edges: {static_graph.num_edges}")
        print(f"  Node features shape: {static_graph.node_features.shape}")
        print(f"  Global features shape: {static_graph.global_features.shape}")
        
        # Solve with MIP
        print("\nSolving with MIP...")
        oracle = MIPOracle(verbose=False, time_limit=30)
        solution = oracle.solve(instance)
        
        if solution:
            print(f"Solution: {solution}")
            
            # Generate training samples
            samples = solution_to_training_samples(instance, solution, graph_builder)
            print(f"\nGenerated {len(samples)} training samples")
            
            if samples:
                print(f"First sample:")
                print(f"  Action node: {samples[0]['action']}")
                print(f"  Operation: {samples[0]['job']} - {samples[0]['op_name']}")
                print(f"  Graph nodes: {samples[0]['graph'].num_nodes}")
