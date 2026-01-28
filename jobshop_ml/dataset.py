"""
Dataset Generation and Management Module
Generates training/validation/test datasets from MIP solutions.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

from data_loader import DataLoader, SchedulingInstance
from mip_oracle import MIPOracle, ScheduleSolution
from graph_builder import GraphBuilder, solution_to_training_samples, SchedulingGraph
import config


class SchedulingDataset(Dataset):
    """
    PyTorch Dataset for imitation learning from MIP solutions.
    Each sample represents a scheduling decision point.
    """
    
    def __init__(self, samples: List[Dict]):
        """
        Initialize dataset.
        
        Args:
            samples: List of training samples from solution_to_training_samples
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            Dictionary with:
                - graph: SchedulingGraph
                - action: Target action (node index)
                - metadata: Additional info (job, op_name, etc.)
        """
        sample = self.samples[idx]
        
        return {
            'graph': sample['graph'],
            'action': torch.tensor(sample['action'], dtype=torch.long),
            'job': sample['job'],
            'op_name': sample['op_name'],
            'step': sample['step']
        }


def collate_scheduling_graphs(batch: List[Dict]) -> Dict:
    """
    Collate function for batching scheduling graphs.
    Creates a single large graph with disconnected components.
    
    Args:
        batch: List of samples from SchedulingDataset
        
    Returns:
        Batched data dictionary
    """
    # Batch graphs by concatenating them
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    global_features_list = []
    actions_list = []
    batch_indices = []
    
    node_offset = 0
    
    for batch_idx, item in enumerate(batch):
        graph = item['graph']
        
        # Node features
        node_features_list.append(graph.node_features)
        
        # Edge indices (shift by node offset)
        if graph.num_edges > 0:
            edge_index_shifted = graph.edge_index + node_offset
            edge_index_list.append(edge_index_shifted)
            edge_features_list.append(graph.edge_features)
        
        # Global features
        global_features_list.append(graph.global_features.unsqueeze(0))
        
        # Action (shift by node offset)
        action_shifted = item['action'] + node_offset
        actions_list.append(action_shifted)
        
        # Batch assignment
        batch_indices.extend([batch_idx] * graph.num_nodes)
        
        node_offset += graph.num_nodes
    
    # Concatenate all
    batched_data = {
        'node_features': torch.cat(node_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long),
        'edge_features': torch.cat(edge_features_list, dim=0) if edge_features_list else torch.zeros((0, 2), dtype=torch.float32),
        'global_features': torch.cat(global_features_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'actions': torch.stack(actions_list),
        'batch_size': len(batch)
    }
    
    return batched_data


class DatasetGenerator:
    """
    Generates datasets by solving instances with MIP oracle.
    """
    
    def __init__(self,
                 data_loader: DataLoader,
                 weights: Dict[str, float] = None,
                 mip_time_limit: int = config.MIP_TIME_LIMIT,
                 cache_dir: str = 'dataset_cache'):
        """
        Initialize dataset generator.
        
        Args:
            data_loader: DataLoader instance
            weights: Objective weights for MIP solver
            mip_time_limit: Time limit for MIP solver
            cache_dir: Directory to cache generated datasets
        """
        self.data_loader = data_loader
        self.weights = weights if weights else config.WEIGHTS
        self.oracle = MIPOracle(weights=self.weights, 
                                time_limit=mip_time_limit,
                                verbose=False)
        self.graph_builder = GraphBuilder()
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def generate_dataset(self,
                        n_instances: int,
                        min_jobs: int,
                        max_jobs: int,
                        seed: int,
                        dataset_name: str) -> SchedulingDataset:
        """
        Generate a dataset by creating random instances and solving them.
        
        Args:
            n_instances: Number of instances to generate
            min_jobs: Minimum jobs per instance
            max_jobs: Maximum jobs per instance
            seed: Random seed
            dataset_name: Name for caching (e.g., 'train', 'val', 'test')
            
        Returns:
            SchedulingDataset
        """
        cache_path = os.path.join(self.cache_dir, f'{dataset_name}.pkl')
        
        # Check cache
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            return SchedulingDataset(samples)
        
        print(f"Generating {dataset_name} dataset with {n_instances} instances...")
        
        # Generate instances
        instances = self.data_loader.generate_random_instances(
            n_instances=n_instances,
            min_jobs=min_jobs,
            max_jobs=max_jobs,
            seed=seed
        )
        
        # Solve each instance and collect training samples
        all_samples = []
        successful_solves = 0
        
        for i, instance in enumerate(tqdm(instances, desc=f"Solving {dataset_name} instances")):
            solution = self.oracle.solve(instance)
            
            if solution is not None:
                # Convert solution to training samples
                samples = solution_to_training_samples(
                    instance, solution, self.graph_builder
                )
                all_samples.extend(samples)
                successful_solves += 1
            else:
                print(f"Warning: Failed to solve instance {i}")
        
        print(f"Successfully solved {successful_solves}/{n_instances} instances")
        print(f"Generated {len(all_samples)} training samples")
        
        # Cache the dataset
        with open(cache_path, 'wb') as f:
            pickle.dump(all_samples, f)
        print(f"Cached dataset to {cache_path}")
        
        return SchedulingDataset(all_samples)
    
    def generate_train_val_test_datasets(self) -> Tuple[SchedulingDataset, SchedulingDataset, SchedulingDataset]:
        """
        Generate train, validation, and test datasets using config parameters.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Training set
        train_dataset = self.generate_dataset(
            n_instances=config.N_TRAIN_INSTANCES,
            min_jobs=config.MIN_JOBS_PER_INSTANCE,
            max_jobs=config.MAX_JOBS_PER_INSTANCE,
            seed=config.RANDOM_SEED,
            dataset_name='train'
        )
        
        # Validation set
        val_dataset = self.generate_dataset(
            n_instances=config.N_VAL_INSTANCES,
            min_jobs=config.MIN_JOBS_PER_INSTANCE,
            max_jobs=config.MAX_JOBS_PER_INSTANCE,
            seed=config.RANDOM_SEED + 1000,
            dataset_name='val'
        )
        
        # Test set (same size as training for now)
        test_dataset = self.generate_dataset(
            n_instances=config.N_TEST_INSTANCES,
            min_jobs=config.MIN_JOBS_PER_INSTANCE,
            max_jobs=config.MAX_JOBS_PER_INSTANCE,
            seed=config.RANDOM_SEED + 2000,
            dataset_name='test'
        )
        
        return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset: SchedulingDataset,
                       val_dataset: SchedulingDataset,
                       test_dataset: SchedulingDataset,
                       batch_size: int = None) -> Tuple:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (uses config if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.TRAINING_CONFIG['batch_size']
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_scheduling_graphs,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scheduling_graphs,
        num_workers=0
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scheduling_graphs,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("=== Dataset Generation Module Test ===\n")
    
    from data_loader import DataLoader
    
    # Initialize
    loader = DataLoader()
    loader.load_data()
    
    # Create dataset generator
    generator = DatasetGenerator(loader)
    
    # Generate a small test dataset
    print("Generating small test dataset...")
    test_dataset = generator.generate_dataset(
        n_instances=3,
        min_jobs=3,
        max_jobs=4,
        seed=42,
        dataset_name='demo'
    )
    
    print(f"\nDataset size: {len(test_dataset)}")
    
    # Test data loader
    if len(test_dataset) > 0:
        demo_loader = TorchDataLoader(
            test_dataset,
            batch_size=2,
            collate_fn=collate_scheduling_graphs
        )
        
        print("\nTesting DataLoader:")
        for batch_idx, batch in enumerate(demo_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Node features: {batch['node_features'].shape}")
            print(f"  Edge index: {batch['edge_index'].shape}")
            print(f"  Actions: {batch['actions'].shape}")
            print(f"  Batch size: {batch['batch_size']}")
            break
