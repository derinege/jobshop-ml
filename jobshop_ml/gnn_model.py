"""
Graph Neural Network Model for Scheduling Policy
Implements a GNN that scores operations for scheduling decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import config


class GraphConvLayer(nn.Module):
    """
    Simple graph convolution layer.
    Implements message passing: h_i' = σ(W_self * h_i + Σ_j W_neigh * h_j)
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Self transform
        h_self = self.W_self(x)
        
        # Neighbor aggregation
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Aggregate messages from neighbors
        h_neigh = torch.zeros(num_nodes, self.W_neigh.out_features, device=x.device)
        if edge_index.size(1) > 0:
            messages = self.W_neigh(x[src])
            h_neigh = h_neigh.index_add(0, dst, messages)
        
        # Combine
        return F.relu(h_self + h_neigh)


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.W_query = nn.Linear(in_dim, out_dim)
        self.W_key = nn.Linear(in_dim, out_dim)
        self.W_value = nn.Linear(in_dim, out_dim)
        self.W_out = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        num_nodes = x.size(0)
        
        # Project to query, key, value
        Q = self.W_query(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.W_key(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.W_value(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        if edge_index.size(1) == 0:
            # No edges, just return transformed features
            return self.W_out(V.view(num_nodes, self.out_dim))
        
        src, dst = edge_index
        
        # Attention scores: (Q_dst * K_src) / sqrt(d)
        scores = (Q[dst] * K[src]).sum(dim=-1) / (self.head_dim ** 0.5)  # (num_edges, num_heads)
        
        # Apply softmax per destination node
        scores_exp = torch.exp(scores - scores.max())
        scores_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        scores_sum = scores_sum.index_add(0, dst, scores_exp)
        
        attention = scores_exp / (scores_sum[dst] + 1e-8)
        attention = self.dropout(attention)
        
        # Weighted aggregation
        messages = V[src] * attention.unsqueeze(-1)  # (num_edges, num_heads, head_dim)
        
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out = out.index_add(0, dst, messages)
        
        # Reshape and output projection
        out = out.view(num_nodes, self.out_dim)
        out = self.W_out(out)
        
        return F.relu(out)


class SchedulingGNN(nn.Module):
    """
    Graph Neural Network for scheduling policy.
    Takes a scheduling state graph and outputs scores for each operation.
    """
    
    def __init__(self, config_dict: Dict = None):
        super().__init__()
        
        if config_dict is None:
            config_dict = config.GNN_CONFIG
        
        self.node_feature_dim = config_dict.get('node_feature_dim', 64)
        self.hidden_dim = config_dict.get('hidden_dim', 128)
        self.num_gnn_layers = config_dict.get('num_gnn_layers', 3)
        self.num_mlp_layers = config_dict.get('num_mlp_layers', 2)
        self.dropout = config_dict.get('dropout', 0.1)
        self.use_attention = config_dict.get('use_attention', True)
        self.num_heads = config_dict.get('num_heads', 4)
        
        # Input projection
        # Note: Actual input dim depends on graph_builder features
        # We'll use a configurable input dim and project to node_feature_dim
        self.input_dim = None  # Will be set dynamically
        self.input_projection = None
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        layer_dims = [self.node_feature_dim] + [self.hidden_dim] * self.num_gnn_layers
        
        for i in range(self.num_gnn_layers):
            if self.use_attention:
                layer = GATLayer(layer_dims[i], layer_dims[i + 1], 
                               num_heads=self.num_heads, dropout=self.dropout)
            else:
                layer = GraphConvLayer(layer_dims[i], layer_dims[i + 1])
            
            self.gnn_layers.append(layer)
        
        # Global pooling (optional, for value prediction)
        # Global features dim will be set dynamically based on actual input
        self.global_feature_dim = None
        self.global_pool = None  # Will be initialized dynamically
        
        # MLP head for operation scoring
        mlp_layers = []
        mlp_dims = [self.hidden_dim] * (self.num_mlp_layers + 1) + [1]
        
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(self.dropout))
        
        self.mlp_head = nn.Sequential(*mlp_layers)
        
        # Value head (for value function approximation in RL)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _initialize_input_projection(self, input_dim: int):
        """Initialize input projection layer based on actual input dimension."""
        if self.input_projection is None:
            self.input_dim = input_dim
            self.input_projection = nn.Linear(input_dim, self.node_feature_dim)
    
    def _initialize_global_pool(self, global_feature_dim: int):
        """Initialize global pooling layer based on actual global feature dimension."""
        if self.global_pool is None:
            self.global_feature_dim = global_feature_dim
            self.global_pool = nn.Sequential(
                nn.Linear(self.hidden_dim + global_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing:
                - node_features: (total_nodes, feature_dim)
                - edge_index: (2, num_edges)
                - global_features: (batch_size, global_dim)
                - batch: (total_nodes,) batch assignment
                
        Returns:
            Dictionary with:
                - scores: (total_nodes,) score for each operation
                - value: (batch_size,) value estimate for each graph
        """
        x = batch['node_features']
        edge_index = batch['edge_index']
        global_features = batch['global_features']
        batch_assignment = batch['batch']
        batch_size = batch['batch_size']
        
        # Initialize input projection if needed
        if self.input_projection is None:
            self._initialize_input_projection(x.size(1))
        
        # Project input features
        x = self.input_projection(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
        
        # Node-level scores
        scores = self.mlp_head(x).squeeze(-1)
        
        # Global pooling for value prediction
        # Sum pooling per graph
        num_nodes = x.size(0)
        device = x.device
        
        # Create one-hot batch matrix for pooling
        batch_onehot = torch.zeros(num_nodes, batch_size, device=device)
        batch_onehot[torch.arange(num_nodes, device=device), batch_assignment] = 1
        
        # Pool: (batch_size, hidden_dim)
        pooled = torch.mm(batch_onehot.t(), x)
        
        # Initialize global pool if needed
        if self.global_pool is None:
            self._initialize_global_pool(global_features.size(1))
        
        # Concatenate with global features
        pooled_with_global = torch.cat([pooled, global_features], dim=1)
        pooled_with_global = self.global_pool(pooled_with_global)
        
        # Value prediction
        value = self.value_head(pooled_with_global).squeeze(-1)
        
        return {
            'scores': scores,
            'value': value
        }
    
    def predict_action(self, 
                      batch: Dict,
                      available_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict which operation to schedule (greedy selection).
        
        Args:
            batch: Batch dictionary
            available_mask: Boolean mask of available operations (total_nodes,)
            
        Returns:
            Selected node indices (batch_size,)
        """
        with torch.no_grad():
            outputs = self.forward(batch)
            scores = outputs['scores']
            
            # Apply mask if provided
            if available_mask is not None:
                scores = scores.masked_fill(~available_mask, float('-inf'))
            
            # Select highest scoring operation per graph
            batch_assignment = batch['batch']
            batch_size = batch['batch_size']
            
            selected_nodes = []
            for b in range(batch_size):
                mask = (batch_assignment == b)
                batch_scores = scores[mask]
                
                if batch_scores.numel() > 0:
                    local_idx = torch.argmax(batch_scores)
                    global_idx = torch.where(mask)[0][local_idx]
                    selected_nodes.append(global_idx)
                else:
                    selected_nodes.append(torch.tensor(-1, device=scores.device))
            
            return torch.stack(selected_nodes)


if __name__ == "__main__":
    # Test the model
    print("=== GNN Model Test ===\n")
    
    # Create dummy batch
    num_nodes = 20
    num_edges = 30
    batch_size = 2
    feature_dim = 25
    global_dim = 20
    
    dummy_batch = {
        'node_features': torch.randn(num_nodes, feature_dim),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'edge_features': torch.randn(num_edges, 2),
        'global_features': torch.randn(batch_size, global_dim),
        'batch': torch.cat([torch.zeros(10, dtype=torch.long), 
                           torch.ones(10, dtype=torch.long)]),
        'batch_size': batch_size
    }
    
    # Create model
    model = SchedulingGNN()
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("\nForward pass:")
    outputs = model(dummy_batch)
    print(f"Scores shape: {outputs['scores'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    
    # Predict action
    print("\nAction prediction:")
    actions = model.predict_action(dummy_batch)
    print(f"Selected actions: {actions}")
