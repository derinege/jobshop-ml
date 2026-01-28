"""
Training Module for Imitation Learning
Trains the GNN policy to mimic optimal MIP solutions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import json

from gnn_model import SchedulingGNN
from dataset import SchedulingDataset, create_data_loaders
import config


class ImitationTrainer:
    """
    Trainer for imitation learning from MIP solutions.
    """
    
    def __init__(self,
                 model: SchedulingGNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cpu',
                 config_dict: Dict = None):
        """
        Initialize trainer.
        
        Args:
            model: SchedulingGNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cpu' or 'cuda')
            config_dict: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        if config_dict is None:
            config_dict = config.TRAINING_CONFIG
        
        self.config = config_dict
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        if self.config['lr_schedule'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs']
            )
        elif self.config['lr_schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['lr_decay_epochs'],
                gamma=self.config['lr_decay_factor']
            )
        else:
            self.scheduler = None
        
        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpoint directory
        self.checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_action_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            scores = outputs['scores']
            value = outputs['value']
            
            # Get targets
            actions = batch['actions']  # (batch_size,)
            
            # Compute action loss (cross-entropy)
            # Need to create score matrix per graph
            batch_assignment = batch['batch']
            batch_size = batch['batch_size']
            
            action_losses = []
            correct = 0
            
            for b in range(batch_size):
                mask = (batch_assignment == b)
                batch_scores = scores[mask]
                batch_action = actions[b]
                
                # Skip if no nodes in this batch
                if batch_scores.numel() == 0:
                    continue
                
                # Find local action index
                global_indices = torch.where(mask)[0]
                local_action = torch.where(global_indices == batch_action)[0]
                
                if local_action.numel() > 0:
                    local_action = local_action[0]
                    batch_loss = self.action_loss_fn(
                        batch_scores.unsqueeze(0), 
                        local_action.unsqueeze(0)
                    )
                    
                    # Check for NaN/Inf
                    if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                        action_losses.append(batch_loss)
                    
                    # Accuracy
                    pred_action = torch.argmax(batch_scores)
                    if pred_action == local_action:
                        correct += 1
            
            # Average action loss (avoid division by zero)
            if len(action_losses) > 0:
                action_loss = torch.stack(action_losses).mean()
            else:
                action_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Value loss (optional, if we have value targets)
            # For now, we don't have value targets from MIP, so skip or use zero
            value_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            loss = (self.config['action_loss_weight'] * action_loss + 
                   self.config['value_loss_weight'] * value_loss)
            
            # Check for NaN/Inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"Warning: NaN/Inf gradients detected, skipping update")
                self.optimizer.zero_grad()
                continue
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics (only if valid)
            action_loss_val = action_loss.item() if not torch.isnan(action_loss) else 0.0
            value_loss_val = value_loss.item() if not torch.isnan(value_loss) else 0.0
            loss_val = loss.item() if not torch.isnan(loss) else 0.0
            
            total_action_loss += action_loss_val
            total_value_loss += value_loss_val
            total_loss += loss_val
            total_accuracy += correct / batch_size if batch_size > 0 else 0.0
            num_batches += 1
        
        # Average metrics (avoid division by zero)
        if num_batches > 0:
            metrics = {
                'loss': total_loss / num_batches,
                'action_loss': total_action_loss / num_batches,
                'value_loss': total_value_loss / num_batches,
                'accuracy': total_accuracy / num_batches
            }
        else:
            metrics = {
                'loss': float('inf'),
                'action_loss': float('inf'),
                'value_loss': 0.0,
                'accuracy': 0.0
            }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_action_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = self._batch_to_device(batch)
                
                outputs = self.model(batch)
                scores = outputs['scores']
                value = outputs['value']
                
                actions = batch['actions']
                batch_assignment = batch['batch']
                batch_size = batch['batch_size']
                
                action_losses = []
                correct = 0
                
                for b in range(batch_size):
                    mask = (batch_assignment == b)
                    batch_scores = scores[mask]
                    batch_action = actions[b]
                    
                    # Skip if no nodes in this batch
                    if batch_scores.numel() == 0:
                        continue
                    
                    global_indices = torch.where(mask)[0]
                    local_action = torch.where(global_indices == batch_action)[0]
                    
                    if local_action.numel() > 0:
                        local_action = local_action[0]
                        batch_loss = self.action_loss_fn(
                            batch_scores.unsqueeze(0),
                            local_action.unsqueeze(0)
                        )
                        
                        # Check for NaN/Inf
                        if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                            action_losses.append(batch_loss)
                        
                        pred_action = torch.argmax(batch_scores)
                        if pred_action == local_action:
                            correct += 1
                
                # Average action loss (avoid division by zero)
                if len(action_losses) > 0:
                    action_loss = torch.stack(action_losses).mean()
                else:
                    action_loss = torch.tensor(0.0, device=self.device)
                value_loss = torch.tensor(0.0, device=self.device)
                
                loss = (self.config['action_loss_weight'] * action_loss + 
                       self.config['value_loss_weight'] * value_loss)
                
                # Track metrics (only if valid)
                action_loss_val = action_loss.item() if not torch.isnan(action_loss) else 0.0
                value_loss_val = value_loss.item() if not torch.isnan(value_loss) else 0.0
                loss_val = loss.item() if not torch.isnan(loss) else 0.0
                
                total_action_loss += action_loss_val
                total_value_loss += value_loss_val
                total_loss += loss_val
                total_accuracy += correct / batch_size if batch_size > 0 else 0.0
                num_batches += 1
        
        # Average metrics (avoid division by zero)
        if num_batches > 0:
            metrics = {
                'loss': total_loss / num_batches,
                'action_loss': total_action_loss / num_batches,
                'value_loss': total_value_loss / num_batches,
                'accuracy': total_accuracy / num_batches
            }
        else:
            metrics = {
                'loss': float('inf'),
                'action_loss': float('inf'),
                'value_loss': 0.0,
                'accuracy': 0.0
            }
        
        return metrics
    
    def train(self, num_epochs: int = None) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Dictionary of training history
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Print progress
            if (epoch + 1) % self.config.get('print_frequency', 10) == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                if self.config['save_best_only']:
                    self.save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pt', num_epochs, val_metrics)
        
        # Save history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


def plot_training_history(history: Dict, save_path: str = 'training_curves.png'):
    """
    Plot training curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_accuracy'], label='Train Accuracy')
        ax2.plot(history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
        plt.close()
        
    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    print("=== Training Module Test ===\n")
    print("This module requires pre-generated datasets.")
    print("Run dataset.py first to generate training data.")
