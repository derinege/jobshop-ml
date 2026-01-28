"""
Main Training Script
Orchestrates the full ML-based job shop scheduling pipeline:
1. Load data
2. Generate datasets from MIP oracle
3. Train GNN policy with imitation learning
4. Evaluate against MIP and heuristics
"""

import torch
import argparse
import os
import sys

from data_loader import DataLoader
from dataset import DatasetGenerator, create_data_loaders
from gnn_model import SchedulingGNN
from training import ImitationTrainer, plot_training_history
from evaluation import Evaluator
import config


def main(args):
    """Main training and evaluation pipeline."""
    
    print("\n" + "="*80)
    print("JOB SHOP SCHEDULING - ML APPROXIMATION PIPELINE")
    print("="*80 + "\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    
    if not args.skip_data_loading:
        print("STEP 1: Loading Excel data...")
        print("-" * 80)
        
        try:
            data_loader = DataLoader(
                islem_tam_path=args.islem_tam_path or config.DATA_PATH_ISLEM_TAM,
                bold_sure_path=args.bold_sure_path or config.DATA_PATH_BOLD_SURE
            )
            data_loader.load_data()
            
            # Show summary
            full_instance = data_loader.create_full_instance()
            print(f"Loaded full instance: {full_instance}")
            
        except FileNotFoundError as e:
            print(f"\n⚠️  ERROR: Data files not found!")
            print(f"    {e}")
            print(f"\n    Please ensure the following files exist:")
            print(f"    - {args.islem_tam_path or config.DATA_PATH_ISLEM_TAM}")
            print(f"    - {args.bold_sure_path or config.DATA_PATH_BOLD_SURE}")
            print(f"\n    You can skip this step with --skip-data-loading if you have cached datasets.")
            return
    else:
        print("STEP 1: Skipping data loading (using cached datasets)\n")
        data_loader = None
    
    # ========================================================================
    # STEP 2: Generate Datasets (or load from cache)
    # ========================================================================
    
    if not args.skip_dataset_generation:
        print("\nSTEP 2: Generating training datasets with MIP oracle...")
        print("-" * 80)
        
        if data_loader is None:
            print("ERROR: Cannot generate datasets without data_loader.")
            print("Either provide data files or use --skip-dataset-generation with cached data.")
            return
        
        generator = DatasetGenerator(
            data_loader,
            weights=config.WEIGHTS,
            mip_time_limit=args.mip_time_limit,
            cache_dir=args.cache_dir
        )
        
        train_dataset, val_dataset, test_dataset = generator.generate_train_val_test_datasets()
        
        print(f"\nDataset sizes:")
        print(f"  Training:   {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test:       {len(test_dataset)} samples")
        
    else:
        print("\nSTEP 2: Loading cached datasets...")
        print("-" * 80)
        
        # Try to load cached datasets
        import pickle
        
        try:
            cache_dir = args.cache_dir
            
            with open(os.path.join(cache_dir, 'train.pkl'), 'rb') as f:
                from dataset import SchedulingDataset
                train_samples = pickle.load(f)
                train_dataset = SchedulingDataset(train_samples)
            
            with open(os.path.join(cache_dir, 'val.pkl'), 'rb') as f:
                val_samples = pickle.load(f)
                val_dataset = SchedulingDataset(val_samples)
            
            with open(os.path.join(cache_dir, 'test.pkl'), 'rb') as f:
                test_samples = pickle.load(f)
                test_dataset = SchedulingDataset(test_samples)
            
            print(f"Loaded datasets from {cache_dir}")
            print(f"  Training:   {len(train_dataset)} samples")
            print(f"  Validation: {len(val_dataset)} samples")
            print(f"  Test:       {len(test_dataset)} samples")
            
        except FileNotFoundError:
            print(f"ERROR: Cached datasets not found in {cache_dir}")
            print("Please run without --skip-dataset-generation first.")
            return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size
    )
    
    # ========================================================================
    # STEP 3: Initialize GNN Model
    # ========================================================================
    
    print("\nSTEP 3: Initializing GNN model...")
    print("-" * 80)
    
    model = SchedulingGNN(config.GNN_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # STEP 4: Train with Imitation Learning
    # ========================================================================
    
    if not args.skip_training:
        print("\nSTEP 4: Training GNN policy with imitation learning...")
        print("-" * 80)
        
        trainer = ImitationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config_dict=config.TRAINING_CONFIG
        )
        
        history = trainer.train(num_epochs=args.num_epochs)
        
        # Plot training curves
        if not args.no_plot:
            plot_training_history(
                history,
                save_path=os.path.join(args.checkpoint_dir, 'training_curves.png')
            )
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
    else:
        print("\nSTEP 4: Skipping training (loading pre-trained model)...")
        print("-" * 80)
        
        # Load pre-trained model (try best_model.pt first, then final_model.pt)
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
        
        checkpoint_path = None
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        elif os.path.exists(final_model_path):
            checkpoint_path = final_model_path
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"  Metrics: {checkpoint['metrics']}")
        else:
            print(f"ERROR: No pre-trained model found at {best_model_path} or {final_model_path}")
            print("Please train a model first without --skip-training")
            return
    
    # ========================================================================
    # STEP 5: Evaluation
    # ========================================================================
    
    if not args.skip_evaluation:
        print("\nSTEP 5: Evaluating trained policy...")
        print("-" * 80)
        
        # Load test instances for evaluation
        if data_loader is None:
            print("Loading data for evaluation...")
            data_loader = DataLoader()
            try:
                data_loader.load_data()
            except FileNotFoundError:
                print("Cannot evaluate without data files.")
                return
        
        # Generate fresh test instances
        print(f"Generating {args.n_eval_instances} test instances...")
        eval_instances = data_loader.generate_random_instances(
            n_instances=args.n_eval_instances,
            min_jobs=args.eval_min_jobs,
            max_jobs=args.eval_max_jobs,
            seed=config.RANDOM_SEED + 5000
        )
        
        # Create evaluator
        evaluator = Evaluator(ml_model=model, device=device)
        
        # Evaluate
        methods_to_compare = ['ml', 'spt', 'lpt', 'fifo']
        if args.compare_with_mip:
            methods_to_compare = ['mip'] + methods_to_compare
        
        print(f"\nComparing methods: {methods_to_compare}")
        summary = evaluator.evaluate_dataset(eval_instances, methods=methods_to_compare)
        
        # Print results
        evaluator.print_comparison_table(summary)
        
    else:
        print("\nSTEP 5: Skipping evaluation")
    
    # ========================================================================
    # Done!
    # ========================================================================
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("  - Adjust hyperparameters in config.py and re-train")
    print("  - Evaluate on larger instances with --eval-max-jobs")
    print("  - Implement full RL training in rl_env.py for further improvement")
    print("  - Use the trained model for production scheduling\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML-based job shop scheduling policy"
    )
    
    # Data paths
    parser.add_argument('--islem-tam-path', type=str, default=None,
                       help='Path to islem_tam_tablo.xlsx')
    parser.add_argument('--bold-sure-path', type=str, default=None,
                       help='Path to bold_islem_sure_tablosu.xlsx')
    
    # Pipeline control
    parser.add_argument('--skip-data-loading', action='store_true',
                       help='Skip data loading (use cached datasets)')
    parser.add_argument('--skip-dataset-generation', action='store_true',
                       help='Skip dataset generation (use cached)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (load pre-trained model)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation')
    
    # Dataset generation
    parser.add_argument('--cache-dir', type=str, default='dataset_cache',
                       help='Directory for dataset cache')
    parser.add_argument('--mip-time-limit', type=int, default=config.MIP_TIME_LIMIT,
                       help='MIP solver time limit (seconds)')
    
    # Training
    parser.add_argument('--batch-size', type=int,
                       default=config.TRAINING_CONFIG['batch_size'],
                       help='Training batch size')
    parser.add_argument('--num-epochs', type=int,
                       default=config.TRAINING_CONFIG['num_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--checkpoint-dir', type=str,
                       default=config.TRAINING_CONFIG['checkpoint_dir'],
                       help='Directory for model checkpoints')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (disable CUDA)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting training curves')
    
    # Evaluation
    parser.add_argument('--n-eval-instances', type=int, default=10,
                       help='Number of instances for evaluation')
    parser.add_argument('--eval-min-jobs', type=int, default=3,
                       help='Min jobs for evaluation instances')
    parser.add_argument('--eval-max-jobs', type=int, default=8,
                       help='Max jobs for evaluation instances')
    parser.add_argument('--compare-with-mip', action='store_true',
                       help='Include MIP in evaluation (slow)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
