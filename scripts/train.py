#!/usr/bin/env python3
"""
Main training script for EUR/USD Transformer model
"""

import argparse
import sys
import os
import json
import logging
from datetime import datetime
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config, update_config_for_timeframe
from utils.data_loader import EURUSDDataLoader
from utils.training import Trainer, create_data_loaders, evaluate_model, DataPreprocessor
from models.transformer import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train EUR/USD Transformer model')
    
    # Data arguments
    parser.add_argument('--timeframe', type=str, default='1h',
                       choices=['5m', '15m', '30m', '1h', '1d', '1w'],
                       help='Timeframe to train on')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for data collection')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date for data collection')
    parser.add_argument('--symbol', type=str, default='EURUSD=X',
                       help='Trading symbol')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='single',
                       choices=['single', 'multi'],
                       help='Model type: single timeframe or multi-timeframe')
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--max-seq-length', type=int, default=None,
                       help='Maximum sequence length (overrides timeframe config)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides timeframe config)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    # Data preprocessing
    parser.add_argument('--scaler-type', type=str, default='standard',
                       choices=['standard', 'minmax'],
                       help='Type of scaler to use')
    parser.add_argument('--sequence-length', type=int, default=None,
                       help='Sequence length for training (overrides max_seq_length)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--save-config', action='store_true',
                       help='Save configuration to output directory')
    
    # Other arguments
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of data')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to existing model for evaluation')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directory and logging"""
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"eurusd_{args.timeframe}_{args.model_type}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    return experiment_dir

def load_or_download_data(args):
    """Load or download data for training"""
    logger.info("Loading/downloading data...")
    
    data_loader = EURUSDDataLoader()
    
    # Check if processed data exists
    processed_file = os.path.join(data_loader.processed_dir, f"processed_{args.timeframe}.pkl")
    
    if os.path.exists(processed_file) and not args.force_download:
        logger.info(f"Loading processed data from {processed_file}")
        import pickle
        with open(processed_file, 'rb') as f:
            data = pickle.load(f)
    else:
        logger.info("Downloading and processing data...")
        
        # Download raw data
        data_dict = data_loader.download_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframes=[args.timeframe]
        )
        
        if args.timeframe not in data_dict:
            raise ValueError(f"No data available for timeframe {args.timeframe}")
        
        # Prepare dataset with technical indicators and labels
        data = data_loader.prepare_dataset(
            data_dict, 
            horizon=24  # Default horizon, will be updated by config
        )[args.timeframe]
    
    logger.info(f"Data loaded: {len(data)} samples, {len(data.columns)} features")
    logger.info(f"Label distribution: {data['label'].value_counts().to_dict()}")
    
    return data

def create_model_config(args, data):
    """Create model configuration"""
    # Start with default config
    config = Config()
    
    # Update for specific timeframe
    config = update_config_for_timeframe(config, args.timeframe)
    
    # Override with command line arguments
    if args.max_seq_length is not None:
        config.model.max_seq_length = args.max_seq_length
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.sequence_length is not None:
        config.model.max_seq_length = args.sequence_length
    
    # Update other parameters
    config.model.d_model = args.d_model
    config.model.n_heads = args.n_heads
    config.model.n_layers = args.n_layers
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.training.warmup_steps = args.warmup_steps
    config.training.patience = args.patience
    
    # Update feature dimension based on actual data
    exclude_cols = ['label', 'future_return']
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    config.model.feature_dim = len(feature_columns)
    
    # Convert config to dictionary for model creation
    model_config = {
        'd_model': config.model.d_model,
        'n_heads': config.model.n_heads,
        'n_layers': config.model.n_layers,
        'd_ff': config.model.d_ff,
        'dropout': config.model.dropout,
        'max_seq_length': config.model.max_seq_length,
        'feature_dim': config.model.feature_dim,
        'num_classes': config.model.num_classes,
        'prediction_horizon': config.model.prediction_horizon,
        'timeframes': config.data.timeframes
    }
    
    training_config = {
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'num_epochs': config.training.num_epochs,
        'warmup_steps': config.training.warmup_steps,
        'gradient_clip_val': config.training.gradient_clip_val,
        'train_split': config.training.train_split,
        'val_split': config.training.val_split,
        'test_split': config.training.test_split,
        'patience': config.training.patience,
        'save_every': config.training.save_every
    }
    
    return model_config, training_config

def main():
    """Main training function"""
    args = parse_args()
    
    # Setup experiment
    experiment_dir = setup_experiment(args)
    
    try:
        # Load or download data
        data = load_or_download_data(args)
        
        # Create configurations
        model_config, training_config = create_model_config(args, data)
        
        # Save configuration if requested
        if args.save_config:
            config_save_path = os.path.join(experiment_dir, 'config.json')
            config_dict = {
                'model_config': model_config,
                'training_config': training_config,
                'args': vars(args)
            }
            with open(config_save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_save_path}")
        
        # Create data preprocessor
        preprocessor = DataPreprocessor(scaler_type=args.scaler_type)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data, training_config, preprocessor
        )
        
        # Save preprocessor
        scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
        preprocessor.save_scaler(scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        if args.evaluate_only:
            # Load existing model for evaluation
            if args.model_path is None:
                args.model_path = os.path.join(experiment_dir, 'best_model.pth')
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found: {args.model_path}")
            
            logger.info(f"Loading model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location='cpu')
            model = create_model(checkpoint['config'], args.model_type)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = evaluate_model(model, test_loader, args.device)
            
            # Save evaluation results
            eval_path = os.path.join(experiment_dir, 'evaluation_results.json')
            with open(eval_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Evaluation results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return
        
        # Create model
        logger.info("Creating model...")
        model = create_model(model_config, args.model_type)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(model, training_config, device=args.device)
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader, experiment_dir)
        
        # Evaluate final model
        logger.info("Evaluating final model...")
        final_metrics = evaluate_model(model, test_loader, args.device)
        
        # Save evaluation results
        eval_path = os.path.join(experiment_dir, 'final_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save training history
        history_path = os.path.join(experiment_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info("Final evaluation results:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save model summary
        summary_path = os.path.join(experiment_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"EUR/USD Transformer Model Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Timeframe: {args.timeframe}\n")
            f.write(f"Model Type: {args.model_type}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Training Samples: {len(train_loader.dataset)}\n")
            f.write(f"Validation Samples: {len(val_loader.dataset)}\n")
            f.write(f"Test Samples: {len(test_loader.dataset)}\n\n")
            f.write(f"Final Metrics:\n")
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        logger.info(f"All results saved to {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
