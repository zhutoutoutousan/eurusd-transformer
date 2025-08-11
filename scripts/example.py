#!/usr/bin/env python3
"""
Example script demonstrating EUR/USD Transformer usage
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config, update_config_for_timeframe
from utils.data_loader import EURUSDDataLoader
from utils.training import Trainer, create_data_loaders, evaluate_model, DataPreprocessor
from models.transformer import create_model

def main():
    """Main example function"""
    print("EUR/USD Transformer Example")
    print("=" * 50)
    
    # 1. Download and prepare data
    print("\n1. Downloading and preparing data...")
    data_loader = EURUSDDataLoader()
    
    # Download data for 1-hour timeframe
    data_dict = data_loader.download_data(
        symbol="EURUSD=X",
        start_date="2020-01-01",
        end_date="2024-12-31",
        timeframes=["1h"]
    )
    
    # Prepare dataset
    prepared_data = data_loader.prepare_dataset(data_dict, horizon=24)
    data = prepared_data["1h"]
    
    print(f"Data shape: {data.shape}")
    print(f"Features: {len(data.columns)}")
    print(f"Label distribution: {data['label'].value_counts().to_dict()}")
    
    # 2. Create model configuration
    print("\n2. Creating model configuration...")
    config = Config()
    config = update_config_for_timeframe(config, "1h")
    
    # Update feature dimension
    exclude_cols = ['label', 'future_return']
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    config.model.feature_dim = len(feature_columns)
    
    print(f"Model dimension: {config.model.d_model}")
    print(f"Feature dimension: {config.model.feature_dim}")
    print(f"Sequence length: {config.model.max_seq_length}")
    
    # 3. Create data loaders
    print("\n3. Creating data loaders...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data, 
        {
            'batch_size': config.training.batch_size,
            'train_split': config.training.train_split,
            'val_split': config.training.val_split,
            'test_split': config.training.test_split
        }, 
        preprocessor
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 4. Create and train model
    print("\n4. Creating and training model...")
    model_config = {
        'd_model': config.model.d_model,
        'n_heads': config.model.n_heads,
        'n_layers': config.model.n_layers,
        'd_ff': config.model.d_ff,
        'dropout': config.model.dropout,
        'max_seq_length': config.model.max_seq_length,
        'feature_dim': config.model.feature_dim,
        'num_classes': config.model.num_classes,
        'prediction_horizon': config.model.prediction_horizon
    }
    
    model = create_model(model_config, 'single')
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    training_config = {
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'num_epochs': 10,  # Small number for demo
        'warmup_steps': config.training.warmup_steps,
        'gradient_clip_val': config.training.gradient_clip_val,
        'patience': config.training.patience,
        'save_every': config.training.save_every
    }
    
    trainer = Trainer(model, training_config, device='auto')
    
    # Train model
    print("Training model (10 epochs for demo)...")
    history = trainer.train(train_loader, val_loader, 'models/checkpoints')
    
    # 5. Evaluate model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, test_loader, 'auto')
    
    print("Test Set Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 6. Make prediction
    print("\n6. Making prediction...")
    model.eval()
    
    # Get a sample from test set
    sample_batch = next(iter(test_loader))
    sample_sequence = sample_batch['sequence'][:1]  # Take first sample
    
    with torch.no_grad():
        outputs = model(sample_sequence)
    
    # Process results
    classification_probs = torch.softmax(outputs['classification_logits'], dim=1)
    predicted_class = torch.argmax(classification_probs, dim=1).item()
    confidence = classification_probs.max().item()
    predicted_return = outputs['regression_output'].squeeze().cpu().numpy()
    
    # Map class to action
    class_to_action = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action = class_to_action[predicted_class]
    
    print(f"Prediction Results:")
    print(f"  Action: {action}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Predicted Return: {predicted_return:.4f}")
    print(f"  Class Probabilities:")
    print(f"    HOLD: {classification_probs[0, 0].item():.3f}")
    print(f"    BUY:  {classification_probs[0, 1].item():.3f}")
    print(f"    SELL: {classification_probs[0, 2].item():.3f}")
    
    print("\nExample completed successfully!")
    print("\nTo run full training:")
    print("python scripts/train.py --timeframe 1h --num-epochs 100")
    print("\nTo make predictions:")
    print("python scripts/predict.py --model-path models/checkpoints/best_model.pth --scaler-path models/checkpoints/scaler.pkl")

if __name__ == "__main__":
    main()
