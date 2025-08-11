"""
Training utilities for EUR/USD Transformer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from tqdm import tqdm
import pickle
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EURUSDDataset(Dataset):
    """Dataset class for EUR/USD time series data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 100, feature_columns: List[str] = None):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with features and labels
            sequence_length: Length of input sequences
            feature_columns: List of feature column names
        """
        self.data = data
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        
        if self.feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['label', 'future_return', 'datetime']
            self.feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare sequences
        self.sequences = []
        self.labels = []
        self.returns = []
        
        for i in range(len(data) - sequence_length):
            # Extract sequence
            sequence = data[self.feature_columns].iloc[i:i+sequence_length].values
            label = data['label'].iloc[i+sequence_length-1]
            future_return = data['future_return'].iloc[i+sequence_length-1]
            
            # Skip if any NaN values
            if not np.isnan(sequence).any() and not np.isnan(label) and not np.isnan(future_return):
                self.sequences.append(sequence)
                self.labels.append(int(label))
                self.returns.append(future_return)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return_value = torch.FloatTensor([self.returns[idx]])
        
        return {
            'sequence': sequence,
            'label': label.squeeze(),
            'return': return_value.squeeze()
        }

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        
    def fit_transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Fit scaler and transform data"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Fit and transform features
        scaled_features = self.scaler.fit_transform(data[feature_columns])
        
        # Create new DataFrame with scaled features
        data_scaled = data.copy()
        data_scaled[feature_columns] = scaled_features
        
        return data_scaled
    
    def transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        scaled_features = self.scaler.transform(data[feature_columns])
        
        data_scaled = data.copy()
        data_scaled[feature_columns] = scaled_features
        
        return data_scaled
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler"""
        if self.scaler is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_scaler(self, filepath: str):
        """Load fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)

class TrainingMetrics:
    """Training metrics tracking"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_mse = []
        self.val_mse = []
        self.learning_rates = []
    
    def update(self, train_loss: float, val_loss: float, train_acc: float, 
               val_acc: float, train_mse: float, val_mse: float, lr: float):
        """Update metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE
        axes[1, 0].plot(self.train_mse, label='Train MSE')
        axes[1, 0].plot(self.val_mse, label='Val MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.learning_rates, label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class Trainer:
    """Main trainer class for EUR/USD transformer"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'auto'):
        """
        Initialize trainer
        
        Args:
            model: Transformer model
            config: Training configuration
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model = model
        self.config = config
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def _get_device(self, device: str) -> torch.device:
        """Get device for training"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            total_iters=self.config['warmup_steps']
        )
        
        # Main scheduler
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'] - self.config['warmup_steps'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        return {'warmup': warmup_scheduler, 'main': main_scheduler}
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        all_predictions = []
        all_labels = []
        all_returns = []
        all_predicted_returns = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].to(self.device)
            returns = batch['return'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate losses
            classification_loss = self.classification_loss(
                outputs['classification_logits'], labels
            )
            regression_loss = self.regression_loss(
                outputs['regression_output'].squeeze(), returns
            )
            
            # Combined loss
            loss = classification_loss + 0.1 * regression_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_val']
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()
            
            # Collect predictions
            pred_labels = torch.argmax(outputs['classification_logits'], dim=1)
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_returns.extend(returns.cpu().numpy())
            all_predicted_returns.extend(outputs['regression_output'].squeeze().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls_loss': f"{classification_loss.item():.4f}",
                'reg_loss': f"{regression_loss.item():.4f}"
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_classification_loss / len(train_loader)
        avg_reg_loss = total_regression_loss / len(train_loader)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        mse = mean_squared_error(all_returns, all_predicted_returns)
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_cls_loss,
            'regression_loss': avg_reg_loss,
            'accuracy': accuracy,
            'mse': mse
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        all_predictions = []
        all_labels = []
        all_returns = []
        all_predicted_returns = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move data to device
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                returns = batch['return'].to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate losses
                classification_loss = self.classification_loss(
                    outputs['classification_logits'], labels
                )
                regression_loss = self.regression_loss(
                    outputs['regression_output'].squeeze(), returns
                )
                
                # Combined loss
                loss = classification_loss + 0.1 * regression_loss
                
                # Update metrics
                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_regression_loss += regression_loss.item()
                
                # Collect predictions
                pred_labels = torch.argmax(outputs['classification_logits'], dim=1)
                all_predictions.extend(pred_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                all_predicted_returns.extend(outputs['regression_output'].squeeze().cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cls_loss': f"{classification_loss.item():.4f}",
                    'reg_loss': f"{regression_loss.item():.4f}"
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        avg_cls_loss = total_classification_loss / len(val_loader)
        avg_reg_loss = total_regression_loss / len(val_loader)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        mse = mean_squared_error(all_returns, all_predicted_returns)
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_cls_loss,
            'regression_loss': avg_reg_loss,
            'accuracy': accuracy,
            'mse': mse
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              save_dir: str = "models/checkpoints") -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Update learning rate
            if epoch < self.config['warmup_steps']:
                self.scheduler['warmup'].step()
            else:
                self.scheduler['main'].step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update metrics tracking
            self.metrics.update(
                train_metrics['loss'], val_metrics['loss'],
                train_metrics['accuracy'], val_metrics['accuracy'],
                train_metrics['mse'], val_metrics['mse'],
                current_lr
            )
            
            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                
                logger.info(f"New best model saved with validation loss: {val_metrics['loss']:.4f}")
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Plot training metrics
        self.metrics.plot_metrics(os.path.join(save_dir, 'training_metrics.png'))
        
        return {
            'train_losses': self.metrics.train_losses,
            'val_losses': self.metrics.val_losses,
            'train_accuracies': self.metrics.train_accuracies,
            'val_accuracies': self.metrics.val_accuracies,
            'train_mse': self.metrics.train_mse,
            'val_mse': self.metrics.val_mse,
            'learning_rates': self.metrics.learning_rates
        }

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'auto') -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_returns = []
    all_predicted_returns = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            returns = batch['return'].to(device)
            
            outputs = model(sequences)
            
            pred_labels = torch.argmax(outputs['classification_logits'], dim=1)
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_returns.extend(returns.cpu().numpy())
            all_predicted_returns.extend(outputs['regression_output'].squeeze().cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    mse = mean_squared_error(all_returns, all_predicted_returns)
    mae = mean_absolute_error(all_returns, all_predicted_returns)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mse': mse,
        'mae': mae
    }

def create_data_loaders(data: pd.DataFrame, config: Dict[str, Any], 
                       preprocessor: DataPreprocessor = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data: DataFrame with features and labels
        config: Configuration dictionary
        preprocessor: Optional data preprocessor
        
    Returns:
        Tuple of train, validation, and test data loaders
    """
    # Get feature columns
    exclude_cols = ['label', 'future_return']
    feature_columns = [col for col in data.columns if col not in exclude_cols]
    
    # Preprocess data if preprocessor is provided
    if preprocessor is not None:
        data = preprocessor.fit_transform(data, feature_columns)
    
    # Create dataset
    dataset = EURUSDDataset(
        data=data,
        sequence_length=config['max_seq_length'],
        feature_columns=feature_columns
    )
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
