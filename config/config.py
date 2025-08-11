"""
Configuration file for EUR/USD Transformer Trading Model
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data configuration parameters"""
    symbol: str = "EUR/USD"
    timeframes: List[str] = None
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_source: str = "yfinance"  # or "ccxt" for live data
    cache_dir: str = "data/cache"
    processed_dir: str = "data/processed"
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["5m", "15m", "30m", "1h", "1d", "1w"]

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Input features
    price_features: List[str] = None
    technical_indicators: List[str] = None
    feature_dim: int = 50
    
    # Output
    prediction_horizon: int = 24  # hours
    num_classes: int = 3  # Buy, Sell, Hold
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = ["open", "high", "low", "close", "volume"]
        if self.technical_indicators is None:
            self.technical_indicators = [
                "sma_20", "sma_50", "ema_12", "ema_26",
                "rsi_14", "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_middle", "bb_lower",
                "stoch_k", "stoch_d", "atr_14", "adx_14",
                "cci_20", "williams_r", "mfi_14"
            ]

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_dir: str = "models/checkpoints"
    save_every: int = 5
    
    # Logging
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "eurusd-transformer"

@dataclass
class InferenceConfig:
    """Inference configuration parameters"""
    model_path: str = "models/best_model.pth"
    batch_size: int = 1
    use_gpu: bool = True
    confidence_threshold: float = 0.7

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()

# Default configuration
config = Config()

# Timeframe-specific configurations
TIMEFRAME_CONFIGS = {
    "5m": {
        "max_seq_length": 1024,
        "prediction_horizon": 12,  # 1 hour ahead
        "batch_size": 64
    },
    "15m": {
        "max_seq_length": 768,
        "prediction_horizon": 8,   # 2 hours ahead
        "batch_size": 48
    },
    "30m": {
        "max_seq_length": 512,
        "prediction_horizon": 6,   # 3 hours ahead
        "batch_size": 32
    },
    "1h": {
        "max_seq_length": 384,
        "prediction_horizon": 24,  # 1 day ahead
        "batch_size": 24
    },
    "1d": {
        "max_seq_length": 256,
        "prediction_horizon": 7,   # 1 week ahead
        "batch_size": 16
    },
    "1w": {
        "max_seq_length": 128,
        "prediction_horizon": 4,   # 1 month ahead
        "batch_size": 8
    }
}

def get_timeframe_config(timeframe: str) -> Dict[str, Any]:
    """Get configuration for specific timeframe"""
    return TIMEFRAME_CONFIGS.get(timeframe, {})

def update_config_for_timeframe(config: Config, timeframe: str):
    """Update configuration for specific timeframe"""
    tf_config = get_timeframe_config(timeframe)
    
    for key, value in tf_config.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config
