"""
Transformer model for EUR/USD time series forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any

class PositionalEncoding(nn.Module):
    """Positional encoding for time series data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """Feed-forward network with residual connection"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TechnicalIndicatorEmbedding(nn.Module):
    """Embedding layer for technical indicators"""
    
    def __init__(self, feature_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class EURUSDTransformer(nn.Module):
    """Main transformer model for EUR/USD forecasting"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Model parameters
        self.d_model = config.get('d_model', 256)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.feature_dim = config.get('feature_dim', 50)
        self.num_classes = config.get('num_classes', 3)
        self.prediction_horizon = config.get('prediction_horizon', 24)
        
        # Embeddings
        self.feature_embedding = TechnicalIndicatorEmbedding(
            self.feature_dim, self.d_model, self.dropout
        )
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.prediction_horizon)
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary with classification and regression outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Output heads
        classification_logits = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        
        return {
            'classification_logits': classification_logits,
            'regression_output': regression_output,
            'hidden_states': x
        }

class MultiTimeframeTransformer(nn.Module):
    """Transformer model that can handle multiple timeframes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.timeframes = config.get('timeframes', ['5m', '15m', '30m', '1h', '1d', '1w'])
        self.d_model = config.get('d_model', 256)
        self.fusion_dim = config.get('fusion_dim', 512)
        
        # Individual transformers for each timeframe
        self.timeframe_transformers = nn.ModuleDict({
            tf: EURUSDTransformer(config) for tf in self.timeframes
        })
        
        # Fusion layer for combining multiple timeframes
        self.fusion_layer = nn.Sequential(
            nn.Linear(len(self.timeframes) * self.d_model, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.fusion_dim, self.d_model)
        )
        
        # Final output heads
        self.final_classification_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.d_model // 2, config.get('num_classes', 3))
        )
        
        self.final_regression_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.d_model // 2, config.get('prediction_horizon', 24))
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple timeframes
        
        Args:
            inputs: Dictionary with timeframe as key and input tensor as value
            
        Returns:
            Dictionary with final outputs
        """
        timeframe_outputs = {}
        
        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe in inputs:
                x = inputs[timeframe]
                outputs = self.timeframe_transformers[timeframe](x)
                timeframe_outputs[timeframe] = outputs
        
        # Combine timeframe outputs
        if timeframe_outputs:
            # Extract pooled representations from each timeframe
            pooled_outputs = []
            for tf in self.timeframes:
                if tf in timeframe_outputs:
                    # Use the last hidden state for pooling
                    hidden_states = timeframe_outputs[tf]['hidden_states']
                    pooled = torch.mean(hidden_states, dim=1)  # (batch_size, d_model)
                    pooled_outputs.append(pooled)
                else:
                    # Create zero tensor for missing timeframes
                    batch_size = next(iter(timeframe_outputs.values()))['hidden_states'].size(0)
                    pooled_outputs.append(torch.zeros(batch_size, self.d_model, device=next(iter(timeframe_outputs.values()))['hidden_states'].device))
            
            # Concatenate and fuse
            combined = torch.cat(pooled_outputs, dim=1)  # (batch_size, n_timeframes * d_model)
            fused = self.fusion_layer(combined)
            
            # Final outputs
            final_classification = self.final_classification_head(fused)
            final_regression = self.final_regression_head(fused)
            
            return {
                'classification_logits': final_classification,
                'regression_output': final_regression,
                'timeframe_outputs': timeframe_outputs,
                'fused_representation': fused
            }
        else:
            raise ValueError("No valid timeframe inputs provided")

def create_model(config: Dict[str, Any], model_type: str = "single") -> nn.Module:
    """
    Factory function to create transformer model
    
    Args:
        config: Model configuration
        model_type: "single" for single timeframe or "multi" for multi-timeframe
        
    Returns:
        Transformer model
    """
    if model_type == "single":
        return EURUSDTransformer(config)
    elif model_type == "multi":
        return MultiTimeframeTransformer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # Test the model
    config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_length': 512,
        'feature_dim': 50,
        'num_classes': 3,
        'prediction_horizon': 24,
        'timeframes': ['5m', '15m', '30m', '1h', '1d', '1w']
    }
    
    # Single timeframe model
    model = create_model(config, "single")
    batch_size, seq_len, feature_dim = 32, 100, 50
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    outputs = model(x)
    print(f"Classification output shape: {outputs['classification_logits'].shape}")
    print(f"Regression output shape: {outputs['regression_output'].shape}")
    
    # Multi-timeframe model
    multi_model = create_model(config, "multi")
    inputs = {
        '1h': torch.randn(batch_size, seq_len, feature_dim),
        '1d': torch.randn(batch_size, seq_len, feature_dim)
    }
    
    multi_outputs = multi_model(inputs)
    print(f"Multi-timeframe classification shape: {multi_outputs['classification_logits'].shape}")
    print(f"Multi-timeframe regression shape: {multi_outputs['regression_output'].shape}")
