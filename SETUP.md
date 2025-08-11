# EUR/USD Transformer Setup Guide

This guide will help you set up and run the EUR/USD Transformer trading model.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 10GB free disk space

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd eurusd-transformer
```

2. **Create a virtual environment**
```bash
# Using conda (recommended)
conda create -n eurusd-transformer python=3.9
conda activate eurusd-transformer

# Or using venv
python -m venv eurusd-transformer
source eurusd-transformer/bin/activate  # On Windows: eurusd-transformer\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Download Data
```bash
python scripts/download_data.py --timeframes 1h 1d --start-date 2020-01-01 --end-date 2024-12-31
```

### 2. Run Example
```bash
python scripts/example.py
```

### 3. Train Model
```bash
# Train on 1-hour timeframe
python scripts/train.py --timeframe 1h --num-epochs 100 --save-config

# Train on daily timeframe
python scripts/train.py --timeframe 1d --num-epochs 50 --save-config

# Train with custom parameters
python scripts/train.py --timeframe 1h --d-model 512 --n-heads 16 --batch-size 64 --learning-rate 5e-5
```

### 4. Make Predictions
```bash
# Single prediction
python scripts/predict.py --model-path models/checkpoints/best_model.pth --scaler-path models/checkpoints/scaler.pkl

# Multi-timeframe predictions
python scripts/predict.py --model-path models/checkpoints/best_model.pth --scaler-path models/checkpoints/scaler.pkl --multi-timeframe

# Continuous predictions
python scripts/predict.py --model-path models/checkpoints/best_model.pth --scaler-path models/checkpoints/scaler.pkl --continuous --interval 300
```

## Configuration

### Timeframe-Specific Settings

The model automatically adjusts parameters based on the timeframe:

| Timeframe | Sequence Length | Prediction Horizon | Batch Size |
|-----------|----------------|-------------------|------------|
| 5m        | 1024           | 12 periods        | 64         |
| 15m       | 768            | 8 periods         | 48         |
| 30m       | 512            | 6 periods         | 32         |
| 1h        | 384            | 24 periods        | 24         |
| 1d        | 256            | 7 periods         | 16         |
| 1w        | 128            | 4 periods         | 8          |

### Model Architecture

- **Transformer Layers**: 6 layers with 8 attention heads
- **Model Dimension**: 256 (configurable)
- **Feed-forward Dimension**: 1024
- **Dropout**: 0.1
- **Activation**: GELU

### Technical Indicators

The model uses 30+ technical indicators:
- Moving averages (SMA, EMA, WMA)
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R
- ATR, ADX, CCI
- Volume-based indicators

## Training Tips

### 1. Data Quality
- Use at least 2 years of historical data
- Ensure data quality and handle missing values
- Consider market regime changes

### 2. Hyperparameter Tuning
- Start with default parameters
- Adjust learning rate based on convergence
- Increase batch size if memory allows
- Use early stopping to prevent overfitting

### 3. Model Selection
- Monitor validation loss and accuracy
- Use ensemble methods for better performance
- Consider different timeframes for different strategies

### 4. Training Monitoring
```bash
# Monitor training with TensorBoard
tensorboard --logdir models/checkpoints

# Use Weights & Biases (optional)
python scripts/train.py --use-wandb --wandb-project eurusd-transformer
```

## Evaluation Metrics

The model provides multiple evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each class (HOLD/BUY/SELL)
- **Recall**: Recall for each class
- **F1-Score**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean squared error for return prediction
- **MAE**: Mean absolute error for return prediction

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Production Deployment

### 1. Model Serving
```python
from scripts.predict import EURUSDPredictor

predictor = EURUSDPredictor(
    model_path='models/checkpoints/best_model.pth',
    scaler_path='models/checkpoints/scaler.pkl'
)

prediction = predictor.predict(symbol='EURUSD=X', timeframe='1h')
```

### 2. Real-time Integration
```python
# Continuous prediction loop
while True:
    prediction = predictor.predict()
    # Process prediction
    time.sleep(300)  # 5 minutes
```

### 3. API Integration
```python
# Flask API example
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict/<timeframe>')
def predict(timeframe):
    prediction = predictor.predict(timeframe=timeframe)
    return jsonify(prediction)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce sequence length
   - Use gradient accumulation

2. **Data Download Issues**
   - Check internet connection
   - Verify symbol name
   - Try different date ranges

3. **Training Convergence**
   - Adjust learning rate
   - Increase warmup steps
   - Check data quality

4. **Prediction Errors**
   - Verify model and scaler paths
   - Check data format
   - Ensure consistent preprocessing

### Performance Optimization

1. **GPU Usage**
   - Use mixed precision training
   - Optimize batch size
   - Use gradient checkpointing

2. **Data Loading**
   - Use multiple workers
   - Enable pin memory
   - Use data prefetching

3. **Model Optimization**
   - Use model quantization
   - Implement model pruning
   - Use TensorRT for inference

## Advanced Usage

### Multi-timeframe Training
```bash
# Train multi-timeframe model
python scripts/train.py --model-type multi --timeframes 1h 1d 1w
```

### Custom Technical Indicators
```python
# Add custom indicators in utils/data_loader.py
def add_custom_indicators(self, df):
    # Your custom indicators here
    return df
```

### Ensemble Methods
```python
# Combine multiple models
predictions = []
for model_path in model_paths:
    model = load_model(model_path)
    pred = model.predict(data)
    predictions.append(pred)

ensemble_prediction = np.mean(predictions, axis=0)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always use proper risk management when trading. The authors are not responsible for any financial losses incurred through the use of this software.
