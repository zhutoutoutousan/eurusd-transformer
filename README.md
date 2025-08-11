# EUR/USD Transformer Trading Model

A comprehensive transformer-based trading system for EUR/USD forex data across multiple timeframes.

## Features

- **Multi-timeframe Analysis**: Supports 5min, 15min, 30min, 1hour, daily, and weekly charts
- **Transformer Architecture**: State-of-the-art transformer model for time series forecasting
- **Technical Indicators**: Comprehensive set of technical analysis indicators
- **Data Pipeline**: Automated data collection and preprocessing
- **Training Pipeline**: Complete training and evaluation framework
- **Real-time Predictions**: Live prediction capabilities

## Timeframes Supported

- 5 minutes (5m)
- 15 minutes (15m) 
- 30 minutes (30m)
- 1 hour (1h)
- Daily (1d)
- Weekly (1w)

## Project Structure

```
eurusd-transformer/
├── data/                   # Data storage
├── models/                 # Model definitions
├── utils/                  # Utility functions
├── config/                 # Configuration files
├── notebooks/              # Jupyter notebooks
├── scripts/                # Training and inference scripts
└── requirements.txt        # Dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download historical data:
```bash
python scripts/download_data.py
```

3. Train the model:
```bash
python scripts/train.py --timeframe 1h
```

4. Make predictions:
```bash
python scripts/predict.py --timeframe 1h
```

## Model Architecture

The transformer model includes:
- Multi-head self-attention mechanism
- Positional encoding for time series
- Technical indicator embeddings
- Multi-timeframe fusion
- Output heads for price prediction and trend classification

## Technical Indicators

- Moving averages (SMA, EMA, WMA)
- RSI, MACD, Bollinger Bands
- Stochastic, Williams %R
- ATR, ADX, CCI
- Volume-based indicators
- Custom forex-specific indicators

## License

MIT License
