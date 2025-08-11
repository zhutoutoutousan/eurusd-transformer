#!/usr/bin/env python3
"""
Prediction script for EUR/USD Transformer model
"""

import argparse
import sys
import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_loader import EURUSDDataLoader
from utils.training import DataPreprocessor
from models.transformer import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EURUSDPredictor:
    """Predictor class for EUR/USD forecasting"""
    
    def __init__(self, model_path: str, scaler_path: str, config_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to fitted scaler
            config_path: Path to model configuration (optional)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        
        # Load model and scaler
        self.model = None
        self.scaler = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        self._load_scaler()
        
        # Initialize data loader
        self.data_loader = EURUSDDataLoader()
        
    def _load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Get configuration
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = config_dict['model_config']
        else:
            self.config = checkpoint.get('config', {})
        
        # Create model
        self.model = create_model(self.config, model_type='single')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_scaler(self):
        """Load fitted scaler"""
        logger.info(f"Loading scaler from {self.scaler_path}")
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Scaler loaded successfully")
    
    def get_latest_data(self, symbol: str = "EURUSD=X", timeframe: str = "1h", 
                       periods: int = 1000) -> pd.DataFrame:
        """
        Get latest market data
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            periods: Number of periods to fetch
            
        Returns:
            DataFrame with latest market data
        """
        logger.info(f"Fetching latest {timeframe} data for {symbol}")
        
        # Map timeframe to yfinance interval
        interval_map = {
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk"
        }
        
        interval = interval_map.get(timeframe, "1h")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{periods}d", interval=interval)
        
        # Clean data
        df = self.data_loader._clean_data(df)
        
        # Add technical indicators
        df = self.data_loader.add_technical_indicators(df)
        
        logger.info(f"Fetched {len(df)} data points")
        return df
    
    def prepare_sequence(self, data: pd.DataFrame, sequence_length: int = None) -> torch.Tensor:
        """
        Prepare input sequence for prediction
        
        Args:
            data: DataFrame with market data
            sequence_length: Length of input sequence
            
        Returns:
            Tensor ready for model input
        """
        if sequence_length is None:
            sequence_length = self.config.get('max_seq_length', 100)
        
        # Get feature columns
        exclude_cols = ['label', 'future_return']
        feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Get the latest sequence
        sequence_data = data[feature_columns].tail(sequence_length)
        
        # Scale the data
        scaled_data = self.scaler.transform(sequence_data)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)  # Add batch dimension
        
        return sequence_tensor
    
    def predict(self, data: pd.DataFrame = None, symbol: str = "EURUSD=X", 
               timeframe: str = "1h") -> Dict[str, Any]:
        """
        Make prediction
        
        Args:
            data: Optional DataFrame with market data (if None, will fetch latest)
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with prediction results
        """
        # Get data if not provided
        if data is None:
            data = self.get_latest_data(symbol, timeframe)
        
        # Prepare sequence
        sequence = self.prepare_sequence(data)
        sequence = sequence.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sequence)
        
        # Process outputs
        classification_probs = torch.softmax(outputs['classification_logits'], dim=1)
        predicted_class = torch.argmax(classification_probs, dim=1).item()
        confidence = classification_probs.max().item()
        
        predicted_return = outputs['regression_output'].squeeze().cpu().numpy()
        
        # Map class to action
        class_to_action = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action = class_to_action[predicted_class]
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_return)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': predicted_return,
            'action': action,
            'confidence': confidence,
            'class_probabilities': {
                'HOLD': classification_probs[0, 0].item(),
                'BUY': classification_probs[0, 1].item(),
                'SELL': classification_probs[0, 2].item()
            }
        }
        
        return result
    
    def predict_multi_timeframe(self, symbol: str = "EURUSD=X") -> Dict[str, Any]:
        """
        Make predictions for multiple timeframes
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with predictions for all timeframes
        """
        timeframes = ["5m", "15m", "30m", "1h", "1d", "1w"]
        predictions = {}
        
        for timeframe in timeframes:
            try:
                prediction = self.predict(symbol=symbol, timeframe=timeframe)
                predictions[timeframe] = prediction
                logger.info(f"{timeframe}: {prediction['action']} (confidence: {prediction['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"Failed to predict for {timeframe}: {e}")
                predictions[timeframe] = None
        
        return predictions

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make EUR/USD predictions')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--scaler-path', type=str, required=True,
                       help='Path to fitted scaler')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to model configuration')
    
    # Prediction arguments
    parser.add_argument('--symbol', type=str, default='EURUSD=X',
                       help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h',
                       choices=['5m', '15m', '30m', '1h', '1d', '1w'],
                       help='Timeframe for prediction')
    parser.add_argument('--multi-timeframe', action='store_true',
                       help='Make predictions for all timeframes')
    
    # Output arguments
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for predictions')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'csv'],
                       help='Output format')
    
    # Real-time arguments
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous predictions')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval between predictions in seconds')
    
    return parser.parse_args()

def save_predictions(predictions: Dict[str, Any], output_file: str, format: str = 'json'):
    """Save predictions to file"""
    if format == 'json':
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    elif format == 'csv':
        # Flatten predictions for CSV
        flat_data = []
        for timeframe, pred in predictions.items():
            if pred is not None:
                flat_data.append({
                    'timestamp': pred['timestamp'],
                    'timeframe': timeframe,
                    'symbol': pred['symbol'],
                    'current_price': pred['current_price'],
                    'predicted_price': pred['predicted_price'],
                    'predicted_return': pred['predicted_return'],
                    'action': pred['action'],
                    'confidence': pred['confidence'],
                    'hold_prob': pred['class_probabilities']['HOLD'],
                    'buy_prob': pred['class_probabilities']['BUY'],
                    'sell_prob': pred['class_probabilities']['SELL']
                })
        
        df = pd.DataFrame(flat_data)
        df.to_csv(output_file, index=False)
    
    logger.info(f"Predictions saved to {output_file}")

def main():
    """Main prediction function"""
    args = parse_args()
    
    try:
        # Initialize predictor
        predictor = EURUSDPredictor(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            config_path=args.config_path
        )
        
        if args.continuous:
            # Continuous prediction mode
            logger.info(f"Starting continuous predictions every {args.interval} seconds")
            logger.info("Press Ctrl+C to stop")
            
            try:
                while True:
                    if args.multi_timeframe:
                        predictions = predictor.predict_multi_timeframe(args.symbol)
                    else:
                        prediction = predictor.predict(symbol=args.symbol, timeframe=args.timeframe)
                        predictions = {args.timeframe: prediction}
                    
                    # Print results
                    print(f"\n{'='*60}")
                    print(f"Predictions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}")
                    
                    for timeframe, pred in predictions.items():
                        if pred is not None:
                            print(f"{timeframe:>4}: {pred['action']:>4} | "
                                  f"Price: {pred['current_price']:.5f} → {pred['predicted_price']:.5f} | "
                                  f"Confidence: {pred['confidence']:.3f}")
                    
                    # Save if output file specified
                    if args.output_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = f"{args.output_file}_{timestamp}.{args.format}"
                        save_predictions(predictions, output_file, args.format)
                    
                    # Wait for next prediction
                    import time
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                logger.info("Continuous prediction stopped by user")
        
        else:
            # Single prediction mode
            if args.multi_timeframe:
                predictions = predictor.predict_multi_timeframe(args.symbol)
            else:
                prediction = predictor.predict(symbol=args.symbol, timeframe=args.timeframe)
                predictions = {args.timeframe: prediction}
            
            # Print results
            print(f"\n{'='*60}")
            print(f"EUR/USD Predictions")
            print(f"{'='*60}")
            
            for timeframe, pred in predictions.items():
                if pred is not None:
                    print(f"\n{timeframe.upper()} Timeframe:")
                    print(f"  Current Price: {pred['current_price']:.5f}")
                    print(f"  Predicted Price: {pred['predicted_price']:.5f}")
                    print(f"  Predicted Return: {pred['predicted_return']:.4f}")
                    print(f"  Action: {pred['action']}")
                    print(f"  Confidence: {pred['confidence']:.3f}")
                    print(f"  Class Probabilities:")
                    for action, prob in pred['class_probabilities'].items():
                        print(f"    {action}: {prob:.3f}")
            
            # Save if output file specified
            if args.output_file:
                save_predictions(predictions, args.output_file, args.format)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()
