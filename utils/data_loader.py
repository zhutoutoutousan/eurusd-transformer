"""
Data loader for EUR/USD historical data with multiple timeframes
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from typing import Dict, List, Optional, Tuple
import ta
from datetime import datetime, timedelta
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EURUSDDataLoader:
    """Data loader for EUR/USD forex data"""
    
    def __init__(self, cache_dir: str = "data/cache", processed_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        self.ensure_directories()
        
        # Timeframe mappings
        self.timeframe_mapping = {
            "5m": "5m",
            "15m": "15m", 
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk"
        }
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_data(self, symbol: str = "EURUSD=X", start_date: str = "2020-01-01", 
                     end_date: str = "2024-12-31", timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download EUR/USD data for multiple timeframes
        
        Args:
            symbol: Trading symbol (default: EURUSD=X for Yahoo Finance)
            start_date: Start date for data collection
            end_date: End date for data collection
            timeframes: List of timeframes to download
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h", "1d", "1w"]
        
        data_dict = {}
        
        for timeframe in timeframes:
            logger.info(f"Downloading {timeframe} data for {symbol}")
            
            # Check cache first
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{timeframe}_{start_date}_{end_date}.pkl")
            
            if os.path.exists(cache_file):
                logger.info(f"Loading {timeframe} data from cache")
                with open(cache_file, 'rb') as f:
                    data_dict[timeframe] = pickle.load(f)
                continue
            
            try:
                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                
                # For intraday data, we need to handle differently
                if timeframe in ["5m", "15m", "30m", "1h"]:
                    # Intraday data has limitations, so we'll get daily and resample
                    df = ticker.history(start=start_date, end=end_date, interval="1d")
                    # Resample to desired timeframe (this is a simplified approach)
                    df = self._resample_to_timeframe(df, timeframe)
                else:
                    # Daily and weekly data
                    interval = self.timeframe_mapping[timeframe]
                    df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                # Clean and process data
                df = self._clean_data(df)
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                
                data_dict[timeframe] = df
                logger.info(f"Successfully downloaded {len(df)} records for {timeframe}")
                
            except Exception as e:
                logger.error(f"Error downloading {timeframe} data: {e}")
                continue
        
        return data_dict
    
    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample daily data to smaller timeframes (simplified approach)"""
        # This is a simplified resampling - in practice, you'd want real intraday data
        if timeframe == "1h":
            # Create hourly data by interpolating
            df_resampled = df.resample('1H').interpolate(method='linear')
        elif timeframe == "30m":
            df_resampled = df.resample('30T').interpolate(method='linear')
        elif timeframe == "15m":
            df_resampled = df.resample('15T').interpolate(method='linear')
        elif timeframe == "5m":
            df_resampled = df.resample('5T').interpolate(method='linear')
        else:
            df_resampled = df
        
        return df_resampled.dropna()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data"""
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, creating dummy data")
                df[col] = df['Close'] if col != 'Volume' else 1000000
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Add datetime index if not present
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by datetime
        df = df.sort_index()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        # Make a copy to avoid modifying original
        df_indicators = df.copy()
        
        # Moving Averages
        df_indicators['sma_20'] = ta.trend.sma_indicator(df_indicators['close'], window=20)
        df_indicators['sma_50'] = ta.trend.sma_indicator(df_indicators['close'], window=50)
        df_indicators['ema_12'] = ta.trend.ema_indicator(df_indicators['close'], window=12)
        df_indicators['ema_26'] = ta.trend.ema_indicator(df_indicators['close'], window=26)
        
        # RSI
        df_indicators['rsi_14'] = ta.momentum.rsi(df_indicators['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df_indicators['close'])
        df_indicators['macd'] = macd.macd()
        df_indicators['macd_signal'] = macd.macd_signal()
        df_indicators['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_indicators['close'])
        df_indicators['bb_upper'] = bb.bollinger_hband()
        df_indicators['bb_middle'] = bb.bollinger_mavg()
        df_indicators['bb_lower'] = bb.bollinger_lband()
        df_indicators['bb_width'] = bb.bollinger_wband()
        df_indicators['bb_percent'] = bb.bollinger_pband()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df_indicators['high'], df_indicators['low'], df_indicators['close'])
        df_indicators['stoch_k'] = stoch.stoch()
        df_indicators['stoch_d'] = stoch.stoch_signal()
        
        # ATR (Average True Range)
        df_indicators['atr_14'] = ta.volatility.average_true_range(df_indicators['high'], df_indicators['low'], df_indicators['close'], window=14)
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df_indicators['high'], df_indicators['low'], df_indicators['close'])
        df_indicators['adx_14'] = adx.adx()
        df_indicators['di_plus'] = adx.adx_pos()
        df_indicators['di_minus'] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        df_indicators['cci_20'] = ta.trend.cci(df_indicators['high'], df_indicators['low'], df_indicators['close'], window=20)
        
        # Williams %R
        df_indicators['williams_r'] = ta.momentum.williams_r(df_indicators['high'], df_indicators['low'], df_indicators['close'])
        
        # Money Flow Index
        df_indicators['mfi_14'] = ta.volume.money_flow_index(df_indicators['high'], df_indicators['low'], df_indicators['close'], df_indicators['volume'], window=14)
        
        # Price-based features
        df_indicators['price_change'] = df_indicators['close'].pct_change()
        df_indicators['high_low_ratio'] = df_indicators['high'] / df_indicators['low']
        df_indicators['close_open_ratio'] = df_indicators['close'] / df_indicators['open']
        
        # Volume indicators
        df_indicators['volume_sma'] = ta.volume.volume_sma(df_indicators['close'], df_indicators['volume'], window=20)
        df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma']
        
        # Remove NaN values created by indicators
        df_indicators = df_indicators.dropna()
        
        return df_indicators
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 24, threshold: float = 0.001) -> pd.DataFrame:
        """
        Create trading labels based on future price movements
        
        Args:
            df: DataFrame with price data
            horizon: Number of periods ahead to look
            threshold: Minimum price change threshold for classification
            
        Returns:
            DataFrame with labels added
        """
        df_labeled = df.copy()
        
        # Calculate future price change
        future_price = df_labeled['close'].shift(-horizon)
        price_change_pct = (future_price - df_labeled['close']) / df_labeled['close']
        
        # Create labels: 0 = Hold, 1 = Buy, 2 = Sell
        labels = np.zeros(len(df_labeled))
        labels[price_change_pct > threshold] = 1  # Buy
        labels[price_change_pct < -threshold] = 2  # Sell
        
        df_labeled['label'] = labels
        df_labeled['future_return'] = price_change_pct
        
        # Remove rows where we don't have future data
        df_labeled = df_labeled.dropna()
        
        return df_labeled
    
    def prepare_dataset(self, data_dict: Dict[str, pd.DataFrame], horizon: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Prepare complete dataset with technical indicators and labels
        
        Args:
            data_dict: Dictionary of DataFrames for different timeframes
            horizon: Prediction horizon for labels
            
        Returns:
            Dictionary of prepared DataFrames
        """
        prepared_data = {}
        
        for timeframe, df in data_dict.items():
            logger.info(f"Preparing dataset for {timeframe}")
            
            # Add technical indicators
            df_with_indicators = self.add_technical_indicators(df)
            
            # Create labels
            df_labeled = self.create_labels(df_with_indicators, horizon=horizon)
            
            # Save processed data
            processed_file = os.path.join(self.processed_dir, f"processed_{timeframe}.pkl")
            with open(processed_file, 'wb') as f:
                pickle.dump(df_labeled, f)
            
            prepared_data[timeframe] = df_labeled
            logger.info(f"Prepared {len(df_labeled)} samples for {timeframe}")
        
        return prepared_data
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns (excluding labels and datetime)"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            'stoch_k', 'stoch_d', 'atr_14', 'adx_14', 'di_plus', 'di_minus',
            'cci_20', 'williams_r', 'mfi_14', 'price_change', 'high_low_ratio',
            'close_open_ratio', 'volume_sma', 'volume_ratio'
        ]

def main():
    """Example usage of the data loader"""
    loader = EURUSDDataLoader()
    
    # Download data for all timeframes
    data_dict = loader.download_data(
        symbol="EURUSD=X",
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    
    # Prepare datasets with technical indicators and labels
    prepared_data = loader.prepare_dataset(data_dict, horizon=24)
    
    # Print summary
    for timeframe, df in prepared_data.items():
        print(f"{timeframe}: {len(df)} samples, {len(df.columns)} features")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        print("-" * 50)

if __name__ == "__main__":
    main()
