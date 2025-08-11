#!/usr/bin/env python3
"""
Data download script for EUR/USD historical data
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import EURUSDDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download EUR/USD historical data')
    
    # Data arguments
    parser.add_argument('--symbol', type=str, default='EURUSD=X',
                       help='Trading symbol')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for data collection')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date for data collection')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['5m', '15m', '30m', '1h', '1d', '1w'],
                       help='Timeframes to download')
    
    # Processing arguments
    parser.add_argument('--horizon', type=int, default=24,
                       help='Prediction horizon for labels')
    parser.add_argument('--threshold', type=float, default=0.001,
                       help='Threshold for classification labels')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for data')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if data exists')
    
    return parser.parse_args()

def main():
    """Main data download function"""
    args = parse_args()
    
    try:
        logger.info("Starting EUR/USD data download and processing")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Timeframes: {args.timeframes}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        
        # Initialize data loader
        data_loader = EURUSDDataLoader(
            cache_dir=os.path.join(args.output_dir, 'cache'),
            processed_dir=os.path.join(args.output_dir, 'processed')
        )
        
        # Download raw data
        logger.info("Downloading raw data...")
        data_dict = data_loader.download_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframes=args.timeframes
        )
        
        # Print data summary
        logger.info("Raw data summary:")
        for timeframe, df in data_dict.items():
            logger.info(f"  {timeframe}: {len(df)} records")
        
        # Prepare datasets with technical indicators and labels
        logger.info("Processing data with technical indicators and labels...")
        prepared_data = data_loader.prepare_dataset(
            data_dict, 
            horizon=args.horizon
        )
        
        # Print processed data summary
        logger.info("Processed data summary:")
        for timeframe, df in prepared_data.items():
            label_dist = df['label'].value_counts().to_dict()
            logger.info(f"  {timeframe}: {len(df)} samples")
            logger.info(f"    Labels: {label_dist}")
            logger.info(f"    Features: {len(df.columns)}")
        
        # Save data summary
        summary_file = os.path.join(args.output_dir, 'data_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("EUR/USD Data Summary\n")
            f.write("===================\n\n")
            f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Symbol: {args.symbol}\n")
            f.write(f"Date Range: {args.start_date} to {args.end_date}\n")
            f.write(f"Prediction Horizon: {args.horizon}\n")
            f.write(f"Classification Threshold: {args.threshold}\n\n")
            
            f.write("Processed Data Summary:\n")
            for timeframe, df in prepared_data.items():
                label_dist = df['label'].value_counts().to_dict()
                f.write(f"\n{timeframe.upper()}:\n")
                f.write(f"  Samples: {len(df)}\n")
                f.write(f"  Features: {len(df.columns)}\n")
                f.write(f"  Labels: {label_dist}\n")
        
        logger.info(f"Data summary saved to {summary_file}")
        logger.info("Data download and processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        raise

if __name__ == "__main__":
    main()
