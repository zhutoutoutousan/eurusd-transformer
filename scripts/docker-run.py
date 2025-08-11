#!/usr/bin/env python3
"""
Docker Compose management script for EUR/USD Transformer
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    
    return result


def build_image():
    """Build the Docker image"""
    print("Building Docker image...")
    run_command("docker-compose build")


def start_services(profile=None, production=False):
    """Start services with specified profile"""
    compose_file = "docker-compose.prod.yml" if production else "docker-compose.yml"
    
    if profile:
        print(f"Starting services with profile: {profile}")
        run_command(f"docker-compose -f {compose_file} --profile {profile} up -d")
    else:
        print("Starting default services...")
        run_command(f"docker-compose -f {compose_file} up -d")


def stop_services(production=False):
    """Stop all services"""
    compose_file = "docker-compose.prod.yml" if production else "docker-compose.yml"
    print("Stopping services...")
    run_command(f"docker-compose -f {compose_file} down")


def download_data():
    """Download EUR/USD data"""
    print("Downloading EUR/USD data...")
    run_command("docker-compose --profile data up download-data")


def train_model(timeframe="1h", epochs=100, batch_size=32, lr=0.001, production=False):
    """Train the model with specified parameters"""
    compose_file = "docker-compose.prod.yml" if production else "docker-compose.yml"
    
    # Override the training command with custom parameters
    cmd = f"python scripts/train.py --timeframe {timeframe} --num-epochs {epochs} --batch-size {batch_size} --learning-rate {lr}"
    
    print(f"Training model with parameters: timeframe={timeframe}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Use docker-compose run for one-off commands
    run_command(f"docker-compose -f {compose_file} run --rm training {cmd}")


def run_prediction(timeframe="1h", continuous=False, interval=300, production=False):
    """Run prediction service"""
    compose_file = "docker-compose.prod.yml" if production else "docker-compose.yml"
    
    cmd = f"python scripts/predict.py --timeframe {timeframe}"
    if continuous:
        cmd += f" --continuous --interval {interval}"
    
    print(f"Running prediction: {cmd}")
    run_command(f"docker-compose -f {compose_file} run --rm prediction {cmd}")


def start_jupyter():
    """Start Jupyter notebook service"""
    print("Starting Jupyter notebook service...")
    run_command("docker-compose --profile jupyter up -d jupyter")
    print("Jupyter Lab is available at: http://localhost:8888")


def start_tensorboard():
    """Start TensorBoard service"""
    print("Starting TensorBoard service...")
    run_command("docker-compose --profile monitoring up -d tensorboard")
    print("TensorBoard is available at: http://localhost:6006")


def show_logs(service="eurusd-transformer", follow=False):
    """Show logs for a specific service"""
    cmd = f"docker-compose logs"
    if follow:
        cmd += " -f"
    cmd += f" {service}"
    run_command(cmd, check=False)


def clean_up():
    """Clean up Docker resources"""
    print("Cleaning up Docker resources...")
    run_command("docker-compose down -v --remove-orphans")
    run_command("docker system prune -f")


def main():
    parser = argparse.ArgumentParser(description="Docker Compose management for EUR/USD Transformer")
    parser.add_argument("action", choices=[
        "build", "start", "stop", "download", "train", "predict", 
        "jupyter", "tensorboard", "logs", "clean", "restart"
    ], help="Action to perform")
    
    parser.add_argument("--profile", choices=["training", "data", "prediction", "jupyter", "monitoring"],
                       help="Docker Compose profile to use")
    parser.add_argument("--production", action="store_true", help="Use production configuration")
    parser.add_argument("--timeframe", default="1h", help="Timeframe for training/prediction")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--continuous", action="store_true", help="Run continuous prediction")
    parser.add_argument("--interval", type=int, default=300, help="Prediction interval in seconds")
    parser.add_argument("--service", default="eurusd-transformer", help="Service name for logs")
    parser.add_argument("--follow", action="store_true", help="Follow logs")
    
    args = parser.parse_args()
    
    try:
        if args.action == "build":
            build_image()
        elif args.action == "start":
            start_services(args.profile, args.production)
        elif args.action == "stop":
            stop_services(args.production)
        elif args.action == "download":
            download_data()
        elif args.action == "train":
            train_model(args.timeframe, args.epochs, args.batch_size, args.learning_rate, args.production)
        elif args.action == "predict":
            run_prediction(args.timeframe, args.continuous, args.interval, args.production)
        elif args.action == "jupyter":
            start_jupyter()
        elif args.action == "tensorboard":
            start_tensorboard()
        elif args.action == "logs":
            show_logs(args.service, args.follow)
        elif args.action == "clean":
            clean_up()
        elif args.action == "restart":
            stop_services(args.production)
            start_services(args.profile, args.production)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
