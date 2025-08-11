# Docker Setup for EUR/USD Transformer

This document provides comprehensive instructions for running the EUR/USD Transformer project using Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for Docker
- 10GB+ free disk space

## Quick Start

### 1. Build the Docker Image

```bash
# Build the base image
docker-compose build

# Or use the management script
python scripts/docker-run.py build
```

### 2. Download Data

```bash
# Download EUR/USD data for all timeframes
docker-compose --profile data up download-data

# Or use the management script
python scripts/docker-run.py download
```

### 3. Start the Application

```bash
# Start the main application
docker-compose up -d

# Or use the management script
python scripts/docker-run.py start
```

## Service Profiles

The Docker Compose setup includes several profiles for different use cases:

### Default Profile
- **eurusd-transformer**: Main application service

### Training Profile
- **training**: Model training service with GPU support

### Data Profile
- **download-data**: Data download and preprocessing service

### Prediction Profile
- **prediction**: Real-time prediction service

### Jupyter Profile
- **jupyter**: Jupyter Lab for interactive development

### Monitoring Profile
- **tensorboard**: TensorBoard for training visualization

## Usage Examples

### Training Models

```bash
# Train with default parameters (1h timeframe)
docker-compose --profile training up training

# Train with custom parameters
docker-compose run --rm training python scripts/train.py \
  --timeframe 15m \
  --num-epochs 200 \
  --batch-size 64 \
  --learning-rate 0.0005

# Or use the management script
python scripts/docker-run.py train --timeframe 15m --epochs 200 --batch-size 64
```

### Running Predictions

```bash
# Single prediction
docker-compose run --rm prediction python scripts/predict.py --timeframe 1h

# Continuous prediction
docker-compose run --rm prediction python scripts/predict.py \
  --timeframe 1h \
  --continuous \
  --interval 300

# Or use the management script
python scripts/docker-run.py predict --timeframe 1h --continuous
```

### Interactive Development

```bash
# Start Jupyter Lab
docker-compose --profile jupyter up -d jupyter
# Access at: http://localhost:8888

# Or use the management script
python scripts/docker-run.py jupyter
```

### Monitoring Training

```bash
# Start TensorBoard
docker-compose --profile monitoring up -d tensorboard
# Access at: http://localhost:6006

# Or use the management script
python scripts/docker-run.py tensorboard
```

## Configuration Files

### Development vs Production

- **docker-compose.yml**: Development configuration
- **docker-compose.prod.yml**: Production configuration with resource limits
- **docker-compose.override.yml**: Development overrides (auto-applied)

### Environment Variables

Key environment variables you can customize:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (set to -1 to disable)

# Python Configuration
PYTHONPATH=/app
PYTHONUNBUFFERED=1

# Model Configuration
MODEL_TYPE=single  # or multi
TIMEFRAME=1h
```

## Volume Mounts

The following directories are mounted as volumes:

- `./data` → `/app/data`: Data storage
- `./models` → `/app/models`: Model checkpoints
- `./logs` → `/app/logs`: Training logs
- `./config` → `/app/config`: Configuration files
- `./notebooks` → `/app/notebooks`: Jupyter notebooks

## Resource Management

### Development (Default)
- No resource limits
- GPU disabled by default
- Source code mounted for development

### Production
- Memory limits: 2-8GB per service
- CPU limits: 0.5-4 cores per service
- GPU enabled
- Read-only config mounts

## Management Script

Use the `scripts/docker-run.py` script for easier management:

```bash
# Show help
python scripts/docker-run.py --help

# Common operations
python scripts/docker-run.py build          # Build image
python scripts/docker-run.py start          # Start services
python scripts/docker-run.py stop           # Stop services
python scripts/docker-run.py download       # Download data
python scripts/docker-run.py train          # Train model
python scripts/docker-run.py predict        # Run predictions
python scripts/docker-run.py jupyter        # Start Jupyter
python scripts/docker-run.py tensorboard    # Start TensorBoard
python scripts/docker-run.py logs --follow  # Follow logs
python scripts/docker-run.py clean          # Clean up
```

## Multi-timeframe Training

Train models for different timeframes:

```bash
# 5-minute data
python scripts/docker-run.py train --timeframe 5m --epochs 150

# 15-minute data
python scripts/docker-run.py train --timeframe 15m --epochs 120

# 30-minute data
python scripts/docker-run.py train --timeframe 30m --epochs 100

# 1-hour data
python scripts/docker-run.py train --timeframe 1h --epochs 80

# Daily data
python scripts/docker-run.py train --timeframe 1d --epochs 50

# Weekly data
python scripts/docker-run.py train --timeframe 1wk --epochs 30
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/docker-run.py train --batch-size 16
   
   # Or use CPU only
   export CUDA_VISIBLE_DEVICES=-1
   ```

2. **Port Conflicts**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :6006
   
   # Use different ports in docker-compose.yml
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER data/ models/ logs/
   ```

4. **Build Failures**
   ```bash
   # Clean and rebuild
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

### Logs and Debugging

```bash
# View logs
docker-compose logs eurusd-transformer

# Follow logs
docker-compose logs -f eurusd-transformer

# Execute commands in container
docker-compose exec eurusd-transformer bash

# Check container status
docker-compose ps
```

## Production Deployment

For production deployment, use the production configuration:

```bash
# Use production config
python scripts/docker-run.py start --production

# Train with production settings
python scripts/docker-run.py train --production --timeframe 1h

# Run prediction service
python scripts/docker-run.py predict --production --continuous
```

### Production Considerations

- Use named volumes for data persistence
- Set appropriate resource limits
- Enable GPU acceleration
- Use restart policies
- Monitor resource usage
- Set up logging aggregation
- Use secrets management for API keys

## Performance Optimization

### GPU Acceleration

```bash
# Enable GPU (requires nvidia-docker)
export CUDA_VISIBLE_DEVICES=0
docker-compose up -d

# Check GPU usage
nvidia-smi
```

### Memory Optimization

```bash
# Adjust batch sizes based on available memory
python scripts/docker-run.py train --batch-size 16  # For 4GB RAM
python scripts/docker-run.py train --batch-size 32  # For 8GB RAM
python scripts/docker-run.py train --batch-size 64  # For 16GB+ RAM
```

### Storage Optimization

```bash
# Use data compression
docker-compose run --rm download-data python scripts/download_data.py --compress

# Clean up old data
docker-compose run --rm eurusd-transformer python -c "
import shutil
shutil.rmtree('data/cache/old_data', ignore_errors=True)
"
```

## Security Considerations

- Never commit API keys to version control
- Use Docker secrets for sensitive data
- Run containers as non-root users
- Regularly update base images
- Scan images for vulnerabilities
- Use read-only file systems where possible

## Monitoring and Logging

### Health Checks

```bash
# Check service health
docker-compose ps

# Monitor resource usage
docker stats
```

### Log Management

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs training

# Export logs
docker-compose logs > logs.txt
```

## Backup and Recovery

### Data Backup

```bash
# Backup data volumes
docker run --rm -v eurusd_data:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz -C /data .

# Backup models
docker run --rm -v eurusd_models:/models -v $(pwd):/backup alpine tar czf /backup/models_backup.tar.gz -C /models .
```

### Restore Data

```bash
# Restore data
docker run --rm -v eurusd_data:/data -v $(pwd):/backup alpine tar xzf /backup/data_backup.tar.gz -C /data

# Restore models
docker run --rm -v eurusd_models:/models -v $(pwd):/backup alpine tar xzf /backup/models_backup.tar.gz -C /models
```
