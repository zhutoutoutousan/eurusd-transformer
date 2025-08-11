# Makefile for EUR/USD Transformer Docker Operations

.PHONY: help build start stop restart clean download train predict jupyter tensorboard logs

# Default target
help:
	@echo "EUR/USD Transformer Docker Management"
	@echo ""
	@echo "Available commands:"
	@echo "  build       - Build Docker image"
	@echo "  start       - Start services"
	@echo "  stop        - Stop services"
	@echo "  restart     - Restart services"
	@echo "  clean       - Clean up Docker resources"
	@echo "  download    - Download EUR/USD data"
	@echo "  train       - Train model (use TIMEFRAME=1h EPOCHS=100)"
	@echo "  predict     - Run predictions (use TIMEFRAME=1h)"
	@echo "  jupyter     - Start Jupyter Lab"
	@echo "  tensorboard - Start TensorBoard"
	@echo "  logs        - Show logs (use SERVICE=eurusd-transformer)"
	@echo ""
	@echo "Examples:"
	@echo "  make train TIMEFRAME=15m EPOCHS=200"
	@echo "  make predict TIMEFRAME=1h"
	@echo "  make logs SERVICE=training"

# Build Docker image
build:
	docker-compose build

# Start services
start:
	docker-compose up -d

# Stop services
stop:
	docker-compose down

# Restart services
restart: stop start

# Clean up Docker resources
clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Download data
download:
	docker-compose --profile data up download-data

# Train model
train:
	@if [ -z "$(TIMEFRAME)" ]; then \
		echo "Usage: make train TIMEFRAME=<timeframe> [EPOCHS=<epochs>] [BATCH_SIZE=<batch_size>]"; \
		echo "Example: make train TIMEFRAME=1h EPOCHS=100 BATCH_SIZE=32"; \
		exit 1; \
	fi
	@EPOCHS=$${EPOCHS:-100}; \
	BATCH_SIZE=$${BATCH_SIZE:-32}; \
	LR=$${LR:-0.001}; \
	docker-compose run --rm training python scripts/train.py \
		--timeframe $(TIMEFRAME) \
		--num-epochs $$EPOCHS \
		--batch-size $$BATCH_SIZE \
		--learning-rate $$LR

# Run predictions
predict:
	@if [ -z "$(TIMEFRAME)" ]; then \
		echo "Usage: make predict TIMEFRAME=<timeframe> [CONTINUOUS=true] [INTERVAL=<seconds>]"; \
		echo "Example: make predict TIMEFRAME=1h CONTINUOUS=true INTERVAL=300"; \
		exit 1; \
	fi
	@CONTINUOUS=$${CONTINUOUS:-false}; \
	INTERVAL=$${INTERVAL:-300}; \
	CMD="python scripts/predict.py --timeframe $(TIMEFRAME)"; \
	if [ "$$CONTINUOUS" = "true" ]; then \
		CMD="$$CMD --continuous --interval $$INTERVAL"; \
	fi; \
	docker-compose run --rm prediction $$CMD

# Start Jupyter Lab
jupyter:
	docker-compose --profile jupyter up -d jupyter
	@echo "Jupyter Lab is available at: http://localhost:8888"

# Start TensorBoard
tensorboard:
	docker-compose --profile monitoring up -d tensorboard
	@echo "TensorBoard is available at: http://localhost:6006"

# Show logs
logs:
	@SERVICE=$${SERVICE:-eurusd-transformer}; \
	docker-compose logs -f $$SERVICE

# Production commands
prod-build:
	docker-compose -f docker-compose.prod.yml build

prod-start:
	docker-compose -f docker-compose.prod.yml up -d

prod-stop:
	docker-compose -f docker-compose.prod.yml down

prod-train:
	@if [ -z "$(TIMEFRAME)" ]; then \
		echo "Usage: make prod-train TIMEFRAME=<timeframe> [EPOCHS=<epochs>]"; \
		exit 1; \
	fi
	@EPOCHS=$${EPOCHS:-100}; \
	docker-compose -f docker-compose.prod.yml run --rm training python scripts/train.py \
		--timeframe $(TIMEFRAME) \
		--num-epochs $$EPOCHS

# Multi-timeframe training shortcuts
train-5m:
	make train TIMEFRAME=5m EPOCHS=150

train-15m:
	make train TIMEFRAME=15m EPOCHS=120

train-30m:
	make train TIMEFRAME=30m EPOCHS=100

train-1h:
	make train TIMEFRAME=1h EPOCHS=80

train-1d:
	make train TIMEFRAME=1d EPOCHS=50

train-1wk:
	make train TIMEFRAME=1wk EPOCHS=30

# Status and monitoring
status:
	docker-compose ps

stats:
	docker stats

# Development helpers
dev-shell:
	docker-compose exec eurusd-transformer bash

dev-logs:
	docker-compose logs -f

# Backup and restore
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@docker run --rm -v eurusd_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/data_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /data . || echo "No data volume found"
	@docker run --rm -v eurusd_models:/models -v $(PWD)/backups:/backup alpine tar czf /backup/models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz -C /models . || echo "No models volume found"
	@echo "Backup completed in backups/ directory"

restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=<backup_file.tar.gz>"; \
		echo "Available backups:"; \
		ls -la backups/ 2>/dev/null || echo "No backups found"; \
		exit 1; \
	fi
	@echo "Restoring from $(BACKUP_FILE)..."
	@if echo "$(BACKUP_FILE)" | grep -q "data"; then \
		docker run --rm -v eurusd_data:/data -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /data; \
	elif echo "$(BACKUP_FILE)" | grep -q "models"; then \
		docker run --rm -v eurusd_models:/models -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /models; \
	else \
		echo "Unknown backup type. Use data_* or models_* backup files."; \
		exit 1; \
	fi
	@echo "Restore completed"
