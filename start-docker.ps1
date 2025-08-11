# PowerShell script for starting EUR/USD Transformer with Docker

param(
    [string]$Action = "help",
    [string]$Timeframe = "1h",
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.001,
    [string]$Service = "eurusd-transformer",
    [switch]$Production,
    [switch]$Continuous,
    [int]$Interval = 300
)

function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Show-Help {
    Write-Host "EUR/USD Transformer Docker Management" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage: .\start-docker.ps1 -Action <action> [options]"
    Write-Host ""
    Write-Host "Actions:"
    Write-Host "  build       - Build Docker image"
    Write-Host "  start       - Start services"
    Write-Host "  stop        - Stop services"
    Write-Host "  restart     - Restart services"
    Write-Host "  clean       - Clean up Docker resources"
    Write-Host "  download    - Download EUR/USD data"
    Write-Host "  train       - Train model"
    Write-Host "  predict     - Run predictions"
    Write-Host "  jupyter     - Start Jupyter Lab"
    Write-Host "  tensorboard - Start TensorBoard"
    Write-Host "  logs        - Show logs"
    Write-Host "  status      - Show service status"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Timeframe <timeframe>     - Timeframe (5m, 15m, 30m, 1h, 1d, 1wk)"
    Write-Host "  -Epochs <epochs>           - Number of training epochs"
    Write-Host "  -BatchSize <batch_size>    - Training batch size"
    Write-Host "  -LearningRate <lr>         - Learning rate"
    Write-Host "  -Service <service>         - Service name for logs"
    Write-Host "  -Production                - Use production configuration"
    Write-Host "  -Continuous                - Run continuous prediction"
    Write-Host "  -Interval <seconds>        - Prediction interval"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start-docker.ps1 -Action build"
    Write-Host "  .\start-docker.ps1 -Action train -Timeframe 15m -Epochs 200"
    Write-Host "  .\start-docker.ps1 -Action predict -Timeframe 1h -Continuous"
    Write-Host "  .\start-docker.ps1 -Action jupyter"
}

function Invoke-DockerCompose {
    param(
        [string]$Command,
        [string]$ComposeFile = "docker-compose.yml"
    )
    
    if ($Production) {
        $ComposeFile = "docker-compose.prod.yml"
    }
    
    $fullCommand = "docker-compose -f $ComposeFile $Command"
    Write-Host "Running: $fullCommand" -ForegroundColor Gray
    
    $result = Invoke-Expression $fullCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        return $false
    }
    return $true
}

function Start-Build {
    Write-Header "Building Docker Image"
    if (Invoke-DockerCompose "build") {
        Write-Success "Docker image built successfully"
    } else {
        Write-Error "Failed to build Docker image"
    }
}

function Start-Services {
    Write-Header "Starting Services"
    if (Invoke-DockerCompose "up -d") {
        Write-Success "Services started successfully"
        Write-Host "Access TensorBoard at: http://localhost:6006" -ForegroundColor Yellow
    } else {
        Write-Error "Failed to start services"
    }
}

function Stop-Services {
    Write-Header "Stopping Services"
    if (Invoke-DockerCompose "down") {
        Write-Success "Services stopped successfully"
    } else {
        Write-Error "Failed to stop services"
    }
}

function Restart-Services {
    Write-Header "Restarting Services"
    Stop-Services
    Start-Services
}

function Clean-Docker {
    Write-Header "Cleaning Docker Resources"
    if (Invoke-DockerCompose "down -v --remove-orphans") {
        Write-Success "Docker resources cleaned"
    } else {
        Write-Error "Failed to clean Docker resources"
    }
    
    Write-Host "Running docker system prune..." -ForegroundColor Gray
    docker system prune -f
}

function Download-Data {
    Write-Header "Downloading EUR/USD Data"
    if (Invoke-DockerCompose "--profile data up download-data") {
        Write-Success "Data downloaded successfully"
    } else {
        Write-Error "Failed to download data"
    }
}

function Train-Model {
    Write-Header "Training Model"
    Write-Host "Parameters: Timeframe=$Timeframe, Epochs=$Epochs, BatchSize=$BatchSize, LearningRate=$LearningRate" -ForegroundColor Gray
    
    $trainCommand = "run --rm training python scripts/train.py --timeframe $Timeframe --num-epochs $Epochs --batch-size $BatchSize --learning-rate $LearningRate"
    
    if (Invoke-DockerCompose $trainCommand) {
        Write-Success "Training completed successfully"
    } else {
        Write-Error "Training failed"
    }
}

function Run-Prediction {
    Write-Header "Running Predictions"
    $predictCommand = "run --rm prediction python scripts/predict.py --timeframe $Timeframe"
    
    if ($Continuous) {
        $predictCommand += " --continuous --interval $Interval"
        Write-Host "Running continuous prediction with interval $Interval seconds" -ForegroundColor Gray
    }
    
    if (Invoke-DockerCompose $predictCommand) {
        Write-Success "Prediction completed successfully"
    } else {
        Write-Error "Prediction failed"
    }
}

function Start-Jupyter {
    Write-Header "Starting Jupyter Lab"
    if (Invoke-DockerCompose "--profile jupyter up -d jupyter") {
        Write-Success "Jupyter Lab started successfully"
        Write-Host "Access Jupyter Lab at: http://localhost:8888" -ForegroundColor Yellow
    } else {
        Write-Error "Failed to start Jupyter Lab"
    }
}

function Start-TensorBoard {
    Write-Header "Starting TensorBoard"
    if (Invoke-DockerCompose "--profile monitoring up -d tensorboard") {
        Write-Success "TensorBoard started successfully"
        Write-Host "Access TensorBoard at: http://localhost:6006" -ForegroundColor Yellow
    } else {
        Write-Error "Failed to start TensorBoard"
    }
}

function Show-Logs {
    Write-Header "Showing Logs for $Service"
    Invoke-DockerCompose "logs -f $Service"
}

function Show-Status {
    Write-Header "Service Status"
    Invoke-DockerCompose "ps"
}

# Main execution
switch ($Action.ToLower()) {
    "help" { Show-Help }
    "build" { Start-Build }
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "clean" { Clean-Docker }
    "download" { Download-Data }
    "train" { Train-Model }
    "predict" { Run-Prediction }
    "jupyter" { Start-Jupyter }
    "tensorboard" { Start-TensorBoard }
    "logs" { Show-Logs }
    "status" { Show-Status }
    default {
        Write-Error "Unknown action: $Action"
        Show-Help
    }
}
