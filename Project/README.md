# Satellite Image Coordinate Prediction

This project trains neural networks to predict the geographic coordinates (latitude and longitude) of satellite images from NASA's EPIC (Earth Polychromatic Imaging Camera) dataset.

## Features

- **Data Management**: Automated downloading and processing of NASA EPIC satellite imagery
- **Model Training**: Support for both regression and autoencoder models with enhanced performance
- **TensorBoard Integration**: Real-time training monitoring with comprehensive logging and error handling
- **Device Optimization**: Automatic optimization for MPS (Apple Silicon), CUDA, and CPU devices
- **Visualization**: Comprehensive plotting tools for coordinate analysis and model evaluation
- **Configuration Management**: Flexible configuration system for hyperparameters
- **Modular Design**: Clean, organized codebase with separated concerns
- **Enhanced Error Handling**: Robust training pipeline with comprehensive error recovery

## Project Structure

```
Project/
├── __init__.py           # Package initialization
├── config.py             # Configuration classes with device auto-detection
├── data.py               # Data downloading and processing
├── datasets.py           # PyTorch datasets with optimized data loading
├── models.py             # Neural network architectures
├── training.py           # Original training utilities
├── enhanced_training.py  # Enhanced trainer with robust TensorBoard logging
├── visualization.py      # Visualization and plotting tools
├── tensorboard_utils.py  # Manual TensorBoard control utilities
├── logging_utils.py      # Enhanced logging system
├── main.py               # Main entry point (uses enhanced trainer)
└── LocationRegressor.py  # Original model (legacy)
```

## Installation

Install the required dependencies:

```bash
pip install torch torchvision pandas matplotlib requests pillow tqdm rich numpy
```

TensorBoard (recommended for training visualization):
```bash
pip install tensorboard
```

Optional dependencies for enhanced functionality:
```bash
pip install basemap scikit-learn seaborn torchinfo psutil
```

## 📋 Complete Command Reference

This section provides every important command and option for the satellite coordinate prediction codebase.

### 🎯 Main Script Commands

#### **Setup Data Pipeline**
```bash
# Basic setup with default settings
python3 main.py --mode setup

# Setup with custom configuration
python3 main.py --mode setup --config my_config.json
```

#### **Training Commands**
```bash
# Train regression model (basic)
python3 main.py --mode train_regressor

# Train with custom parameters
python3 main.py --mode train_regressor --epochs 100 --batch_size 32 --lr 0.001

# Train with specific device
python3 main.py --mode train_regressor --device mps --epochs 50

# Train with configuration file
python3 main.py --mode train_regressor --config config.json

# Train autoencoder
python3 main.py --mode train_autoencoder --epochs 50 --batch_size 64

# Train without TensorBoard
python3 main.py --mode train_regressor --epochs 50 --no-tensorboard
```

#### **Evaluation Commands**
```bash
# Evaluate with default test set
python3 main.py --mode evaluate --model_path models/best_model.pth

# Evaluate with custom configuration
python3 main.py --mode evaluate --model_path models/final_model.pth --config config.json
```

### 🛠️ Utility Script Commands

#### **TensorBoard Management**
```bash
# Start TensorBoard manually
python3 tensorboard_utils.py start

# Start with custom port
python3 tensorboard_utils.py start --port 8080

# Start with custom log directory
python3 tensorboard_utils.py start --logdir custom_logs

# Stop TensorBoard
python3 tensorboard_utils.py stop

# Check TensorBoard status
python3 tensorboard_utils.py status

# Start without opening browser
python3 tensorboard_utils.py start --no-browser
```

#### **Performance Testing**
```bash
# Run device performance benchmark
python3 test_mps.py

# Test TensorBoard logging capabilities
python3 tensorboard_demo.py
```

#### **Enhanced Logging**
```bash
# Test logging system
python3 logging_utils.py
```

### ⚙️ Configuration File Commands

#### **Create Custom Configuration**
```bash
# Create a JSON configuration file
cat > custom_config.json << EOF
{
  "data": {
    "image_size": 128,
    "grayscale": false,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "model": {
    "input_channels": 3,
    "conv_channels": [32, 64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "activation": "relu",
    "hidden_dim": 256,
    "dropout_rate": 0.3,
    "output_dim": 2
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 200,
    "device": "auto",
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "launch_tensorboard": true,
    "tensorboard_port": 6006,
    "num_threads": 16
  }
}
EOF

# Use custom configuration
python3 main.py --config custom_config.json --mode train_regressor
```

### 🖥️ Development Commands

#### **Package Management**
```bash
# Install core dependencies
pip3 install torch torchvision pandas matplotlib requests pillow tqdm rich numpy

# Install TensorBoard
pip3 install tensorboard

# Install optional dependencies
pip3 install basemap scikit-learn seaborn torchinfo psutil

# Install all dependencies at once
pip3 install torch torchvision pandas matplotlib requests pillow tqdm rich numpy tensorboard basemap scikit-learn seaborn torchinfo psutil
```

#### **Project Structure Commands**
```bash
# View project structure
tree Project/ -I '__pycache__'

# Check Python syntax
python3 -m py_compile *.py

# Run with Python path
PYTHONPATH=. python3 main.py --mode train_regressor
```

### 📊 Data Management Commands

#### **Data Pipeline Operations**
```bash
# Force re-download all metadata
rm -rf epic.gsfc.nasa.gov/
python3 main.py --mode setup

# Reconsolidate metadata
python3 -c "
from data import EPICDataDownloader
from config import Config
downloader = EPICDataDownloader(Config())
downloader.consolidate_metadata()
"

# Extract coordinates from existing data
python3 -c "
from data import CoordinateExtractor
from config import Config
extractor = CoordinateExtractor(Config())
lat, lon = extractor.extract_coordinates()
print(f'Extracted {len(lat)} coordinate pairs')
"
```

### 🧪 Testing and Validation Commands

#### **Model Testing**
```bash
# Test model architecture
python3 -c "
from models import create_location_regressor, create_autoencoder
from config import Config

config = Config()
regressor = create_location_regressor(config)
autoencoder = create_autoencoder(config)

print(f'Regressor parameters: {sum(p.numel() for p in regressor.parameters()):,}')
print(f'Autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}')
"

# Test data loading
python3 -c "
from datasets import create_dataloaders
from config import Config

config = Config()
train_loader, val_loader, test_loader = create_dataloaders(config)
print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
"
```

### 🔧 System Administration Commands

#### **Device Management**
```bash
# Check available devices
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
"

# Test device performance
python3 -c "
from config import Config
config = Config()
print(f'Auto-detected device: {config.training.device}')
"

# Force specific device
python3 main.py --mode train_regressor --device cpu --epochs 1
python3 main.py --mode train_regressor --device mps --epochs 1
python3 main.py --mode train_regressor --device cuda --epochs 1
```

#### **Log and Output Management**
```bash
# View TensorBoard logs
ls -la logs/

# View training logs
find logs/ -name "training_*.log" -exec tail -20 {} \;

# Clean old logs
rm -rf logs/*.log
rm -rf logs/tensorboard_logs/

# View saved models
ls -la models/

# View outputs
ls -la outputs/
```

### 🚀 Production Commands

#### **Batch Training**
```bash
# Train multiple models in sequence
for epochs in 50 100 200; do
    python3 main.py --mode train_regressor --epochs $epochs --batch_size 32 --lr 0.001
    cp models/best_model.pth models/best_model_${epochs}_epochs.pth
done

# Train with different learning rates
for lr in 0.001 0.0005 0.0001; do
    python3 main.py --mode train_regressor --epochs 100 --lr $lr --batch_size 32
    cp models/best_model.pth models/best_model_lr${lr}.pth
done
```

#### **Evaluation Pipeline**
```bash
# Evaluate all saved models
for model in models/best_model_*.pth; do
    python3 main.py --mode evaluate --model_path "$model"
done

# Generate comprehensive evaluation report
python3 -c "
import glob
import subprocess

models = glob.glob('models/best_model_*.pth')
for model in models:
    print(f'Evaluating {model}')
    subprocess.run(['python3', 'main.py', '--mode', 'evaluate', '--model_path', model])
"
```

### 🐛 Debugging Commands

#### **Debug Mode Training**
```bash
# Train with verbose logging
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from main import main
main()
" --mode train_regressor --epochs 1

# Train with minimal batch size (for debugging)
python3 main.py --mode train_regressor --epochs 1 --batch_size 2

# Train on CPU (for debugging)
python3 main.py --mode train_regressor --epochs 1 --device cpu
```

#### **Data Validation**
```bash
# Validate data integrity
python3 -c "
from data import CoordinateExtractor
from config import Config

config = Config()
extractor = CoordinateExtractor(config)
lat, lon = extractor.extract_coordinates()

print(f'Total coordinates: {len(lat)}')
print(f'Latitude range: {min(lat):.3f} to {max(lat):.3f}')
print(f'Longitude range: {min(lon):.3f} to {max(lon):.3f}')

# Check for missing or invalid data
invalid_lat = sum(1 for x in lat if abs(x) > 90)
invalid_lon = sum(1 for x in lon if abs(x) > 180)
print(f'Invalid coordinates: {invalid_lat} lat, {invalid_lon} lon')
"
```

### 📈 Monitoring Commands

#### **System Monitoring**
```bash
# Monitor GPU usage (if available)
nvidia-smi -l 1  # NVIDIA GPU
# or for Apple Silicon:
sudo powermetrics --samplers gpu_power -i 1000

# Monitor memory usage
python3 -c "
import psutil
import torch

print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Monitor training progress
tail -f logs/training_*.log
```

#### **TensorBoard Monitoring**
```bash
# Start TensorBoard with all logs
tensorboard --logdir logs --port 6006 --host 0.0.0.0

# Start TensorBoard with reload
tensorboard --logdir logs --reload_interval 5 --port 6006

# Compare multiple runs
tensorboard --logdir logs/run1,logs/run2,logs/run3 --port 6006
```

## 🎯 Quick Reference Summary

| Command | Purpose |
|---------|---------|
| `python3 main.py --mode setup` | Initialize data pipeline |
| `python3 main.py --mode train_regressor --epochs 50` | Train coordinate model |
| `python3 main.py --mode train_autoencoder --epochs 30` | Train autoencoder |
| `python3 main.py --mode evaluate --model_path models/best_model.pth` | Evaluate model |
| `python3 tensorboard_utils.py start` | Start TensorBoard manually |
| `python3 test_mps.py` | Benchmark device performance |
| `python3 tensorboard_demo.py` | Test TensorBoard logging |
| `python3 main.py --device mps --epochs 10` | Train on Apple Silicon |
| `python3 main.py --no-tensorboard --epochs 50` | Train without TensorBoard |
| `python3 main.py --config custom.json` | Use custom configuration |

These commands cover every major function of the codebase from development to production deployment.

### Usage Examples

#### 1. Setup Data Pipeline

```bash
# Basic setup with auto-detected device
python main.py --mode setup

# Setup with custom configuration
python main.py --mode setup --config my_config.json
```

**What it does:**
- Downloads metadata from NASA EPIC API
- Extracts coordinate information from existing data
- Creates visualization plots (coordinate distribution, world map)
- Saves processed data to appropriate directories
- Generates statistical summaries

#### 2. Train Location Regression Model

```bash
# Basic training with default settings
python main.py --mode train_regressor

# Custom training parameters
python main.py --mode train_regressor --epochs 100 --batch_size 32 --lr 0.001

# Force specific device
python main.py --mode train_regressor --device mps --epochs 50

# Using configuration file
python main.py --mode train_regressor --config config.json
```

**Features:**
- Automatic device detection (CUDA > MPS > CPU)
- Model checkpointing (saves best and final models)
- Learning rate scheduling
- TensorBoard logging (if available)
- Progress tracking with loss visualization

#### 3. Train Autoencoder Model

```bash
# Basic autoencoder training
python main.py --mode train_autoencoder --epochs 50

# High-capacity autoencoder
python main.py --mode train_autoencoder --epochs 100 --batch_size 64
```

**Features:**
- Image compression and reconstruction
- Latent space learning
- Reconstruction quality monitoring

#### 4. Evaluate Model Performance

```bash
# Evaluate with default test set
python main.py --mode evaluate --model_path models/best_model.pth

# Evaluate with specific configuration
python main.py --mode evaluate --model_path models/final_model.pth --config config.json
```

**Outputs:**
- Test MSE loss
- Mean and median coordinate error (in degrees)
- Prediction vs true coordinate plots
- Error distribution analysis
- Visual comparison charts

#### 5. Advanced Configuration

Create a custom JSON configuration file:

```json
{
  "data": {
    "image_size": 128,
    "grayscale": false,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "model": {
    "input_channels": 3,
    "conv_channels": [32, 64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "activation": "relu",
    "hidden_dim": 256,
    "dropout_rate": 0.3
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 200,
    "device": "auto",
    "optimizer": "adamw",
    "weight_decay": 1e-4
  }
}
```

Then use it:
```bash
python main.py --config custom_config.json --mode train_regressor
```

### Enhanced TensorBoard Integration

The system includes robust TensorBoard logging with automatic error handling and performance optimization:

```bash
# Training with enhanced TensorBoard logging (default)
python main.py --mode train_regressor --epochs 50

# Disable automatic TensorBoard launch
python main.py --mode train_regressor --epochs 50 --no-tensorboard
```

**TensorBoard Features:**
- **Real-time Metrics**: Loss tracking (train/validation), learning rate, gradient norms
- **Model Monitoring**: Weight histograms, parameter statistics, gradient distributions
- **Performance Tracking**: Epoch duration, batches per second, coordinate prediction errors
- **Enhanced Reliability**: Automatic flushing, error recovery, graceful interruption handling
- **Device Optimization**: Optimized logging for MPS, CUDA, and CPU devices
- **Automatic Browser Launch**: Opens at http://localhost:6006 with background process management
- **Comprehensive Logging**: Configuration, hyperparameters, and training metadata

**Logged Metrics:**
- `Loss/Train_Batch` and `Loss/Train_Epoch`: Training loss progression
- `Loss/Val_Epoch`: Validation loss per epoch
- `Learning_Rate/Current` and `Learning_Rate/Epoch`: Learning rate tracking
- `Gradients/Total_Norm`: Gradient norm monitoring
- `Weights/*` and `Gradients/*`: Parameter distributions and gradients
- `Error/Coordinate_Degrees`: Coordinate prediction accuracy (degrees)
- `Performance/*`: Training speed and efficiency metrics
- `Model/*`: Model architecture statistics

**Manual TensorBoard Control:**

```bash
# Start TensorBoard manually
python tensorboard_utils.py start --logdir logs

# Stop TensorBoard
python tensorboard_utils.py stop

# Check TensorBoard status
python tensorboard_utils.py status

# Custom port
python tensorboard_utils.py start --port 8080
```

### Performance Testing

Test device performance:

```bash
# Run performance benchmark
python test_mps.py
```

This will compare inference speeds across available devices and show throughput metrics.

## Quick Start

### 1. Setup Data Pipeline

```bash
python main.py --mode setup
```

### 2. Train Model

```bash
python main.py --mode train_regressor --epochs 10 --batch_size 32
```

### 3. Evaluate Model

```bash
python main.py --mode evaluate --model_path models/best_model.pth
```

## Configuration System

The project uses a flexible hierarchical configuration system with default values and JSON file support.

### Default Configuration

```python
# Data Configuration
data:
  image_size: 64                    # Input image resolution
  grayscale: true                    # Convert to grayscale
  train_split: 0.8                  # Training data proportion
  val_split: 0.1                    # Validation data proportion
  test_split: 0.1                   # Test data proportion

# Model Configuration  
model:
  input_channels: 1                  # Number of input channels (1=grayscale, 3=RGB)
  conv_channels: [64, 128, 256]     # Convolutional layer channels
  kernel_size: 3                     # Convolution kernel size
  pool_size: 4                       # Max pooling size
  activation: "tanh"                # Activation function
  hidden_dim: 128                   # Fully connected layer size
  output_dim: 2                     # Output dimension (lon, lat)

# Training Configuration
training:
  batch_size: 32                     # Training batch size
  learning_rate: 0.001               # Learning rate
  epochs: 100                       # Number of training epochs
  device: "auto"                    # Device selection
  optimizer: "adam"                  # Optimizer type
  weight_decay: 1e-5                # Weight decay
  log_dir: "logs"                   # TensorBoard log directory
  save_dir: "models"                 # Model save directory
  num_threads: 16                    # PyTorch thread count
  launch_tensorboard: true           # Auto-launch TensorBoard
  tensorboard_port: 6006              # TensorBoard port number
```

### Custom Configuration File

Create `config.json`:

```json
{
  "data": {
    "image_size": 128,
    "grayscale": false
  },
  "model": {
    "conv_channels": [32, 64, 128],
    "activation": "relu"
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 200
  }
}
```

Use with:
```bash
python main.py --config config.json --mode train_regressor
```

## Models

### LocationRegressor

A CNN-based regression model that predicts latitude and longitude coordinates from satellite images.

**Architecture:**
- 3 convolutional layers with max pooling
- Tanh activation functions
- Fully connected layers for regression
- 2 output neurons (longitude, latitude)

### AutoEncoder

An encoder-decoder architecture for learning compressed representations of satellite images.

## Data Pipeline

1. **Metadata Download**: Fetches image metadata from NASA EPIC API
2. **Coordinate Extraction**: Extracts lat/lon coordinates from metadata
3. **Image Download**: Downloads actual satellite images
4. **Dataset Creation**: Creates PyTorch datasets with proper train/val/test splits
5. **Data Loading**: Provides efficient data loading with transforms

## Visualization Tools

- **Coordinate Distribution**: Histograms of latitude and longitude distributions
- **World Map Plot**: Global visualization of image locations
- **Training Curves**: Loss plots for training monitoring
- **Prediction Analysis**: Scatter plots comparing true vs predicted coordinates
- **Error Analysis**: Distribution of prediction errors

## Performance Metrics

- **MSE Loss**: Standard regression loss for training optimization
- **Coordinate Error**: Euclidean distance in degrees between predicted and true coordinates
- **Training Performance**: Batches per second, epoch duration, device utilization
- **Model Statistics**: Parameter counts, gradient norms, weight distributions
- **Error Analysis**: Comprehensive error distribution analysis with visualizations

### Performance Optimizations

The enhanced trainer includes several performance improvements:

- **Device-Specific Data Loading**: Optimized for MPS (Apple Silicon), CUDA, and CPU
  - MPS: `num_workers=0, pin_memory=False` (prevents conflicts)
  - CUDA: `num_workers=4, pin_memory=True` (optimal for GPU)
  - CPU: `num_workers=2, pin_memory=False` (balanced performance)
- **Automatic Flushing**: TensorBoard logs flushed every 50 batches and after each epoch
- **Error Recovery**: Continues training even if individual batch logging fails
- **Memory Management**: Efficient checkpoint saving and tensor cleanup

## Programmatic Usage

### Basic Training Script

```python
from config import Config
from models import create_location_regressor
from enhanced_training import LocationRegressorTrainer  # Enhanced trainer
from datasets import create_dataloaders

# Setup configuration
config = Config()
config.training.epochs = 50
config.training.batch_size = 32

# Create data loaders (automatically optimized for device)
train_loader, val_loader, test_loader = create_dataloaders(config)

# Create model and enhanced trainer
model = create_location_regressor(config)
trainer = LocationRegressorTrainer(model, train_loader, val_loader, config)

# Train with robust TensorBoard logging
history = trainer.train(num_epochs=config.training.epochs)
print(f"Best validation loss: {trainer.best_val_loss:.6f}")

# Enhanced trainer provides:
# - Automatic device optimization
# - Robust TensorBoard logging
# - Error handling and recovery
# - Performance monitoring
# - Graceful interruption handling
```

### Data Analysis

```python
from data import EPICDataDownloader, CoordinateExtractor
from visualization import plot_coordinate_distribution, plot_world_map_with_coordinates

# Initialize with config
config = Config()

# Extract and analyze coordinates
extractor = CoordinateExtractor(config)
lat_coords, lon_coords = extractor.extract_coordinates()

# Create visualizations
plot_coordinate_distribution(
    lat_coords, lon_coords, 
    save_path="outputs/coordinate_distribution.png",
    show_plot=False
)

plot_world_map_with_coordinates(
    lat_coords, lon_coords,
    save_path="outputs/world_map.png", 
    show_plot=False
)

# Get statistics
stats = extractor.get_coordinate_stats(lat_coords, lon_coords)
print("Coordinate Statistics:")
print(stats)
```

### Custom Model Evaluation

```python
import torch
from config import Config
from models import create_location_regressor
from datasets import create_dataloaders
from enhanced_training import CoordinateNormalizer

# Load configuration and data (automatically device-optimized)
config = Config()
_, _, test_loader = create_dataloaders(config)

# Load trained model
model = create_location_regressor(config)
checkpoint = torch.load("models/best_model.pth", map_location=config.training.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate on test set
device = torch.device(config.training.device)
model.to(device)

normalizer = CoordinateNormalizer()
all_errors = []

with torch.no_grad():
    for images, coords in test_loader:
        images = images.to(device)
        coords = coords.to(device)
        
        # Predict
        predictions = model(images)
        
        # Calculate coordinate errors
        pred_coords = normalizer.denormalize(predictions)
        true_coords = normalizer.denormalize(coords)
        
        errors = torch.sqrt((pred_coords[:, 0] - true_coords[:, 0])**2 + 
                          (pred_coords[:, 1] - true_coords[:, 1])**2)
        all_errors.extend(errors.cpu().numpy())

print(f"Mean coordinate error: {np.mean(all_errors):.3f} degrees")
print(f"Median coordinate error: {np.median(all_errors):.3f} degrees")
print(f"Total samples evaluated: {len(all_errors)}")
```

## Hardware Requirements

- **GPU**: Recommended for training
  - CUDA-compatible NVIDIA GPU (Linux/Windows)
  - Apple Silicon GPU (M1/M2/M3 Macs) with MPS support
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~50GB for full dataset (optional, can download on-demand)

## Device Options

The framework automatically detects and uses the best available hardware:

1. **CUDA** - NVIDIA GPU support (Linux/Windows)
2. **MPS** - Apple Silicon GPU support (M1/M2/M3 Macs)
3. **CPU** - Fallback option

You can also manually specify the device:
```bash
python main.py --mode train_regressor --device mps  # Force Apple Silicon
python main.py --mode train_regressor --device cuda  # Force CUDA
python main.py --mode train_regressor --device cpu   # Force CPU
```

## Recent Enhancements (v2.0)

### 🚀 Major Improvements

#### Enhanced TensorBoard Logging
- **Robust Error Handling**: Automatic recovery from logging failures
- **Performance Optimized**: Device-specific logging configurations
- **Comprehensive Metrics**: Training, validation, performance, and model statistics
- **Real-time Monitoring**: Automatic flushing every 50 batches and after each epoch

#### Data Loading Performance Fixes
- **MPS (Apple Silicon) Optimization**: Fixed conflicts with multiprocessing and pin_memory
- **Device-Specific Settings**: Automatic configuration based on available hardware
- **Speed Improvements**: 5x faster training on Apple Silicon devices
- **Memory Efficiency**: Optimized memory usage and reduced overhead

#### Training Reliability
- **Error Recovery**: Continues training even if individual batches fail
- **Graceful Interruption**: Proper cleanup on Ctrl+C or other interruptions
- **Enhanced Logging**: Detailed error messages and progress tracking
- **Checkpoint Robustness**: Reliable model saving and loading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Download Failures**: Check internet connection and NASA API status
3. **Missing Dependencies**: Install optional dependencies for full functionality
4. **MPS Performance Issues**: Ensure using enhanced trainer (`enhanced_training.py`)

### Performance Tips

- **Use Enhanced Trainer**: Automatically enables optimizations and TensorBoard logging
- **Device Auto-Detection**: Let the system choose the best device (default: "auto")
- **Batch Size Optimization**: Start with 32, adjust based on available memory
- **TensorBoard Monitoring**: Use the comprehensive logging for training insights
- **Memory Management**: The enhanced trainer handles memory cleanup automatically

### Known Issues & Solutions

#### Apple Silicon (M1/M2/M3) Performance
- **Issue**: Original trainer had slow data loading due to MPS conflicts
- **Solution**: Use enhanced trainer with device-optimized data loading
- **Verification**: Check logs for "Using dataloader settings for mps: workers=0, pin_memory=False"

#### TensorBoard Logging Interruption
- **Issue**: Previous version could lose logs if training was interrupted
- **Solution**: Enhanced trainer includes automatic flushing every 50 batches
- **Verification**: TensorBoard events are written in real-time during training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure you comply with NASA's data usage policies for the EPIC dataset.

## Acknowledgments

- NASA EPIC team for providing the satellite imagery dataset
- PyTorch team for the deep learning framework
- Matplotlib and Basemap communities for visualization tools