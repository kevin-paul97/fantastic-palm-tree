# Satellite Image Coordinate Prediction

A deep learning project that predicts geographic coordinates from NASA EPIC satellite images using convolutional neural networks.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/kevin-paul97/fantastic-palm-tree.git
cd fantastic-palm-tree/Project
pip install -r requirements.txt

# Download data and train model
python3 main.py --mode setup
python3 main.py --mode train_regressor

# Evaluate model
python3 main.py --mode evaluate --model_path models/best_model.pth
```

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [Single Image Testing](#single-image-testing)
- [Downloading Recent Data](#downloading-recent-data)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)

## ✨ Features

- **🛰️ NASA EPIC Data**: Downloads and processes satellite imagery from NASA's EPIC API
- **🧠 Deep Learning**: CNN models for coordinate prediction (Location Regressor & AutoEncoder)
- **📊 Comprehensive Evaluation**: Multiple metrics including Haversine distance calculations
- **🗺️ Rich Visualizations**: World maps, coordinate distributions, error analysis
- **📈 Professional Reporting**: JSON, Markdown, CSV evaluation reports
- **🎯 Single Image Testing**: Test individual predictions with visual feedback
- **🔄 Recent Data Downloads**: Get latest satellite images automatically

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch (with MPS/CUDA support recommended)
- Git

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/kevin-paul97/fantastic-palm-tree.git
cd fantastic-palm-tree/Project
```

2. **Install Dependencies**
```bash
pip install torch torchvision matplotlib numpy pandas requests basemap tqdm tensorboard
```

3. **Verify Installation**
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

## 📁 Data Setup

### Option 1: Complete Historical Data
```bash
# Download all available satellite data (may take time)
python3 main.py --mode setup
```

### Option 2: Recent Data Only
```bash
# Download most recent 7 days of images
python3 main.py --mode download_recent --num_days 7

# Download latest 100 images
python3 main.py --mode download_latest --num_images 100
```

### Data Structure
```
Project/
├── data/
│   ├── raw/                    # Complete metadata
│   ├── images/                 # Daily image metadata
│   └── combined/               # Consolidated metadata
├── models/                     # Trained model checkpoints
├── outputs/                    # Visualizations and reports
└── logs/                       # Training logs
```

## 🏋️ Training Models

### Location Regressor (Coordinate Prediction)
```bash
# Basic training
python3 main.py --mode train_regressor

# Custom training parameters
python3 main.py --mode train_regressor --epochs 50 --batch_size 32 --lr 0.001

# Specify device
python3 main.py --mode train_regressor --device mps  # Apple Silicon
python3 main.py --mode train_regressor --device cuda  # NVIDIA GPU
python3 main.py --mode train_regressor --device cpu   # CPU fallback
```

### AutoEncoder (Feature Learning)
```bash
# Train autoencoder
python3 main.py --mode train_autoencoder

# Custom parameters
python3 main.py --mode train_autoencoder --epochs 100 --batch_size 64
```

### Training Features
- **📊 TensorBoard Integration**: Automatic training visualization
- **🔄 Learning Rate Scheduling**: Adaptive learning rate adjustment
- **💾 Checkpointing**: Save best and final models automatically
- **📱 Device Optimization**: Auto-detects MPS > CUDA > CPU

## 📊 Model Evaluation

### Comprehensive Evaluation
```bash
# Evaluate trained model
python3 main.py --mode evaluate --model_path models/best_model.pth

# Evaluate with custom config
python3 main.py --mode evaluate --model_path models/final_model.pth --config config.json
```

### Evaluation Output
- **📈 Performance Metrics**: MSE, coordinate errors, Haversine distances
- **🗺️ Visualizations**: Error distributions, prediction maps
- **📋 Reports**: JSON, Markdown, CSV with detailed statistics
- **🎯 Accuracy Analysis**: Geographic accuracy thresholds

### Example Evaluation Results
```
Test MSE: 0.017530
Mean coordinate error: 48.784 degrees
Mean Haversine distance: 5189.8 km
Accuracy within 100 km: 0.33%
```

## 🎯 Single Image Testing

### Test Individual Predictions
```bash
# Test single image with visualization
python3 test_single_prediction.py --model_path models/best_model.pth --show

# Test without showing plot
python3 test_single_prediction.py --model_path models/best_model.pth
```

### Test Multiple Images
```bash
# Test 6 random images
python3 test_multiple_predictions.py --model_path models/best_model.pth --num_samples 6 --show

# Test 3 images without showing
python3 test_multiple_predictions.py --model_path models/best_model.pth --num_samples 3
```

### Visualization Features
- **🖼️ Side-by-side Display**: Original image + world map
- **📍 Coordinate Markers**: 
  - 🟢 Green circle = True coordinates
  - ❌ Red X = Predicted coordinates
  - 📏 Blue line = Error distance
- **📏 Error Metrics**: Haversine distance in kilometers
- **🗺️ Professional Maps**: Miller projection with geographic features

### Example Single Prediction Output
```
============================================================
SINGLE IMAGE PREDICTION RESULTS
============================================================
True Coordinates:  (19.69°, 0.05°)
Pred Coordinates:  (5.77°, -2.71°)
Longitude Error:   13.92°
Latitude Error:    2.76°
Distance Error:    1577.7 km
============================================================
```

## 🔄 Downloading Recent Data

### Most Recent Days
```bash
# Download last 7 days (default)
python3 main.py --mode download_recent

# Download last 14 days
python3 main.py --mode download_recent --num_days 14

# Download last 30 days
python3 main.py --mode download_recent --num_days 30
```

### Latest N Images
```bash
# Download latest 100 images (default)
python3 main.py --mode download_latest

# Download latest 50 images
python3 main.py --mode download_latest --num_images 50

# Download latest 200 images
python3 main.py --mode download_latest --num_images 200
```

### Download Features
- **🆕 Newest Metadata**: Automatically gets most recent satellite data
- **📊 Auto-Visualization**: Creates coordinate distribution maps
- **📈 Statistics**: Shows geographic coverage and data quality
- **🔄 Auto-Consolidation**: Merges new data with existing datasets

## ⚙️ Configuration

### Default Configuration
The project uses sensible defaults, but you can customize via `config.json`:

```json
{
  "data": {
    "api_base_url": "https://api.nasa.gov/EPIC/api/v1.0",
    "image_size": [64, 64],
    "grayscale": true,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "model": {
    "input_channels": 1,
    "conv_channels": [32, 64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "hidden_dim": 256,
    "output_dim": 2
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "device": "auto"
  }
}
```

### Custom Config Usage
```bash
# Use custom configuration
python3 main.py --mode train_regressor --config my_config.json

# Override specific parameters
python3 main.py --mode train_regressor --epochs 100 --lr 0.0005 --batch_size 64
```

## 📂 Project Structure

```
Project/
├── main.py                    # Main CLI interface
├── config.py                  # Configuration management
├── models.py                  # Neural network architectures
├── datasets.py                # Data loading and coordinate handling
├── training.py                # Basic training utilities
├── enhanced_training.py       # Advanced training with TensorBoard
├── visualization.py           # Plotting and visualization tools
├── evaluation_reporter.py     # Comprehensive evaluation reports
├── data.py                    # Data downloading and processing
├── logging_utils.py          # Enhanced logging system
├── test_single_prediction.py  # Single image testing tool
├── test_multiple_predictions.py # Multiple image testing tool
├── data/                      # Data directories
│   ├── raw/                   # Complete metadata
│   ├── images/                # Daily image metadata
│   └── combined/              # Consolidated metadata
├── models/                    # Saved model checkpoints
├── outputs/                   # Generated visualizations and reports
├── logs/                      # Training and evaluation logs
└── requirements.txt           # Python dependencies
```

## 📈 Performance & Results

### Current Model Performance
- **📊 Dataset**: ~3,000 satellite images with coordinates
- **🎯 Mean Error**: ~5,190 km (geographic distance)
- **📏 Median Error**: ~3,800 km
- **🎯 Precision**: ~0.33% predictions within 100 km
- **📈 Training Time**: ~10-15 minutes on Apple Silicon M2

### Model Architecture
- **🧠 CNN**: 3 convolutional layers (32→64→128 channels)
- **📏 Input**: 64×64 grayscale satellite images
- **🎯 Output**: 2 coordinates (longitude, latitude)
- **🔄 Normalization**: Coordinate normalization to [0,1] range

### Evaluation Metrics
- **📏 MSE Loss**: Normalized coordinate mean squared error
- **🌍 Haversine Distance**: Geographic distance in kilometers
- **📐 Coordinate Error**: Euclidean distance in degrees
- **🎯 Accuracy Thresholds**: Percentage within specific distances

## 🔧 Troubleshooting

### Common Issues

**Q: "No module named 'torch'"**
```bash
pip install torch torchvision
```

**Q: "Basemap not found"**
```bash
pip install basemap
# Or alternative: pip install cartopy
```

**Q: "CUDA out of memory"**
```bash
# Reduce batch size
python3 main.py --mode train_regressor --batch_size 16

# Or use CPU
python3 main.py --mode train_regressor --device cpu
```

**Q: "Download failed"**
```bash
# Check internet connection and NASA API status
# Try again - NASA API sometimes has rate limits
python3 main.py --mode setup
```

**Q: "Poor prediction accuracy"**
- Model needs more training data
- Try different hyperparameters
- Consider data augmentation
- Check for data quality issues

### Performance Tips

1. **🚀 Use GPU**: Enable MPS (Apple Silicon) or CUDA (NVIDIA)
2. **📦 Batch Size**: Optimize for your GPU memory
3. **🔄 Learning Rate**: Start with 0.001, adjust as needed
4. **📊 More Data**: Download recent images for better training
5. **🎯 Regularization**: Use weight decay to prevent overfitting

## 📚 Advanced Usage

### Custom Training Scripts
```python
from config import Config
from enhanced_training import LocationRegressorTrainer
from datasets import create_dataloaders

# Load config and data
config = Config()
train_loader, val_loader, test_loader = create_dataloaders(config)

# Create and train model
from models import create_location_regressor
model = create_location_regressor(config)

trainer = LocationRegressorTrainer(model, train_loader, val_loader, config)
history = trainer.train(num_epochs=100)
```

### Custom Evaluation
```python
from evaluation_reporter import EvaluationReporter
import torch

# Load model and evaluate
model = torch.load('models/best_model.pth')
reporter = EvaluationReporter('models/best_model.pth', config)

# Generate custom report
report_path = reporter.generate_comprehensive_report(
    predictions, targets, mse_loss, "custom_outputs"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **🛰️ NASA**: For the EPIC satellite imagery API
- **🧠 PyTorch**: Deep learning framework
- **📊 Matplotlib**: Visualization library
- **🌍 Basemap**: Geographic mapping tools

## 📞 Support

For questions, issues, or contributions:
- 🐛 Report bugs via GitHub Issues
- 💬 Start discussions in GitHub Discussions
- 📧 Contact: [your-email@example.com]

---

**🚀 Happy satellite image coordinate prediction!**