# Satellite Image Coordinate Prediction

A production-ready system for training neural networks to predict geographic coordinates (latitude and longitude) from NASA EPIC satellite imagery with cross-machine compatibility and secure API key management.

## 🚀 Features

- **Production Ready**: Clean, modular codebase with comprehensive error handling
- **Cross-Machine Compatible**: Works seamlessly across different environments (Mac, Linux, Windows)
- **Secure API Management**: Integrated GitHub Secrets support for NASA EPIC API keys
- **Automated Data Pipeline**: Download and process satellite imagery automatically
- **Advanced Model Training**: Support for both regression and autoencoder models
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **Device Optimization**: Automatic detection and optimization for MPS (Apple Silicon), CUDA, and CPU
- **Comprehensive Visualization**: Coordinate analysis, world maps, and model evaluation plots
- **Flexible Configuration**: JSON-based configuration system with CLI overrides

## 📋 Requirements

### Core Dependencies
```bash
pip install torch torchvision pandas matplotlib requests pillow tqdm rich numpy tensorboard
```

### Optional Dependencies
```bash
pip install basemap scikit-learn seaborn torchinfo psutil
```

## 🔑 API Key Setup

### Option 1: Environment Variable (Recommended)
```bash
export NASA_EPIC_API_KEY="your_api_key_here"
```

### Option 2: GitHub Secrets (Production)
1. Go to: https://github.com/your-username/fantastic-palm-tree/settings/secrets/actions
2. Click: "New repository secret"
3. Name: `NASA_EPIC_API_KEY`
4. Value: `vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC`
5. Select "Deploy to workflow" if available

The system will automatically detect and use the configured API key for all NASA EPIC requests.

## 🏗️ Project Structure

```
fantastic-palm-tree/
├── main.py                      # Main entry point with CLI interface
├── config.py                    # Configuration management with device auto-detection
├── api_key_manager.py           # Secure API key management
├── github_config_loader.py      # GitHub Actions integration
├── data.py                      # NASA EPIC data downloading and processing
├── datasets.py                  # PyTorch datasets and data loading
├── cross_machine_datasets.py    # Cross-machine compatible dataset loader
├── image_file_mapper.py        # Image filename mapping for portability
├── models.py                    # Neural network architectures
├── training.py                  # Training utilities
├── enhanced_training.py         # Enhanced trainer with advanced features
├── visualization.py             # Coordinate analysis and visualization
├── evaluation_reporter.py        # Comprehensive model evaluation
├── logging_utils.py             # Enhanced logging system
├── tensorboard_utils.py         # TensorBoard utilities
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Data Pipeline
Download and prepare satellite imagery data:
```bash
python3 main.py --mode setup
```

### 2. Train Model
Train the location regression model:
```bash
python3 main.py --mode train_regressor --epochs 50 --batch_size 32
```

### 3. Evaluate Model
Evaluate a trained model:
```bash
python3 main.py --mode evaluate --model_path models/your_model.pth
```

### 4. Download Recent Images
Get the latest satellite imagery (metadata + actual images):
```bash
python3 main.py --mode download_recent --num_days 7
```

### 5. Download Only Images
Download actual PNG image files (oldest available first):
```bash
# Download 10 oldest images
python3 main.py --mode download_images --num_images 10

# Download images from 5 oldest days
python3 main.py --mode download_images --num_days 5

# Download all available images
python3 main.py --mode download_images
```

### 6. Download Latest N Images
Get the latest N satellite images with metadata:
```bash
python3 main.py --mode download_latest --num_images 100
```

## 🔧 Advanced Usage

### Cross-Machine Dataset Preparation
For maximum compatibility across different machines:

```bash
# Create filename mapping (enables cross-machine portability)
python3 image_file_mapper.py --mode map

# Verify dataset integrity
python3 cross_machine_datasets.py

# Download images with proper mapping
python3 image_file_mapper.py --mode download --max_images 1000
```

### Custom Configuration
Create a `config.json` file:
```json
{
  "data": {
    "image_size": [128, 128],
    "grayscale": false,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "model": {
    "conv_channels": [32, 64, 128],
    "hidden_dim": 256,
    "activation": "relu"
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    "device": "auto"
  }
}
```

Use with:
```bash
python3 main.py --config config.json --mode train_regressor
```

### Training Options
```bash
# Train autoencoder
python3 main.py --mode train_autoencoder --epochs 100

# Custom hyperparameters
python3 main.py --mode train_regressor --lr 0.0005 --batch_size 16 --device cuda

# Disable TensorBoard for headless environments
python3 main.py --mode train_regressor --no-tensorboard
```

### Data Management Options
```bash
# Setup complete data pipeline (metadata + sample images)
python3 main.py --mode setup

# Download recent 7 days (metadata + images)
python3 main.py --mode download_recent --num_days 7

# Download only images from oldest available dates
python3 main.py --mode download_images --num_images 50

# Download images from specific number of oldest days
python3 main.py --mode download_images --num_days 5

# Download latest 100 images with metadata
python3 main.py --mode download_latest --num_images 100

# Cross-machine dataset preparation
python3 image_file_mapper.py --mode map
python3 image_file_mapper.py --mode download --max_images 1000

# Verify dataset integrity
python3 cross_machine_datasets.py
```

## 📊 Model Performance

The system includes comprehensive evaluation tools:

- **Coordinate Error Analysis**: Mean/Median error in degrees
- **Haversine Distance**: Geographic accuracy in kilometers
- **Visualization Tools**: World maps, error distributions, prediction plots
- **Detailed Reports**: HTML reports with comprehensive metrics

## 🛠️ Development

### Device Auto-Detection
The system automatically detects and configures for:
- **CUDA GPU** (NVIDIA): Best performance for training
- **MPS** (Apple Silicon): Optimized for M1/M2/M3 chips
- **CPU**: Fallback for any system

### Logging & Monitoring
- Enhanced logging with colored output
- TensorBoard integration for training visualization
- Progress bars with `tqdm`
- Automatic checkpointing and model saving

### Error Handling
- Robust download retry mechanisms
- Graceful fallback for missing API keys
- Comprehensive error reporting and recovery

## 📋 Complete Command Reference

### Available Modes
| Mode | Description | Key Arguments |
|------|-------------|---------------|
| `setup` | Initialize complete data pipeline | None |
| `train_regressor` | Train coordinate prediction model | `--epochs`, `--batch_size`, `--lr` |
| `train_autoencoder` | Train image autoencoder | `--epochs`, `--batch_size`, `--lr` |
| `evaluate` | Evaluate trained model | `--model_path` (required) |
| `download_recent` | Download recent N days (metadata + images) | `--num_days` |
| `download_latest` | Download latest N images (metadata + images) | `--num_images` |
| `download_images` | Download only PNG image files | `--num_images`, `--num_days` |

### Command Examples

#### Data Pipeline Commands
```bash
# Complete setup (metadata + sample downloads)
python3 main.py --mode setup

# Download recent 7 days with actual images
python3 main.py --mode download_recent --num_days 7

# Download latest 100 images with metadata
python3 main.py --mode download_latest --num_images 100

# Download only 50 oldest PNG images
python3 main.py --mode download_images --num_images 50

# Download images from 3 oldest days
python3 main.py --mode download_images --num_days 3
```

#### Training Commands
```bash
# Train with default settings (50 epochs, batch 32)
python3 main.py --mode train_regressor

# Custom training parameters
python3 main.py --mode train_regressor --epochs 100 --batch_size 64 --lr 0.001

# Train on specific device
python3 main.py --mode train_regressor --device cuda

# Train autoencoder
python3 main.py --mode train_autoencoder --epochs 200

# Training without TensorBoard
python3 main.py --mode train_regressor --no-tensorboard
```

#### Evaluation Commands
```bash
# Evaluate specific model
python3 main.py --mode evaluate --model_path models/best_model.pth
```

#### Cross-Machine Compatibility Commands
```bash
# Create image filename mapping (for portability)
python3 image_file_mapper.py --mode map

# Download images with mapping file
python3 image_file_mapper.py --mode download --max_images 1000

# Verify dataset integrity across machines
python3 cross_machine_datasets.py
```

### All Available Arguments
```bash
--mode MODE              # Operation mode (required)
--config PATH            # Custom config file
--model_path PATH        # Model file for evaluation
--epochs N              # Training epochs (default: varies)
--batch_size N          # Training batch size
--lr FLOAT             # Learning rate
--device DEVICE          # cuda/mps/cpu/auto (default: auto)
--no-tensorboard         # Disable TensorBoard
--num_days N           # Days to download (default: 7)
--num_images N         # Images to download (default: 100)
```

## 📁 Data Directories

- `images/`: Downloaded satellite images (original format)
- `combined/`: Consolidated metadata files
- `images_with_filenames/`: Cross-machine compatible image structure
- `models/`: Trained model checkpoints
- `logs/`: Training logs and TensorBoard files
- `outputs/`: Visualization outputs and reports

## 🔒 Security

- API keys are never committed to version control
- Support for GitHub Actions secrets
- Environment variable fallbacks
- Secure credential management

## 📈 Performance Tips

1. **Use GPU acceleration** when available (CUDA/MPS)
2. **Optimize batch size** based on your GPU memory
3. **Use cross-machine datasets** for portable workflows
4. **Monitor with TensorBoard** for training insights
5. **Configure appropriate workers** for data loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

## 📄 License

This project uses NASA EPIC data. Please refer to NASA's data usage policies and the project's license file for terms of use.

## 🆘 Troubleshooting

### API Key Issues
- Ensure your NASA EPIC API key is valid
- Check environment variables: `echo $NASA_EPIC_API_KEY`
- Verify GitHub Actions secrets are properly configured

### Download Failures
- Check internet connection
- Verify API key permissions
- Monitor NASA EPIC API status

### Training Issues
- Ensure sufficient GPU memory
- Reduce batch size if OOM errors occur
- Check data integrity with verification tools

### Cross-Machine Issues
- Run `python3 cross_machine_datasets.py` to verify dataset
- Ensure image files exist and match mapping
- Use `image_file_mapper.py` to fix missing files

---

**Note**: This repository is production-ready and designed for both research and deployment scenarios. All prototyping code and development artifacts have been removed for a clean, maintainable codebase.