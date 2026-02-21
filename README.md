# Satellite Image Coordinate Prediction

A PyTorch CNN that predicts geographic coordinates (latitude/longitude) from NASA EPIC satellite imagery.

## Requirements

```
torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn
```

Optional: `basemap` (world map plots), `seaborn`, `psutil`

## Setup

1. Install dependencies: `pip install torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn`
2. (Optional) Set your NASA API key: `export NASA_EPIC_API_KEY="your_key_here"`

## Usage

```bash
# Download metadata and set up data pipeline
python main.py setup

# Download recent satellite images
python main.py download 7              # last 7 days (default)
python main.py download 30             # last 30 days

# Train the regressor
python main.py train regressor
python main.py train regressor --epochs 50

# Evaluate a trained model
python main.py evaluate models/regressor_final.pth

# Run exploratory data analysis
python eda.py

# Test predictions
python test_predictions.py --model_path models/regressor_final.pth
python test_predictions.py --model_path models/regressor_final.pth --num_samples 6

# World map with predictions colored by error
python test_predictions.py --model_path models/regressor_final.pth --num_samples 100 --world_map
```

### CLI Options

```
--epochs N          Override training epochs
--batch-size N      Override batch size
--lr RATE           Override learning rate
--device DEVICE     Force device (auto/cuda/mps/cpu)
--no-tensorboard    Disable TensorBoard auto-launch
--config FILE       Load config from JSON file
```

## Project Structure

```
main.py                 CLI entry point
config.py               Configuration (data, model, training)
models.py               LocationRegressor CNN
training.py             Trainer with TensorBoard logging
data.py                 NASA EPIC API client + async image downloader
datasets.py             PyTorch Dataset + DataLoader + CoordinateNormalizer
eda.py                  Exploratory data analysis (PCA, statistics, overview plot)
visualization.py        Plotting (distributions, world maps, training curves)
evaluation_reporter.py  Evaluation metrics + report generation (JSON/MD/CSV)
tensorboard_utils.py    TensorBoard start/stop utilities
api_key_manager.py      NASA API key management
test_predictions.py     Prediction testing + world error map
```

## Data Organization

```
images/YYYY-MM-DD/*.png     Satellite images organized by date
combined/YYYY-MM-DD.json    Metadata per date
models/                     Saved model checkpoints
logs/tensorboard/           TensorBoard run logs
outputs/                    Evaluation reports and plots
```

## Configuration

Default config is defined in `config.py`. Override via CLI flags or a JSON config file:

```json
{
  "data": {"image_size": 64, "grayscale": true},
  "model": {"conv_channels": [64, 128, 256], "hidden_dim": 128},
  "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 100}
}
```

Device is auto-detected: CUDA > MPS (Apple Silicon) > CPU.

## Exploratory Data Analysis

Run `python eda.py` to generate a comprehensive 4x4 overview figure with:
- Mean/std images, pixel and coordinate distributions
- PCA (2 components) colored by longitude and latitude
- Brightness-coordinate correlations and summary statistics tables

Output: `outputs/eda_overview.png` + TensorBoard metrics under `logs/tensorboard/eda_*/`.

## TensorBoard

Training and EDA logs are saved to `logs/tensorboard/`. View with:

```bash
tensorboard --logdir logs/tensorboard
python tensorboard_utils.py start
```
