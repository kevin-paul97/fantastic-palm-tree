# Satellite Image Coordinate Prediction

A PyTorch CNN that predicts geographic coordinates (latitude/longitude) from NASA EPIC satellite imagery.

## Requirements

```
torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard
```

Optional: `basemap` (world map plots), `seaborn`, `psutil`

## Setup

1. Install dependencies: `pip install torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard`
2. (Optional) Set your NASA API key: `export NASA_EPIC_API_KEY="your_key_here"`

## Usage

```bash
# Download metadata and set up data pipeline
python main.py setup

# Download satellite images
python main.py download recent 7       # last 7 days
python main.py download latest 50      # 50 latest images
python main.py download all            # everything

# Train models
python main.py train regressor         # coordinate prediction CNN
python main.py train autoencoder       # image autoencoder

# Evaluate a trained model
python main.py evaluate models/best_model.pth

# Test predictions with visualization
python test_predictions.py --model_path models/best_model.pth
python test_predictions.py --model_path models/best_model.pth --num_samples 6
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
models.py               LocationRegressor CNN + AutoEncoder
training.py             Unified trainer with TensorBoard logging
data.py                 NASA EPIC API client + image downloader
datasets.py             PyTorch Dataset + DataLoader + CoordinateNormalizer
visualization.py        Plotting (distributions, world maps, training curves)
evaluation_reporter.py  Evaluation metrics + report generation (JSON/MD/CSV)
tensorboard_utils.py    TensorBoard start/stop utilities
api_key_manager.py      NASA API key management
test_predictions.py     Prediction testing with visualization
```

## Data Organization

```
images/YYYY-MM-DD/*.png     Satellite images organized by date
combined/YYYY-MM-DD.json    Consolidated metadata per date
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

## TensorBoard

Training logs are saved to `logs/tensorboard/`. View with:

```bash
tensorboard --logdir logs/tensorboard
python tensorboard_utils.py start
```
