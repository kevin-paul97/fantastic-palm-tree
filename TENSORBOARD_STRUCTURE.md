# TensorBoard Directory Structure

## New Organized Structure

```
logs/
├── tensorboard/
│   ├── run_20260209_093153/
│   │   └── events.out.tfevents.*
│   ├── run_20260209_103045/
│   │   └── events.out.tfevents.*
│   └── ... (timestamped runs)
└── (old mixed files - can be cleaned up)

models/
├── best_model.pth
├── regressor_final.pth
├── autoencoder_final.pth
└── ... (saved model files)
```

## Benefits

1. **Separation of Concerns**: TensorBoard logs and model files are in separate directories
2. **Timestamped Runs**: Each training session gets its own timestamped subdirectory
3. **Easy Navigation**: Clear view of different training runs
4. **Clean Organization**: No more mixed files in logs directory
5. **TensorBoard Commands**: Easy to launch TensorBoard for specific runs

## Usage

### View latest run:
```bash
# Using utility script (auto-detects latest run)
python3 tensorboard_utils.py start

# Or manually
tensorboard --logdir logs/tensorboard/run_$(ls -t logs/tensorboard/run_* | head -1 | cut -d'_' -f2-)
```

### View all runs:
```bash
# Using utility script
python3 tensorboard_utils.py start --logdir logs/tensorboard

# Or manually
tensorboard --logdir logs/tensorboard/
```

### Stop TensorBoard:
```bash
# Using utility script
python3 tensorboard_utils.py stop

# Check status
python3 tensorboard_utils.py status
```

### Training command (auto-launches TensorBoard):
```bash
python3 main.py train regressor --epochs 100

# Disable auto-launch
python3 main.py train regressor --epochs 100 --no-tensorboard
```

This will:
- Create a new timestamped directory in `logs/tensorboard/run_YYYYMMDD_HHMMSS/`
- Save model files to `models/` directory
- Log the exact directory path for easy reference
- Launch TensorBoard automatically (if enabled) pointing to the correct run