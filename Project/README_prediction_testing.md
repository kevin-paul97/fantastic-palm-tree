# Single Image Prediction Testing Tools

This directory contains tools for testing individual satellite image predictions and visualizing results on world maps.

## Available Scripts

### 1. `test_single_prediction.py`
Tests a single satellite image and creates a side-by-side visualization showing:
- The original satellite image
- World map with true and predicted coordinates marked
- Error calculation in both degrees and kilometers

**Usage:**
```bash
python test_single_prediction.py --model_path models/best_model.pth [--show] [--output_dir outputs]
```

**Arguments:**
- `--model_path`: Path to trained model (required)
- `--config`: Path to config file (optional)
- `--output_dir`: Output directory for visualizations (default: outputs)
- `--show`: Display plots interactively

**Output:**
- `outputs/single_prediction_test.png` - Side-by-side visualization
- Console output with detailed error metrics

### 2. `test_multiple_predictions.py`
Tests multiple random satellite images and creates a grid visualization showing:
- Multiple satellite images
- Corresponding world maps with coordinate predictions
- Error distances for each prediction

**Usage:**
```bash
python test_multiple_predictions.py --model_path models/best_model.pth --num_samples 6 [--show]
```

**Arguments:**
- `--model_path`: Path to trained model (required)
- `--config`: Path to config file (optional)
- `--output_dir`: Output directory for visualizations (default: outputs)
- `--num_samples`: Number of test samples to evaluate (default: 6)
- `--show`: Display plots interactively

**Output:**
- `outputs/multiple_predictions_test_{N}.png` - Grid visualization
- Console output with individual and summary statistics

## Visualization Features

### World Map Markers
- 🟢 **Green Circle**: True coordinates (actual satellite location)
- ❌ **Red X**: Predicted coordinates (model output)
- 📏 **Blue Dashed Line**: Error distance line
- 🟨 **Yellow Label**: Error distance in kilometers

### Error Metrics
Both scripts provide comprehensive error analysis:
- **Coordinate Error**: Euclidean distance in degrees
- **Distance Error**: Haversine distance in kilometers (geographically accurate)
- **Longitude/Latitude Errors**: Individual component errors

## Example Output

```
============================================================
SINGLE IMAGE PREDICTION RESULTS
============================================================
True Coordinates:  (19.69°, 0.05°)
Pred Coordinates:  (5.77°, -2.71°)
Longitude Error:   13.92°
Latitude Error:    2.76°
Coordinate Error:  14.19°
Distance Error:    1577.7 km
============================================================
```

## Model Performance Context

Based on current model evaluation:
- **Mean Error**: ~5,190 km
- **Median Error**: ~3,800 km  
- **Individual predictions**: Range from 1,500 km to 12,000+ km
- **Accuracy within 100 km**: ~0.33% of predictions

This indicates the model needs significant improvement for precise coordinate prediction, though it can identify general geographic regions.

## Technical Notes

### Coordinate Handling
- True coordinates come directly from dataset (real-world coordinates)
- Model outputs are normalized and must be denormalized for comparison
- Haversine distance calculation provides accurate geographic distance measurements
- Longitude wraparound is properly handled for errors across ±180° boundary

### Visualization Maps
- Uses Miller cylindrical projection for global coverage
- Includes coastlines, countries, and geographic features
- Grid lines every 30° latitude and 60° longitude
- High-resolution output (300 DPI) for detailed analysis