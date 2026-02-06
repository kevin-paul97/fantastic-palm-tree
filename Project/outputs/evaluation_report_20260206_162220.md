# Satellite Image Coordinate Prediction - Evaluation Report

## Evaluation Summary
- **Timestamp**: 2026-02-06T16:22:20.226263
- **Model Path**: models/best_model.pth
- **Dataset Size**: 300 samples
- **Device**: mps

## Model Performance

### Overall Metrics
- **MSE Loss**: 0.011763

### Coordinate Errors (Degrees)
| Metric | Longitude | Latitude | Combined |
|--------|-----------|----------|----------|
| Mean | 37.657° | 3.379° | 38.195° |
| Median | 26.853° | 2.791° | 27.337° |
| Std Dev | 35.750° | 2.733° | 35.441° |
| Min | 0.068° | 0.012° | 0.353° |
| Max | 164.925° | 14.012° | 164.928° |
| 25th Percentile | - | - | 12.486° |
| 75th Percentile | - | - | 49.467° |
| 95th Percentile | - | - | 116.832° |
| 99th Percentile | - | - | 150.406° |

### Geographic Distance (Haversine - Kilometers)
| Metric | Distance |
|--------|----------|
| Mean | 4051.7 km |
| Median | 2917.1 km |
| Std Dev | 3635.7 km |
| Min | 39.2 km |
| Max | 15990.9 km |
| 25th Percentile | 1313.6 km |
| 75th Percentile | 5221.6 km |
| 95th Percentile | 12466.0 km |
| 99th Percentile | 14794.7 km |

## Accuracy Analysis

### Geographic Accuracy Thresholds
| Distance Threshold | Count | Percentage |
|-------------------|-------|------------|
| ≤ 1 km | 0 | 0.00% |
| ≤ 10 km | 0 | 0.00% |
| ≤ 100 km | 2 | 0.67% |
| ≤ 1000 km | 51 | 17.00% |

## Performance Assessment

### Interpretation
- **Excellent Performance**: Mean error < 50 km, > 50% predictions within 100 km
- **Good Performance**: Mean error < 200 km, > 30% predictions within 100 km  
- **Moderate Performance**: Mean error < 500 km, > 20% predictions within 100 km
- **Poor Performance**: Mean error > 500 km, < 20% predictions within 100 km

### Current Model Assessment
Mean error: 4051.7 km
Accuracy within 100 km: 0.67%

*This report was generated automatically by the evaluation system.*
