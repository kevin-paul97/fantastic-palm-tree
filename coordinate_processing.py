"""
Consolidated coordinate processing utilities.
Centralizes all coordinate normalization, denormalization, and error calculation logic.
"""

import torch
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CoordinateProcessor:
    """
    Centralized coordinate processing for satellite image coordinate prediction.
    Handles normalization, denormalization, and all distance/error calculations.
    """
    
    def __init__(self, coordinate_range: Optional[Dict[str, float]] = None):
        """
        Initialize coordinate processor.
        
        Args:
            coordinate_range: Optional dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        """
        if coordinate_range:
            self.min_lat = coordinate_range['min_lat']
            self.max_lat = coordinate_range['max_lat']
            self.min_lon = coordinate_range['min_lon']
            self.max_lon = coordinate_range['max_lon']
            self.coord_range = {
                'min_lat': self.min_lat,
                'max_lat': self.max_lat,
                'min_lon': self.min_lon,
                'max_lon': self.max_lon
            }
        else:
            # Earth coordinate bounds
            self.min_lat = -90.0
            self.max_lat = 90.0
            self.min_lon = -180.0
            self.max_lon = 180.0
            self.coord_range = {
                'min_lat': self.min_lat,
                'max_lat': self.max_lat,
                'min_lon': self.min_lon,
                'max_lon': self.max_lon
            }
    
    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize real-world coordinates to [-1, 1] range.
        
        Args:
            coords: Tensor of shape (N, 2) with real-world coordinates [lon, lat]
            
        Returns:
            Normalized coordinates in [-1, 1] range
        """
        if coords.shape[-1] != 2:
            raise ValueError(f"Expected 2D coordinates, got shape {coords.shape}")
        
        # Normalize longitude (column 0)
        lon_norm = 2.0 * (coords[..., 0] - self.min_lon) / (self.max_lon - self.min_lon) - 1.0
        
        # Normalize latitude (column 1)
        lat_norm = 2.0 * (coords[..., 1] - self.min_lat) / (self.max_lat - self.min_lat) - 1.0
        
        normalized = torch.stack([lon_norm, lat_norm], dim=-1)
        return normalized.clamp(-1.0, 1.0)
    
    def denormalize(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize coordinates from [-1, 1] to real-world coordinates.
        
        Args:
            coords_norm: Normalized coordinates in [-1, 1] range
            
        Returns:
            Real-world coordinates [lon, lat]
        """
        if coords_norm.shape[-1] != 2:
            raise ValueError(f"Expected 2D normalized coordinates, got shape {coords_norm.shape}")
        
        # Denormalize longitude (column 0)
        lon = (coords_norm[..., 0] + 1.0) / 2.0 * (self.max_lon - self.min_lon) + self.min_lon
        
        # Denormalize latitude (column 1)
        lat = (coords_norm[..., 1] + 1.0) / 2.0 * (self.max_lat - self.min_lat) + self.min_lat
        
        denormalized = torch.stack([lon, lat], dim=-1)
        return denormalized
    
    def compute_coordinate_error_degrees(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute coordinate error in degrees for each prediction.
        
        Args:
            pred_coords: Predicted coordinates [lon, lat]
            true_coords: True coordinates [lon, lat]
            
        Returns:
            Error in degrees for each coordinate pair
        """
        if pred_coords.shape != true_coords.shape:
            raise ValueError(f"Shape mismatch: pred {pred_coords.shape} vs true {true_coords.shape}")
        
        # Compute Euclidean distance in coordinate space
        lon_error = pred_coords[..., 0] - true_coords[..., 0]
        lat_error = pred_coords[..., 1] - true_coords[..., 1]
        
        # Convert to degrees (approximate)
        coord_error_degrees = torch.sqrt(lon_error**2 + lat_error**2)
        return coord_error_degrees
    
    def haversine_distance(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculate Haversine distance between predicted and true coordinates.
        
        Args:
            pred_coords: Predicted coordinates [lon, lat] in degrees
            true_coords: True coordinates [lon, lat] in degrees
            
        Returns:
            Haversine distance in kilometers
        """
        if pred_coords.shape != true_coords.shape:
            raise ValueError(f"Shape mismatch: pred {pred_coords.shape} vs true {true_coords.shape}")
        
        # Convert to radians
        pred_lon_rad = torch.deg2rad(pred_coords[..., 0])
        pred_lat_rad = torch.deg2rad(pred_coords[..., 1])
        true_lon_rad = torch.deg2rad(true_coords[..., 0])
        true_lat_rad = torch.deg2rad(true_coords[..., 1])
        
        # Haversine formula
        dlat = true_lat_rad - pred_lat_rad
        dlon = true_lon_rad - pred_lon_rad
        
        a = torch.sin(dlat / 2)**2 + torch.cos(pred_lat_rad) * torch.cos(true_lat_rad) * torch.sin(dlon / 2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Earth's radius in kilometers
        R = 6371.0
        distance = R * c
        
        return distance
    
    def compute_comprehensive_metrics(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            pred_coords: Predicted coordinates [lon, lat]
            true_coords: True coordinates [lon, lat]
            
        Returns:
            Dictionary with all metrics
        """
        # Compute various error metrics
        coord_errors = self.compute_coordinate_error_degrees(pred_coords, true_coords)
        haversine_distances = self.haversine_distance(pred_coords, true_coords)
        
        # Move to CPU for calculations
        coord_errors_cpu = coord_errors.abs().cpu()
        haversine_cpu = haversine_distances.abs().cpu()
        
        metrics = {
            # Coordinate errors (degrees)
            'mean_coordinate_error_deg': coord_errors_cpu.mean().item(),
            'median_coordinate_error_deg': coord_errors_cpu.median().item(),
            'std_coordinate_error_deg': coord_errors_cpu.std().item(),
            'min_coordinate_error_deg': coord_errors_cpu.min().item(),
            'max_coordinate_error_deg': coord_errors_cpu.max().item(),
            
            # Haversine distances (kilometers)
            'mean_haversine_km': haversine_cpu.mean().item(),
            'median_haversine_km': haversine_cpu.median().item(),
            'std_haversine_km': haversine_cpu.std().item(),
            'min_haversine_km': haversine_cpu.min().item(),
            'max_haversine_km': haversine_cpu.max().item(),
            
            # Percentiles
            'haversine_p25': haversine_cpu.quantile(0.25).item(),
            'haversine_p75': haversine_cpu.quantile(0.75).item(),
            'haversine_p95': haversine_cpu.quantile(0.95).item(),
            'haversine_p99': haversine_cpu.quantile(0.99).item(),
            
            # Accuracy thresholds (common benchmarks)
            'predictions_within_1km': int((haversine_cpu <= 1).sum()),
            'predictions_within_10km': int((haversine_cpu <= 10).sum()),
            'predictions_within_100km': int((haversine_cpu <= 100).sum()),
            'predictions_within_1000km': int((haversine_cpu <= 1000).sum()),
            
            # Percentage accuracy
            'percentage_within_1km': float((haversine_cpu <= 1).float().mean() * 100),
            'percentage_within_10km': float((haversine_cpu <= 10).float().mean() * 100),
            'percentage_within_100km': float((haversine_cpu <= 100).float().mean() * 100),
            'percentage_within_1000km': float((haversine_cpu <= 1000).float().mean() * 100),
            
            # Total samples
            'total_samples': len(haversine_cpu)
        }
        
        return metrics
    
    def format_metrics_summary(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics into a readable summary string.
        
        Args:
            metrics: Metrics dictionary from compute_comprehensive_metrics
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Coordinate Prediction Evaluation Summary:
==========================================
Haversine Distance (km):
  Mean:     {metrics['mean_haversine_km']:.1f}
  Median:   {metrics['median_haversine_km']:.1f}
  Std Dev:  {metrics['std_haversine_km']:.1f}
  Range:    {metrics['min_haversine_km']:.1f} - {metrics['max_haversine_km']:.1f}
  Percentiles: 25th={metrics['haversine_p25']:.1f}, 75th={metrics['haversine_p75']:.1f}, 95th={metrics['haversine_p95']:.1f}

Coordinate Error (degrees):
  Mean:     {metrics['mean_coordinate_error_deg']:.4f}
  Median:   {metrics['median_coordinate_error_deg']:.4f}
  Range:    {metrics['min_coordinate_error_deg']:.4f} - {metrics['max_coordinate_error_deg']:.4f}

Accuracy Benchmarks:
  Within 1km:   {metrics['percentage_within_1km']:.1f}% ({metrics['predictions_within_1km']}/{metrics['total_samples']})
  Within 10km:  {metrics['percentage_within_10km']:.1f}% ({metrics['predictions_within_10km']}/{metrics['total_samples']})
  Within 100km: {metrics['percentage_within_100km']:.1f}% ({metrics['predictions_within_100km']}/{metrics['total_samples']})
  Within 1000km: {metrics['percentage_within_1000km']:.1f}% ({metrics['predictions_within_1000km']}/{metrics['total_samples']})

Total Samples: {metrics['total_samples']}
==========================================
"""
        return summary
    
    @staticmethod
    def from_training_data(train_loader) -> 'CoordinateProcessor':
        """
        Create CoordinateProcessor with ranges computed from training data.
        
        Args:
            train_loader: Training dataloader
            
        Returns:
            CoordinateProcessor with data-specific ranges
        """
        all_coords = []
        for images, coords in train_loader:
            all_coords.append(coords)
        
        if not all_coords:
            logger.warning("No training data found, using default Earth bounds")
            return CoordinateProcessor()
        
        all_coords_tensor = torch.cat(all_coords, dim=0)
        
        coord_range = {
            'min_lat': float(all_coords_tensor[:, 1].min()),
            'max_lat': float(all_coords_tensor[:, 1].max()),
            'min_lon': float(all_coords_tensor[:, 0].min()),
            'max_lon': float(all_coords_tensor[:, 0].max())
        }
        
        logger.info(f"Computed coordinate ranges from training data: {coord_range}")
        return CoordinateProcessor(coord_range)


# Backward compatibility alias
CoordinateNormalizer = CoordinateProcessor