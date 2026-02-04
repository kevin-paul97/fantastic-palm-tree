"""
Utility functions for visualization and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.basemap import Basemap
import pandas as pd
from typing import List, Tuple, Optional
try:
    import seaborn as sns
except ImportError:
    sns = None
from pathlib import Path

from datasets import CoordinateNormalizer


def plot_coordinate_distribution(
    lat_coords: List[float],
    lon_coords: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot distribution of latitude and longitude coordinates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Latitude distribution
    ax1.hist(lat_coords, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Latitude (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Latitude Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Longitude distribution
    ax2.hist(lon_coords, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Longitude Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_world_map_with_coordinates(
    lat_coords: List[float],
    lon_coords: List[float],
    title: str = "Satellite Image Coordinates",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    highlight_points: Optional[List[Tuple[float, float]]] = None
) -> None:
    """Plot coordinates on world map using Basemap."""
    fig = plt.figure(figsize=(12, 8))
    
    # Set up orthographic map projection
    m = Basemap(projection='ortho', lat_0=30, lon_0=10, resolution='c')
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')
    
    # Draw lat/lon grid lines
    m.drawmeridians(np.arange(0, 360, 30))
    m.drawparallels(np.arange(-90, 90, 30))
    
    # Plot coordinates
    if lat_coords and lon_coords:
        x, y = m(lon_coords, lat_coords)
        m.scatter(x, y, marker='D', color='m', s=10, alpha=0.6, 
                  label="Satellite coordinates")
    
    # Highlight specific points
    if highlight_points:
        for i, (lat, lon) in enumerate(highlight_points):
            x, y = m(lon, lat)
            m.scatter(x, y, marker='o', color='red', s=100, 
                      label=f"Point {i+1}")
    
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_coordinate_predictions(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot true vs predicted coordinates with error metrics."""
    # Convert to numpy and denormalize
    normalizer = CoordinateNormalizer()
    
    true_coords_np = normalizer.denormalize(true_coords).cpu().numpy()
    pred_coords_np = normalizer.denormalize(pred_coords).cpu().numpy()
    
    # Calculate longitude errors with wraparound handling
    lon_direct_diff = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lon_wrap_diff = 360.0 - lon_direct_diff
    lon_errors = np.minimum(lon_direct_diff, lon_wrap_diff)
    
    # Calculate latitude errors (no wraparound needed)
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    
    # Calculate Euclidean distance errors in degrees
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Longitude comparison (index 0)
    ax1.scatter(true_coords_np[:, 0], pred_coords_np[:, 0], alpha=0.6, 
               c=distance_errors, cmap='viridis', s=30)
    ax1.plot([true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             [true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             'r--', label='Perfect Prediction', linewidth=2)
    ax1.set_xlabel('True Longitude (degrees)')
    ax1.set_ylabel('Predicted Longitude (degrees)')
    ax1.set_title(f'Longitude Predictions\nMAE: {lon_errors.mean():.3f}°')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Latitude comparison (index 1)
    ax2.scatter(true_coords_np[:, 1], pred_coords_np[:, 1], alpha=0.6,
               c=distance_errors, cmap='viridis', s=30)
    ax2.plot([true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             [true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel('True Latitude (degrees)')
    ax2.set_ylabel('Predicted Latitude (degrees)')
    ax2.set_title(f'Latitude Predictions\nMAE: {lat_errors.mean():.3f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Add colorbar for distance errors
    cbar = fig.colorbar(ax1.collections[0], ax=[ax1, ax2], 
                        label='Distance Error (degrees)', 
                        orientation='horizontal', pad=0.1, aspect=30)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot distribution of coordinate prediction errors."""
    # Convert to numpy and denormalize
    normalizer = CoordinateNormalizer()
    
    true_coords_np = normalizer.denormalize(true_coords).cpu().numpy()
    pred_coords_np = normalizer.denormalize(pred_coords).cpu().numpy()
    
    # Calculate errors using proper coordinate error calculation
    normalizer = CoordinateNormalizer()
    
    # Convert tensors to numpy if needed
    if isinstance(true_coords, torch.Tensor):
        true_coords_np = true_coords.cpu().numpy()
    else:
        true_coords_np = true_coords
        
    if isinstance(pred_coords, torch.Tensor):
        pred_coords_np = pred_coords.cpu().numpy()
    else:
        pred_coords_np = pred_coords
    
    # Use proper longitude error calculation
    lon_errors = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    
    # Account for longitude wraparound
    lon_direct_diff = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lon_wrap_diff = 360.0 - lon_direct_diff
    lon_errors = np.minimum(lon_direct_diff, lon_wrap_diff)
    
    # Calculate Euclidean distance errors in degrees
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Longitude error distribution
    ax1.hist(lon_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Longitude Error (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Longitude Error\nMean: {lon_errors.mean():.3f}°, Median: {np.median(lon_errors):.3f}°')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(lon_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax1.axvline(np.median(lon_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax1.legend()
    
    # Latitude error distribution
    ax2.hist(lat_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Latitude Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Latitude Error\nMean: {lat_errors.mean():.3f}°, Median: {np.median(lat_errors):.3f}°')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(lat_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax2.axvline(np.median(lat_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax2.legend()
    
    # Distance error distribution
    ax3.hist(distance_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Distance Error (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Euclidean Distance Error\nMean: {distance_errors.mean():.3f}°, Median: {np.median(distance_errors):.3f}°')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(distance_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax3.axvline(np.median(distance_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_coordinate_statistics_table(
    lat_coords: List[float],
    lon_coords: List[float]
) -> pd.DataFrame:
    """Create a statistics table for coordinate data."""
    df_lat = pd.DataFrame(lat_coords)
    df_lon = pd.DataFrame(lon_coords)
    
    stats = pd.DataFrame({
        "Latitude": df_lat.describe()[0],
        "Longitude": df_lon.describe()[0]
    })
    
    return stats


def visualize_sample_images(
    images: torch.Tensor,
    coords: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Visualize sample satellite images with their coordinates."""
    # Denormalize coordinates
    normalizer = CoordinateNormalizer()
    coords_np = normalizer.denormalize(coords).cpu().numpy()
    
    # Select random samples
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        coords_np = coords_np[indices]
    
    # Create subplot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = [axes]
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # Convert tensor to image
        img = images[i].squeeze().cpu().numpy()
        if img.ndim == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
        
        # Plot image
        if rows == 1:
            if num_samples > 1:
                ax = axes[col]
            else:
                ax = axes
        else:
            ax = axes[row][col]
        
        if not isinstance(ax, list):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.axis('off')
            ax.set_title(f'Lon: {coords_np[i, 0]:.2f}°, Lat: {coords_np[i, 1]:.2f}°')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            if num_samples > 1:
                ax = axes[col]
            else:
                ax = axes
        else:
            ax = axes[row][col]
        if not isinstance(ax, list):
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_predictions_on_world_map(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_points: int = 1000
) -> None:
    """Plot prediction errors on world map for geographic error analysis."""
    # Convert to numpy and denormalize
    normalizer = CoordinateNormalizer()
    
    true_coords_np = normalizer.denormalize(true_coords).cpu().numpy()
    pred_coords_np = normalizer.denormalize(pred_coords).cpu().numpy()
    
    # Calculate errors
    lon_errors = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    # Sample points if too many for visualization
    if len(true_coords_np) > max_points:
        indices = np.random.choice(len(true_coords_np), max_points, replace=False)
        true_coords_np = true_coords_np[indices]
        pred_coords_np = pred_coords_np[indices]
        distance_errors = distance_errors[indices]
    
    fig = plt.figure(figsize=(15, 10))
    
    # Set up world map
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')
    
    # Convert coordinates to map coordinates
    lon_true, lat_true = true_coords_np[:, 0], true_coords_np[:, 1]
    x, y = m(lon_true, lat_true)
    
    # Create scatter plot with color representing error magnitude
    scatter = m.scatter(x, y, c=distance_errors, s=20, alpha=0.7, 
                       cmap='Reds', edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca(), label='Prediction Error (degrees)', 
                       orientation='vertical', pad=0.02, shrink=0.8)
    
    # Add grid lines
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=10)
    m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0], fontsize=10)
    
    # Add title with statistics
    title = (f'Coordinate Prediction Errors Worldwide\n'
            f'Mean Error: {distance_errors.mean():.3f}°, '
            f'Median Error: {np.median(distance_errors):.3f}°, '
            f'Max Error: {distance_errors.max():.3f}°')
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()