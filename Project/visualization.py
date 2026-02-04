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
    """Plot true vs predicted coordinates."""
    # Convert to numpy and denormalize
    normalizer = CoordinateNormalizer()
    
    true_coords_np = normalizer.denormalize(true_coords).cpu().numpy()
    pred_coords_np = normalizer.denormalize(pred_coords).cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Longitude comparison
    ax1.scatter(true_coords_np[:, 0], pred_coords_np[:, 0], alpha=0.6)
    ax1.plot([true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             [true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             'r--', label='Perfect Prediction')
    ax1.set_xlabel('True Longitude')
    ax1.set_ylabel('Predicted Longitude')
    ax1.set_title('Longitude Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Latitude comparison
    ax2.scatter(true_coords_np[:, 1], pred_coords_np[:, 1], alpha=0.6)
    ax2.plot([true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             [true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             'r--', label='Perfect Prediction')
    ax2.set_xlabel('True Latitude')
    ax2.set_ylabel('Predicted Latitude')
    ax2.set_title('Latitude Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
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
    
    # Calculate errors
    lon_errors = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    
    # Calculate Euclidean distance errors
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Longitude error distribution
    ax1.hist(lon_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Longitude Error (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Longitude Error Distribution\nMean: {lon_errors.mean():.3f}°')
    ax1.grid(True, alpha=0.3)
    
    # Latitude error distribution
    ax2.hist(lat_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Latitude Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Latitude Error Distribution\nMean: {lat_errors.mean():.3f}°')
    ax2.grid(True, alpha=0.3)
    
    # Distance error distribution
    ax3.hist(distance_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Distance Error (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Distance Error Distribution\nMean: {distance_errors.mean():.3f}°')
    ax3.grid(True, alpha=0.3)
    
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