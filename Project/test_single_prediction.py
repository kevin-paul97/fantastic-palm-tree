#!/usr/bin/env python3
"""
Test single image prediction and visualize on world map.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
from pathlib import Path
import logging

from config import Config
from datasets import create_dataloaders, CoordinateNormalizer
from models import create_location_regressor
from visualization import plot_world_map_with_coordinates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(config, model_path: str):
    """Load trained model from checkpoint."""
    model = create_location_regressor(config)
    checkpoint = torch.load(model_path, map_location=config.training.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(torch.device(config.training.device))
    return model


def predict_single_image(model, image, true_coords, config):
    """Predict coordinates for a single image."""
    device = torch.device(config.training.device)
    
    # Prepare image tensor
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    
    image = image.to(device)
    true_coords = true_coords.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image)
    
    # Denormalize prediction
    normalizer = CoordinateNormalizer()
    pred_coords = normalizer.denormalize(prediction)
    
    # True coordinates are already in real-world coordinates (from dataloader)
    # No need to denormalize them again
    true_coords_denorm = true_coords.squeeze().cpu().numpy()
    pred_coords_np = pred_coords.squeeze().cpu().numpy()
    
    return pred_coords_np, true_coords_denorm


def plot_prediction_comparison(
    image_tensor,
    true_coords,
    pred_coords,
    save_path: str = None,
    show_plot: bool = True
):
    """Plot image with coordinates marked on world map."""
    # Convert image tensor to numpy for display
    img_np = image_tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Image subplot
    ax1 = plt.subplot(1, 2, 1)
    if img_np.ndim == 2:
        ax1.imshow(img_np, cmap='gray')
    else:
        ax1.imshow(img_np)
    ax1.set_title('Satellite Image')
    ax1.axis('off')
    
    # World map subplot
    ax2 = plt.subplot(1, 2, 2)
    
    # Create world map
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')
    
    # Convert coordinates to map coordinates
    true_lon, true_lat = true_coords[0], true_coords[1]
    pred_lon, pred_lat = pred_coords[0], pred_coords[1]
    
    # Plot true coordinates
    x_true, y_true = m(true_lon, true_lat)
    m.scatter(x_true, y_true, marker='o', color='green', s=100, 
               label=f'True: ({true_lon:.2f}°, {true_lat:.2f}°)',
               edgecolors='black', linewidth=2, zorder=5)
    
    # Plot predicted coordinates
    x_pred, y_pred = m(pred_lon, pred_lat)
    m.scatter(x_pred, y_pred, marker='x', color='red', s=100,
               label=f'Pred: ({pred_lon:.2f}°, {pred_lat:.2f}°)',
               linewidths=3, zorder=5)
    
    # Draw line between true and predicted
    m.plot([x_true, x_pred], [y_true, y_pred], 'b--', alpha=0.7, linewidth=2,
            label=f'Error: {calculate_distance(true_coords, pred_coords):.1f} km')
    
    # Add grid lines
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=10)
    m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0], fontsize=10)
    
    ax2.set_title('Coordinate Prediction on World Map')
    ax2.legend(loc='upper right')
    
    # Add error information
    error_km = calculate_distance(true_coords, pred_coords)
    error_deg = np.sqrt((true_lon - pred_lon)**2 + (true_lat - pred_lat)**2)
    
    fig.text(0.5, 0.02, 
             f'Prediction Error: {error_deg:.3f}° ({error_km:.1f} km)',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_distance(coord1, coord2):
    """Calculate Haversine distance between two coordinates in km."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Handle longitude wraparound
    if dlon > math.pi:
        dlon = dlon - 2*math.pi
    elif dlon < -math.pi:
        dlon = dlon + 2*math.pi
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return 6371.0 * c  # Earth radius in km


def main():
    parser = argparse.ArgumentParser(description="Test single image prediction")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--show", action="store_true",
                       help="Show plots")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(config, args.model_path)
    
    # Create dataloader
    _, _, test_loader = create_dataloaders(config, batch_size=1, num_workers=0)
    
    # Get single test sample
    data_iter = iter(test_loader)
    image, true_coords = next(data_iter)
    
    # Make prediction
    logger.info("Making prediction...")
    pred_coords, true_coords_denorm = predict_single_image(
        model, image, true_coords, config
    )
    
    # Calculate error
    error_km = calculate_distance(true_coords_denorm, pred_coords)
    error_deg = np.sqrt((true_coords_denorm[0] - pred_coords[0])**2 + 
                      (true_coords_denorm[1] - pred_coords[1])**2)
    
    # Print results
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION RESULTS")
    print("="*60)
    print(f"True Coordinates:  ({true_coords_denorm[0]:.4f}°, {true_coords_denorm[1]:.4f}°)")
    print(f"Pred Coordinates:  ({pred_coords[0]:.4f}°, {pred_coords[1]:.4f}°)")
    print(f"Longitude Error:   {abs(true_coords_denorm[0] - pred_coords[0]):.4f}°")
    print(f"Latitude Error:    {abs(true_coords_denorm[1] - pred_coords[1]):.4f}°")
    print(f"Coordinate Error:  {error_deg:.4f}°")
    print(f"Distance Error:    {error_km:.1f} km")
    print("="*60)
    
    # Create visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / "single_prediction_test.png"
    logger.info(f"Saving visualization to {save_path}")
    
    plot_prediction_comparison(
        image.squeeze(),
        true_coords_denorm,
        pred_coords,
        save_path=str(save_path),
        show_plot=args.show
    )
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()