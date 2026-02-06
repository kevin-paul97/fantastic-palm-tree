#!/usr/bin/env python3
"""
Test multiple single image predictions and visualize on world maps.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pathlib import Path
import logging
import random

from config import Config
from datasets import create_dataloaders, CoordinateNormalizer
from models import create_location_regressor

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
    
    # True coordinates are already in real-world coordinates
    true_coords_denorm = true_coords.squeeze().cpu().numpy()
    pred_coords_np = pred_coords.squeeze().cpu().numpy()
    
    return pred_coords_np, true_coords_denorm


def calculate_distance(coord1, coord2):
    """Calculate Haversine distance between two coordinates in km."""
    import math
    
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


def plot_multiple_predictions(
    images,
    true_coords_list,
    pred_coords_list,
    save_path: str = None,
    show_plot: bool = True
):
    """Plot multiple image predictions with coordinates on world maps."""
    num_samples = len(images)
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    for i in range(num_samples):
        # Create subplot for image
        ax_img = plt.subplot(rows, cols*2, i*2 + 1)
        
        # Convert image tensor to numpy for display
        img_np = images[i].squeeze().cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        if img_np.ndim == 2:
            ax_img.imshow(img_np, cmap='gray')
        else:
            ax_img.imshow(img_np)
        
        ax_img.set_title(f'Image {i+1}')
        ax_img.axis('off')
        
        # Create subplot for world map
        ax_map = plt.subplot(rows, cols*2, i*2 + 2)
        
        # Create world map
        m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')
        
        # Draw map features
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')
        
        # Get coordinates
        true_coords = true_coords_list[i]
        pred_coords = pred_coords_list[i]
        
        true_lon, true_lat = true_coords[0], true_coords[1]
        pred_lon, pred_lat = pred_coords[0], pred_coords[1]
        
        # Plot true coordinates
        x_true, y_true = m(true_lon, true_lat)
        m.scatter(x_true, y_true, marker='o', color='green', s=80, 
                   label=f'True', edgecolors='black', linewidth=1.5, zorder=5)
        
        # Plot predicted coordinates
        x_pred, y_pred = m(pred_lon, pred_lat)
        m.scatter(x_pred, y_pred, marker='x', color='red', s=80,
                   label=f'Pred', linewidths=2, zorder=5)
        
        # Draw line between true and predicted
        error_km = calculate_distance(true_coords, pred_coords)
        m.plot([x_true, x_pred], [y_true, y_pred], 'b--', alpha=0.7, linewidth=1.5)
        
        # Add error text
        ax_map.text(0.02, 0.98, f'{error_km:.0f} km', 
                   transform=ax_map.transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   va='top')
        
        ax_map.set_title(f'Prediction Map {i+1}')
        if i == 0:  # Only add legend to first map
            ax_map.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test multiple single image predictions")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=6,
                       help="Number of test samples to evaluate")
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
    
    # Get multiple test samples
    all_data = list(test_loader)
    random.shuffle(all_data)
    sample_data = all_data[:args.num_samples]
    
    # Make predictions
    logger.info(f"Making predictions for {args.num_samples} samples...")
    images = []
    true_coords_list = []
    pred_coords_list = []
    
    for i, (image, true_coords) in enumerate(sample_data):
        pred_coords, true_coords_denorm = predict_single_image(
            model, image, true_coords, config
        )
        
        images.append(image.squeeze())
        true_coords_list.append(true_coords_denorm)
        pred_coords_list.append(pred_coords)
        
        # Calculate error
        error_km = calculate_distance(true_coords_denorm, pred_coords)
        error_deg = np.sqrt((true_coords_denorm[0] - pred_coords[0])**2 + 
                          (true_coords_denorm[1] - pred_coords[1])**2)
        
        print(f"Sample {i+1}:")
        print(f"  True:  ({true_coords_denorm[0]:.2f}°, {true_coords_denorm[1]:.2f}°)")
        print(f"  Pred:  ({pred_coords[0]:.2f}°, {pred_coords[1]:.2f}°)")
        print(f"  Error: {error_km:.1f} km ({error_deg:.3f}°)")
        print()
    
    # Create visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / f"multiple_predictions_test_{args.num_samples}.png"
    logger.info(f"Saving visualization to {save_path}")
    
    plot_multiple_predictions(
        images,
        true_coords_list,
        pred_coords_list,
        save_path=str(save_path),
        show_plot=args.show
    )
    
    # Calculate average error
    errors = [calculate_distance(true_coords_list[i], pred_coords_list[i]) 
              for i in range(len(pred_coords_list))]
    
    print("="*60)
    print(f"RESULTS SUMMARY ({len(errors)} samples)")
    print("="*60)
    print(f"Average error: {np.mean(errors):.1f} km")
    print(f"Median error:  {np.median(errors):.1f} km")
    print(f"Min error:     {np.min(errors):.1f} km")
    print(f"Max error:     {np.max(errors):.1f} km")
    print("="*60)
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()