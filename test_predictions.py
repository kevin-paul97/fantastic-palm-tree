#!/usr/bin/env python3
"""
Test model predictions on satellite images - single or multiple samples.
"""

import argparse
import math
import random
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pathlib import Path
import logging

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

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    true_coords = true_coords.to(device)

    with torch.no_grad():
        prediction = model(image)

    normalizer = CoordinateNormalizer()
    pred_coords = normalizer.denormalize(prediction)

    true_coords_denorm = true_coords.squeeze().cpu().numpy()
    pred_coords_np = pred_coords.squeeze().cpu().numpy()

    return pred_coords_np, true_coords_denorm


def haversine_km(coord1, coord2):
    """Calculate Haversine distance between two [lon, lat] coordinates in km."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    if dlon > math.pi:
        dlon -= 2 * math.pi
    elif dlon < -math.pi:
        dlon += 2 * math.pi

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


def plot_single(image_tensor, true_coords, pred_coords, save_path=None, show_plot=True):
    """Plot image alongside prediction on world map."""
    img_np = image_tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
    ax1.set_title('Satellite Image')
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawmapboundary(fill_color='lightblue')

    true_lon, true_lat = true_coords[0], true_coords[1]
    pred_lon, pred_lat = pred_coords[0], pred_coords[1]

    x_true, y_true = m(true_lon, true_lat)
    m.scatter(x_true, y_true, marker='o', color='green', s=100,
              label=f'True: ({true_lon:.2f}, {true_lat:.2f})',
              edgecolors='black', linewidth=2, zorder=5)

    x_pred, y_pred = m(pred_lon, pred_lat)
    m.scatter(x_pred, y_pred, marker='x', color='red', s=100,
              label=f'Pred: ({pred_lon:.2f}, {pred_lat:.2f})',
              linewidths=3, zorder=5)

    error_km = haversine_km(true_coords, pred_coords)
    m.plot([x_true, x_pred], [y_true, y_pred], 'b--', alpha=0.7, linewidth=2,
           label=f'Error: {error_km:.1f} km')

    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=10)
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10)
    ax2.set_title('Coordinate Prediction')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiple(images, true_list, pred_list, save_path=None, show_plot=True):
    """Plot multiple image predictions with world maps."""
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    for i in range(n):
        ax_img = plt.subplot(rows, cols * 2, i * 2 + 1)
        img_np = images[i].squeeze().cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        ax_img.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        ax_img.set_title(f'Image {i + 1}')
        ax_img.axis('off')

        ax_map = plt.subplot(rows, cols * 2, i * 2 + 2)
        m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')

        true_lon, true_lat = true_list[i][0], true_list[i][1]
        pred_lon, pred_lat = pred_list[i][0], pred_list[i][1]

        x_true, y_true = m(true_lon, true_lat)
        m.scatter(x_true, y_true, marker='o', color='green', s=80,
                  label='True', edgecolors='black', linewidth=1.5, zorder=5)

        x_pred, y_pred = m(pred_lon, pred_lat)
        m.scatter(x_pred, y_pred, marker='x', color='red', s=80,
                  label='Pred', linewidths=2, zorder=5)

        error_km = haversine_km(true_list[i], pred_list[i])
        m.plot([x_true, x_pred], [y_true, y_pred], 'b--', alpha=0.7, linewidth=1.5)

        ax_map.text(0.02, 0.98, f'{error_km:.0f} km',
                    transform=ax_map.transAxes, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), va='top')
        ax_map.set_title(f'Map {i + 1}')
        if i == 0:
            ax_map.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test model predictions on satellite images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of test samples (1 for single, >1 for multiple)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = Config.from_dict(json.load(f))
    else:
        config = Config()

    model = load_model(config, args.model_path)
    _, _, test_loader = create_dataloaders(config, batch_size=1, num_workers=0)

    all_data = list(test_loader)
    random.shuffle(all_data)
    samples = all_data[:args.num_samples]

    images, true_list, pred_list = [], [], []

    for i, (image, true_coords) in enumerate(samples):
        pred, true_denorm = predict_single_image(model, image, true_coords, config)
        images.append(image.squeeze())
        true_list.append(true_denorm)
        pred_list.append(pred)

        error_km = haversine_km(true_denorm, pred)
        error_deg = np.sqrt((true_denorm[0] - pred[0]) ** 2 + (true_denorm[1] - pred[1]) ** 2)
        print(f"Sample {i + 1}: True=({true_denorm[0]:.2f}, {true_denorm[1]:.2f})  "
              f"Pred=({pred[0]:.2f}, {pred[1]:.2f})  Error={error_km:.1f} km ({error_deg:.3f} deg)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.num_samples == 1:
        save_path = output_dir / "single_prediction_test.png"
        plot_single(images[0], true_list[0], pred_list[0],
                    save_path=str(save_path), show_plot=args.show)
    else:
        save_path = output_dir / f"multiple_predictions_{args.num_samples}.png"
        plot_multiple(images, true_list, pred_list,
                      save_path=str(save_path), show_plot=args.show)

    errors = [haversine_km(true_list[i], pred_list[i]) for i in range(len(pred_list))]
    print(f"\nMean={np.mean(errors):.1f} km  Median={np.median(errors):.1f} km  "
          f"Min={np.min(errors):.1f} km  Max={np.max(errors):.1f} km")


if __name__ == "__main__":
    main()
