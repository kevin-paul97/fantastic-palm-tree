"""
Main script for training satellite image coordinate prediction models.
"""

import logging
import argparse
import torch
from pathlib import Path

from config import Config
from data import EPICDataDownloader, CoordinateExtractor
from datasets import create_dataloaders
from models import create_location_regressor, create_autoencoder
from training import LocationRegressorTrainer, AutoEncoderTrainer
from visualization import (
    plot_coordinate_distribution,
    plot_world_map_with_coordinates,
    create_coordinate_statistics_table
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup enhanced logging if available
try:
    from logging_utils import setup_logging
    setup_logging("INFO")
except ImportError:
    pass


def setup_data_pipeline(config):
    """Setup the complete data pipeline."""
    logger.info("Setting up data pipeline...")
    
    # Initialize downloader
    downloader = EPICDataDownloader(config)
    
    # Download metadata if needed
    if not Path(config.data.raw_data_dir).exists():
        logger.info("Downloading metadata...")
        downloader.download_metadata()
    
    # Download daily data if needed
    if not Path(config.data.combined_dir).exists():
        logger.info("Downloading daily data...")
        downloader.download_all_images()
        downloader.consolidate_metadata()
    
    # Extract coordinates for analysis
    extractor = CoordinateExtractor(config)
    lat_coords, lon_coords = extractor.extract_coordinates()
    
    # Create visualizations
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    plot_coordinate_distribution(
        lat_coords, lon_coords,
        save_path=str(output_dir / "coordinate_distribution.png"),
        show_plot=False
    )
    
    plot_world_map_with_coordinates(
        lat_coords, lon_coords,
        save_path=str(output_dir / "world_map_coordinates.png"),
        show_plot=False
    )
    
    # Print statistics
    stats = create_coordinate_statistics_table(lat_coords, lon_coords)
    logger.info("Coordinate Statistics:\n" + str(stats))
    
    logger.info("Data pipeline setup complete!")
    return lat_coords, lon_coords


def train_location_regressor(config):
    """Train the location regression model."""
    logger.info("Training LocationRegressor model...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    # Create model
    model = create_location_regressor(config)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create trainer
    trainer = LocationRegressorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    history = trainer.train(num_epochs=config.training.epochs)
    
    logger.info(f"Training complete! Best validation loss: {trainer.best_val_loss:.6f}")
    return model, history


def train_autoencoder(config):
    """Train the autoencoder model."""
    logger.info("Training AutoEncoder model...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    # Create model
    model = create_autoencoder(config)
    logger.info(f"AutoEncoder has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create trainer
    trainer = AutoEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    history = trainer.train(num_epochs=config.training.epochs)
    
    logger.info(f"AutoEncoder training complete! Best validation loss: {trainer.best_val_loss:.6f}")
    return model, history


def evaluate_model(config, model_path: str):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model from {model_path}...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    # Load model
    model = create_location_regressor(config)
    checkpoint = torch.load(model_path, map_location=config.training.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    model.eval()
    device = torch.device(config.training.device)
    model.to(device)
    
    from datasets import CoordinateNormalizer
    normalizer = CoordinateNormalizer()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, coords in test_loader:
            images = images.to(device)
            coords = coords.to(device)
            
            predictions = model(images)
            
            all_predictions.append(predictions)
            all_targets.append(coords)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mse_loss = torch.nn.functional.mse_loss(all_predictions, normalizer.normalize(all_targets))
    
    # Denormalize for coordinate error calculation
    pred_coords = normalizer.denormalize(all_predictions)
    true_coords = normalizer.denormalize(all_targets)
    
    # Calculate coordinate errors in degrees
    lon_errors = torch.abs(pred_coords[:, 0] - true_coords[:, 0])
    lat_errors = torch.abs(pred_coords[:, 1] - true_coords[:, 1])
    coord_errors = torch.sqrt(lon_errors**2 + lat_errors**2)
    
    logger.info(f"Test MSE: {mse_loss:.6f}")
    logger.info(f"Mean coordinate error: {coord_errors.mean():.3f} degrees")
    logger.info(f"Median coordinate error: {coord_errors.median():.3f} degrees")
    
    # Additional statistics
    logger.info(f"Longitude mean error: {lon_errors.mean():.3f} degrees")
    logger.info(f"Latitude mean error: {lat_errors.mean():.3f} degrees")
    logger.info(f"Min coordinate error: {coord_errors.min():.3f} degrees")
    logger.info(f"Max coordinate error: {coord_errors.max():.3f} degrees")
    
    # Create visualizations
    from visualization import plot_coordinate_predictions, plot_error_distribution
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    plot_coordinate_predictions(
        all_targets, all_predictions,
        save_path=str(output_dir / "coordinate_predictions.png"),
        show_plot=False
    )
    
    plot_error_distribution(
        all_targets, all_predictions,
        save_path=str(output_dir / "error_distribution.png"),
        show_plot=False
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Satellite Image Coordinate Prediction")
    parser.add_argument("--mode", choices=["setup", "train_regressor", "train_autoencoder", "evaluate"], 
                       default="train_regressor", help="Mode to run")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to trained model for evaluation")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], 
                       help="Training device (auto-detects: cuda > mps > cpu)")
    parser.add_argument("--no-tensorboard", action="store_true", 
                       help="Disable automatic TensorBoard launch")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    if args.no_tensorboard:
        config.training.launch_tensorboard = False
    
    # Ensure directories exist
    import os
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Set number of threads
    torch.set_num_interop_threads(config.training.num_threads)
    torch.set_num_threads(config.training.num_threads)
    
    # Run specified mode
    if args.mode == "setup":
        setup_data_pipeline(config)
    elif args.mode == "train_regressor":
        train_location_regressor(config)
    elif args.mode == "train_autoencoder":
        train_autoencoder(config)
    elif args.mode == "evaluate":
        if not args.model_path:
            raise ValueError("Model path required for evaluation")
        evaluate_model(config, args.model_path)


if __name__ == "__main__":
    main()