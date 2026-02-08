"""
Simplified main script with consolidated commands and improved structure.
"""

import logging
import argparse
import torch
from pathlib import Path
from typing import Optional

from config import Config
from data import EPICDataDownloader, CoordinateExtractor
from datasets import create_dataloaders
from models import create_location_regressor, create_autoencoder
from training import UnifiedTrainer, LocationRegressorTrainer, AutoEncoderTrainer
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
    if not Path(config.data.images_dir).exists():
        logger.info("Downloading daily data...")
        downloader.download_all_images()
    
    logger.info("Data pipeline setup complete!")
    
    # Create coordinate statistics
    extractor = CoordinateExtractor(config)
    stats = extractor.coordinate_statistics
    logger.info(f"Coordinate range: Lat [{stats['min_lat']:.3f}, {stats['max_lat']:.3f}], "
                f"Lon [{stats['min_lon']:.3f}, {stats['max_lon']:.3f}]")
    
    # Create visualizations
    logger.info("Creating coordinate distribution plots...")
    plot_coordinate_distribution(stats)
    plot_world_map_with_coordinates(stats)
    create_coordinate_statistics_table(stats)


def train_model(config, model_type: str = "regressor"):
    """Unified training function for both model types."""
    logger.info(f"Training {model_type} model...")
    
    # Create model and data
    if model_type == "regressor":
        model = create_location_regressor(config)
        trainer_class = LocationRegressorTrainer
    elif model_type == "autoencoder":
        model = create_autoencoder(config)
        trainer_class = AutoEncoderTrainer
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Setup directories
    import os
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Create trainer
    trainer = trainer_class(model, train_loader, val_loader, config)
    
    try:
        # Train model
        results = trainer.train()
        logger.info(f"Training complete! Best validation loss: {results['best_val_loss']:.4f}")
        
        # Save final model
        final_model_path = os.path.join(config.training.save_dir, f"{model_type}_final.pth")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Model saved to: {final_model_path}")
        
        return final_model_path
        
    finally:
        # Cleanup
        trainer.cleanup()


def evaluate_model_performance(config, model_path: str):
    """Unified model evaluation with comprehensive metrics."""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model and create data
    model = create_location_regressor(config)
    model.load_state_dict(torch.load(model_path, map_location=config.training.device))
    
    _, _, test_loader = create_dataloaders(config)
    
    # Create trainer for evaluation
    _, _, test_loader = create_dataloaders(config)
    trainer = LocationRegressorTrainer(model, None, test_loader, config)
    
    # Evaluate with coordinate metrics
    metrics = trainer.evaluate_coordinates(test_loader)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Mean coordinate error: {metrics['mean_coordinate_error_deg']:.4f}°")
    logger.info(f"  Median coordinate error: {metrics['median_coordinate_error_deg']:.4f}°")
    logger.info(f"  Mean Haversine distance: {metrics['mean_haversine_km']:.1f} km")
    logger.info(f"  Median Haversine distance: {metrics['median_haversine_km']:.1f} km")
    
    return metrics


def download_data(config, mode: str = "recent", num_days: int = 7, num_images: int = 100):
    """Consolidated data download function."""
    logger.info(f"Downloading data in {mode} mode...")
    
    downloader = EPICDataDownloader(config)
    
    if mode == "recent":
        # Download recent days
        logger.info(f"Downloading images from last {num_days} days...")
        downloader.download_recent_days(num_days)
        
    elif mode == "latest":
        # Download latest images
        logger.info(f"Downloading {num_images} latest images...")
        downloader.download_latest_images(num_images)
        
    elif mode == "all":
        # Download all available data
        logger.info("Downloading all available images...")
        downloader.download_all_images()
    
    else:
        raise ValueError(f"Unknown download mode: {mode}")
    
    logger.info("Download complete!")


def main():
    """Main function with simplified command structure."""
    parser = argparse.ArgumentParser(
        description="Satellite Image Coordinate Prediction - Consolidated Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                    # Setup complete data pipeline
  %(prog)s train regressor          # Train location regressor
  %(prog)s train autoencoder        # Train autoencoder
  %(prog)s evaluate model.pth       # Evaluate trained model
  %(prog)s download recent 7        # Download last 7 days
  %(prog)s download latest 50       # Download 50 latest images
        """
    )
    
    # Main command
    parser.add_argument("command", 
                       choices=["setup", "train", "evaluate", "download"],
                       help="Command to execute")
    
    # Command-specific arguments
    parser.add_argument("target", nargs="?",
                       help="Target for command (model type or download mode)")
    parser.add_argument("value", nargs="?", type=int,
                       help="Value for command (days/images count)")
    
    # Configuration and overrides
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
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
    
    # Set threading
    torch.set_num_interop_threads(config.training.num_threads)
    torch.set_num_threads(config.training.num_threads)
    
    try:
        # Execute command
        if args.command == "setup":
            setup_data_pipeline(config)
            
        elif args.command == "train":
            if not args.target or args.target not in ["regressor", "autoencoder"]:
                raise ValueError("Training requires specifying 'regressor' or 'autoencoder'")
            train_model(config, args.target)
            
        elif args.command == "evaluate":
            if not args.target:
                raise ValueError("Evaluation requires specifying model path")
            evaluate_model_performance(config, args.target)
            
        elif args.command == "download":
            if not args.target or args.target not in ["recent", "latest", "all"]:
                raise ValueError("Download requires specifying 'recent', 'latest', or 'all'")
            
            num_days = args.value if args.value else 7
            num_images = args.value if args.value else 100
            download_data(config, args.target, num_days, num_images)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())