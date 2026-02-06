#!/usr/bin/env python3
"""
Enhanced logging utilities for better training visibility.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Setup comprehensive logging for training."""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Enhanced logger for training metrics and progress."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logger for training-specific logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log = self.log_dir / f"training_{timestamp}.log"
        
        setup_logging(
            log_file=str(training_log),
            console_output=False  # Avoid duplicate console output
        )
        self.logger = logging.getLogger("training")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        if self.logger:
            self.logger.info(f"Starting Epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float, duration: float):
        """Log epoch completion."""
        if self.logger:
            self.logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Duration: {duration:.2f}s"
            )
    
    def log_model_save(self, model_path: str, is_best: bool = False):
        """Log model checkpoint."""
        if self.logger:
            status = "best" if is_best else "final"
            self.logger.info(f"Saved {status} model checkpoint: {model_path}")
    
    def log_training_complete(self, total_time: float, best_val_loss: float):
        """Log training completion."""
        if self.logger:
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.logger.info(
                f"Training completed - "
                f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}, "
                f"Best validation loss: {best_val_loss:.6f}"
            )
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int):
        """Log dataset information."""
        if self.logger:
            self.logger.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def log_model_info(self, model_name: str, total_params: int, trainable_params: int):
        """Log model information."""
        if self.logger:
            self.logger.info(
                f"Model: {model_name} - "
                f"Total parameters: {total_params:,}, "
                f"Trainable parameters: {trainable_params:,}"
            )
    
    def log_device_info(self, device: str):
        """Log training device information."""
        if self.logger:
            self.logger.info(f"Training device: {device}")
    
    def log_config_summary(self, config_dict: dict):
        """Log configuration summary."""
        if self.logger:
            self.logger.info("Training Configuration:")
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    self.logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"    {sub_key}: {sub_value}")
                else:
                    self.logger.info(f"  {key}: {value}")


def create_progress_callback(total_steps: int, description: str = "Processing"):
    """Create a progress callback for long-running operations."""
    
    def callback(current_step: int):
        progress = (current_step / total_steps) * 100
        print(f"\r{description}: {progress:.1f}% ({current_step}/{total_steps})", end="", flush=True)
        
        if current_step == total_steps:
            print()  # New line when complete
    
    return callback


if __name__ == "__main__":
    # Test logging setup
    setup_logging("INFO", "logs/test.log")
    logger = logging.getLogger("test")
    
    logger.info("Test logging system")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test training logger
    training_logger = TrainingLogger("logs")
    training_logger.log_epoch_start(0, 10)
    training_logger.log_epoch_end(0, 0.5, 0.4, 120.5)
    training_logger.log_training_complete(7200, 0.25)