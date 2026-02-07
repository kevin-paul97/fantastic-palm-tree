"""
Enhanced training utilities with improved TensorBoard logging and error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.utils.data import DataLoader
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time
from tqdm import tqdm

from models import LocationRegressor, AutoEncoder
from datasets import CoordinateNormalizer

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced trainer class with robust TensorBoard logging."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(config.training.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = self._setup_criterion()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Ensure log directory exists
        import os
        os.makedirs(config.training.log_dir, exist_ok=True)
        
        # Setup tensorboard writer with enhanced error handling
        if SummaryWriter is not None:
            try:
                # Ensure log directory exists and is writable
                import os
                os.makedirs(config.training.log_dir, exist_ok=True)
                
                # Test write access
                test_file = os.path.join(config.training.log_dir, 'test_write')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    raise PermissionError(f"Cannot write to log directory {config.training.log_dir}: {e}")
                
                self.writer = SummaryWriter(
                    log_dir=config.training.log_dir,
                    comment=f'_device_{config.training.device}_batch_{config.training.batch_size}',
                    flush_secs=10  # Auto-flush every 10 seconds
                )
                
                # Test writer functionality
                self.writer.add_scalar('test/init', 1.0, 0)
                self.writer.flush()
                self.writer.add_scalar('test/init', 2.0, 1)
                self.writer.flush()
                
                # Log configuration
                self._log_config_to_tensorboard()
                
                # Launch tensorboard automatically if enabled
                if config.training.launch_tensorboard:
                    self._launch_tensorboard(config.training.log_dir)
                    
                logger.info("TensorBoard writer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TensorBoard writer: {e}")
                self.writer = None
        else:
            self.writer = None
            logger.warning("TensorBoard SummaryWriter not available")
        
        # Setup coordinate normalizer
        self.coord_normalizer = CoordinateNormalizer()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Tensorboard process tracking
        self.tensorboard_process = None
    
    def _launch_tensorboard(self, log_dir: str) -> None:
        """Launch TensorBoard server for training visualization."""
        try:
            import subprocess
            import webbrowser
            import time
            import os
            import sys
            
            # Check if log directory exists
            if not os.path.exists(log_dir):
                logger.warning(f"Log directory does not exist: {log_dir}")
                return
            
            # Check if tensorboard is already running on this port
            tensorboard_port = 6006
            try:
                # Check if port is in use
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', tensorboard_port))
                sock.close()
                
                if result == 0:
                    logger.info(f"TensorBoard already running on http://localhost:{tensorboard_port}")
                    try:
                        webbrowser.open(f"http://localhost:{tensorboard_port}")
                    except Exception:
                        logger.info("Could not open browser automatically")
                    return
            except Exception:
                pass
            
            # Test tensorboard availability
            test_cmd = [sys.executable, '-c', 'import tensorboard; print("OK")']
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    raise ImportError("TensorBoard not properly installed")
            except Exception as e:
                raise ImportError(f"TensorBoard not available: {e}")
            
            # Launch tensorboard with enhanced error handling
            cmd = [
                sys.executable, '-m', 'tensorboard.main',
                '--logdir', log_dir,
                '--port', str(tensorboard_port),
                '--host', 'localhost',
                '--reload_interval', '5'  # Reload every 5 seconds
            ]
            
            logger.info(f"Starting TensorBoard with command: {' '.join(cmd)}")
            
            # Start tensorboard process with better error handling
            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            # Wait for process to start and check for errors
            time.sleep(3)
            
            # Check if process is still running
            if self.tensorboard_process.poll() is None:
                logger.info("TensorBoard process started successfully")
                try:
                    webbrowser.open(f"http://localhost:{tensorboard_port}")
                    logger.info("TensorBoard opened in browser")
                except Exception:
                    logger.info("Could not open browser automatically - open http://localhost:{tensorboard_port} manually")
            else:
                stdout, stderr = self.tensorboard_process.communicate()
                logger.error(f"TensorBoard failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                self.tensorboard_process = None
                return
            
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        except Exception as e:
            logger.warning(f"Failed to launch TensorBoard: {e}")
            self.tensorboard_process = None
    
    def stop_tensorboard(self) -> None:
        """Stop TensorBoard process if running."""
        if self.tensorboard_process:
            try:
                # Check if process is still running
                if self.tensorboard_process.poll() is None:
                    self.tensorboard_process.terminate()
                    try:
                        self.tensorboard_process.wait(timeout=5)
                        logger.info("TensorBoard stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.tensorboard_process.kill()
                        self.tensorboard_process.wait()
                        logger.info("TensorBoard killed forcefully")
                else:
                    logger.info("TensorBoard process already terminated")
                    
            except Exception as e:
                logger.warning(f"Error stopping TensorBoard: {e}")
            finally:
                self.tensorboard_process = None
                
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop TensorBoard
            self.stop_tensorboard()
            
            # Close TensorBoard writer
            if self.writer is not None:
                try:
                    self.writer.close()
                    logger.info("TensorBoard writer closed")
                except Exception as e:
                    logger.warning(f"Error closing TensorBoard writer: {e}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    def _log_config_to_tensorboard(self) -> None:
        """Log training configuration to TensorBoard."""
        if self.writer is None:
            return
        
        try:
            import json
            
            # Log hyperparameters
            config_dict = {
                'data': {
                    'image_size': self.config.data.image_size,
                    'grayscale': self.config.data.grayscale,
                    'train_split': self.config.data.train_split,
                    'val_split': self.config.data.val_split,
                    'test_split': self.config.data.test_split
                },
                'model': {
                    'input_channels': self.config.model.input_channels,
                    'conv_channels': self.config.model.conv_channels,
                    'kernel_size': self.config.model.kernel_size,
                    'pool_size': self.config.model.pool_size,
                    'activation': self.config.model.activation,
                    'hidden_dim': self.config.model.hidden_dim,
                    'output_dim': self.config.model.output_dim
                },
                'training': {
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'optimizer': self.config.training.optimizer,
                    'weight_decay': self.config.training.weight_decay,
                    'device': self.config.training.device,
                    'num_threads': self.config.training.num_threads
                }
            }
            
            # Log as text for easy reading
            config_text = json.dumps(config_dict, indent=2)
            self.writer.add_text('Config/Training_Configuration', config_text, 0)
            
            # Log individual hyperparameters
            hparams = {
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'optimizer': self.config.training.optimizer,
                'image_size': self.config.data.image_size,
                'grayscale': self.config.data.grayscale,
                'device': self.config.training.device
            }
            
            # Log model architecture info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            model_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_type': self.model.__class__.__name__
            }
            
            # Combine all hyperparameters
            all_hparams = {**hparams, **model_info}
            
            # Write hyperparameters to TensorBoard
            self.writer.add_hparams(all_hparams, {'metric/placeholder': 0})
            
            logger.info("Configuration logged to TensorBoard")
            
        except Exception as e:
            logger.warning(f"Failed to log configuration to TensorBoard: {e}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        return nn.MSELoss()
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler.ReduceLROnPlateau]:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
    
    def _log_batch_to_tensorboard(self, loss: float, batch_idx: int) -> None:
        """Log batch-level metrics to TensorBoard with error handling."""
        if self.writer is None:
            return
        
        try:
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss, global_step)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate/Current', current_lr, global_step)
            
            # Log batch statistics every 10 batches
            if batch_idx % 10 == 0:
                # Log gradient norms
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.writer.add_scalar('Gradients/Total_Norm', total_norm, global_step)
                
                # Log weight and gradient histograms
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(f'Weights/{name}', param.data, global_step)
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)
            
            # Flush every 50 batches to ensure logs are written
            if batch_idx % 50 == 0:
                self.writer.flush()
                
        except Exception as e:
            logger.warning(f"Failed to log batch statistics to TensorBoard: {e}")
    
    def _log_epoch_to_tensorboard(self, train_loss: float, val_loss: float, epoch_time: float) -> None:
        """Log epoch-level metrics to TensorBoard with error handling."""
        if self.writer is None:
            return
        
        try:
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/Val_Epoch', val_loss, self.current_epoch)
            self.writer.add_scalar('Learning_Rate/Epoch', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
            # Log epoch summary statistics
            self.writer.add_scalar('Performance/Epoch_Duration', epoch_time, self.current_epoch)
            self.writer.add_scalar('Performance/Batches_per_Second', len(self.train_loader) / epoch_time, self.current_epoch)
            
            # Log model parameter statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.writer.add_scalar('Model/Total_Parameters', total_params, self.current_epoch)
            self.writer.add_scalar('Model/Trainable_Parameters', trainable_params, self.current_epoch)
            
            # Flush to ensure data is written to disk
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log epoch statistics to TensorBoard: {e}")
    
    def train_epoch(self) -> float:
        """Train for one epoch with enhanced error handling."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        try:
            for batch_idx, (images, coords) in enumerate(pbar):
                try:
                    images = images.to(self.device)
                    coords = coords.to(self.device)
                    
                    # Normalize coordinates
                    coords_norm = self.coord_normalizer.normalize(coords)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, coords_norm)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # Log to tensorboard
                    self._log_batch_to_tensorboard(loss.item(), batch_idx)
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in training epoch {self.current_epoch + 1}: {e}")
            raise
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch with enhanced error handling."""
        self.model.eval()
        total_loss = 0.0
        
        try:
            with torch.no_grad():
                for batch_idx, (images, coords) in enumerate(self.val_loader):
                    try:
                        images = images.to(self.device)
                        coords = coords.to(self.device)
                        
                        # Normalize coordinates
                        coords_norm = self.coord_normalizer.normalize(coords)
                        
                        # Forward pass
                        outputs = self.model(images)
                        loss = self.criterion(outputs, coords_norm)
                        
                        total_loss += loss.item()
                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error in validation epoch {self.current_epoch + 1}: {e}")
            raise
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Train model for specified number of epochs with enhanced logging."""
        # Initialize enhanced logging
        try:
            from logging_utils import TrainingLogger
            self.training_logger = TrainingLogger(self.config.training.log_dir)
            self.training_logger.log_device_info(str(self.device))
            self.training_logger.log_dataset_info(
                len(self.train_loader) if hasattr(self.train_loader, '__len__') else 0,
                len(self.val_loader) if hasattr(self.val_loader, '__len__') else 0,
                0  # Will be set when test_loader is available
            )
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.training_logger.log_model_info(
                self.model.__class__.__name__,
                total_params,
                trainable_params
            )
        except ImportError:
            self.training_logger = None
        
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Track epoch start time
                epoch_start_time = time.time()
                
                # Train
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Learning rate scheduler step
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Calculate epoch duration
                epoch_end_time = time.time()
                training_time = epoch_end_time - epoch_start_time
                
                # Log epoch metrics
                self._log_epoch_to_tensorboard(train_loss, val_loss, training_time)
                
                # Enhanced logging
                if hasattr(self, 'training_logger') and self.training_logger:
                    self.training_logger.log_epoch_end(epoch, train_loss, val_loss, training_time)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                
                # Log progress
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Save final model
            self.save_checkpoint('final_model.pth')
            
            # Close tensorboard writer
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception as e:
                    logger.warning(f"Error closing TensorBoard writer: {e}")
            
            # Stop tensorboard process
            self.stop_tensorboard()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint with error handling."""
        try:
            checkpoint_path = Path(self.config.training.save_dir) / filename
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'config': self.config
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint with error handling."""
        try:
            checkpoint_path = Path(self.config.training.save_dir) / filename
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


# Enhanced trainer classes for specific models
class LocationRegressorTrainer(EnhancedTrainer):
    """Enhanced trainer specifically for LocationRegressor model."""
    
    def compute_coordinate_error(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute coordinate prediction error in degrees."""
        # Denormalize predictions and targets
        outputs_denorm = self.coord_normalizer.denormalize(outputs)
        targets_denorm = self.coord_normalizer.denormalize(targets)
        
        # Compute Euclidean distance in degrees
        diff = outputs_denorm - targets_denorm
        distance = torch.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
        
        return distance.mean()
    
    def validate_epoch(self) -> float:
        """Validate with additional coordinate error metrics."""
        self.model.eval()
        total_loss = 0.0
        total_coord_error = 0.0
        
        try:
            with torch.no_grad():
                for images, coords in self.val_loader:
                    try:
                        images = images.to(self.device)
                        coords = coords.to(self.device)
                        
                        # Normalize coordinates
                        coords_norm = self.coord_normalizer.normalize(coords)
                        
                        # Forward pass
                        outputs = self.model(images)
                        loss = self.criterion(outputs, coords_norm)
                        
                        # Compute coordinate error
                        coord_error = self.compute_coordinate_error(outputs, coords_norm)
                        
                        total_loss += loss.item()
                        total_coord_error += coord_error.item()
                    except Exception as e:
                        logger.error(f"Error in validation batch: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error in validation epoch: {e}")
            raise
        
        avg_loss = total_loss / len(self.val_loader)
        avg_coord_error = total_coord_error / len(self.val_loader)
        
        # Log coordinate error
        if self.writer is not None:
            try:
                self.writer.add_scalar('Error/Coordinate_Degrees', avg_coord_error, self.current_epoch)
                
                # Log additional metrics
                self.writer.add_scalar('Error/Coordinate_Degrees_Rolling', avg_coord_error, self.current_epoch)
                
                # Log model weights periodically (every 5 epochs)
                if self.current_epoch % 5 == 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.writer.add_histogram(f'Weights_Epoch/{name}', param.data, self.current_epoch)
                            self.writer.add_scalar(f'Weights_Mean/{name}', param.data.mean(), self.current_epoch)
                            self.writer.add_scalar(f'Weights_Std/{name}', param.data.std(), self.current_epoch)
            except Exception as e:
                logger.warning(f"Failed to log coordinate error: {e}")
        
        return avg_loss


class AutoEncoderTrainer(EnhancedTrainer):
    """Enhanced trainer for AutoEncoder model."""
    
    def _setup_criterion(self) -> nn.Module:
        """Setup reconstruction loss."""
        return nn.MSELoss()
    
    def train_epoch(self) -> float:
        """Train autoencoder for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        try:
            for batch_idx, (images, _) in enumerate(pbar):
                try:
                    images = images.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    reconstructed = self.model(images)
                    loss = self.criterion(reconstructed, images)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # Log to tensorboard
                    self._log_batch_to_tensorboard(loss.item(), batch_idx)
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in training epoch {self.current_epoch + 1}: {e}")
            raise
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate autoencoder for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        try:
            with torch.no_grad():
                for images, _ in self.val_loader:
                    try:
                        images = images.to(self.device)
                        
                        # Forward pass
                        reconstructed = self.model(images)
                        loss = self.criterion(reconstructed, images)
                        
                        total_loss += loss.item()
                    except Exception as e:
                        logger.error(f"Error in validation batch: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error in validation epoch {self.current_epoch + 1}: {e}")
            raise
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss