"""
Unified training utilities combining basic and enhanced functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import os
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
from datetime import datetime

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None

from torch.utils.data import DataLoader

from models import LocationRegressor
from datasets import CoordinateNormalizer
from tensorboard_utils import is_port_available, start_tensorboard

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified trainer class with configurable enhancement levels."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: Optional[str] = None,
        enhanced_mode: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(config.training.device)
        self.enhanced_mode = enhanced_mode
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = self._setup_criterion()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup coordinate normalizer
        self._setup_coordinate_normalizer()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # TensorBoard setup
        self.tensorboard_process = None
        self.writer = None
        if SummaryWriter is not None:
            self._setup_tensorboard()
        else:
            logger.warning("TensorBoard not available")
    
    def _setup_coordinate_normalizer(self):
        """Setup coordinate normalizer from training data."""
        # Get coordinate ranges from training data
        all_coords = []
        for images, coords in self.train_loader:
            all_coords.append(coords)
        
        if all_coords:
            all_coords_tensor = torch.cat(all_coords, dim=0)
            self.coord_normalizer = CoordinateNormalizer(all_coords_tensor)
        else:
            self.coord_normalizer = CoordinateNormalizer()
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer with configurable error handling."""
        try:
            # Create timestamped run directory for organized logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tensorboard_run_dir = os.path.join(self.config.training.log_dir, f"run_{timestamp}")
            os.makedirs(self.tensorboard_run_dir, exist_ok=True)
            
            # Ensure base log directory exists and is writable
            os.makedirs(self.config.training.log_dir, exist_ok=True)
            
            if self.enhanced_mode:
                # Enhanced error checking
                test_file = os.path.join(self.config.training.log_dir, 'test_write')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    raise PermissionError(f"Cannot write to log directory {self.config.training.log_dir}: {e}")
                
                # Create writer with enhanced settings
                if SummaryWriter is not None:
                    self.writer = SummaryWriter(
                        log_dir=self.tensorboard_run_dir,
                        comment=f'_device_{self.config.training.device}_batch_{self.config.training.batch_size}',
                        flush_secs=10  # Auto-flush every 10 seconds
                    )
                    
                    # Test writer functionality
                    if self.writer is not None:
                        self.writer.add_scalar('test/init', 1.0, 0)
                        self.writer.flush()
                        self.writer.add_scalar('test/init', 2.0, 1)
                        self.writer.flush()
                else:
                    self.writer = None
            else:
                # Basic setup
                if SummaryWriter is not None:
                    self.writer = SummaryWriter(
                        log_dir=self.tensorboard_run_dir,
                        comment=f'_device_{self.config.training.device}_batch_{self.config.training.batch_size}'
                    )
                else:
                    self.writer = None
            
            # Log configuration
            self._log_config_to_tensorboard()
            
            # Launch tensorboard automatically if enabled
            if self.config.training.launch_tensorboard:
                self._launch_tensorboard()
                
            logger.info(f"TensorBoard writer initialized successfully")
            logger.info(f"Logs directory: {self.tensorboard_run_dir}")
            logger.info(f"View with: tensorboard --logdir {self.tensorboard_run_dir}")
            logger.info(f"Or view all runs with: tensorboard --logdir {os.path.dirname(self.tensorboard_run_dir)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard writer: {e}")
            self.writer = None
    
    def _launch_tensorboard(self):
        """Launch TensorBoard server with enhanced error handling."""
        try:
            if self.enhanced_mode:
                # Enhanced launch with multiple fallback methods - launch the specific run directory
                success = start_tensorboard(self.tensorboard_run_dir, 6006, open_browser=False)
                if success:
                    logger.info(f"TensorBoard launched for current run: {self.tensorboard_run_dir}")
                    logger.info(f"Access at: http://localhost:6006")
                else:
                    logger.warning("Failed to auto-launch TensorBoard, you can launch manually:")
                    logger.info(f"tensorboard --logdir {self.tensorboard_run_dir}")
                if success:
                    # Store the process info (simplified version)
                    self.tensorboard_process = True
                else:
                    self.tensorboard_process = None
            else:
                # Basic launch
                import subprocess
                import webbrowser
                import sys
                
                # Check if port is available
                if not is_port_available(6006):
                    logger.warning("Port 6006 is in use, TensorBoard may not start properly")
                
                # Launch TensorBoard using python -m tensorboard.main (more reliable)
                cmd = [sys.executable, "-m", "tensorboard.main", 
                       "--logdir", self.tensorboard_run_dir,
                       "--port", "6006",
                       "--host", "localhost"]
                
                self.tensorboard_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait a moment to check if it started successfully
                time.sleep(2)
                if self.tensorboard_process.poll() is not None:
                    stdout, stderr = self.tensorboard_process.communicate()
                    logger.error(f"TensorBoard failed to start: {stderr}")
                    self.tensorboard_process = None
                else:
                    logger.info(f"TensorBoard launched on http://localhost:6006")
                    
                # Auto-open browser if enabled (handle missing config gracefully)
                if hasattr(self.config.training, 'open_browser') and self.config.training.open_browser:
                    webbrowser.open('http://localhost:6006')
                        
        except Exception as e:
            logger.error(f"Failed to launch TensorBoard: {e}")
            self.tensorboard_process = None
    
    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        optimizer_type = self.config.training.optimizer.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _setup_criterion(self):
        """Setup loss function based on configuration."""
        criterion_type = self.config.training.loss_function.lower()
        
        if criterion_type == 'mse':
            return nn.MSELoss()
        elif criterion_type == 'l1':
            return nn.L1Loss()
        elif criterion_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {criterion_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on configuration."""
        scheduler_type = self.config.training.scheduler.lower()
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.training.gamma,
                patience=self.config.training.step_size
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _log_config_to_tensorboard(self):
        """Log training configuration to TensorBoard."""
        if self.writer is None:
            return
            
        # Model configuration
        model_config = {
            'model/hidden_dim': self.config.model.hidden_dim,
            'model/input_channels': self.config.model.input_channels,
            'model/type': self.model.__class__.__name__,
        }
        
        # Training configuration
        training_config = {
            'training/learning_rate': self.config.training.learning_rate,
            'training/batch_size': self.config.training.batch_size,
            'training/max_epochs': self.config.training.max_epochs,
            'training/optimizer': self.config.training.optimizer,
            'training/loss_function': self.config.training.loss_function,
            'training/scheduler': self.config.training.scheduler,
            'training/gradient_clipping': self.config.training.gradient_clipping,
            'training/weight_decay': self.config.training.weight_decay,
            'training/step_size': getattr(self.config.training, 'step_size', 20),
            'training/gamma': getattr(self.config.training, 'gamma', 0.5),
            'training/device': str(self.device),
            'training/enhanced_mode': str(self.enhanced_mode),
        }
        
        # Data configuration (if available)
        data_config = {}
        try:
            if hasattr(self.config, 'data') and self.config.data:
                data_config = {
                    'data/image_size': getattr(self.config.data, 'image_size', 64),
                    'data/train_split': getattr(self.config.data, 'train_split', 0.8),
                    'data/val_split': getattr(self.config.data, 'val_split', 0.1),
                    'data/test_split': getattr(self.config.data, 'test_split', 0.1),
                }
        except:
            pass
        
        # Log all configurations
        all_configs = {**model_config, **training_config, **data_config}
        for key, value in all_configs.items():
            self.writer.add_text(key, str(value))
        
        # Log model architecture summary
        self.writer.add_text('model/architecture', str(self.model))
        
        # Count and log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting training for {self.config.training.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {total_params}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc='Training', leave=False)):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('train/loss', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('val/loss', avg_loss, self.current_epoch)
        
        return avg_loss

    def train(self):
        """Train the model for the specified number of epochs."""
        logger.info(f"Starting training for {self.config.training.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in tqdm(range(self.config.training.max_epochs), desc="Training epochs"):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log learning rate
            if self.writer is not None:
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
            
            # Log epoch results with progress information
            logger.info(f'Epoch {epoch+1}/{self.config.training.max_epochs}: '
                       f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Enhanced logging
            if self.enhanced_mode and self.writer is not None:
                self.writer.add_scalar('train/val_loss_ratio', train_loss/val_loss, epoch)
                
                # Log gradient norm (enhanced feature)
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.writer.add_scalar('train/gradient_norm', total_norm, epoch)
        
        logger.info('Training completed!')
        
        # Initialize variables to avoid unbound errors
        train_improvement = 0
        val_improvement = 0
        epochs_to_best = self.config.training.max_epochs
        
        # Log final metrics to TensorBoard
        if self.writer is not None:
            # Final hyperparameters summary
            final_hparams = {
                'model_type': self.model.__class__.__name__,
                'total_epochs': self.config.training.max_epochs,
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'train_improvement_percent': train_improvement,
                'val_improvement_percent': val_improvement,
                'epochs_to_best': epochs_to_best
            }
            
            # Calculate improvement metrics
            if len(self.train_losses) > 1:
                train_improvement = ((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0]) * 100
            
            if len(self.val_losses) > 1:
                val_improvement = ((self.val_losses[0] - self.best_val_loss) / self.val_losses[0]) * 100
                epochs_to_best = self.val_losses.index(self.best_val_loss) + 1 if self.best_val_loss in self.val_losses else self.config.training.max_epochs
            
            # Log final hyperparameters
            for key, value in final_hparams.items():
                self.writer.add_text(f'final/{key}', str(value))
            
            # Log loss curve statistics
            if self.train_losses:
                self.writer.add_scalar('final/train_loss_mean', sum(self.train_losses)/len(self.train_losses), self.config.training.max_epochs-1)
                self.writer.add_scalar('final/train_loss_min', min(self.train_losses), self.config.training.max_epochs-1)
                self.writer.add_scalar('final/train_loss_max', max(self.train_losses), self.config.training.max_epochs-1)
                self.writer.add_scalar('final/train_loss_final', self.train_losses[-1], self.config.training.max_epochs-1)
                
                if self.val_losses:
                    self.writer.add_scalar('final/val_loss_mean', sum(self.val_losses)/len(self.val_losses), self.config.training.max_epochs-1)
                    self.writer.add_scalar('final/val_loss_min', min(self.val_losses), self.config.training.max_epochs-1)
                    self.writer.add_scalar('final/val_loss_max', max(self.val_losses), self.config.training.max_epochs-1)
                    self.writer.add_scalar('final/val_loss_final', self.val_losses[-1], self.config.training.max_epochs-1)
                
                # Log training efficiency metrics
                self.writer.add_scalar('final/train_improvement_percent', train_improvement, self.config.training.max_epochs-1)
                self.writer.add_scalar('final/val_improvement_percent', val_improvement, self.config.training.max_epochs-1)
                
                # Log convergence metrics
                self.writer.add_scalar('final/epochs_to_best_val', epochs_to_best, self.config.training.max_epochs-1)
                self.writer.add_scalar('final/convergence_rate', epochs_to_best / self.config.training.max_epochs, self.config.training.max_epochs-1)
                
                # Create summary table
                summary_text = f"""
 Training Summary:
 - Model: {final_hparams['model_type']}
 - Total Epochs: {final_hparams['total_epochs']}
 - Best Val Loss: {final_hparams['best_val_loss']:.6f}
 - Final Train Loss: {final_hparams['final_train_loss']:.6f}
 - Final Val Loss: {final_hparams['final_val_loss']:.6f}
 - Train Improvement: {train_improvement:.2f}% (if applicable)
 - Val Improvement: {val_improvement:.2f}% (if applicable)
 - Epochs to Best: {epochs_to_best}
 - Convergence Rate: {epochs_to_best / final_hparams['total_epochs']:.2f}
 - Optimizer: {getattr(self.config.training, 'optimizer', 'N/A')}
 - Learning Rate: {getattr(self.config.training, 'learning_rate', 'N/A')}
 - Batch Size: {getattr(self.config.training, 'batch_size', 'N/A')}
 - Loss Function: {getattr(self.config.training, 'loss_function', 'N/A')}
 - Scheduler: {getattr(self.config.training, 'scheduler', 'N/A')}
 - Device: {str(self.device)}
            """
                
                self.writer.add_text('final/training_summary', summary_text)
                
                # Log model complexity metrics
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                self.writer.add_scalar('final/total_parameters', total_params, self.config.training.max_epochs-1)
                self.writer.add_scalar('final/trainable_parameters', trainable_params, self.config.training.max_epochs-1)
                self.writer.add_scalar('final/non_trainable_parameters', non_trainable_params, self.config.training.max_epochs-1)
                self.writer.add_scalar('final/trainable_ratio', trainable_params / total_params if total_params > 0 else 0, self.config.training.max_epochs-1)
                
                # Log model complexity metrics
                if hasattr(self.model, 'layers') or hasattr(self.model, 'modules'):
                    total_layers = len(list(self.model.modules()))
                    self.writer.add_scalar('final/total_layers', total_layers, self.config.training.max_epochs-1)
                
                # Ensure all data is written to TensorBoard
                self.writer.flush()
                logger.info("Final training metrics logged to TensorBoard")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_hyperparameters': {
                'model_type': self.model.__class__.__name__,
                'total_epochs': self.config.training.max_epochs,
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'train_improvement_percent': train_improvement,
                'val_improvement_percent': val_improvement,
                'epochs_to_best': epochs_to_best
            }
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            # Save essential config values as dict instead of full config object
            'config_dict': {
                'model': {
                    'input_channels': self.config.model.input_channels,
                    'hidden_dim': self.config.model.hidden_dim,
                    'conv_channels': self.config.model.conv_channels,
                    'kernel_size': self.config.model.kernel_size,
                    'pool_size': self.config.model.pool_size,
                    'activation': self.config.model.activation,
                    'output_dim': self.config.model.output_dim
                },
                'training': {
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'max_epochs': self.config.training.max_epochs,
                    'optimizer': self.config.training.optimizer,
                    'weight_decay': self.config.training.weight_decay,
                    'loss_function': self.config.training.loss_function,
                    'scheduler': self.config.training.scheduler,
                    'step_size': self.config.training.step_size,
                    'gamma': self.config.training.gamma,
                    'gradient_clipping': self.config.training.gradient_clipping,
                    'device': self.config.training.device
                }
            }
        }
        
        filepath = os.path.join(self.config.training.save_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = os.path.join(self.config.training.save_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Handle PyTorch 2.6+ weights_only security
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            if "weights_only" in str(e):
                # Fallback for PyTorch 2.6+
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            else:
                raise e
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f'Checkpoint loaded: {filepath}')
    
    def cleanup(self):
        """Cleanup resources."""
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        # Stop TensorBoard server
        if self.tensorboard_process is not None:
            logger.info("Stopping TensorBoard server...")
            if isinstance(self.tensorboard_process, subprocess.Popen):
                try:
                    # Try graceful shutdown first
                    self.tensorboard_process.terminate()
                    
                    # Wait up to 5 seconds for graceful shutdown
                    try:
                        self.tensorboard_process.wait(timeout=5)
                        logger.info("TensorBoard server stopped gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        logger.warning("TensorBoard server did not stop gracefully, force killing...")
                        self.tensorboard_process.kill()
                        self.tensorboard_process.wait()
                        logger.info("TensorBoard server force killed")
                        
                except Exception as e:
                    logger.error(f"Error stopping TensorBoard server: {e}")
                    # Try to kill the process as a last resort
                    try:
                        self.tensorboard_process.kill()
                    except:
                        pass
                finally:
                    self.tensorboard_process = None
            else:
                # Simple boolean flag case
                logger.info("TensorBoard server was launched externally")
                self.tensorboard_process = None


class LocationRegressorTrainer(UnifiedTrainer):
    """Specialized trainer for location regression models."""
    
    def __init__(self, model: LocationRegressor, train_loader: DataLoader, 
                 val_loader: DataLoader, config, device: Optional[str] = None, enhanced_mode: bool = True):
        super().__init__(model, train_loader, val_loader, config, device, enhanced_mode)
    
    def evaluate_coordinates(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate coordinate prediction accuracy."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating coordinates"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(images)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        
            # Calculate coordinate errors
            coord_errors = self.coord_normalizer.compute_coordinate_error_degrees(
                all_predictions, all_targets
            )
            haversine_distances = self.coord_normalizer.compute_haversine_distance(
                all_predictions, all_targets
            )
            
            return {
                'mean_coordinate_error_deg': coord_errors.mean().item(),
                'median_coordinate_error_deg': coord_errors.median().item(),
                'mean_haversine_km': haversine_distances.mean().item(),
                'median_haversine_km': haversine_distances.median().item()
            }


