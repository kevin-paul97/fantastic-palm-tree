"""
Neural network models for satellite image coordinate prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LocationRegressor(nn.Module):
    """CNN for predicting latitude and longitude from satellite images."""
    
    def __init__(
        self,
        input_channels: Optional[int] = None,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 4,
        activation: str = "tanh",
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout_rate: float = 0.2,
        config=None
    ):
        super(LocationRegressor, self).__init__()
        
        # Handle config object or individual parameters
        if config is not None:
            self.input_channels = config.input_channels
            self.conv_channels = config.conv_channels
            self.kernel_size = config.kernel_size
            self.pool_size = config.pool_size
            self.activation = config.activation.lower()
            self.hidden_dim = config.hidden_dim
            self.output_dim = config.output_dim
            self.dropout_rate = dropout_rate
        else:
            if conv_channels is None:
                conv_channels = [64, 128, 256]
            if input_channels is None:
                input_channels = 1  # Default grayscale
            
            self.input_channels = input_channels
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size
            self.pool_size = pool_size
            self.activation = activation.lower()
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.dropout_rate = dropout_rate
        
        # Build convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Calculate flattened size after convolutions
        # Assuming input size of 64x64 (as per config)
        self.flattened_size = self._calculate_flattened_size(64, 64)
        
        # Build fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
            # Remove Sigmoid - let model learn proper range, we'll constrain during training
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build convolutional layers."""
        layers = nn.ModuleList()
        
        in_channels = self.input_channels
        
        for out_channels in self.conv_channels:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=1),
                nn.MaxPool2d(self.pool_size),
                self._get_activation_layer()
            )
            layers.append(conv_block)
            in_channels = out_channels
            
        return layers
    
    def _get_activation_layer(self) -> nn.Module:
        """Get activation layer based on configuration."""
        if self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _calculate_flattened_size(self, height: int, width: int) -> int:
        """Calculate the size of flattened features after convolutions."""
        h, w = height, width
        
        for _ in self.conv_channels:
            h = h // self.pool_size
            w = w // self.pool_size
        
        return h * w * self.conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x


def create_location_regressor(config) -> LocationRegressor:
    """Create a LocationRegressor model from configuration."""
    return LocationRegressor(config=config.model)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (1, 1, 64, 64)):
    """Print a summary of the model architecture."""
    try:
        from torchinfo import summary
        summary(model, input_size=input_size)
    except ImportError:
        print(f"Model: {model.__class__.__name__}")
        print(f"Trainable parameters: {count_parameters(model):,}")
        
        # Print model structure
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                print(f"  {name}: {module}")