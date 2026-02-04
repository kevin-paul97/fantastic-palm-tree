"""
Neural network models for satellite image coordinate prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LocationRegressor(nn.Module):
    """
    A regression model to predict the location of a given image with two output nodes.
    The CNN model sees the image in black and white and predicts the x and y coordinates 
    (centroid of the earth). It has two output nodes, one for x and one for y.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 4,
        activation: str = "tanh",
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout_rate: float = 0.2
    ):
        super(LocationRegressor, self).__init__()
        
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        
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
        )
        
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


class AutoEncoder(nn.Module):
    """
    Autoencoder for learning compressed representations of satellite images.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        encoded_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        super(AutoEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.encoded_dim = encoded_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, encoded_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 8 * 8 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)


def create_location_regressor(config) -> LocationRegressor:
    """Create a LocationRegressor model from configuration."""
    return LocationRegressor(
        input_channels=config.model.input_channels,
        conv_channels=config.model.conv_channels,
        kernel_size=config.model.kernel_size,
        pool_size=config.model.pool_size,
        activation=config.model.activation,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim
    )


def create_autoencoder(config) -> AutoEncoder:
    """Create an AutoEncoder model from configuration."""
    return AutoEncoder(
        input_channels=config.model.input_channels,
        encoded_dim=config.model.hidden_dim
    )


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