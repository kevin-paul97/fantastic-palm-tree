"""
PyTorch datasets for satellite image coordinate prediction.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SatelliteImageDataset(Dataset):
    """Dataset for satellite images with coordinate labels."""
    
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_seed: int = 42
    ):
        self.image_dir = Path(image_dir)
        self.metadata_dir = Path(metadata_dir)
        self.transform = transform
        self.split = split
        
        # Load all metadata and create image-coordinate pairs
        self.samples = self._load_samples()
        
        # Split data
        self.samples = self._split_data(train_split, val_split, random_seed)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Tuple[str, Tuple[float, float]]]:
        """Load image paths and their corresponding coordinates."""
        samples = []
        
        for json_file in self.metadata_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                date = json_file.stem
                
                for item in data:
                    image_name = item.get("image")
                    coords = item.get("centroid_coordinates", {})
                    
                    if image_name and coords.get("lat") and coords.get("lon"):
                        image_path = self.image_dir / "earth" / f"{image_name}.png"
                        
                        if image_path.exists():
                            lat = float(coords["lat"])
                            lon = float(coords["lon"])
                            samples.append((str(image_path), (lat, lon)))
                            
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        return samples
    
    def _split_data(
        self, 
        train_split: float, 
        val_split: float, 
        random_seed: int
    ) -> List[Tuple[str, Tuple[float, float]]]:
        """Split data into train/val/test sets."""
        import random
        
        random.seed(random_seed)
        random.shuffle(self.samples)
        
        n_samples = len(self.samples)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        if self.split == "train":
            return self.samples[:n_train]
        elif self.split == "val":
            return self.samples[n_train:n_train + n_val]
        elif self.split == "test":
            return self.samples[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """Get an image and its coordinates."""
        image_path, (lat, lon) = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (64, 64), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = transforms.ToTensor()(image)
        
        # Convert coordinates to tensor [lon, lat] for consistent processing
        coords = torch.tensor([lon, lat], dtype=torch.float32)
        
        return image, coords


def create_transforms(image_size: int = 64, grayscale: bool = True) -> transforms.Compose:
    """Create image transformation pipeline."""
    transform_list = []
    
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if grayscale:
        transform_list.append(transforms.Grayscale())
    
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    config,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Determine optimal dataloader settings based on device
    import torch
    device = torch.device(config.training.device)
    
    # Optimize dataloader settings for MPS vs CUDA vs CPU
    if device.type == 'mps':
        # MPS has issues with multiprocessing and pin_memory
        num_workers = 0  # Single-threaded for MPS
        pin_memory = False
        persistent_workers = False
    elif device.type == 'cuda':
        # CUDA benefits from multiprocessing and pin_memory
        num_workers = min(num_workers, 4)  # Cap workers to prevent oversubscription
        pin_memory = True
        persistent_workers = True
    else:
        # CPU: moderate workers, no pin_memory needed
        num_workers = min(num_workers, 2)
        pin_memory = False
        persistent_workers = num_workers > 0
    
    logger.info(f"Using dataloader settings for {device.type}: "
                f"workers={num_workers}, pin_memory={pin_memory}")
    
    # Create transforms
    transform = create_transforms(
        image_size=config.data.image_size,
        grayscale=config.data.grayscale
    )
    
    # Create datasets
    train_dataset = SatelliteImageDataset(
        image_dir=config.data.download_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="train",
        train_split=config.data.train_split,
        val_split=config.data.val_split
    )
    
    val_dataset = SatelliteImageDataset(
        image_dir=config.data.download_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="val",
        train_split=config.data.train_split,
        val_split=config.data.val_split
    )
    
    test_dataset = SatelliteImageDataset(
        image_dir=config.data.download_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="test",
        train_split=config.data.train_split,
        val_split=config.data.val_split
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class CoordinateNormalizer:
    """Normalizes and denormalizes coordinate values."""
    
    def __init__(self):
        self.lat_min = -90.0
        self.lat_max = 90.0
        self.lon_min = -180.0
        self.lon_max = 180.0
    
    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1] range."""
        # coords shape: [batch_size, 2] where [:, 0] = lon, [:, 1] = lat
        lon = coords[:, 0]
        lat = coords[:, 1]
        
        # Normalize longitude from [-180, 180] to [0, 1]
        lon_norm = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        
        # Normalize latitude from [-90, 90] to [0, 1]
        lat_norm = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        
        # Clamp to [0, 1] range to handle any edge cases
        lon_norm = torch.clamp(lon_norm, 0.0, 1.0)
        lat_norm = torch.clamp(lat_norm, 0.0, 1.0)
        
        return torch.stack([lon_norm, lat_norm], dim=1)
    
    def denormalize(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates from [0, 1] to original range."""
        lon_norm = coords_norm[:, 0]
        lat_norm = coords_norm[:, 1]
        
        lon = lon_norm * (self.lon_max - self.lon_min) + self.lon_min
        lat = lat_norm * (self.lat_max - self.lat_min) + self.lat_min
        
        return torch.stack([lon, lat], dim=1)
    
    def compute_longitude_error(self, pred_lon: torch.Tensor, true_lon: torch.Tensor) -> torch.Tensor:
        """Compute longitude error accounting for wraparound at ±180°."""
        # Calculate direct difference
        diff = torch.abs(pred_lon - true_lon)
        
        # Account for wraparound: if difference > 180°, use the shorter path
        wrap_diff = 360.0 - diff
        
        # Take the minimum of direct and wraparound differences
        lon_error = torch.minimum(diff, wrap_diff)
        
        return lon_error
    
    def compute_coordinate_error_degrees(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """Compute coordinate prediction error in degrees with proper longitude handling."""
        # pred_coords and true_coords are in real-world coordinates [lon, lat]
        pred_lon, pred_lat = pred_coords[:, 0], pred_coords[:, 1]
        true_lon, true_lat = true_coords[:, 0], true_coords[:, 1]
        
        # Longitude error with wraparound handling
        lon_error = self.compute_longitude_error(pred_lon, true_lon)
        
        # Latitude error (no wraparound needed)
        lat_error = torch.abs(pred_lat - true_lat)
        
        # Euclidean distance
        distance_error = torch.sqrt(lon_error**2 + lat_error**2)
        
        return distance_error