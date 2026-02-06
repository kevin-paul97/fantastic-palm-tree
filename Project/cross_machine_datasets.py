#!/usr/bin/env python3
"""
Enhanced dataset loader that works with mapped image filenames.
This makes the code portable across different machines.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)


class CrossMachineCompatibleDataset(Dataset):
    """Dataset class that works with mapped image filenames for cross-machine compatibility."""
    
    def __init__(self, config, mapping_file: str = "image_filename_mapping.json", 
                 image_root_dir: str = "images_with_filenames", 
                 transform=None, split: str = "train"):
        self.config = config.data
        self.transform = transform or self._get_default_transform()
        self.split = split
        
        # Load image filename mapping
        self.mapping_file = Path(mapping_file)
        self.image_root_dir = Path(image_root_dir)
        
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}. "
                               "Run image_file_mapper.py first!")
        
        with open(self.mapping_file, 'r') as f:
            self.image_mapping = json.load(f)
        
        # Extract valid image entries (those with coordinates)
        self.valid_images = []
        for image_name, metadata in self.image_mapping.items():
            coords = metadata.get('coordinates', {})
            if coords.get('lat') is not None and coords.get('lon') is not None:
                self.valid_images.append({
                    'image_name': image_name,
                    'filename': metadata['filename'],
                    'full_path': Path(metadata['full_path']),
                    'coordinates': coords
                })
        
        logger.info(f"Loaded {len(self.valid_images)} valid image entries from mapping")
        
        # Split data
        self._create_train_val_test_splits()
    
    def _get_default_transform(self):
        """Get default image transformations."""
        transform_list = []
        
        # Resize
        if self.config.image_size:
            transform_list.append(transforms.Resize(self.config.image_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Grayscale conversion
        if self.config.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        return transforms.Compose(transform_list)
    
    def _create_train_val_test_splits(self):
        """Create train/validation/test splits."""
        # Shuffle data
        random.shuffle(self.valid_images)
        
        total_samples = len(self.valid_images)
        train_size = int(total_samples * self.config.train_split)
        val_size = int(total_samples * self.config.val_split)
        
        self.train_indices = self.valid_images[:train_size]
        self.val_indices = self.valid_images[train_size:train_size + val_size]
        self.test_indices = self.valid_images[train_size + val_size:]
        
        logger.info(f"Split {total_samples} samples: "
                   f"train={len(self.train_indices)}, "
                   f"val={len(self.val_indices)}, "
                   f"test={len(self.test_indices)}")
    
    def _load_image(self, image_entry: dict) -> torch.Tensor:
        """Load image from file using mapped filename."""
        image_path = image_entry['full_path']
        
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            # Return dummy image
            return torch.zeros((1, self.config.image_size[0], self.config.image_size[1]))
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Apply transforms
            image = self.transform(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            return torch.zeros((1, self.config.image_size[0], self.config.image_size[1]))
    
    def _load_coordinates(self, image_entry: dict) -> torch.Tensor:
        """Load coordinates from image entry."""
        coords = image_entry['coordinates']
        lat = coords.get('lat', 0.0)
        lon = coords.get('lon', 0.0)
        return torch.tensor([lon, lat], dtype=torch.float32)
    
    def __len__(self) -> int:
        """Get dataset size for current split."""
        if self.split == "train":
            return len(self.train_indices)
        elif self.split == "val":
            return len(self.val_indices)
        else:  # test
            return len(self.test_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and coordinates for current split."""
        if self.split == "train":
            image_entry = self.train_indices[idx]
        elif self.split == "val":
            image_entry = self.val_indices[idx]
        else:  # test
            image_entry = self.test_indices[idx]
        
        # Load image and coordinates
        image = self._load_image(image_entry)
        coordinates = self._load_coordinates(image_entry)
        
        return image, coordinates


def create_cross_machine_dataloaders(
    config, 
    mapping_file: str = "image_filename_mapping.json",
    image_root_dir: str = "images_with_filenames",
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders that work across different machines."""
    
    # Set optimal dataloader settings based on device
    device = config.training.device
    
    # Optimize for different devices
    if device == "mps":
        num_workers = 0  # MPS has issues with multiprocessing
        pin_memory = False
    elif device == "cuda":
        num_workers = min(num_workers, 4)  # Don't exceed reasonable limit
        pin_memory = True
    else:  # cpu
        num_workers = 2
        pin_memory = False
    
    logger.info(f"Using dataloader settings for {device}: workers={num_workers}, pin_memory={pin_memory}")
    
    # Create datasets
    train_dataset = CrossMachineCompatibleDataset(
        config, mapping_file, image_root_dir, split="train"
    )
    val_dataset = CrossMachineCompatibleDataset(
        config, mapping_file, image_root_dir, split="val"
    )
    test_dataset = CrossMachineCompatibleDataset(
        config, mapping_file, image_root_dir, split="test"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def verify_dataset_integrity(
    mapping_file: str = "image_filename_mapping.json",
    image_root_dir: str = "images_with_filenames"
) -> dict:
    """Verify that all mapped images exist and have valid coordinates."""
    
    mapping_path = Path(mapping_file)
    image_root = Path(image_root_dir)
    
    if not mapping_path.exists():
        return {"error": f"Mapping file not found: {mapping_path}"}
    
    # Load mapping
    with open(mapping_path, 'r') as f:
        image_mapping = json.load(f)
    
    stats = {
        'total_entries': len(image_mapping),
        'valid_coordinates': 0,
        'existing_files': 0,
        'missing_files': 0,
        'errors': []
    }
    
    for image_name, metadata in image_mapping.items():
        coords = metadata.get('coordinates', {})
        lat, lon = coords.get('lat'), coords.get('lon')
        
        # Check coordinates
        if lat is not None and lon is not None:
            stats['valid_coordinates'] += 1
        else:
            stats['errors'].append(f"Invalid coordinates for {image_name}")
            continue
        
        # Check file existence
        image_path = Path(metadata['full_path'])
        if image_path.exists():
            stats['existing_files'] += 1
        else:
            stats['missing_files'] += 1
            stats['errors'].append(f"Missing image file: {image_path}")
    
    # Generate summary
    success_rate = (stats['existing_files'] / stats['total_entries']) * 100 if stats['total_entries'] > 0 else 0
    
    print(f"\n📊 Dataset Integrity Report:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Valid coordinates: {stats['valid_coordinates']}")
    print(f"   Existing files: {stats['existing_files']}")
    print(f"   Missing files: {stats['missing_files']}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if stats['errors']:
        print(f"\n⚠️  Errors found:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more errors")
    
    return stats


def main():
    import argparse
    from config import Config
    
    parser = argparse.ArgumentParser(description="Verify cross-machine dataset")
    parser.add_argument("--mapping_file", type=str, default="image_filename_mapping.json",
                       help="Path to image filename mapping")
    parser.add_argument("--image_root_dir", type=str, default="images_with_filenames",
                       help="Root directory for images")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Verify dataset
    stats = verify_dataset_integrity(args.mapping_file, args.image_root_dir)
    
    # Try to create a dataloader as additional test
    try:
        print("\n🔄 Testing dataloader creation...")
        train_loader, val_loader, test_loader = create_cross_machine_dataloaders(
            config, args.mapping_file, args.image_root_dir, batch_size=4
        )
        
        # Test loading a batch
        train_images, train_coords = next(iter(train_loader))
        print(f"✅ Successfully loaded training batch: {train_images.shape}, {train_coords.shape}")
        
        val_images, val_coords = next(iter(val_loader))
        print(f"✅ Successfully loaded validation batch: {val_images.shape}, {val_coords.shape}")
        
        test_images, test_coords = next(iter(test_loader))
        print(f"✅ Successfully loaded test batch: {test_images.shape}, {test_coords.shape}")
        
        print("\n🎉 Dataset is ready for cross-machine usage!")
        
    except Exception as e:
        print(f"\n❌ Dataloader test failed: {e}")
        return 1
    
    return 0 if stats['existing_files'] > 0 else 1


if __name__ == "__main__":
    exit(main())