#!/usr/bin/env python3
"""
Demo script to showcase cross-machine dataset functionality.
This creates a small sample dataset to demonstrate how the system works.
"""

import json
import numpy as np
from pathlib import Path
import logging

from cross_machine_datasets import CrossMachineCompatibleDataset
from config import Config

logger = logging.getLogger(__name__)


def create_sample_metadata(num_samples: int = 20) -> dict:
    """Create sample metadata for demonstration."""
    sample_metadata = {}
    
    for i in range(num_samples):
        image_name = f"epic_1b_demo_{i:06d}.png"
        date = f"2025-01-{(i%30)+1:02d}"
        
        # Generate random coordinates
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        
        sample_metadata[image_name] = {
            'filename': image_name,
            'full_path': f"sample_images/{date}/{image_name}",
            'date': date,
            'coordinates': {
                'lat': float(lat),
                'lon': float(lon)
            }
        }
    
    return sample_metadata


def create_sample_images(sample_metadata: dict) -> None:
    """Create sample image files."""
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    for date in set(entry['date'] for entry in sample_metadata.values()):
        date_dir = sample_dir / date
        date_dir.mkdir(exist_ok=True)
        
        for image_name, metadata in sample_metadata.items():
            if metadata['date'] == date:
                # Create a dummy image (64x64 grayscale)
                image_path = date_dir / image_name
                
                if not image_path.exists():
                    # Generate random image data
                    image_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                    from PIL import Image
                    img = Image.fromarray(image_data, mode='L')
                    img.save(image_path)
                    logger.info(f"Created sample image: {image_path}")


def demonstrate_cross_machine_functionality():
    """Demonstrate the cross-machine dataset functionality."""
    print("🎯 Cross-Machine Dataset Functionality Demo")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample metadata and images...")
    sample_metadata = create_sample_metadata(20)
    create_sample_images(sample_metadata)
    
    # Save sample metadata mapping
    mapping_file = Path("sample_image_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"   Created {len(sample_metadata)} sample images")
    print(f"   Saved mapping to: {mapping_file}")
    
    # Test dataset creation
    print("\n2. Testing cross-machine dataset...")
    try:
        from config import Config
        config = Config()
        
        # Create dataset
        dataset = CrossMachineCompatibleDataset(
            config, 
            mapping_file="sample_image_mapping.json",
            image_root_dir="sample_images",
            split="train"
        )
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Dataset loaded successfully!")
        
        # Test data loading
        for i in range(min(3, len(dataset))):
            image, coords = dataset[i]
            print(f"   Sample {i+1}:")
            print(f"     Image shape: {image.shape}")
            print(f"     Coordinates: ({coords[0]:.2f}, {coords[1]:.2f})")
        
        # Test dataloader creation
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_images, batch_coords = next(iter(dataloader))
        
        print(f"\n3. Testing dataloader:")
        print(f"   Batch shape: {batch_images.shape}")
        print(f"   Batch coordinates shape: {batch_coords.shape}")
        
        print("\n✅ Cross-machine functionality working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    print("\n4. Usage Instructions:")
    print("   To use with your own data:")
    print("   1. Run: python3 image_file_mapper.py --mode download --max_images 100")
    print("   2. Then: python3 cross_machine_datasets.py --verify")
    print("   3. Finally: Use CrossMachineCompatibleDataset in your training script")
    
    print("\n" + "=" * 50)
    print("🎉 Demo Complete!")
    return True


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo cross-machine dataset functionality")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample dataset for demonstration")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of sample images to create")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.create_sample:
        demonstrate_cross_machine_functionality()
    else:
        print("Use --create_sample to run the demonstration")
        print("Example: python3 demo_cross_machine.py --create_sample --num_samples 50")


if __name__ == "__main__":
    main()