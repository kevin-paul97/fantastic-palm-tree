#!/usr/bin/env python3
"""
Image filename mapping utility for cross-machine compatibility.
Creates proper image files from consolidated metadata with correct naming.
"""

import json
import os
import requests
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import shutil

logger = logging.getLogger(__name__)


class ImageFileMapper:
    """Maps consolidated metadata to actual image files with proper naming."""
    
    def __init__(self, config):
        self.config = config.data
        self.combined_dir = Path(self.config.combined_dir)
        self.images_dir = Path(self.config.images_dir)
        self.base_url = "https://api.nasa.gov/EPIC/archive/natural"
        
    def load_consolidated_metadata(self) -> Dict[str, List[dict]]:
        """Load all consolidated metadata files."""
        metadata_by_date = {}
        
        if not self.combined_dir.exists():
            logger.error(f"Combined directory not found: {self.combined_dir}")
            return {}
        
        # Load all JSON files
        for json_file in self.combined_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    date = json_file.stem  # Filename without extension is the date
                    metadata_by_date[date] = data
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded metadata for {len(metadata_by_date)} dates")
        return metadata_by_date
    
    def create_image_filename(self, image_data: dict) -> str:
        """Create proper image filename from image metadata."""
        image_name = image_data.get('image', '')
        date = image_data.get('date', '')
        
        # EPIC image format: epic_1b_YYYYMMDDHHMMSS.png
        if image_name:
            # Extract the base name without extension
            base_name = image_name.replace('.png', '')
            return f"{base_name}.png"
        else:
            # Fallback format
            timestamp = image_data.get('centroid_coordinates', {}).get('satellite_timestamp', '')
            if timestamp:
                clean_timestamp = timestamp.replace(':', '').replace('-', '')
                return f"epic_1b_{clean_timestamp}.png"
            else:
                return f"epic_{date.replace('-', '')}.png"
    
    def download_image(self, image_data: dict, output_dir: Path) -> bool:
        """Download a single image using proper filename."""
        try:
            image_name = image_data.get('image', '')
            date = image_data.get('date', '')
            
            if not image_name:
                logger.warning(f"No image name for data on {date}")
                return False
            
            # Create output directory for this date
            date_dir = output_dir / date
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate proper filename
            filename = self.create_image_filename(image_data)
            output_path = date_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"Image already exists: {output_path}")
                return True
            
            # Construct download URL
            # EPIC archive URL structure: https://api.nasa.gov/EPIC/archive/natural/2023/01/01/epic_1b_20230101000101.png
            year, month, day = date[:4], date[5:7], date[8:10]
            
            # Ensure image_name includes .png extension
            if not image_name.endswith('.png'):
                image_name = image_name + '.png'
            
            download_url = f"{self.base_url}/{year}/{month}/{day}/{image_name}"
            
            # Download image
            logger.debug(f"Downloading: {download_url}")
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {image_data.get('image', 'unknown')}: {e}")
            return False
    
    def create_all_image_files(self, output_dir: Optional[Path] = None, 
                           max_images: Optional[int] = None) -> bool:
        """Create all image files from consolidated metadata."""
        if output_dir is None:
            output_dir = Path("images_with_filenames")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load consolidated metadata
        metadata_by_date = self.load_consolidated_metadata()
        if not metadata_by_date:
            logger.error("No consolidated metadata found")
            return False
        
        total_images = 0
        successful_downloads = 0
        
        # Count total images first
        for date, images in metadata_by_date.items():
            total_images += len(images)
        
        if max_images:
            total_images = min(total_images, max_images)
            logger.info(f"Will download up to {max_images} images")
        
        logger.info(f"Starting download of {total_images} images...")
        
        # Download images with progress bar
        with tqdm(total=total_images, desc="Downloading images") as pbar:
            for date, images in metadata_by_date.items():
                for image_data in images:
                    if max_images and successful_downloads >= max_images:
                        break
                    
                    if self.download_image(image_data, output_dir):
                        successful_downloads += 1
                    
                    pbar.update(1)
                
                if max_images and successful_downloads >= max_images:
                    break
        
        logger.info(f"Successfully downloaded {successful_downloads}/{total_images} images")
        logger.info(f"Images saved to: {output_dir}")
        
        return successful_downloads > 0
    
    def create_metadata_mapping(self, output_file: Optional[Path] = None) -> bool:
        """Create a mapping file with original metadata -> image filenames."""
        if output_file is None:
            output_file = Path("image_filename_mapping.json")
        
        metadata_by_date = self.load_consolidated_metadata()
        if not metadata_by_date:
            return False
        
        mapping = {}
        image_count = 0
        
        for date, images in metadata_by_date.items():
            for image_data in images:
                image_name = image_data.get('image', '')
                if image_name:
                    proper_filename = self.create_image_filename(image_data)
                    date_dir = f"images_with_filenames/{date}"
                    full_path = f"{date_dir}/{proper_filename}"
                    
                    # Add coordinates for easier reference
                    coords = image_data.get('centroid_coordinates', {})
                    mapping[image_name] = {
                        'filename': proper_filename,
                        'full_path': full_path,
                        'date': date,
                        'coordinates': {
                            'lat': coords.get('lat'),
                            'lon': coords.get('lon')
                        }
                    }
                    image_count += 1
        
        # Save mapping
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Created filename mapping for {image_count} images: {output_file}")
        return True
    
    def verify_image_files(self, images_dir: Path) -> Dict[str, int]:
        """Verify which image files exist and generate statistics."""
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return {}
        
        stats = {
            'total_images': 0,
            'existing_files': 0,
            'missing_files': 0,
            'by_date': {}
        }
        
        # Load consolidated metadata
        metadata_by_date = self.load_consolidated_metadata()
        
        for date, images in metadata_by_date.items():
            date_dir = images_dir / date
            date_stats = {'total': len(images), 'existing': 0, 'missing': 0}
            
            for image_data in images:
                image_name = image_data.get('image', '')
                if image_name:
                    filename = self.create_image_filename(image_data)
                    filepath = date_dir / filename
                    
                    if filepath.exists():
                        date_stats['existing'] += 1
                        stats['existing_files'] += 1
                    else:
                        date_stats['missing'] += 1
                        stats['missing_files'] += 1
                    
                    stats['total_images'] += 1
            
            stats['by_date'][date] = date_stats
        
        logger.info(f"Verification complete:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Existing files: {stats['existing_files']}")
        logger.info(f"  Missing files: {stats['missing_files']}")
        
        return stats


def main():
    import argparse
    from config import Config
    
    parser = argparse.ArgumentParser(description="Create image files from consolidated metadata")
    parser.add_argument("--mode", choices=["download", "verify", "map"], 
                       default="download", help="Operation mode")
    parser.add_argument("--output_dir", type=str, default="images_with_filenames",
                       help="Output directory for images")
    parser.add_argument("--max_images", type=int, 
                       help="Maximum number of images to download")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--mapping_file", type=str, default="image_filename_mapping.json",
                       help="Output file for filename mapping")
    
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
    
    # Create mapper
    mapper = ImageFileMapper(config)
    
    if args.mode == "download":
        output_dir = Path(args.output_dir)
        success = mapper.create_all_image_files(
            output_dir, 
            max_images=args.max_images
        )
        
        if success:
            # Create mapping file
            mapper.create_metadata_mapping(Path(args.mapping_file))
            print(f"\n✅ Successfully created image files in: {output_dir}")
            print(f"📋 Filename mapping saved to: {args.mapping_file}")
        else:
            print("❌ Failed to create image files")
    
    elif args.mode == "verify":
        images_dir = Path(args.output_dir)
        stats = mapper.verify_image_files(images_dir)
        
        print(f"\n📊 Verification Results:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Existing files: {stats['existing_files']}")
        print(f"   Missing files: {stats['missing_files']}")
        print(f"   Success rate: {stats['existing_files']/max(stats['total_images'], 1)*100:.1f}%")
    
    elif args.mode == "map":
        mapping_file = Path(args.mapping_file)
        success = mapper.create_metadata_mapping(mapping_file)
        
        if success:
            print(f"\n📋 Filename mapping created: {mapping_file}")
        else:
            print("❌ Failed to create mapping file")


if __name__ == "__main__":
    main()