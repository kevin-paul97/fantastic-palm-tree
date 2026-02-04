"""
Data downloading and processing utilities for NASA EPIC satellite images.
"""

import json
import os
import shutil
from typing import List, Tuple, Optional
import requests
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EPICDataDownloader:
    """Downloads and manages NASA EPIC satellite image data."""
    
    def __init__(self, config):
        self.config = config.data
        self.base_url = self.config.api_base_url
        self.raw_data_dir = Path(self.config.raw_data_dir)
        self.images_dir = Path(self.config.images_dir)
        self.combined_dir = Path(self.config.combined_dir)
        
    def download_metadata(self) -> bool:
        """Download the complete metadata file from NASA EPIC API."""
        try:
            url = f"{self.base_url}/all"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create directories
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata_path = self.raw_data_dir / "all.json"
            with open(metadata_path, 'w') as f:
                json.dump(response.json(), f)
            
            logger.info(f"Downloaded metadata with {len(response.json())} entries")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download metadata: {e}")
            return False
    
    def load_metadata(self) -> List[dict]:
        """Load metadata from local file."""
        metadata_path = self.raw_data_dir / "all.json"
        
        if not metadata_path.exists():
            if not self.download_metadata():
                raise FileNotFoundError("Metadata file not found and download failed")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def extract_dates(self, metadata: List[dict]) -> List[str]:
        """Extract unique dates from metadata."""
        dates = list({item["date"] for item in metadata})
        dates.sort()
        return dates
    
    def download_daily_data(self, date: str) -> bool:
        """Download image data for a specific date."""
        try:
            url = f"{self.base_url}/date/{date}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create date directory
            date_dir = self.images_dir / date
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # Save daily data
            daily_data_path = date_dir / f"{date}.json"
            with open(daily_data_path, 'w') as f:
                json.dump(response.json(), f)
            
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download data for {date}: {e}")
            return False
    
    def download_all_images(self, progress_callback=None) -> bool:
        """Download all image metadata."""
        metadata = self.load_metadata()
        dates = self.extract_dates(metadata)
        
        success_count = 0
        for i, date in enumerate(dates):
            if self.download_daily_data(date):
                success_count += 1
            
            if progress_callback:
                progress_callback(i + 1, len(dates))
        
        logger.info(f"Downloaded data for {success_count}/{len(dates)} dates")
        return success_count > 0
    
    def consolidate_metadata(self) -> bool:
        """Consolidate all daily JSON files into combined directory."""
        try:
            # Create combined directory
            self.combined_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all date directories
            date_dirs = [d for d in self.images_dir.iterdir() 
                        if d.is_dir() and d.name != ".DS_Store"]
            date_dirs.sort()
            
            # Copy all daily JSON files
            for date_dir in date_dirs:
                json_file = date_dir / f"{date_dir.name}.json"
                if json_file.exists():
                    dest_file = self.combined_dir / json_file.name
                    shutil.copy2(json_file, dest_file)
            
            logger.info(f"Consolidated {len(date_dirs)} metadata files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to consolidate metadata: {e}")
            return False


class CoordinateExtractor:
    """Extracts and processes coordinate data from metadata."""
    
    def __init__(self, config):
        self.config = config.data
        self.combined_dir = Path(self.config.combined_dir)
    
    def extract_coordinates(self) -> Tuple[List[float], List[float]]:
        """Extract latitude and longitude coordinates from all metadata files."""
        lat_coordinates = []
        lon_coordinates = []
        
        if not self.combined_dir.exists():
            raise FileNotFoundError(f"Combined directory not found: {self.combined_dir}")
        
        # Process all JSON files
        for json_file in self.combined_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    coords = item.get("centroid_coordinates", {})
                    lat = coords.get("lat")
                    lon = coords.get("lon")
                    
                    if lat is not None and lon is not None:
                        lat_coordinates.append(float(lat))
                        lon_coordinates.append(float(lon))
                        
            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")
        
        logger.info(f"Extracted {len(lat_coordinates)} coordinate pairs")
        return lat_coordinates, lon_coordinates
    
    def get_coordinate_stats(self, lat_coords: List[float], lon_coords: List[float]) -> pd.DataFrame:
        """Get statistical summary of coordinates."""
        df_lat = pd.DataFrame(lat_coords)
        df_lat.columns = ["latitude"]
        df_lon = pd.DataFrame(lon_coords)
        df_lon.columns = ["longitude"]
        
        stats = pd.DataFrame({
            "latitude": df_lat.describe()["latitude"],
            "longitude": df_lon.describe()["longitude"]
        })
        
        return stats


class ImageDownloader:
    """Downloads actual image files based on metadata."""
    
    def __init__(self, config):
        self.config = config.data
        self.download_dir = Path(self.config.download_dir)
    
    def download_image(self, image_name: str, date: str) -> bool:
        """Download a single image."""
        try:
            # Construct image URL
            url = f"https://epic.gsfc.nasa.gov/archive/enhanced/{date[:4]}/{date[5:7]}/{date[8:10]}/png/{image_name}.png"
            
            # Create target directory
            target_dir = self.download_dir / "earth"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save image
            image_path = target_dir / f"{image_name}.png"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {image_name}: {e}")
            return False
    
    def download_images_from_metadata(self, progress_callback=None) -> bool:
        """Download all images from metadata files."""
        combined_dir = Path(self.config.combined_dir)
        
        if not combined_dir.exists():
            raise FileNotFoundError("Combined metadata directory not found")
        
        image_count = 0
        success_count = 0
        
        # Process all metadata files
        for json_file in combined_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            date = json_file.stem
            
            for item in data:
                image_name = item.get("image")
                if image_name:
                    image_count += 1
                    if self.download_image(image_name, date):
                        success_count += 1
                
                if progress_callback:
                    progress_callback(success_count, image_count)
        
        logger.info(f"Downloaded {success_count}/{image_count} images")
        return success_count > 0