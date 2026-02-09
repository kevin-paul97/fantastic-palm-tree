"""
Data downloading and processing utilities for NASA EPIC satellite images.
"""

import json
import os
import shutil
from typing import List, Tuple, Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
            # Setup retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            url = f"{self.base_url}/all"
            response = session.get(url, timeout=60)
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
            # Setup retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            url = f"{self.base_url}/date/{date}"
            response = session.get(url, timeout=60)
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
    
    def download_recent_images(self, num_days: int = 7, progress_callback=None) -> bool:
        """Download most recent images including actual image files."""
        # First, download metadata
        metadata_success = self.download_all_images(progress_callback)
        
        if metadata_success:
            # Then download the actual image files
            from config import Config
            config = Config()
            config.data = self.config
            image_downloader = ImageDownloader(config)
            return image_downloader.download_recent_images(num_days, progress_callback)
        
        return False
    
    def download_latest_images(self, num_images: int = 100) -> bool:
        """Download the latest N images from the most recent dates."""
        metadata = self.load_metadata()
        dates = self.extract_dates(metadata)
        
        # Start from the most recent date and work backwards
        downloaded_images = 0
        target_date_idx = len(dates) - 1
        
        while downloaded_images < num_images and target_date_idx >= 0:
            date = dates[target_date_idx]
            logger.info(f"Downloading date {date} for more images ({downloaded_images}/{num_images})")
            
            if self.download_daily_data(date):
                # Check how many images we got from this date
                date_dir = self.images_dir / date
                daily_json = date_dir / f"{date}.json"
                
                if daily_json.exists():
                    with open(daily_json, 'r') as f:
                        daily_data = json.load(f)
                        images_in_date = len(daily_data)
                        downloaded_images += images_in_date
                        logger.info(f"Got {images_in_date} images from {date}")
            
            target_date_idx -= 1
        
        # Auto-consolidate after downloading
        if downloaded_images > 0:
            self.consolidate_metadata()
        
        return downloaded_images > 0
    
    def scan_available_data(self) -> Dict[str, Any]:
        """Scan images directory and return available dates and statistics."""
        stats = {
            "total_dates": 0,
            "total_images": 0,
            "available_dates": [],
            "images_per_date": {}
        }
        
        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return stats
            
        # Scan date folders
        date_dirs = [d for d in self.images_dir.iterdir() 
                    if d.is_dir() and d.name != ".DS_Store"]
        date_dirs.sort()
        
        for date_dir in date_dirs:
            date = date_dir.name
            image_files = list(date_dir.glob("*.png"))
            image_count = len(image_files)
            
            if image_count > 0:
                stats["available_dates"].append(date)
                stats["images_per_date"][date] = image_count
                stats["total_images"] += image_count
        
        stats["total_dates"] = len(stats["available_dates"])
        
        logger.info(f"Found {stats['total_images']} images across {stats['total_dates']} dates")
        return stats
    
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
    """Downloads actual image files based on metadata with API authentication."""
    
    def __init__(self, config):
        self.config = config.data
        self.images_dir = Path(self.config.images_dir)
        
        # Set up authenticated session
        try:
            from api_key_manager import get_authenticated_requests_session
            self.session = get_authenticated_requests_session()
        except ImportError:
            logger.warning("API key manager not available, using unauthenticated requests")
            self.session = requests.Session()
    
    def download_image(self, image_name: str, date: str) -> bool:
        """Download a single image with proper filename handling."""
        try:
            # Ensure image_name has .png extension
            if not image_name.endswith('.png'):
                image_name = f"{image_name}.png"

            year, month, day = date[:4], date[5:7], date[8:10]
            
            # Use the original EPIC archive domain (not API endpoint)
            # Format: https://epic.gsfc.nasa.gov/archive/natural/2019/05/30/png/epic_RGB_20190530011359.png
            url = f"https://epic.gsfc.nasa.gov/archive/natural/{year}/{month}/{day}/png/{image_name}"
            
            # Create target directory structure: images/YYYY-MM-DD/
            date_dir = self.images_dir / date
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image with proper filename
            image_path = date_dir / image_name
            
            # Skip if already exists
            if image_path.exists():
                logger.debug(f"Image already exists: {image_path}")
                return True
            
            # Download image with authentication
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Successfully downloaded: {image_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {image_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {image_name}: {e}")
            return False
    
    def download_images_from_date(self, date: str, progress_callback=None) -> Tuple[int, int]:
        """Download all images from a specific date metadata."""
        combined_dir = Path(self.config.combined_dir)
        metadata_file = combined_dir / f"{date}.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found for date {date}")
            return 0, 0
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            image_count = 0
            success_count = 0
            
            for item in data:
                image_name = item.get("image")
                if image_name:
                    image_count += 1
                    if self.download_image(image_name, date):
                        success_count += 1
                
                if progress_callback:
                    progress_callback(success_count, image_count)
            
            logger.info(f"Downloaded {success_count}/{image_count} images for {date}")
            return success_count, image_count
            
        except Exception as e:
            logger.error(f"Failed to process metadata for {date}: {e}")
            return 0, 0
    
    def download_images_from_metadata(self, progress_callback=None, max_images: Optional[int] = None) -> bool:
        """Download images from oldest metadata files first."""
        combined_dir = Path(self.config.combined_dir)
        
        if not combined_dir.exists():
            raise FileNotFoundError("Combined metadata directory not found")
        
        # Get all metadata files in chronological order (oldest first)
        json_files = sorted(combined_dir.glob("*.json"))
        if not json_files:
            logger.error("No metadata files found")
            return False
        
        total_images = 0
        total_success = 0
        
        # Process each date's metadata from oldest to newest
        for json_file in json_files:
            date = json_file.stem
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Count images for this date
                date_images = [item for item in data if item.get("image")]
                date_image_count = len(date_images)
                
                # Check if we've hit the limit BEFORE processing this date
                if max_images and (total_images + date_image_count) > max_images:
                    # Only download up to the limit
                    remaining_images = max_images - total_images
                    date_images = date_images[:remaining_images]
                    date_image_count = remaining_images
                
                # Download images for this date
                for item in date_images:
                    image_name = item.get("image")
                    if image_name:
                        total_images += 1
                        if self.download_image(image_name, date):
                            total_success += 1
                    
                    if progress_callback:
                        progress_callback(total_success, total_images)
                    
                    # Stop processing dates if we've reached the limit
                    if max_images and total_images >= max_images:
                        break
                        
                # Break out of outer loop if limit reached
                if max_images and total_images >= max_images:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                continue
        
        logger.info(f"Oldest-first download complete: {total_success}/{total_images} images")
        return total_success > 0
    
    def download_recent_images(self, num_days: int = 7, progress_callback=None) -> bool:
        """Download images from the oldest N days instead of recent."""
        combined_dir = Path(self.config.combined_dir)
        
        if not combined_dir.exists():
            raise FileNotFoundError("Combined metadata directory not found")
        
        # Get oldest metadata files first
        json_files = sorted(combined_dir.glob("*.json"))
        oldest_files = json_files[:num_days] if len(json_files) >= num_days else json_files
        
        total_images = 0
        total_success = 0
        
        for json_file in oldest_files:
            date = json_file.stem
            logger.info(f"Downloading images for {date} (oldest available)")
            
            success, count = self.download_images_from_date(date, progress_callback)
            total_success += success
            total_images += count
        
        logger.info(f"Oldest images download complete: {total_success}/{total_images} images")
        return total_success > 0