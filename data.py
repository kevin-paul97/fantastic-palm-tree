"""
Data downloading and processing utilities for NASA EPIC satellite images.

API reference: https://epic.gsfc.nasa.gov/about/api

Endpoints:
  Metadata:  https://epic.gsfc.nasa.gov/api/natural/available
             https://epic.gsfc.nasa.gov/api/natural/date/YYYY-MM-DD
  Images:    https://epic.gsfc.nasa.gov/archive/natural/YYYY/MM/DD/png/{name}.png
"""

import json
import requests
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

API_BASE = "https://epic.gsfc.nasa.gov/api/natural"
ARCHIVE_BASE = "https://epic.gsfc.nasa.gov/archive/natural"


class EPICDataDownloader:
    """Downloads metadata and images from the NASA EPIC API."""

    def __init__(self, config):
        self.config = config.data
        self.images_dir = Path(self.config.images_dir)
        self.combined_dir = Path(self.config.combined_dir)

    # ── Metadata ──

    def fetch_available_dates(self) -> List[str]:
        """Fetch list of all available dates from the API."""
        url = f"{API_BASE}/available"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dates = response.json()
        dates.sort()
        logger.info(f"Found {len(dates)} available dates")
        return dates

    def fetch_date_metadata(self, date: str) -> List[dict]:
        """Fetch full metadata for a specific date and save to combined dir."""
        url = f"{API_BASE}/date/{date}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        self.combined_dir.mkdir(parents=True, exist_ok=True)
        with open(self.combined_dir / f"{date}.json", "w") as f:
            json.dump(data, f)

        logger.info(f"Fetched metadata for {date}: {len(data)} images")
        return data

    # ── Image downloads ──

    def download_image(self, image_name: str, date: str) -> bool:
        """Download a single image from the EPIC archive.

        Args:
            image_name: Image identifier from metadata (e.g. 'epic_1b_20150613110250').
            date: Date string 'YYYY-MM-DD'.
        """
        if not image_name.endswith(".png"):
            image_name = f"{image_name}.png"

        date_dir = self.images_dir / date
        image_path = date_dir / image_name

        if image_path.exists():
            return True

        year, month, day = date[:4], date[5:7], date[8:10]
        url = f"{ARCHIVE_BASE}/{year}/{month}/{day}/png/{image_name}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download {image_name}: {e}")
            return False

        date_dir.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(response.content)

        logger.debug(f"Downloaded: {image_path}")
        return True

    def download_date_images(self, date: str) -> Tuple[int, int]:
        """Download all images for a date. Fetches metadata first if needed.

        Returns:
            (success_count, total_count)
        """
        metadata_path = self.combined_dir / f"{date}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
        else:
            data = self.fetch_date_metadata(date)

        images = [item["image"] for item in data if item.get("image")]
        success = sum(1 for name in images if self.download_image(name, date))
        logger.info(f"{date}: downloaded {success}/{len(images)} images")
        return success, len(images)

    def download_metadata(self) -> bool:
        """Download metadata for all available dates."""
        dates = self.fetch_available_dates()
        logger.info(f"Downloading metadata for {len(dates)} dates")
        
        for date in dates:
            try:
                self.fetch_date_metadata(date)
            except Exception as e:
                logger.error(f"Failed to download metadata for {date}: {e}")
                return False
                
        logger.info("Metadata download complete")
        return True

    # ── Download functionality ──

    def download_recent(self, num_days: int = 7) -> bool:
        """Download images from the most recent N days."""
        dates = self.fetch_available_dates()
        recent_dates = dates[-num_days:]
        logger.info(f"Downloading images for {len(recent_dates)} most recent dates")

        total_success, total_count = 0, 0
        for date in recent_dates:
            success, count = self.download_date_images(date)
            total_success += success
            total_count += count

        logger.info(f"Recent download complete: {total_success}/{total_count} images")
        return total_success > 0

    # ── Utilities ──

    def scan_available_data(self) -> Dict[str, Any]:
        """Scan local images directory and return statistics."""
        stats = {
            "total_dates": 0,
            "total_images": 0,
            "available_dates": [],
            "images_per_date": {},
        }

        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return stats

        for date_dir in sorted(self.images_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name == ".DS_Store":
                continue
            image_files = list(date_dir.glob("*.png"))
            if image_files:
                stats["available_dates"].append(date_dir.name)
                stats["images_per_date"][date_dir.name] = len(image_files)
                stats["total_images"] += len(image_files)

        stats["total_dates"] = len(stats["available_dates"])
        logger.info(f"Found {stats['total_images']} images across {stats['total_dates']} dates")
        return stats


class CoordinateExtractor:
    """Extracts coordinate data from saved metadata in combined/ directory."""

    def __init__(self, config):
        self.combined_dir = Path(config.data.combined_dir)

    def extract_coordinates(self) -> Tuple[List[float], List[float]]:
        """Extract latitude and longitude from all metadata files."""
        lat_coordinates = []
        lon_coordinates = []

        if not self.combined_dir.exists():
            raise FileNotFoundError(f"Combined directory not found: {self.combined_dir}")

        for json_file in sorted(self.combined_dir.glob("*.json")):
            try:
                with open(json_file) as f:
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
        stats = pd.DataFrame({
            "latitude": pd.Series(lat_coords).describe(),
            "longitude": pd.Series(lon_coords).describe(),
        })
        return stats
