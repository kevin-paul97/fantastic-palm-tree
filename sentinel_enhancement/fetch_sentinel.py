"""
Fetch a globally-distributed set of Copernicus Sentinel-2 true-color quicklooks
with precise scene-center coordinates, to be used as an auxiliary geolocation
dataset for transfer-learning the EPIC LocationRegressor.

Why Sentinel-2:
  - Copernicus / ESA mission with global, precisely geolocated optical imagery.
  - The STAC catalogue exposes a small true-color JPEG `thumbnail` asset per
    scene that is downloadable ANONYMOUSLY (no CDSE OAuth token required).
  - Each STAC item carries a `bbox`, giving an exact scene-center (lon, lat)
    label for the image -> coordinate regression task.

Output:
  sentinel_data/images/<id>.jpg          true-color quicklook
  sentinel_data/labels.json              { "<id>.jpg": [lon, lat], ... }
  sentinel_data/manifest.json            richer per-item metadata
"""

import json
import os
import sys
import time
import random
import concurrent.futures as cf
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sentinel_data"
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Target number of usable images.
TARGET = int(os.environ.get("SENTINEL_TARGET", "900"))
# Max scenes to request per grid cell.
PER_CELL = int(os.environ.get("SENTINEL_PER_CELL", "6"))
CLOUD_LT = float(os.environ.get("SENTINEL_CLOUD_LT", "25"))
DATE_RANGE = os.environ.get("SENTINEL_DATERANGE", "2025-04-01T00:00:00Z/2026-07-15T00:00:00Z")

UA = {"User-Agent": "epic-transfer-learning/1.0", "Content-Type": "application/json"}


def http_json(url, payload, timeout=40, retries=3):
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urlrequest.Request(url, data=data, headers=UA, method="POST")
            with urlrequest.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"  ! search failed: {e}", flush=True)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def download(url, dest, timeout=45, retries=2):
    for attempt in range(retries):
        try:
            req = urlrequest.Request(url, headers={"User-Agent": UA["User-Agent"]})
            with urlrequest.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            if len(blob) < 1200:  # too small to be a real quicklook
                return False
            dest.write_bytes(blob)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(1.0)
    return False


def build_grid():
    """Global grid of bbox cells over land-bearing latitudes."""
    cells = []
    # Skip the ice-covered extremes; S2 covers land + coastal water.
    lat_edges = list(range(-56, 72, 16))   # 16-degree lat bands
    lon_edges = list(range(-180, 180, 20))  # 20-degree lon bands
    for la in lat_edges:
        for lo in lon_edges:
            cells.append((lo, la, lo + 20, la + 16))
    random.Random(42).shuffle(cells)
    return cells


def scene_center(bbox):
    lon = (bbox[0] + bbox[2]) / 2.0
    lat = (bbox[1] + bbox[3]) / 2.0
    return round(lon, 4), round(lat, 4)


def main():
    grid = build_grid()
    print(f"Sentinel-2 fetch: {len(grid)} grid cells, target {TARGET} images, "
          f"cloud<{CLOUD_LT}%", flush=True)

    # 1) Collect candidate (id, thumb_url, lon, lat) via STAC search.
    candidates = {}
    for i, cell in enumerate(grid):
        if len(candidates) >= TARGET * 1.4:
            break
        payload = {
            "collections": [COLLECTION],
            "bbox": list(cell),
            "datetime": DATE_RANGE,
            "query": {"eo:cloud_cover": {"lt": CLOUD_LT}},
            "limit": PER_CELL,
        }
        res = http_json(STAC_SEARCH, payload)
        n_new = 0
        if res:
            for feat in res.get("features", []):
                fid = feat.get("id")
                assets = feat.get("assets", {})
                thumb = assets.get("thumbnail") or {}
                href = thumb.get("href")
                bbox = feat.get("bbox")
                if not (fid and href and bbox):
                    continue
                if fid in candidates:
                    continue
                lon, lat = scene_center(bbox)
                candidates[fid] = {
                    "id": fid,
                    "url": href,
                    "lon": lon,
                    "lat": lat,
                    "cloud": feat.get("properties", {}).get("eo:cloud_cover"),
                }
                n_new += 1
        if (i + 1) % 20 == 0 or n_new:
            print(f"  cell {i+1}/{len(grid)} -> +{n_new} (total candidates {len(candidates)})",
                  flush=True)
        time.sleep(0.05)

    print(f"Collected {len(candidates)} candidates. Downloading thumbnails...", flush=True)

    # 2) Download thumbnails concurrently.
    labels = {}
    manifest = []
    items = list(candidates.values())

    def fetch_one(item):
        fname = f"{item['id']}.jpg"
        dest = IMG_DIR / fname
        if dest.exists() and dest.stat().st_size > 1200:
            return fname, item
        ok = download(item["url"], dest)
        return (fname, item) if ok else (None, item)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch_one, it) for it in items]
        for fut in cf.as_completed(futs):
            fname, item = fut.result()
            done += 1
            if fname:
                labels[fname] = [item["lon"], item["lat"]]
                manifest.append({"file": fname, "lon": item["lon"],
                                 "lat": item["lat"], "cloud": item["cloud"]})
            if done % 50 == 0:
                print(f"  downloaded {len(labels)} / attempted {done}", flush=True)
            if len(labels) >= TARGET:
                break

    # Merge with any previously-downloaded labels so successive runs are additive.
    labels_path = OUT_DIR / "labels.json"
    merged = {}
    if labels_path.exists():
        try:
            merged = json.loads(labels_path.read_text())
        except Exception:
            merged = {}
    # Also recover any image on disk that already has a coord in the backup.
    backup = OUT_DIR / "labels_run1_backup.json"
    if backup.exists():
        try:
            for k, v in json.loads(backup.read_text()).items():
                merged.setdefault(k, v)
        except Exception:
            pass
    merged.update(labels)
    # Keep only labels whose image actually exists on disk.
    merged = {k: v for k, v in merged.items() if (IMG_DIR / k).exists()}

    labels_path.write_text(json.dumps(merged, indent=1))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE: {len(labels)} new this run; {len(merged)} total Sentinel-2 images in {IMG_DIR}", flush=True)
    print(f"labels.json -> {OUT_DIR/'labels.json'}", flush=True)


if __name__ == "__main__":
    main()
