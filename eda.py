"""
Exploratory Data Analysis for satellite image coordinate prediction.

Standalone module — run with: python eda.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import logging
from datetime import datetime
from typing import Tuple, Dict

from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import Config
from datasets import SatelliteImageDataset, create_transforms

logger = logging.getLogger(__name__)


def load_all_data(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images and coordinates across train/val/test splits."""
    transform = create_transforms(
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    datasets = []
    for split in ("train", "val", "test"):
        ds = SatelliteImageDataset(
            image_dir=config.data.images_dir,
            metadata_dir=config.data.combined_dir,
            transform=transform,
            split=split,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
        )
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=64, shuffle=False, num_workers=0)

    all_images, all_coords = [], []
    for imgs, coords in loader:
        all_images.append(imgs.numpy())
        all_coords.append(coords.numpy())

    images = np.concatenate(all_images, axis=0)  # (N, C, H, W)
    coords = np.concatenate(all_coords, axis=0)  # (N, 2) [lon, lat]

    logger.info(f"Loaded {len(images)} samples, shape {images.shape}")
    return images, coords


def compute_image_statistics(images: np.ndarray) -> Dict:
    """Compute comprehensive pixel-level and per-image statistics."""
    N = images.shape[0]
    flat = images.reshape(N, -1)  # (N, C*H*W)

    per_image_means = flat.mean(axis=1)
    per_image_stds = flat.std(axis=1)

    all_pixels = flat.ravel()
    # Subsample for higher-order stats to keep it fast
    rng = np.random.default_rng(42)
    pixel_sample = rng.choice(all_pixels, size=min(500_000, len(all_pixels)), replace=False)

    mean_image = images.mean(axis=0)  # (C, H, W)
    std_image = images.std(axis=0)

    return {
        "global_mean": float(all_pixels.mean()),
        "global_std": float(all_pixels.std()),
        "global_var": float(all_pixels.var()),
        "global_min": float(all_pixels.min()),
        "global_max": float(all_pixels.max()),
        "global_skew": float(scipy_stats.skew(pixel_sample)),
        "global_kurtosis": float(scipy_stats.kurtosis(pixel_sample)),
        "mean_image": mean_image,
        "std_image": std_image,
        "per_image_means": per_image_means,
        "per_image_stds": per_image_stds,
        "pixel_sample": pixel_sample,
    }


def compute_coordinate_statistics(coords: np.ndarray) -> Dict:
    """Compute statistics for longitude and latitude."""
    result = {}
    for i, name in enumerate(["lon", "lat"]):
        vals = coords[:, i]
        result[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "var": float(np.var(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "skew": float(scipy_stats.skew(vals)),
            "kurtosis": float(scipy_stats.kurtosis(vals)),
        }
    return result


def compute_pca(images: np.ndarray) -> Dict:
    """Run PCA to 2 components on flattened images."""
    N = images.shape[0]
    flat = images.reshape(N, -1).astype(np.float64)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(flat)

    return {
        "pca_2d": pca_2d,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "singular_values": pca.singular_values_,
    }


def compute_kmeans(images: np.ndarray, pca_results: Dict, k_range=(2, 9)) -> Dict:
    """Run K-Means clustering, selecting optimal K via silhouette score."""
    N = images.shape[0]
    flat = images.reshape(N, -1).astype(np.float64)

    k_min, k_max = k_range
    ks = list(range(k_min, k_max))
    inertias = []
    silhouettes = []

    pca_2d = pca_results["pca_2d"]

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(flat)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(pca_2d, labels, sample_size=min(5000, N))))
        logger.info(f"  K={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.3f}")

    best_idx = int(np.argmax(silhouettes))
    best_k = ks[best_idx]
    logger.info(f"Best K={best_k} (silhouette={silhouettes[best_idx]:.3f})")

    # Refit with best K
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_best.fit_predict(flat)
    centers = km_best.cluster_centers_  # (K, C*H*W)

    return {
        "labels": labels,
        "centers": centers,
        "best_k": best_k,
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_silhouette": silhouettes[best_idx],
    }


def compute_correlations(
    image_stats: Dict, coords: np.ndarray, pca_results: Dict
) -> Dict:
    """Compute Pearson correlations between features and coordinates."""
    pca_2d = pca_results["pca_2d"]
    lon, lat = coords[:, 0], coords[:, 1]
    means = image_stats["per_image_means"]

    result = {}
    pairs = [
        ("pc1_lon", pca_2d[:, 0], lon),
        ("pc1_lat", pca_2d[:, 0], lat),
        ("pc2_lon", pca_2d[:, 1], lon),
        ("pc2_lat", pca_2d[:, 1], lat),
        ("brightness_lon", means, lon),
        ("brightness_lat", means, lat),
    ]
    for name, a, b in pairs:
        r, p = scipy_stats.pearsonr(a, b)
        result[name] = {"r": float(r), "p": float(p)}

    return result


def log_to_tensorboard(
    image_stats: Dict,
    coord_stats: Dict,
    pca_results: Dict,
    corr_results: Dict,
    kmeans_results: Dict,
    n_samples: int,
    config: Config,
) -> None:
    """Log all EDA metrics to TensorBoard."""
    if SummaryWriter is None:
        logger.warning("TensorBoard not available, skipping logging")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.training.log_dir, f"eda_{timestamp}")
    writer = SummaryWriter(log_dir)

    writer.add_scalar("eda/dataset/total_samples", n_samples, 0)

    # Image stats
    for key in ("global_mean", "global_std", "global_var", "global_min",
                "global_max", "global_skew", "global_kurtosis"):
        writer.add_scalar(f"eda/image/{key}", image_stats[key], 0)

    # Coordinate stats
    for axis in ("lon", "lat"):
        for key in ("mean", "std", "var", "min", "max", "skew", "kurtosis"):
            writer.add_scalar(f"eda/coords/{axis}_{key}", coord_stats[axis][key], 0)

    # PCA
    writer.add_scalar("eda/pca/explained_variance_pc1",
                       float(pca_results["explained_variance_ratio"][0]), 0)
    writer.add_scalar("eda/pca/explained_variance_pc2",
                       float(pca_results["explained_variance_ratio"][1]), 0)
    writer.add_scalar("eda/pca/total_explained_variance",
                       pca_results["total_explained_variance"], 0)

    # Correlations
    for name, vals in corr_results.items():
        writer.add_scalar(f"eda/correlation/{name}_r", vals["r"], 0)
        writer.add_scalar(f"eda/correlation/{name}_p", vals["p"], 0)

    # K-Means
    writer.add_scalar("eda/kmeans/best_k", kmeans_results["best_k"], 0)
    writer.add_scalar("eda/kmeans/best_silhouette", kmeans_results["best_silhouette"], 0)
    for i, (k, inertia, sil) in enumerate(zip(
            kmeans_results["ks"], kmeans_results["inertias"], kmeans_results["silhouettes"])):
        writer.add_scalar("eda/kmeans/inertia", inertia, k)
        writer.add_scalar("eda/kmeans/silhouette", sil, k)

    # Mean image
    mean_img = image_stats["mean_image"]
    if mean_img.ndim == 3 and mean_img.shape[0] == 1:
        mean_img_tb = np.repeat(mean_img, 3, axis=0)  # (3, H, W) for TB
    else:
        mean_img_tb = mean_img
    writer.add_image("eda/mean_image", mean_img_tb, 0)

    # Summary text
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Samples | {n_samples} |",
        f"| Pixel Mean | {image_stats['global_mean']:.4f} |",
        f"| Pixel Std | {image_stats['global_std']:.4f} |",
        f"| PCA Total Var | {pca_results['total_explained_variance']*100:.1f}% |",
        f"| Lon Range | [{coord_stats['lon']['min']:.1f}, {coord_stats['lon']['max']:.1f}] |",
        f"| Lat Range | [{coord_stats['lat']['min']:.1f}, {coord_stats['lat']['max']:.1f}] |",
    ]
    writer.add_text("eda/summary", "\n".join(lines), 0)

    writer.flush()
    writer.close()
    logger.info(f"TensorBoard logs saved to {log_dir}")


def create_overview_figure(
    images: np.ndarray,
    coords: np.ndarray,
    image_stats: Dict,
    coord_stats: Dict,
    pca_results: Dict,
    corr_results: Dict,
    kmeans_results: Dict,
    save_path: str,
) -> None:
    """Create a single large overview figure with 5x4 subplots."""
    fig, axes = plt.subplots(5, 4, figsize=(24, 25))
    fig.suptitle(
        "Exploratory Data Analysis — Satellite Image Coordinate Prediction",
        fontsize=16, fontweight="bold", y=0.98,
    )

    N, C, H, W = images.shape
    rng = np.random.default_rng(42)

    # ── Row 0: Image Visualization ──────────────────────────────────────
    # (0,0) Mean image
    ax = axes[0, 0]
    mean_img = image_stats["mean_image"][0] if C == 1 else image_stats["mean_image"].transpose(1, 2, 0)
    ax.imshow(mean_img, cmap="gray")
    ax.set_title("Mean Image")
    ax.axis("off")

    # (0,1) Std image
    ax = axes[0, 1]
    std_img = image_stats["std_image"][0] if C == 1 else image_stats["std_image"].transpose(1, 2, 0)
    im = ax.imshow(std_img, cmap="hot")
    ax.set_title("Pixel Std Dev")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (0,2) & (0,3) Sample images
    sample_idxs = rng.choice(N, size=2, replace=False)
    for col, idx in zip([2, 3], sample_idxs):
        ax = axes[0, col]
        img = images[idx, 0] if C == 1 else images[idx].transpose(1, 2, 0)
        ax.imshow(img, cmap="gray")
        lon, lat = coords[idx]
        ax.set_title(f"Sample: Lon={lon:.1f}, Lat={lat:.1f}", fontsize=10)
        ax.axis("off")

    # ── Row 1: Distributions ────────────────────────────────────────────
    # (1,0) Pixel intensity histogram
    ax = axes[1, 0]
    ax.hist(image_stats["pixel_sample"], bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title("Pixel Intensity Distribution")
    ax.set_xlabel("Pixel Value [0-1]")
    ax.set_ylabel("Count")
    stats_text = (f"mean={image_stats['global_mean']:.3f}\n"
                  f"std={image_stats['global_std']:.3f}\n"
                  f"skew={image_stats['global_skew']:.2f}\n"
                  f"kurt={image_stats['global_kurtosis']:.2f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # (1,1) Per-image mean brightness
    ax = axes[1, 1]
    means = image_stats["per_image_means"]
    ax.hist(means, bins=50, color="darkorange", edgecolor="none", alpha=0.8)
    ax.axvline(means.mean(), color="red", ls="--", lw=1.5, label=f"mean={means.mean():.3f}")
    ax.axvline(np.median(means), color="blue", ls="--", lw=1.5, label=f"median={np.median(means):.3f}")
    ax.set_title("Mean Brightness per Image")
    ax.set_xlabel("Mean Pixel Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # (1,2) Longitude distribution
    ax = axes[1, 2]
    lon = coords[:, 0]
    ax.hist(lon, bins=50, color="royalblue", edgecolor="none", alpha=0.8)
    ax.axvline(lon.mean(), color="red", ls="--", lw=1.5)
    ax.axvline(np.median(lon), color="navy", ls="--", lw=1.5)
    ax.set_title("Longitude Distribution")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Count")
    cs = coord_stats["lon"]
    txt = (f"mean={cs['mean']:.1f}\nstd={cs['std']:.1f}\n"
           f"skew={cs['skew']:.2f}\nkurt={cs['kurtosis']:.2f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # (1,3) Latitude distribution
    ax = axes[1, 3]
    lat = coords[:, 1]
    ax.hist(lat, bins=50, color="crimson", edgecolor="none", alpha=0.8)
    ax.axvline(lat.mean(), color="red", ls="--", lw=1.5)
    ax.axvline(np.median(lat), color="darkred", ls="--", lw=1.5)
    ax.set_title("Latitude Distribution")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Count")
    cs = coord_stats["lat"]
    txt = (f"mean={cs['mean']:.1f}\nstd={cs['std']:.1f}\n"
           f"skew={cs['skew']:.2f}\nkurt={cs['kurtosis']:.2f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Row 2: PCA & Spatial ────────────────────────────────────────────
    pca_2d = pca_results["pca_2d"]

    # (2,0) Lon vs Lat scatter
    ax = axes[2, 0]
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.3, c="teal")
    ax.set_title("Coordinate Coverage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    # (2,1) PCA colored by longitude
    ax = axes[2, 1]
    sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], s=4, alpha=0.4,
                    c=coords[:, 0], cmap="viridis")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Longitude")
    r_val = corr_results["pc1_lon"]["r"]
    ax.set_title(f"PCA colored by Longitude (r={r_val:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # (2,2) PCA colored by latitude
    ax = axes[2, 2]
    sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], s=4, alpha=0.4,
                    c=coords[:, 1], cmap="coolwarm")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Latitude")
    r_val = corr_results["pc1_lat"]["r"]
    ax.set_title(f"PCA colored by Latitude (r={r_val:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # (2,3) PCA explained variance
    ax = axes[2, 3]
    evr = pca_results["explained_variance_ratio"] * 100
    bars = ax.bar(["PC1", "PC2"], evr, color=["#4C72B0", "#DD8452"], edgecolor="black", lw=0.5)
    for bar, val in zip(bars, evr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"PCA Explained Variance (total={evr.sum():.1f}%)")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_ylim(0, max(evr) * 1.2)

    # ── Row 3: Correlations & Tables ────────────────────────────────────
    per_means = image_stats["per_image_means"]

    # (3,0) Brightness vs Longitude
    ax = axes[3, 0]
    ax.scatter(coords[:, 0], per_means, s=4, alpha=0.3, c="seagreen")
    z = np.polyfit(coords[:, 0], per_means, 1)
    x_line = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", lw=2)
    cr = corr_results["brightness_lon"]
    ax.set_title(f"Brightness vs Longitude (r={cr['r']:.3f})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Mean Brightness")

    # (3,1) Brightness vs Latitude
    ax = axes[3, 1]
    ax.scatter(coords[:, 1], per_means, s=4, alpha=0.3, c="coral")
    z = np.polyfit(coords[:, 1], per_means, 1)
    x_line = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r-", lw=2)
    cr = corr_results["brightness_lat"]
    ax.set_title(f"Brightness vs Latitude (r={cr['r']:.3f})")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Mean Brightness")

    # (3,2) Statistics summary table
    ax = axes[3, 2]
    ax.axis("off")
    ax.set_title("Summary Statistics", fontweight="bold")
    col_labels = ["Pixels", "Longitude", "Latitude"]
    row_labels = ["Mean", "Std", "Variance", "Min", "Max", "Skewness", "Kurtosis"]
    cs_lon = coord_stats["lon"]
    cs_lat = coord_stats["lat"]
    table_data = [
        [f"{image_stats['global_mean']:.4f}", f"{cs_lon['mean']:.2f}", f"{cs_lat['mean']:.2f}"],
        [f"{image_stats['global_std']:.4f}", f"{cs_lon['std']:.2f}", f"{cs_lat['std']:.2f}"],
        [f"{image_stats['global_var']:.4f}", f"{cs_lon['var']:.2f}", f"{cs_lat['var']:.2f}"],
        [f"{image_stats['global_min']:.4f}", f"{cs_lon['min']:.2f}", f"{cs_lat['min']:.2f}"],
        [f"{image_stats['global_max']:.4f}", f"{cs_lon['max']:.2f}", f"{cs_lat['max']:.2f}"],
        [f"{image_stats['global_skew']:.3f}", f"{cs_lon['skew']:.3f}", f"{cs_lat['skew']:.3f}"],
        [f"{image_stats['global_kurtosis']:.3f}", f"{cs_lon['kurtosis']:.3f}", f"{cs_lat['kurtosis']:.3f}"],
    ]
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                     cellLoc="center", loc="center", colWidths=[0.28, 0.28, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # (3,3) Correlation summary table
    ax = axes[3, 3]
    ax.axis("off")
    ax.set_title("Correlations (Pearson)", fontweight="bold")
    corr_rows = ["PC1-Lon", "PC1-Lat", "PC2-Lon", "PC2-Lat", "Bright-Lon", "Bright-Lat"]
    corr_keys = ["pc1_lon", "pc1_lat", "pc2_lon", "pc2_lat", "brightness_lon", "brightness_lat"]
    corr_data = []
    for key in corr_keys:
        r = corr_results[key]["r"]
        p = corr_results[key]["p"]
        sig = "*" if p < 0.05 else ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        corr_data.append([f"{r:.4f}", f"{p:.2e}", sig])
    table = ax.table(cellText=corr_data, rowLabels=corr_rows,
                     colLabels=["r", "p-value", "sig"],
                     cellLoc="center", loc="center", colWidths=[0.22, 0.28, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # ── Row 4: K-Means Clustering ─────────────────────────────────────
    labels = kmeans_results["labels"]
    best_k = kmeans_results["best_k"]
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_k))

    # (4,0) Elbow + Silhouette curves
    ax = axes[4, 0]
    ks = kmeans_results["ks"]
    ax2 = ax.twinx()
    ax.plot(ks, kmeans_results["inertias"], "o-", color="#4C72B0", lw=2, label="Inertia")
    ax2.plot(ks, kmeans_results["silhouettes"], "s-", color="#DD8452", lw=2, label="Silhouette")
    ax.axvline(best_k, color="red", ls="--", lw=1.5, alpha=0.7, label=f"Best K={best_k}")
    ax.set_title("K-Means: Elbow & Silhouette")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia", color="#4C72B0")
    ax2.set_ylabel("Silhouette Score", color="#DD8452")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # (4,1) PCA colored by cluster
    ax = axes[4, 1]
    for c in range(best_k):
        mask = labels == c
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=4, alpha=0.4,
                   color=cluster_colors[c], label=f"C{c} (n={mask.sum()})")
    ax.set_title(f"PCA by K-Means Cluster (K={best_k})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=3, loc="best")

    # (4,2) Lon/Lat colored by cluster
    ax = axes[4, 2]
    for c in range(best_k):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.4,
                   color=cluster_colors[c], label=f"C{c}")
    ax.set_title("Geographic Distribution by Cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, markerscale=3, loc="best")

    # (4,3) Cluster centers as images
    ax = axes[4, 3]
    centers = kmeans_results["centers"]
    center_imgs = centers.reshape(best_k, C, H, W)
    # Tile cluster center images horizontally
    if C == 1:
        tiles = [center_imgs[c, 0] for c in range(best_k)]
    else:
        tiles = [center_imgs[c].transpose(1, 2, 0) for c in range(best_k)]
    tile_row = np.concatenate(tiles, axis=1)
    ax.imshow(tile_row, cmap="gray")
    # Add cluster labels
    for c in range(best_k):
        ax.text(c * W + W / 2, -2, f"C{c}", ha="center", fontsize=9, fontweight="bold",
                color=cluster_colors[c])
    ax.set_title("Cluster Centers")
    ax.axis("off")

    # Footer
    mode = "Grayscale" if C == 1 else "RGB"
    fig.text(0.5, 0.01,
             f"N = {N} samples  |  Image size: {H}x{W}  |  {mode}  |  "
             f"PCA explained variance: {pca_results['total_explained_variance']*100:.1f}%  |  "
             f"K-Means K={best_k} (silhouette={kmeans_results['best_silhouette']:.3f})",
             ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Overview figure saved to {save_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = Config()
    os.makedirs("outputs", exist_ok=True)

    logger.info("Starting Exploratory Data Analysis...")

    # Load data
    logger.info("Loading dataset (all splits)...")
    images, coords = load_all_data(config)
    logger.info(f"Loaded {len(images)} samples")

    # Compute statistics
    logger.info("Computing image statistics...")
    image_stats = compute_image_statistics(images)

    logger.info("Computing coordinate statistics...")
    coord_stats = compute_coordinate_statistics(coords)

    # PCA
    logger.info("Running PCA (2 components)...")
    pca_results = compute_pca(images)

    # Correlations
    logger.info("Computing correlations...")
    corr_results = compute_correlations(image_stats, coords, pca_results)

    # K-Means clustering
    logger.info("Running K-Means clustering (K=2..8)...")
    kmeans_results = compute_kmeans(images, pca_results)

    # TensorBoard
    logger.info("Logging to TensorBoard...")
    log_to_tensorboard(image_stats, coord_stats, pca_results, corr_results,
                       kmeans_results, len(images), config)

    # Overview figure
    logger.info("Creating overview figure...")
    save_path = "outputs/eda_overview.png"
    create_overview_figure(images, coords, image_stats, coord_stats,
                           pca_results, corr_results, kmeans_results, save_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Total samples:    {len(images)}")
    print(f"Image shape:      {images.shape[1:]} ({'Grayscale' if images.shape[1]==1 else 'RGB'})")
    print(f"Pixel mean:       {image_stats['global_mean']:.4f}")
    print(f"Pixel std:        {image_stats['global_std']:.4f}")
    print(f"Pixel skewness:   {image_stats['global_skew']:.3f}")
    print(f"Pixel kurtosis:   {image_stats['global_kurtosis']:.3f}")
    print(f"Longitude range:  [{coord_stats['lon']['min']:.2f}, {coord_stats['lon']['max']:.2f}]")
    print(f"Latitude range:   [{coord_stats['lat']['min']:.2f}, {coord_stats['lat']['max']:.2f}]")
    print(f"PCA explained:    {pca_results['total_explained_variance']*100:.1f}%"
          f"  (PC1={pca_results['explained_variance_ratio'][0]*100:.1f}%,"
          f" PC2={pca_results['explained_variance_ratio'][1]*100:.1f}%)")

    print(f"K-Means K:        {kmeans_results['best_k']}"
          f"  (silhouette={kmeans_results['best_silhouette']:.3f})")
    cluster_sizes = np.bincount(kmeans_results["labels"])
    print(f"Cluster sizes:    {', '.join(f'C{i}={s}' for i, s in enumerate(cluster_sizes))}")

    # Find strongest correlation
    best_key = max(corr_results, key=lambda k: abs(corr_results[k]["r"]))
    best = corr_results[best_key]
    print(f"Strongest corr:   {best_key} (r={best['r']:.4f}, p={best['p']:.2e})")

    print(f"\nFigure saved:     {save_path}")
    print("=" * 60)

    logger.info("EDA complete!")


if __name__ == "__main__":
    main()
