# Cross-Machine Dataset Tutorial

This tutorial explains how to use the cross-machine compatible dataset functionality that makes your code work on different machines by using proper image filenames instead of internal references.

## 🎯 What Problem Does This Solve?

**Before**: The original code used internal file references that were machine-specific, making it impossible to share datasets or run the code on different machines.

**After**: The new system creates a mapping between metadata and actual image files with proper filenames, making the code portable across machines.

## 📋 Prerequisites

- Completed basic setup (see main README)
- Have consolidated metadata in `data/combined/`
- Want to make your dataset portable

## 🚀 Quick Start

### 1. Create Image Files with Proper Filenames

```bash
# Download 100 images with proper filenames
python3 image_file_mapper.py --mode download --max_images 100

# Download latest 50 images
python3 image_file_mapper.py --mode download_latest --num_images 50

# Download recent 7 days of images
python3 image_file_mapper.py --mode download_recent --num_days 7
```

### 2. Verify Your Dataset

```bash
# Verify all images exist and have valid coordinates
python3 cross_machine_datasets.py --verify

# Verify with custom paths
python3 cross_machine_datasets.py --verify --mapping_file my_mapping.json --image_root_dir my_images
```

### 3. Use Cross-Machine Dataset in Your Code

```python
from cross_machine_datasets import create_cross_machine_dataloaders
from config import Config

# Load configuration
config = Config()

# Create cross-machine compatible dataloaders
train_loader, val_loader, test_loader = create_cross_machine_dataloaders(
    config,
    mapping_file="image_filename_mapping.json",
    image_root_dir="images_with_filenames",
    batch_size=32
)

# Use in training
for images, coordinates in train_loader:
    # Your training code here
    print(f"Batch: {images.shape}, {coordinates.shape}")
    break
```

## 📁 File Structure After Setup

```
Project/
├── image_filename_mapping.json     # Maps metadata -> image files
├── images_with_filenames/           # Downloaded images with proper names
│   ├── 2025-01-28/                  # Date directories
│   │   ├── epic_1b_20250128000101.png
│   │   └── epic_1b_20250128000201.png
│   └── 2025-01-29/
├── data/combined/                   # Original metadata
└── cross_machine_datasets.py        # Dataset loader
```

## 🔧 Detailed Usage

### Image File Mapper

The `image_file_mapper.py` script creates proper image files from consolidated metadata:

```bash
# Download modes
python3 image_file_mapper.py --mode download --max_images 100
python3 image_file_mapper.py --mode download_latest --num_images 50
python3 image_file_mapper.py --mode download_recent --num_days 7

# Verification mode
python3 image_file_mapper.py --mode verify

# Create mapping only
python3 image_file_mapper.py --mode map
```

**Options:**
- `--mode`: `download`, `download_latest`, `download_recent`, `verify`, `map`
- `--max_images`: Maximum number of images to download
- `--num_days`: Number of recent days to download
- `--output_dir`: Directory to save images
- `--mapping_file`: Path to mapping file

### Cross-Machine Dataset

The `cross_machine_datasets.py` provides portable dataset loading:

```python
from cross_machine_datasets import CrossMachineCompatibleDataset

# Create dataset directly
dataset = CrossMachineCompatibleDataset(
    config,
    mapping_file="image_filename_mapping.json",
    image_root_dir="images_with_filenames",
    split="train"
)

# Or use convenience function
from cross_machine_datasets import create_cross_machine_dataloaders
train_loader, val_loader, test_loader = create_cross_machine_dataloaders(config)
```

**Features:**
- ✅ Works on any machine with the image files
- ✅ Handles missing files gracefully
- ✅ Automatic train/val/test splits
- ✅ Device-optimized dataloader settings
- ✅ Coordinate validation and error handling

## 🧪 Demo and Testing

### Run Demo

```bash
# Create sample dataset to test functionality
python3 demo_cross_machine.py --create_sample --num_samples 20
```

### Test Your Setup

```bash
# Test dataset integrity
python3 cross_machine_datasets.py --verify

# Test with sample data
python3 demo_cross_machine.py --create_sample
```

## 📊 Example Output

### Successful Download
```
✅ Successfully created image files in: images_with_filenames
📋 Filename mapping saved to: image_filename_mapping.json
```

### Verification Results
```
📊 Dataset Integrity Report:
   Total entries: 39409
   Valid coordinates: 39409
   Existing files: 39409
   Missing files: 0
   Success rate: 100.0%
```

### Dataset Loading
```
Dataset size: 2396
Sample 1: Image shape: torch.Size([1, 64, 64]), Coordinates: (19.69, 0.05)
Batch shape: torch.Size([32, 1, 64, 64])
```

## 🔍 Troubleshooting

### Common Issues

**Q: "Mapping file not found"**
```bash
# Run the image mapper first
python3 image_file_mapper.py --mode download --max_images 10
```

**Q: "Missing image files"**
```bash
# Verify which files are missing
python3 image_file_mapper.py --mode verify

# Download missing files
python3 image_file_mapper.py --mode download --max_images 50
```

**Q: "Invalid coordinates"**
```bash
# Check coordinate validation
python3 cross_machine_datasets.py --verify

# Look for specific errors in the output
```

**Q: "NASA API 403 Forbidden"**
- This is a rate limiting issue with NASA's API
- Try again later or use existing data
- The demo creates sample data to test functionality

### Performance Tips

1. **📦 Batch Size**: Optimize for your GPU memory
2. **🔄 Workers**: Set to 0 for MPS, 2-4 for CUDA/CPU
3. **📁 File Organization**: Keep images in date directories
4. **💾 Memory**: Use appropriate batch sizes for your system

## 🎯 Integration with Existing Code

### Replace Old Dataset Loading

**Before:**
```python
from datasets import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config)
```

**After:**
```python
from cross_machine_datasets import create_cross_machine_dataloaders
train_loader, val_loader, test_loader = create_cross_machine_dataloaders(config)
```

### Update Training Scripts

```python
# In your training script
from cross_machine_datasets import create_cross_machine_dataloaders

# Create portable dataloaders
train_loader, val_loader, test_loader = create_cross_machine_dataloaders(
    config,
    mapping_file="image_filename_mapping.json",
    image_root_dir="images_with_filenames"
)

# Your existing training code works unchanged
for epoch in range(epochs):
    for images, coords in train_loader:
        # Training logic here
        pass
```

## 📦 Sharing Your Dataset

### What to Share

1. **Image Files**: The `images_with_filenames/` directory
2. **Mapping File**: `image_filename_mapping.json`
3. **Configuration**: Your `config.json` (optional)

### How to Share

```bash
# Create a shareable archive
tar -czf satellite_dataset.tar.gz images_with_filenames/ image_filename_mapping.json

# On another machine:
tar -xzf satellite_dataset.tar.gz
python3 cross_machine_datasets.py --verify
```

### What NOT to Share

- ❌ Original metadata files (machine-specific paths)
- ❌ Model checkpoints (can be retrained)
- ❌ Logs and temporary files

## 🎉 Benefits

### ✅ Cross-Machine Compatibility
- Works on any system with the image files
- No machine-specific path dependencies
- Easy to share and deploy

### ✅ Robust Error Handling
- Graceful handling of missing files
- Coordinate validation
- Device optimization

### ✅ Professional Dataset Management
- Proper file naming conventions
- Metadata-image mapping
- Verification tools

### ✅ Easy Integration
- Drop-in replacement for existing datasets
- Same API as original dataset
- Minimal code changes required

---

**🚀 Your dataset is now portable and ready for cross-machine deployment!**