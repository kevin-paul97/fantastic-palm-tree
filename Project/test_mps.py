#!/usr/bin/env python3
"""
Quick test script to verify MPS performance on Apple Silicon.
"""

import torch
import time
from config import Config
from models import create_location_regressor
from datasets import create_dataloaders

def test_mps_performance():
    """Test MPS vs CPU performance."""
    print("=== MPS Performance Test ===")
    
    # Test device detection
    config = Config()
    print(f"Auto-detected device: {config.training.device}")
    
    # Create small test data
    batch_size = 16
    input_tensor = torch.randn(batch_size, 1, 64, 64)
    
    devices_to_test = []
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append("mps")
    devices_to_test.append("cpu")
    
    model = create_location_regressor(config)
    
    for device in devices_to_test:
        print(f"\n--- Testing {device.upper()} ---")
        
        # Move model and data to device
        test_model = create_location_regressor(config)
        test_model.to(device)
        test_input = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = test_model(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = test_model(test_input)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {batch_size / (avg_time / 1000):.1f} samples/sec")
    
    print("\n=== Real Training Test ===")
    
    # Quick training test with 1 epoch
    config.training.epochs = 1
    config.training.batch_size = 32
    
    train_loader, val_loader, test_loader = create_dataloaders(config, batch_size=32)
    
    model = create_location_regressor(config)
    model.to(config.training.device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {config.training.device} for 1 epoch...")
    
    start_time = time.time()
    model.train()
    last_loss = None
    
    for batch_idx, (images, coords) in enumerate(train_loader):
        images = images.to(config.training.device)
        coords = coords.to(config.training.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Normalize coordinates
        from datasets import CoordinateNormalizer
        normalizer = CoordinateNormalizer()
        coords_norm = normalizer.normalize(coords)
        
        loss = criterion(outputs, coords_norm)
        loss.backward()
        optimizer.step()
        
        last_loss = loss.item()
        
        if batch_idx >= 10:  # Just test first 10 batches
            break
    
    end_time = time.time()
    
    print(f"Training time for 10 batches: {end_time - start_time:.2f} seconds")
    print(f"Average time per batch: {(end_time - start_time) / 10:.3f} seconds")
    if last_loss is not None:
        print(f"Loss after 10 batches: {last_loss:.6f}")

if __name__ == "__main__":
    test_mps_performance()