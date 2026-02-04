#!/usr/bin/env python3
"""
TensorBoard logging demonstration script.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

def demonstrate_tensorboard_logging():
    """Demonstrate comprehensive TensorBoard logging capabilities."""
    
    # Create a simple model for demonstration
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = DemoModel()
    
    # Create SummaryWriter with timestamped log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/demo_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard logs being written to: {log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir {log_dir}")
    
    # 1. Log scalars (training curves)
    for epoch in range(10):
        # Simulate training and validation loss
        train_loss = 1.0 * np.exp(-epoch * 0.3) + 0.1 * np.random.rand()
        val_loss = 1.0 * np.exp(-epoch * 0.25) + 0.15 * np.random.rand()
        learning_rate = 0.01 * (0.95 ** epoch)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Log accuracy
        accuracy = 1 - val_loss
        writer.add_scalar('Accuracy', accuracy, epoch)
    
    # 2. Log histograms (parameter distributions)
    with torch.no_grad():
        # Generate some fake parameter data
        weights = torch.randn(100, 50)
        gradients = torch.randn(100, 50) * 0.01
        
        writer.add_histogram('weights/fc1', weights, 0)
        writer.add_histogram('gradients/fc1', gradients, 0)
    
    # 3. Log images
    # Create sample images
    for i in range(5):
        # Generate a fake 28x28 image
        img = torch.randn(1, 28, 28)
        writer.add_image(f'sample_images/image_{i}', img, i)
    
    # 4. Log graphs
    dummy_input = torch.randn(1, 10)
    writer.add_graph(model, dummy_input)
    
    # 5. Log text and hyperparameters
    writer.add_text('Model/Architecture', str(model), 0)
    
    # Log hyperparameters
    hparams = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'optimizer': 'adam',
        'model_type': 'DemoModel'
    }
    metrics = {'final_accuracy': 0.95, 'final_loss': 0.05}
    writer.add_hparams(hparams, metrics)
    
    # 6. Log embeddings (optional)
    # Create some sample embeddings
    embedding_data = torch.randn(20, 10)  # 20 items, 10-dimensional
    metadata = [f'item_{i}' for i in range(20)]
    writer.add_embedding(embedding_data, metadata=metadata, tag='demo_embeddings')
    
    # 7. Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
    
    # 8. Log custom metrics
    for step in range(100):
        # Simulate gradient norms
        grad_norm = torch.randn(1).item() * 0.1 + 0.05
        writer.add_scalar('Gradients/Total_Norm', abs(grad_norm), step)
        
        # Simulate learning rate schedule
        if step % 10 == 0:
            writer.add_scalar('Scheduler/Step', step, step)
    
    writer.close()
    print("TensorBoard logging completed!")
    print(f"Run: tensorboard --logdir {log_dir} --port 6006")

if __name__ == "__main__":
    demonstrate_tensorboard_logging()