# A regression model to predict the location of a given image with two output nodes
# The CNN model sees the image in black and white and predicts the x and y coordinates (controid of the earth)
# It has two output nodes, one for x and one for y

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationRegressor(nn.Module):
    def __init__(self):
        super(LocationRegressor, self).__init__()

        # Using nn.Sequential to define the model
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.Tanh(), # Next the linear layers for regression
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.model(x)


