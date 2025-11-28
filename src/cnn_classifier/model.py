"""CNN model architecture for candlestick prediction."""

import torch
import torch.nn as nn


class CandlestickCNN(nn.Module):
    """
    CNN architecture from the paper.

    Architecture:
        Conv2D(32, 3x3) → ReLU → MaxPool2D(2x2)
        Conv2D(48, 3x3) → ReLU → MaxPool2D(2x2) → Dropout(0.25)
        Conv2D(64, 3x3) → ReLU → MaxPool2D(2x2)
        Conv2D(96, 3x3) → ReLU → MaxPool2D(2x2) → Dropout(0.25)
        Flatten
        Dense(256) → ReLU → Dropout(0.5)
        Dense(2) → Softmax
    """

    def __init__(self, image_size: int = 50) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(48, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 96, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.25)

        # Calculate flattened size dynamically
        # Each conv (3x3, no padding): size - 2
        # Each pool (2x2): size // 2
        size = image_size
        size = (size - 2) // 2  # conv1 + pool1
        size = (size - 2) // 2  # conv2 + pool2
        size = (size - 2) // 2  # conv3 + pool3
        size = (size - 2) // 2  # conv4 + pool4
        flat_size = size * size * 96

        self.fc1 = nn.Linear(flat_size, 256)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Conv block 3
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # Conv block 4
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop5(x)
        x = self.fc2(x)

        return x
