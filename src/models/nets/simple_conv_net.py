"""
File: simple_conv_net.py
Creation Date: 2023-01-07

A simple net consisting of a few Conv2d and Dense layers
"""
import torch
from torch import nn


class SimpleConvNet(nn.Module):
    def __init__(self, name: str = 'simple_conv', output_size: int = 1):
        super().__init__()

        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.model_2 = nn.Sequential(
            nn.Linear(in_features=6 * 16 * 16, out_features=1000),
            nn.Linear(1000, 500),
            nn.Linear(500, 250),
            nn.Linear(250, 120),
            nn.Linear(120, 60),
            nn.Linear(60, output_size)
        )

    def forward(self, x: torch.Tensor):
        x = self.model_1(x)
        x = x.view(-1, 6 * 16 * 16)
        return self.model_2(x)
