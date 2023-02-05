"""
File: resnet.py
Creation Date: 2023-01-10

A ResNet adaptation for use in transfer learning purposes.
Comes from PyTorch's integration of different ResNet
levels.
"""
import torch
from torch import nn
from torchvision.models import resnet50


class ResNet(nn.Module):
    def __init__(self, output_size: int = 1):
        super().__init__()

        self.resnet = resnet50(pretrained=True)
        # We need to freeze all the layers so they
        # don't get trained and waste time
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Once it's frozen, we adapt the fully-connected
        # layer to the output size we need. This will
        # get trained.
        self.resnet.fc = nn.Linear(2048, output_size)
    
    def forward(self, x: torch.Tensor):
        return self.resnet(x)