"""
File: vit.py
Creation Date: 2023-03-08

Contains definitions for the vision transformer (ViT) with 16 x 16 image segments.
Also allows for a pretrained model to see how it fares when fine-tuned.
"""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ViT(nn.Module):
    def __init__(self, output_size: int = 1, pretrained: bool = True):
        super().__init__()

        if pretrained:
            # Uses ImageNet V1's weights. Others are available...
            self.vit = vit_b_16(weights='IMAGENET1K_V1')
        else:
            self.vit = vit_b_16()
        # Change the last layer to output output_size neurons
        self.vit.heads = nn.Sequential(nn.Linear(in_features=768, out_features=output_size, bias=True))

    def forward(self, x: torch.Tensor):
        return self.vit(x)
