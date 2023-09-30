"""
File: src/models/nets/efficientnet.py
Creation Date: 2023-09-23

Contains definitions for various levels of EfficientNet. Levels range from 0 to 7. Level 0 has been shown
to have similar performance to ResNet-50, while having 5x fewer parameters, and 11x fewer FLOPS.
Options are present to use either finetuned weights, random weights, and to train and freeze the layers.
"""
import torch
import torch.nn as nn
from torchvision.models.efficientnet import *

from typing import Literal, Dict, Tuple, Callable

class EfficientNet(nn.Module):
    def __init__(self, level: int = 0, weights: str = Literal['random', 'pretrained'], finetune: bool = False,
                 output_size: int = 1):
        super().__init__()
        # TODO: Separate this into another Module class, because it changes according to the input image size.
        # Current image size: 401
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 3, (3, 3), stride=2, padding=24),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=3)
        )
        # Create a map for all the different levels, 0 through 7
        # Another thing which needs to be saved is the number of nodes in the last layer and the dropout, so that
        # it can be matched accordingly with however many class outputs we have.
        en_map: Dict[int, Tuple[Callable, int, float]] = {
            0: (efficientnet_b0, 1280, 0.2),
            1: (efficientnet_b1, 1280, 0.2),
            2: (efficientnet_b2, 1408, 0.3),
            3: (efficientnet_b3, 1536, 0.3),
            4: (efficientnet_b4, 1792, 0.4),
            5: (efficientnet_b5, 2048, 0.4),
            6: (efficientnet_b6, 2304, 0.5),
            7: (efficientnet_b7, 2560, 0.5)
        }
        # Grab the correct model, and if we need pretrained, use the "DEFAULT" keyword
        if weights == 'random':
            model, lin_features, rate = en_map[level]
            self.model: nn.Module = model()
        else:
            model, lin_features, rate = en_map[level]
            self.model: nn.Module = model(weights='DEFAULT')

        # Freeze the entire model if we need to
        if not finetune:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        # Change the last layer to match the output size
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=rate),
            nn.Linear(in_features=lin_features, out_features=output_size)
        )

    def forward(self, x):
        return self.model(self.model1(x))
