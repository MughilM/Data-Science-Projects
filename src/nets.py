"""
File: nets.py
Creation Date: 2025-06-19

Contains all the basic neural net definitions, so that these can be passed onto the
LightningModules.
"""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# A simple convolutional net for very basic training
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

# ResNet-50 pretrained on ImageNet, and ability to switch out the output layer
# and only train that.
class ResNet(nn.Module):
    def __init__(self, name: str = 'resnet50', output_size: int = 1, finetune: bool = False):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # We need to freeze all the layers, so they
        # don't get trained and waste time
        if not finetune:
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False
        # Once it's frozen, we adapt the fully-connected
        # layer to the output size we need. This will
        # get trained.
        self.resnet.fc = nn.Linear(2048, output_size)

    def forward(self, x: torch.Tensor):
        return self.resnet(x)
