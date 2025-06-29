"""
File: nets.py
Creation Date: 2025-06-19

Contains all the basic neural net definitions, so that these can be passed onto the
LightningModules.
"""
from typing import Literal, Dict, Tuple, Callable

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16
from torchvision.models.efficientnet import *

from src.utils.special_modules import ASPP, _SimpleSegmentationModel, Pix2PixUpsample, MobileNetEncoder

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

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


class UMobileNet(nn.Module):
    def __init__(self, image_size, in_image_channels, output_classes):
        super().__init__()
        self.output_classes = output_classes
        self.encoder = MobileNetEncoder(in_image_channels)
        # Calculate the corresponding image sizes of the layer outputs
        # in encoder. They divide by 2. If it's odd, then we take the ceiling.
        self.image_sizes = [image_size]
        for _ in range(5):
            if self.image_sizes[-1] % 2 == 1:
                self.image_sizes.append((self.image_sizes[-1] + 1) // 2)
            else:
                self.image_sizes.append(self.image_sizes[-1] // 2)

        self.up_stack = nn.ModuleList([
            Pix2PixUpsample(320, 512, 4),
            Pix2PixUpsample(576 + 512, 256, 4),
            Pix2PixUpsample(192 + 256, 128, 4),
            Pix2PixUpsample(144 + 128, 64, 4)
        ])

        self.last_conv = nn.ConvTranspose2d(in_channels=96 + 64, out_channels=output_classes, kernel_size=4,
                                            stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        # Push through encoder
        skips = list(self.encoder(x).values())
        # We start with block 16, and we'll also reverse the remaining
        x = skips[-1]
        skips = skips[::-1][1:]

        for up, skip_connection in zip(self.up_stack, skips):
            x = up(x)
            # Because the upsample cleanly multiplies by 2, there's
            # a chance that the resulting image size and the corresponding
            # image output from the encoder are off by 1. In this case, we center
            # crop the larger image to match.
            if x.shape[-1] > skip_connection.shape[-1]:
                x = TF.center_crop(x, output_size=skip_connection.shape[-1])
            # Now we can concatenate
            x = torch.cat([x, skip_connection], dim=1)
        x = self.last_conv(x)
        # Same issue here...
        if x.shape[-1] != self.image_sizes[0]:
            x = TF.center_crop(x, output_size=self.image_sizes[0])
        return x

class ViT(nn.Module):
    def __init__(self, name: str = 'vit_b_165', output_size: int = 1, pretrained: bool = True):
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