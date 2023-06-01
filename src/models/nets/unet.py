import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in mobilenet.parameters():
            param.requires_grad = False
        self.layers = {
            'features.2.conv.0': 'block_1',
            'features.4.conv.0': 'block_3',
            'features.7.conv.0': 'block_6',
            'features.14.conv.0': 'block_13',
            'features.17.conv.2': 'block_16'
        }
        self.model = create_feature_extractor(mobilenet, return_nodes=self.layers)

    def forward(self, x):
        return self.model(x)


class Pix2PixUpsample(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size):
        super().__init__()

        self.conv2d = nn.ConvTranspose2d(in_chan, out_chan, kernel_size, stride=2, padding=1, bias=False)
        # Initialize weights with mean 0 and std 0.02
        nn.init.normal_(self.conv2d.weight, mean=0, std=0.02)

        self.model = nn.Sequential(
            self.conv2d,
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.output_classes = output_classes
        self.encoder = Encoder()

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
            x = torch.cat([x, skip_connection], dim=1)
        x = self.last_conv(x)
        return x