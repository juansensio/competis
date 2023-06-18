import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional 

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
        
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        self.convT = nn.ConvTranspose2d(
            in_channels,
            skip_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv1 = Conv2dReLU(
            skip_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    # def forward(self, x, skip: Optional[torch.Tensor]=None): # hace que el cat pete al hacer el scripting
    def forward(self, x, skip):
        x = self.convT(x)
        x = torch.cat([x, skip], dim=1) # peta el scripting
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=1,
        use_batchnorm=True,
    ):
        super().__init__()
        in_channels = channels[:-1]
        skip_channels = in_channels[1:] + in_channels[-1:]
        blocks = [
            DecoderBlock(in_ch, skip_ch, skip_ch, use_batchnorm)
            for in_ch, skip_ch in zip(in_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(skip_channels[-1], out_channels, kernel_size=1)

    def forward(self, features):
        features = features[::-1]
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(features[i], features[i+1] if i+1 < len(features) else None)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.out_conv(x)
        return x
