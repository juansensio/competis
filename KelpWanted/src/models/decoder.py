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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        up_channels = (
            out_channels  # smp usa in_channels, unet paper usa out_channels...
        )
        self.convT = nn.ConvTranspose2d(
            in_channels,
            up_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv1 = Conv2dReLU(
            skip_channels + up_channels if skip_channels is not None else up_channels,
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
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # peta el scripting
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1, use_batchnorm=True):
        super().__init__()
        in_channels = encoder_channels[::-1]
        in_channels[0] = in_channels[0]
        skip_channels = in_channels[1:] + [None]
        out_channels = skip_channels[:-1] + [skip_channels[-2] // 2]
        blocks = [
            DecoderBlock(
                in_ch,
                skip_ch if skip_ch is not None else None,
                out_ch,
                use_batchnorm,
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, features[i + 1] if i + 1 < len(features) else None)
        return self.out_conv(x)


class PAFPN(nn.Module):
    def __init__(self, encoder_channels, channels, use_batchnorm=True):
        super().__init__()
        print(encoder_channels)
        self.up_blocks = nn.ModuleList(
            [
                Conv2dReLU(
                    encoder_channels[i] + encoder_channels[i - 1],
                    encoder_channels[i - 1],
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=use_batchnorm,
                )
                for i in reversed(range(1, len(encoder_channels)))
            ]
        )
        self.down_blocks = nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv2dReLU(
                        channels[i] if i == len(channels) - 1 else channels[i] * 2,
                        (channels[i - 1]),
                        kernel_size=3,
                        padding=1,
                        use_batchnorm=use_batchnorm,
                    ),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                for i in reversed(range(1, len(channels)))
            ]
        )

    def forward(self, features):
        # up
        x = features[-1]
        up = [x]
        for i in reversed(range(1, len(features))):
            xr = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x = torch.cat((features[i - 1], xr), 1)
            x = self.up_blocks[::-1][i - 1](x)
            up.append(x)

        for i in range(len(up)):
            print(up[i].shape)

        # down
        x = up[-1]
        down = [x]
        for i in reversed(range(1, len(up))):
            x = self.blocks[::-1][i - 1](x)
            x = torch.cat((x, up[i - 1]), 1)
            down.append(x)
        return down
