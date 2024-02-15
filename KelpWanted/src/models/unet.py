import torch
from .encoder import Encoder
from .decoder import Decoder, PAFPN


class Unet(torch.nn.Module):
    def __init__(
        self, encoder="resnet18", pretrained=True, in_chans=3, freeze=False, pafpn=False
    ):
        super().__init__()
        self.freeze = freeze
        self.encoder = Encoder(encoder, pretrained, in_chans)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.channels = [
            self.encoder.encoder.feature_info.channels(i)
            for i in range(len(self.encoder.encoder.feature_info))
        ]
        self.accum = self.channels[-1:]
        for i in reversed(range(len(self.channels) - 1)):
            self.accum.append(self.accum[-1] + self.channels[i])
        down_channels = self.accum[-1:]
        for i in range(len(self.accum) - 1):
            down_channels.append(self.accum[::-1][i + 1] * 2)
        self.pafpn = pafpn
        if pafpn:
            self.pafpn = PAFPN(self.channels, self.accum)
        self.decoder = Decoder(self.encoder.channels if not pafpn else down_channels)

    def forward(self, x):
        B = x.size(0)
        if self.freeze:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        if self.pafpn:
            features = self.pafpn(features)
        return self.decoder(features)
