import torch
from .encoder import Encoder
from .decoder import Decoder
from einops import rearrange


class Unet(torch.nn.Module):
    def __init__(
        self, encoder="resnet18", pretrained=True, in_chans=3, t=1, freeze=False
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
        self.decoder = Decoder(self.encoder.channels, t=t)

    def forward(self, x):
        B = x.size(0)
        x = rearrange(x, "b h w t c -> (b t) c h w")
        if self.freeze:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        features = [rearrange(f, "(b t) c h w -> b (t c) h w", b=B) for f in features]
        return self.decoder(features)
