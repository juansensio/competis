import torch
import timm


class Encoder(torch.nn.Module):
    def __init__(self, encoder, pretrained=True, in_chans=3):
        super().__init__()
        self.encoder = timm.create_model(
            encoder,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
        )
        self.channels = [
            self.encoder.feature_info.channels(i)
            for i in range(len(self.encoder.feature_info))
        ]

    def forward(self, x):
        return self.encoder(x)
