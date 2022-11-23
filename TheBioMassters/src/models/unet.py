import segmentation_models_pytorch as smp
from .base import BaseModule
import torch

class UNet(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.unet = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels,
            classes=1,
        )

    def forward(self, s1s, s2s):
        if s1s is not None:
            x = self.unet(s1s.squeeze(1))
        else:
            x = self.unet(s2s.squeeze(1))
        return torch.sigmoid(x).squeeze(1)