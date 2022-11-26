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

    def forward(self, x, y=None):
        s1s, s2s = x
        if s1s is not None:
            if y is not None:
                for trans in self.transforms:
                    s1s, y = trans(s1s, y)
            x = self.unet(s1s.squeeze(1))
            # return s1s, y
        else:
            if y is not None:
                for trans in self.transforms:
                    s2s, y = trans(s2s, y)
            x = self.unet(s2s.squeeze(1))
            # return s2s, y
        return torch.sigmoid(x).squeeze(1), y

