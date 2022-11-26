import segmentation_models_pytorch as smp
from .base import BaseModule
import torch

class UNet2(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.unet1 = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels_s1,
            classes=1,
        )
        self.unet2 = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels_s2,
            classes=1,
        )
        self.conv_out = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x, y=None):
        s1s, s2s = x
        if y is not None:
            x = torch.cat((s1s, s2s), dim=2)
            for trans in self.transforms:
                x, y = trans(x, y)
            s1s, s2s = torch.split(x, [self.hparams.in_channels_s1, self.hparams.in_channels_s2], dim=2)
        x1 = self.unet1(s1s.squeeze(1)) # una sola fecha
        x2 = self.unet2(s2s.squeeze(1)) # una sola fecha
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_out(x)
        return torch.sigmoid(x).squeeze(1), y

