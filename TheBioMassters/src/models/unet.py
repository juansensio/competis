import segmentation_models_pytorch as smp
from .base import BaseModule
import torch
from einops import rearrange


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
        x = torch.tensor([], device=self.device, dtype=torch.float32)
        if s2s is not None:
            x = torch.cat((x, s2s), dim=2)
        if s1s is not None:
            x = torch.cat((x, s1s), dim=2)
        if y is not None:
            for trans in self.transforms:
                x, y = trans(x, y)
        x = rearrange(x, 'b l c h w -> b (l c) h w')
        x = self.unet(x)
        return torch.sigmoid(x).squeeze(1), y
