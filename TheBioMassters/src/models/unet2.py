import segmentation_models_pytorch as smp
from .base import BaseModule
import torch
from einops import rearrange


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
        self.conv_out = torch.nn.Conv2d(
            2*self.hparams.seq_len, 1, 3, padding=1)

    def forward(self, x, y=None):
        s1s, s2s = x
        B, L, _, _, _ = s2s.shape
        if y is not None:
            x = torch.cat((s1s, s2s), dim=2)
            for trans in self.transforms:
                x, y = trans(x, y)
            s1s, s2s = torch.split(
                x, [self.hparams.in_channels_s1, self.hparams.in_channels_s2], dim=2)
        x1 = self.unet1(rearrange(s1s, 'b l c h w -> (b l) c h w'))
        x2 = self.unet2(rearrange(s2s, 'b l c h w -> (b l) c h w'))
        x = torch.cat((x1, x2), dim=1)
        x = rearrange(x, '(b l) c h w -> b (l c) h w', b=B)
        x = self.conv_out(x)
        return torch.sigmoid(x).squeeze(1), y
