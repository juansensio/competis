from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from .base import BaseModule
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange


class UNetDF(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        encoder_depth = 5
        self.encoder1 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s1,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        self.encoder2 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s2,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        # default values for unet
        decoder_use_batchnorm = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        # double decoder channels for feature fusion
        # assuming encoder1 and encoder2 have the same number of out_channels
        self.decoder = UnetDecoder(
            encoder_channels=[
                2*c*self.hparams.seq_len for c in self.encoder1.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if self.hparams.encoder.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        activation = None
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )
        self.name = "u-{}".format(self.hparams.encoder)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x, y=None):
        s1s, s2s = x
        B, L, _, _, _ = s1s.shape
        if y is not None:
            x = torch.cat((s1s, s2s), dim=2)
            for trans in self.transforms:
                x, y = trans(x, y)
            s1s, s2s = torch.split(
                x, [self.hparams.in_channels_s1, self.hparams.in_channels_s2], dim=2)
        s1s = rearrange(s1s, 'b l c h w -> (b l) c h w')
        s2s = rearrange(s2s, 'b l c h w -> (b l) c h w')
        features1 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L)
                     for f in self.encoder1(s1s)]
        features2 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L)
                     for f in self.encoder2(s2s)]
        features = []
        for f1, f2 in zip(features1, features2):
            features.append(torch.cat([f1, f2], dim=1))
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1), y
