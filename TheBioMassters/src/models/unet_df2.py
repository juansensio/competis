from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange


class UNetDF2(torch.nn.Module):
    def __init__(self, encoder='resnet18', pretrained='imagenet'):
        super().__init__()
        encoder_depth = 5
        self.encoder1 = get_encoder(
            encoder,
            in_channels=3,
            depth=encoder_depth,
            weights=pretrained,
        )
        self.encoder2 = get_encoder(
            encoder,
            in_channels=3,
            depth=encoder_depth,
            weights=pretrained,
        )
        # default values for unet
        decoder_use_batchnorm = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        # double decoder channels for feature fusion
        # assuming encoder1 and encoder2 have the same number of out_channels
        self.decoder = UnetDecoder(
            encoder_channels=[
                2*c*12 for c in self.encoder1.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        activation = None
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )
        self.name = "u-{}".format(encoder)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x1, x2):
        B, L, H, W, C = x1.shape
        x1 = rearrange(x1, 'b l h w c -> (b l) c h w')
        x2 = rearrange(x2, 'b l h w c -> (b l) c h w')
        f1 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L)
              for f in self.encoder1(x1)]
        f2 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L)
              for f in self.encoder2(x2)]
        f = [torch.cat([f1, f2], dim=1)
             for f1, f2 in zip(f1, f2)]
        decoder_output = self.decoder(*f)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)
