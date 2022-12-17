from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from .base import BaseModule, BaseModule2
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange
from .ltae import LTAE
import torch.nn.functional as F


class UNetLTAE(BaseModule2):
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
            encoder_channels=self.encoder1.out_channels,
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
        last_c = self.encoder1.out_channels[-1]
        self.ltae = LTAE(
            in_channels=last_c,
            len_max_seq=self.hparams.seq_len*2,
            return_att=True,
            n_neurons=[last_c*2, last_c],
            d_model=last_c*2,
            n_head=self.hparams.n_head,
        )

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x1, x2):
        B, L, _, _, _ = x1.shape
        # apply encoder to all images
        x1 = rearrange(x1, 'b l h w c -> (b l) c h w')
        x2 = rearrange(x2, 'b l h w c -> (b l) c h w')
        f1s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.encoder1(x1)]
        f2s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.encoder2(x2)]
        # apply attn to last feature maps
        f = torch.cat([f1s[-1], f2s[-1]], dim=1)
        f = rearrange(f, 'b t c h w -> (b h w) t c')
        out, att = self.ltae(f)
        out = rearrange(out, '(b h w) c -> b c h w', h=8, w=8)
        att = rearrange(att, 'nh (b h w) t -> (nh b) 1 t h w', h=8, w=8)
        features = [out]
        # interpolate and apply attention maps to the rest of feature maps
        for f1, f2 in zip(f1s[1:-1][::-1], f2s[1:-1][::-1]):
            f = torch.cat([f1, f2], dim=1)
            B, L, C, H, W = f.shape
            att = F.interpolate(att, scale_factor=(1, 2, 2), mode="nearest")
            f = rearrange(f, 'b t c h w -> (b h w) t c')
            f = torch.stack(f.split(
                f.shape[-1] // self.hparams.n_head, dim=-1)).view(self.hparams.n_head * B * H * W, L, -1)
            att = rearrange(att, 'nhb 1 t h w -> (nhb h w) 1 t')
            out = torch.matmul(att, f)
            out = out.view(self.hparams.n_head, B*H*W, 1, -1).squeeze(dim=2)
            out = out.permute(1, 0, 2).contiguous().view(
                B*H*W, -1)  # Concatenate heads
            out = rearrange(out, '(b h w) c -> b c h w', h=H, w=W)
            att = rearrange(att, '(nhb h w) 1 t -> nhb 1 t h w', h=H, w=W)
            features.append(out)
        features.append(0)
        features = features[::-1]
        # apply decoder and return output
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)


class UNetLTAE1(BaseModule):
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
            encoder_channels=self.encoder1.out_channels,
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
        last_c = self.encoder1.out_channels[-1]
        self.ltae = LTAE(
            in_channels=last_c,
            len_max_seq=self.hparams.seq_len*2,
            return_att=True,
            n_neurons=[last_c*2, last_c],
            d_model=last_c*2,
            n_head=self.hparams.n_head,
        )

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        x1, x2 = x
        B, L, _, _, _ = x1.shape
        # apply encoder to all images
        x1 = rearrange(x1, 'b l c h w -> (b l) c h w')
        x2 = rearrange(x2, 'b l c h w -> (b l) c h w')
        f1s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.encoder1(x1)]
        f2s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.encoder2(x2)]
        # apply attn to last feature maps
        f = torch.cat([f1s[-1], f2s[-1]], dim=1)
        f = rearrange(f, 'b t c h w -> (b h w) t c')
        out, att = self.ltae(f)
        out = rearrange(out, '(b h w) c -> b c h w', h=8, w=8)
        att = rearrange(att, 'nh (b h w) t -> (nh b) 1 t h w', h=8, w=8)
        features = [out]
        # interpolate and apply attention maps to the rest of feature maps
        for f1, f2 in zip(f1s[1:-1][::-1], f2s[1:-1][::-1]):
            f = torch.cat([f1, f2], dim=1)
            B, L, C, H, W = f.shape
            att = F.interpolate(att, scale_factor=(1, 2, 2), mode="nearest")
            f = rearrange(f, 'b t c h w -> (b h w) t c')
            f = torch.stack(f.split(
                f.shape[-1] // self.hparams.n_head, dim=-1)).view(self.hparams.n_head * B * H * W, L, -1)
            att = rearrange(att, 'nhb 1 t h w -> (nhb h w) 1 t')
            out = torch.matmul(att, f)
            out = out.view(self.hparams.n_head, B*H*W, 1, -1).squeeze(dim=2)
            out = out.permute(1, 0, 2).contiguous().view(
                B*H*W, -1)  # Concatenate heads
            out = rearrange(out, '(b h w) c -> b c h w', h=H, w=W)
            att = rearrange(att, '(nhb h w) 1 t -> nhb 1 t h w', h=H, w=W)
            features.append(out)
        features.append(0)
        features = features[::-1]
        # apply decoder and return output
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)
