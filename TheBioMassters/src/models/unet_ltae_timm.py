from .base import BaseModule2
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange
from .ltae import LTAE
import torch.nn.functional as F
import timm
import torch.nn as nn

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        # elif name == "argmax":
        #     self.activation = ArgMax(**params)
        # elif name == "argmax2d":
        #     self.activation = ArgMax(dim=1, **params)
        # elif name == "clamp":
        #     self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class UNetLTAET(BaseModule2):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.encoder1 = timm.create_model(self.hparams.encoder, pretrained=self.hparams.pretrained, in_chans=self.hparams.in_channels_s1, features_only=True)
        self.encoder2 = timm.create_model(self.hparams.encoder, pretrained=self.hparams.pretrained, in_chans=self.hparams.in_channels_s2, features_only=True)
        encoder_channels=[f['num_chs'] for f in self.encoder1.feature_info]
        # default values for unet
        decoder_use_batchnorm = True
        decoder_channels = (160, 80, 40, 20, 10)
        decoder_attention_type = None
        # double decoder channels for feature fusion
        # assuming encoder1 and encoder2 have the same number of out_channels
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
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
        last_c = self.encoder1.feature_info[-1]['num_chs']
        self.ltae = LTAE(
            in_channels=last_c,
            len_max_seq=self.hparams.seq_len*2,
            return_att=True,
            n_neurons=[last_c*2, last_c],
            d_model=last_c*2,
            n_head=self.hparams.n_head,
        )

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

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
        for f1, f2 in zip(f1s[:-1][::-1], f2s[:-1][::-1]):
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