from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from .base import BaseModule
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange
from .attention import PatchEmbedding, Block

class UNetA(BaseModule):
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
        self.patch_embed1 = torch.nn.ModuleList([
            # PatchEmbedding(256, 32, 2, self.hparams.n_embed),
            PatchEmbedding(128, 16, 64, self.hparams.n_embed),
            PatchEmbedding(64, 8, 64, self.hparams.n_embed),
            PatchEmbedding(32, 4, 128, self.hparams.n_embed),
            PatchEmbedding(16, 2, 256, self.hparams.n_embed),
            PatchEmbedding(8, 1, 512, self.hparams.n_embed)
        ])
        self.patch_embed2 = torch.nn.ModuleList([
            # PatchEmbedding(256, 32, 2, self.hparams.n_embed),
            PatchEmbedding(128, 16, 64, self.hparams.n_embed),
            PatchEmbedding(64, 8, 64, self.hparams.n_embed),
            PatchEmbedding(32, 4, 128, self.hparams.n_embed),
            PatchEmbedding(16, 2, 256, self.hparams.n_embed),
            PatchEmbedding(8, 1, 512, self.hparams.n_embed)
        ])
        self.attns = torch.nn.ModuleList([
            # Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
            Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
            Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
            Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
            Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
            Block(self.hparams.n_embed, self.hparams.n_embed, self.hparams.n_heads),
        ])
        self.querys = torch.nn.ParameterList([
            # torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed)),
            torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed)),
            torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed)),
            torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed)),
            torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed)),
            torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed))
        ])
        self.conv_ts = torch.nn.ModuleList([
            # torch.nn.ConvTranspose2d(64, 2, 32, stride=32),
            torch.nn.ConvTranspose2d(self.hparams.n_embed, 64, 16, stride=16),
            torch.nn.ConvTranspose2d(self.hparams.n_embed, 64, 8, stride=8),
            torch.nn.ConvTranspose2d(self.hparams.n_embed, 128, 4, stride=4),
            torch.nn.ConvTranspose2d(self.hparams.n_embed, 256, 2, stride=2),
            torch.nn.ConvTranspose2d(self.hparams.n_embed, 512, 1, stride=1)
        ])
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 64, self.hparams.n_embed))
        self.fpos_embed = torch.nn.Parameter(torch.randn(1, 2*64*self.hparams.seq_len, self.hparams.n_embed))

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        s1s, s2s = x
        B, L, _, _, _ = s1s.shape
        s1s = rearrange(s1s, 'b l c h w -> (b l) c h w')
        s2s = rearrange(s2s, 'b l c h w -> (b l) c h w')
        f1s = self.encoder1(s1s)
        f2s = self.encoder2(s2s)
        features = [0]
        for f1, f2, pe1, pe2, attn, o, ct in zip(f1s[1:], f2s[1:], self.patch_embed1, self.patch_embed2, self.attns, self.querys, self.conv_ts):
            f1e = rearrange(pe1(f1), '(b l) n e -> b (l n) e', b=B)
            f2e = rearrange(pe2(f2), '(b l) n e -> b (l n) e', b=B)
            fe = torch.cat([f1e, f2e], dim=1) + self.fpos_embed.repeat(B, 1, 1)
            q = o.repeat(B, 1, 1) + self.pos_embed.repeat(B, 1, 1)
            fa = attn(fe, q)
            fa = rearrange(fa, 'b n e -> b e n')
            fa = fa.view(B, self.hparams.n_embed, 8, 8)
            fo = ct(fa)
            features.append(fo)
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)
