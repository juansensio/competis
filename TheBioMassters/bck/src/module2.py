import torch
from .module import BiFpn
from .module import get_feature_info
from .attention import Block
from einops import rearrange
from .module import BaseModule
import timm
import torch.nn as nn
from .attention import PatchEmbedding


def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Transformer(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        # sentinel 1 encoder
        self.encoder1 = timm.create_model(
            self.hparams.encoder,
            features_only=True,
            in_chans=self.hparams.in_channels_s1,
            pretrained=self.hparams.pretrained,
            out_indices=(0, 1, 2, 3, 4)
        )
        # sentinel 2 encoder
        self.encoder2 = timm.create_model(
            self.hparams.encoder,
            features_only=True,
            in_chans=self.hparams.in_channels_s2,
            pretrained=self.hparams.pretrained,
            out_indices=(0, 1, 2, 3, 4)
        )
        feature_info1 = get_feature_info(self.encoder1)
        feature_info2 = get_feature_info(self.encoder2)
        # bifpns
        self.bifpn1 = BiFpn(feature_info1)
        self.bifpn2 = BiFpn(feature_info2)
        # feature embedders
        self.patch_sizes = (16, 8, 4, 2, 1)
        self.sizes = (128, 64, 32, 16, 8)
        self.fe1 = nn.ModuleList([
            PatchEmbedding(s, ps, 64, self.hparams.embed_dim)
            for i, (s, ps) in enumerate(zip(self.sizes, self.patch_sizes))
        ])
        self.fe2 = nn.ModuleList([
            PatchEmbedding(s, ps, 64, self.hparams.embed_dim)
            for i, (s, ps) in enumerate(zip(self.sizes, self.patch_sizes))
        ])
        self.fpos_embed = nn.Parameter(
            torch.randn(1, 320, self.hparams.embed_dim))
        # output query
        self.query = nn.Parameter(torch.randn(1, 1, 256, 256))
        self.query_embedding = PatchEmbedding(
            256, 32, 1, self.hparams.embed_dim)  # 32 here is 1 in the lower level
        self.pos_embed = nn.Parameter(torch.randn(
            1, self.query_embedding.n_patches, self.hparams.embed_dim))
        # attention blocks
        self.first_attn = Block(
            kv_dim=self.hparams.embed_dim,
            q_dim=self.hparams.embed_dim,
            n_heads=self.hparams.n_heads,
            # attn_pdrop=attn_pdrop,
            # resid_pdrop=resid_pdrop
        )
        self.self_attention_blocks = nn.ModuleList([
            Block(
                kv_dim=self.hparams.embed_dim,
                q_dim=self.hparams.embed_dim,
                n_heads=self.hparams.n_heads,
                # attn_pdrop=attn_pdrop,
                # resid_pdrop=resid_pdrop
            ) for _ in range(self.hparams.num_blocks)
        ])
        # decoder and output head
        # depende del embedding dim ! (este sirve para 256)
        decoder_channels = (64, 32, 16, 8, 4)
        blocks = []
        for i in range(1, len(decoder_channels)):
            in_channels = decoder_channels[i - 1]
            out_channels = decoder_channels[i]
            blocks.append(decoder_block(in_channels, out_channels))
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(
            decoder_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, xs1, xs2):
        B, L, C, H, W = xs1.shape
        x1 = self.bifpn1(self.encoder1(
            rearrange(xs1, 'b l c h w -> (b l) c h w')))
        x2 = self.bifpn2(self.encoder2(
            rearrange(xs2, 'b l c h w -> (b l) c h w')))
        e1, e2 = [], []
        for i, (f1, f2) in enumerate(zip(x1, x2)):
            e1.append(self.fe1[i](f1))
            e2.append(self.fe2[i](f2))
        e1, e2 = torch.cat(e1, dim=1), torch.cat(e2, dim=1)
        e1 = rearrange(e1, '(b l) n e -> b (l n) e', b=B)
        e2 = rearrange(e2, '(b l) n e -> b (l n) e', b=B)
        # repetir embed para cada imagen en la serie temporal
        fpos_embed = self.fpos_embed.repeat(B, self.hparams.seq_len, 1)
        fe = torch.cat([e1 + fpos_embed, e2 + fpos_embed], dim=1)
        fe = torch.cat([e1, e2], dim=1)
        query = self.query_embedding(
            self.query.repeat(B, 1, 1, 1)) + self.pos_embed
        x = self.first_attn(fe, query)
        for self_attn_layer in self.self_attention_blocks:
            x = self_attn_layer(x, x)
        x = rearrange(x, 'b n (h w) -> b n h w', h=int(self.hparams.embed_dim **
                      0.5), w=int(self.hparams.embed_dim**0.5))
        for block in self.decoder:
            x = block(x)
        return torch.sigmoid(self.head(x)).squeeze(1)

    def shared_step(self, batch):
        s1, s2, labels = batch
        y_hat = self(s1, s2)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        metric = torch.mean(torch.sqrt(
            torch.sum((y_hat * 12905.3 - labels * 12905.3)**2, dim=(1, 2))))
        # y_hat = y_hat * 63.456604 + 63.32611
        # labels = labels * 63.456604 + 63.32611
        # metric = torch.mean(torch.sqrt(
        #     torch.sum((y_hat - labels)**2, dim=(1, 2))))
        return loss, metric

    def predict(self, x1, x2):
        self.eval()
        with torch.no_grad():
            return self(x1, x2)
