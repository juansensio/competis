from .module import BaseModule
import timm
import torch.nn as nn

def decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

from einops import rearrange
import torch 
from .attention import Block

class Transformer(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        # sentinel 1 encoder
        self.encoder1 = timm.create_model(
            self.hparams.encoder, 
            features_only=True, 
            in_chans=self.hparams.in_channels_s1,
            pretrained=self.hparams.pretrained
        )
        # sentinel 2 encoder
        self.encoder2 = timm.create_model(
            self.hparams.encoder, 
            features_only=True, 
            in_chans=self.hparams.in_channels_s2,
            pretrained=self.hparams.pretrained
        )
        # feature embedders
        self.patch_size = 8
        self.sizes = (128, 64, 32, 16, 8)
        self.fe = nn.ModuleList([
            PatchEmbedding(s, self.patch_size, self.encoder1.feature_info[i]['num_chs'], self.hparams.embed_dim)
            for i, s in enumerate(self.sizes)
        ])
        # output query
        self.query = nn.Parameter(torch.randn(1, 1, 256, 256))
        self.query_embedding = PatchEmbedding(256, 8, 1, self.hparams.embed_dim)
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
        decoder_channels=(1024, 512, 256, 128, 64) # depende del embedding dim ! (este sirve para 256)
        blocks = []
        for i in range(1, len(decoder_channels)):
            in_channels = decoder_channels[i - 1]
            out_channels = decoder_channels[i]
            blocks.append(decoder_block(in_channels, out_channels))
        self.decoder = nn.ModuleList(blocks) 
        self.head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1, stride=1)

        # TODO: add positional encoding

    def forward(self, xs1, xs2):
        B, L, C, H, W = xs1.shape
        xs1 = rearrange(xs1, 'b l c h w -> (b l) c h w')
        x1 = self.encoder1(xs1)
        xs2 = rearrange(xs2, 'b l c h w -> (b l) c h w')
        x2 = self.encoder2(xs2)
        e1, e2 = [], []
        for i, (f1, f2) in enumerate(zip(x1, x2)):
            f1e, f2e = self.fe[i](f1), self.fe[i](f2)
            e1.append(f1e)
            e2.append(f2e)
        e1 = torch.cat(e1, dim=1)
        e2 = torch.cat(e2, dim=1)
        e1 = rearrange(e1, '(b l) n e -> b (l n) e', b=B)
        e2 = rearrange(e2, '(b l) n e -> b (l n) e', b=B)
        fe = torch.cat([e1, e2], dim=1)
        x = self.first_attn(fe, self.query_embedding(self.query.repeat(B,1,1,1)))
        for self_attn_layer in self.self_attention_blocks:
            x = self_attn_layer(x, x)
        x = rearrange(x, 'b n (h w) -> b n h w', h=int(self.hparams.embed_dim**0.5), w=int(self.hparams.embed_dim**0.5))
        for block in self.decoder:
            x = block(x)
        return torch.sigmoid(self.head(x)).squeeze(1)

    def shared_step(self, batch):
        s1, s2, labels = batch
        y_hat = self(s1, s2)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        metric = torch.mean(torch.sqrt(
            torch.sum((y_hat* 12905.3 - labels* 12905.3)**2, dim=(1, 2))))
        return loss, metric

    def predict(self, x1, x2):
        self.eval()
        with torch.no_grad():
            return self(x1, x2)