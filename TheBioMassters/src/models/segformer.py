from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from .base import BaseModule, BaseModule2
from .unet_decoder import UnetDecoder
import torch
from einops import rearrange
from .ltae import LTAE
import torch.nn.functional as F
from transformers import  SegformerForSemanticSegmentation


class SegFormer(BaseModule2):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.segformer1 = SegformerForSemanticSegmentation.from_pretrained( "nvidia/mit-b0", num_channels=2, ignore_mismatched_sizes=True, num_labels=1)
        self.segformer2 = SegformerForSemanticSegmentation.from_pretrained( "nvidia/mit-b0", num_channels=6, ignore_mismatched_sizes=True, num_labels=1)
        last_c = 256
        self.n_head = 16
        self.ltae = LTAE(
            in_channels=last_c,
            len_max_seq=24,
            return_att=True,
            n_neurons=[last_c*2, last_c],
            d_model=last_c*2,
            n_head=self.n_head,
        )

    def forward(self, x1, x2):
        B, L, _, _, _ = x1.shape
        # apply encoder to all images in time series
        x1 = rearrange(x1, 'b l h w c -> (b l) c h w')
        x2 = rearrange(x2, 'b l h w c -> (b l) c h w')
        f1s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.segformer1(x1, output_hidden_states=True).hidden_states]
        f2s = [rearrange(f, '(b l) c h w -> b l c h w', b=B, l=L)
               for f in self.segformer2(x2, output_hidden_states=True).hidden_states]
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
                f.shape[-1] // self.n_head, dim=-1)).view(self.n_head * B * H * W, L, -1)
            att = rearrange(att, 'nhb 1 t h w -> (nhb h w) 1 t')
            out = torch.matmul(att, f)
            out = out.view(self.n_head, B*H*W, 1, -1).squeeze(dim=2)
            out = out.permute(1, 0, 2).contiguous().view(
                B*H*W, -1)  # Concatenate heads
            out = rearrange(out, '(b h w) c -> b c h w', h=H, w=W)
            att = rearrange(att, '(nhb h w) 1 t -> nhb 1 t h w', h=H, w=W)
            features.append(out)
        features = features[::-1]
        # apply decoder and return output
        output = self.segformer1.decode_head(features)
        return torch.nn.functional.interpolate(torch.sigmoid(output), scale_factor=4, mode="bilinear", align_corners=False).squeeze(1)

