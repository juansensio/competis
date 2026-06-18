# Adapted from TimeSformer-pytorch by Phil Wang (lucidrains)
# https://github.com/lucidrains/TimeSformer-pytorch
# Licensed under MIT License
#
# Source files:
#   - timesformer_pytorch/timesformer_pytorch.py
#   - timesformer_pytorch/rotary.py

import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from src.model.SaTformer.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding


# helpers

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    """
    Apply layer norm, then some other fn.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):

    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out  = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads

        self.scale = dim_head ** -0.5

        # [D_h * h???]
        inner_dim  = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        
        # x: (b, n, D)

        h       = self.heads
        qkv     = self.to_qkv(x)
        assert qkv.shape[-1] % 3 == 0, \
            f"Error: expected attention qkv tensor to have a dim=-1 evenly divisble by 3; got: {qkv.shape}"

        # [b, n, D]
        q, k, v = qkv.chunk(3, dim = -1)

        # [b, n, D] -> [b * num_heads, n, D_h];
        # - split tokens along patch dim to perform multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # q / sqrt(D_h); for scaled dot product attention
        q = q * self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        # - [b, h * n]
        cls_out = attn(cls_q, k, v, mask = cls_mask)

        # rearrange across time or space; [b, (f * n), d]
        # * n: space
        # * f: time
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_, mask = mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes

class SaTformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size   = 32,
        patch_size   = 4,
        channels     = 11,
        depth        = 12,
        heads        = 8,
        dim_head     = 64,
        attn_dropout = 0.1,
        ff_dropout   = 0.1,
        rotary_emb   = True,
        shift_tokens = False,
        attn="ST"
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches   = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim     = channels   * patch_size ** 2

        self.attn = attn

        self.heads              = heads
        self.patch_size         = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token          = nn.Parameter(torch.randn(1, dim))

        # NOTE: we do not support rotary embeddings
        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):

            ff           = FeedForward(dim, dropout = ff_dropout)

            if self.attn == "T->S" or self.attn == "S->T":
                time_attn    = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
                spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            elif self.attn == "ST":
                # NOTE: OURS
                full_st_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            elif self.attn =="ST^2":
                # NOTE: OURS
                full_st_attn_1 = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
                full_st_attn_2 = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            else:
                raise Exception()

            if shift_tokens:

                if self.attn == "T->S" or self.attn == "S->T":
                    time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))
                elif self.attn == "ST":
                    # NOTE: OURS
                    full_st_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (full_st_attn, ff))
                elif self.attn =="ST^2":
                    # NOTE: OURS
                    full_st_attn_1, ff = map(lambda t: PreTokenShift(num_frames, t), (full_st_attn_1, ff))
                    full_st_attn_2, ff = map(lambda t: PreTokenShift(num_frames, t), (full_st_attn_2, ff))
                else:
                    raise Exception()

            if self.attn == "T->S" or self.attn == "S->T":
                time_attn, spatial_attn, ff     = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))
            elif self.attn == "ST":
                # NOTE: OURS)
                full_st_attn, ff = map(lambda t: PreNorm(dim, t), (full_st_attn, ff))
            elif self.attn =="ST^2":
                # NOTE: OURS)
                full_st_attn_1, ff = map(lambda t: PreNorm(dim, t), (full_st_attn_1, ff))
                full_st_attn_2, ff = map(lambda t: PreNorm(dim, t), (full_st_attn_2, ff))
            else:
                raise Exception()

            if self.attn == "T->S" or self.attn == "S->T":
                self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))
            elif self.attn == "ST":
                # NOTE: OURS
                self.layers.append(nn.ModuleList([full_st_attn, ff]))
            elif self.attn =="ST^2":
                # NOTE: OURS
                self.layers.append(nn.ModuleList([full_st_attn_1, full_st_attn_2, ff]))
            else: raise Exception()

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, mask = None, return_patches = False):

        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings

        video  = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        tokens = self.to_patch_embedding(video)

        # add cls token

        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x         = torch.cat((cls_token, tokens), dim = 1)

        # positional embedding

        frame_pos_emb = None
        image_pos_emb = None

        if not self.use_rotary_emb:
            embs = torch.arange(x.shape[1]).to(device)
            x += self.pos_emb(embs)
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # calculate masking for uneven number of frames

        frame_mask    = None
        cls_attn_mask = None
        if exists(mask):
            mask_with_cls = F.pad(mask, (1, 0), value = True)
            frame_mask    = repeat(mask_with_cls, 'b f -> (b h n) () f'  , n = n, h = self.heads)
            cls_attn_mask = repeat(mask,          'b f -> (b h) () (f n)', n = n, h = self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        if self.attn == "T->S" or self.attn == "S->T":
            for (time_attn, spatial_attn, ff) in self.layers:
                if self.attn == "T->S":
                    x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
                    x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
                else:
                    x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
                    x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
                x = ff(x) + x
        elif self.attn == "ST":
            for (full_st_attention, ff) in self.layers:
                # NOTE: ours
                # simply don't rearrange dims
                x = full_st_attention(x, 'b (f n) d', 'b (f n) d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
        elif self.attn =="ST^2":
            # time and space attention
            # NOTE: ours
            for (full_st_attention_1, full_st_attention_2, ff) in self.layers:
                # simply don't rearrange dims
                x = full_st_attention_1(x, 'b (f n) d', 'b (f n) d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
                x = full_st_attention_2(x, 'b (f n) d', 'b (f n) d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
                x = ff(x) + x # skip

        # # time and space attention
        # # NOTE: ours
        # for (full_st_attention, ff) in self.layers:

        #     # x = time_attn(   x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
        #     # x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f,                    cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            
        #     # NOTE: ours
        #     # simply don't rearrange dims
        #     x = full_st_attention(x, 'b (f n) d', 'b (f n) d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
        #     x = ff(x) + x # skip

        # # time -> space attention
        # for (time_attn, spatial_attn, ff) in self.layers:
        #     x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
        #     x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
        #     x = ff(x) + x

        # # space -> time attention
        # for (time_attn, spatial_attn, ff) in self.layers:
        #     x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
        #     x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
        #     x = ff(x) + x

        if return_patches:
            return x[:, 1:]

        cls_token = x[:, 0]
        return self.to_out(cls_token)


if __name__ == "__main__":

    model = SaTformer(
        dim          = 512,
        channels     = 11,
        image_size   = 32,
        patch_size   = 4,
        num_frames   = 4,
        num_classes  = 1,
        depth        = 12,
        heads        = 8,
        dim_head     = 64,
        attn_dropout = 0.1,
        ff_dropout   = 0.1,
        rotary_emb=False,
    )

    video = torch.randn(4, 4, 11, 32, 32)   # (batch x frames x channels x height x width)
    mask  = None                            # (batch x frame) - use a mask if there are variable length videos in the same batch
    pred  = model(video, mask = mask)       # (B, N_classes)
