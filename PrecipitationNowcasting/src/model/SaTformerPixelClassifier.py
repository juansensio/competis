import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

LOG_NORM_DIVISOR = 4.58


class SaTformerPixelClassifier(nn.Module):
    def __init__(self, satformer_model, num_classes=64, decoder_dim=128):
        super().__init__()
        self.encoder = satformer_model
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, decoder_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.GELU(),
            nn.Upsample(size=(41, 41), mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=3, padding=1),
        )

        bin_values = torch.linspace(0.0, 1.0, num_classes)
        self.register_buffer('bin_values', bin_values)
        self.register_buffer('bin_physical', torch.expm1(bin_values * LOG_NORM_DIVISOR))

    def forward(self, x):
        b, f, _, h, w = x.shape
        p = self.encoder.patch_size
        hp, wp = h // p, w // p

        patch_tokens = self.encoder(x, return_patches=True)
        grid_features = rearrange(patch_tokens, 'b (f h w) d -> b f d h w', f=f, h=hp, w=wp)

        # Use the most recent frame rather than mean-pooling all frames
        spatial_features = grid_features[:, -1]

        return self.decoder(spatial_features)

    def predict_physical(self, logits):
        """Expected rain rate in mm/h from per-pixel logits."""
        probs = F.softmax(logits, dim=1)
        bin_physical = self.bin_physical.view(1, self.num_classes, 1, 1)
        return torch.sum(probs * bin_physical, dim=1)

    def predict_regression(self, x):
        """Returns continuous precipitation in mm/h."""
        logits = self.forward(x)
        return self.predict_physical(logits)
