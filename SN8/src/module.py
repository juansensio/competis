import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch
from einops import rearrange
import torch.nn as nn


class Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            features_only=True
        )
        self.polys_head = torch.nn.Conv2d(512, 148+1, 3, padding=1)
        self.lines_head = torch.nn.Conv2d(512, 28+1, 3, padding=1)
        self.polys_mask_head = torch.nn.Conv2d(512, 148+1, 3, padding=1)
        self.lines_mask_head = torch.nn.Conv2d(512, 28+1, 3, padding=1)

    def forward(self, x):
        x = x.float() / 255
        fs = self.backbone(x)
        polys = self.polys_head(fs[-1])
        lines = self.lines_head(fs[-1])
        polys = rearrange(polys, 'b c h w -> b c (h w)')
        lines = rearrange(lines, 'b c h w -> b c (h w)')
        polys_mask = self.polys_mask_head(fs[-1])
        lines_mask = self.lines_mask_head(fs[-1])
        polys_mask = rearrange(polys_mask, 'b c h w -> b c (h w)')
        lines_mask = rearrange(lines_mask, 'b c h w -> b c (h w)')
        return lines, polys, lines_mask, polys_mask

    def predict(self, x):
        with torch.no_grad():
            lines, polys, lines_mask, polys_mask = self(x)
            lines = torch.sigmoid(lines)
            lines[..., -1] = lines[..., -1] > 0.5
            polys = torch.sigmoid(polys)
            polys[..., -1] = polys[..., -1] > 0.5
            lines_mask = torch.sigmoid(lines_mask) > 0.5
            polys_mask = torch.sigmoid(polys_mask) > 0.5
            return lines, polys, lines_mask, polys_mask

    def shared_step(self, batch, batch_idx):
        x, y1, y2, mask1, mask2 = batch
        y1_hat, y2_hat, mask1_hat, mask2_hat = self(x)
        lines_geom_loss = F.l1_loss(torch.sigmoid(
            y1_hat[..., :-1])*mask1[..., :-1], y1[..., :-1]*mask1[..., :-1])
        lines_class_loss = F.binary_cross_entropy_with_logits(
            y1_hat[..., -1], y1[..., -1], weight=mask1[..., -1])
        polys_geom_loss = F.l1_loss(torch.sigmoid(
            y2_hat[..., :-1])*mask2[..., :-1], y2[..., :-1]*mask2[..., :-1])
        polys_class_loss = F.binary_cross_entropy_with_logits(
            y2_hat[..., -1], y2[..., -1], weight=mask2[..., -1])
        lines_mask_loss = F.binary_cross_entropy_with_logits(
            mask1_hat, mask1.float())
        polys_mask_loss = F.binary_cross_entropy_with_logits(
            mask2_hat, mask2.float())
        return lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss, lines_mask_loss, polys_mask_loss

    def training_step(self, batch, batch_idx):
        lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss, lines_mask_loss, polys_mask_loss = self.shared_step(
            batch, batch_idx)
        loss = lines_geom_loss + polys_geom_loss + lines_class_loss + \
            polys_class_loss + lines_mask_loss + polys_mask_loss
        self.log('loss', loss)
        # self.log('lines_geom_loss', lines_geom_loss, prog_bar=True)
        # self.log('lines_class_loss', lines_class_loss, prog_bar=True)
        # self.log('polys_geom_loss', polys_geom_loss, prog_bar=True)
        # self.log('polys_class_loss', polys_class_loss, prog_bar=True)
        # self.log('lines_mask_loss', lines_mask_loss, prog_bar=True)
        # self.log('polys_mask_loss', polys_mask_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss, lines_mask_loss, polys_mask_loss = self.shared_step(
            batch, batch_idx)
        loss = lines_geom_loss + polys_geom_loss + lines_class_loss + \
            polys_class_loss + lines_mask_loss + polys_mask_loss
        self.log('val_loss', loss)
        # self.log('val_lines_geom_loss', lines_geom_loss, prog_bar=True)
        # self.log('val_lines_class_loss', lines_class_loss, prog_bar=True)
        # self.log('val_polys_geom_loss', polys_geom_loss, prog_bar=True)
        # self.log('val_polys_class_loss', polys_class_loss, prog_bar=True)
        # self.log('val_lines_mask_loss', lines_mask_loss, prog_bar=True)
        # self.log('val_polys_mask_loss', polys_mask_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(),
                                                                 **self.hparams['optimizer_params'])
        return optimizer
