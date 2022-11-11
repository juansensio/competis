import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch
from einops import rearrange
import torch.nn as nn


class Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        max_poly_len, max_line_len, num_classes = 146, 74, 3
        self.save_hyperparameters(hparams)
        self.backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            features_only=True
        )
        self.polys_geom_head = torch.nn.Conv2d(512, max_poly_len, 1, padding=0)
        self.lines_geom_head = torch.nn.Conv2d(512, max_line_len, 1, padding=0)
        self.polys_class_head = torch.nn.Conv2d(512, num_classes, 1, padding=0)
        self.lines_class_head = torch.nn.Conv2d(512, num_classes, 1, padding=0)

    def forward(self, x):
        x = x.float() / 255
        fs = self.backbone(x)
        polys = self.polys_geom_head(fs[-1])
        lines = self.lines_geom_head(fs[-1])
        polys = rearrange(polys, 'b c h w -> b c (h w)')
        lines = rearrange(lines, 'b c h w -> b c (h w)')
        polys_class = self.polys_class_head(fs[-1])
        lines_class = self.lines_class_head(fs[-1])
        polys_class = rearrange(polys_class, 'b c h w -> b c (h w)')
        lines_class = rearrange(lines_class, 'b c h w -> b c (h w)')
        return lines, polys, lines_class, polys_class

    def predict(self, x):
        with torch.no_grad():
            lines, polys, lines_cls, polys_cls = self(x)
            lines_cls = torch.argmax(lines_cls, 1)
            polys_cls = torch.argmax(polys_cls, 1)
            return lines, polys, lines_cls, polys_cls

    def shared_step(self, batch, batch_idx):
        x, lines_y, polys_y, lines_cls_y, polys_cls_y = batch
        lines_y_hat, polys_y_hat, lines_cls_y_hat, polys_cls_y_hat = self(x)
        mask = lines_cls_y != 0
        mask = mask.unsqueeze(1).expand(
            mask.shape[0], lines_y.shape[1], mask.shape[1])
        lines_geom_loss = F.l1_loss(lines_y_hat*mask, lines_y*mask)
        lines_class_loss = F.cross_entropy(lines_cls_y_hat, lines_cls_y)
        mask = polys_cls_y != 0
        mask = mask.unsqueeze(1).expand(
            mask.shape[0], polys_y.shape[1], mask.shape[1])
        polys_geom_loss = F.l1_loss(polys_y_hat*mask, polys_y*mask)
        polys_class_loss = F.cross_entropy(polys_cls_y_hat, polys_cls_y)
        return lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss

    def training_step(self, batch, batch_idx):
        lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss = self.shared_step(
            batch, batch_idx)
        loss = lines_geom_loss + lines_class_loss + polys_geom_loss + polys_class_loss
        self.log('loss', loss)
        # self.log('lines_geom_loss', lines_geom_loss, prog_bar=True)
        # self.log('lines_class_loss', lines_class_loss, prog_bar=True)
        # self.log('polys_geom_loss', polys_geom_loss, prog_bar=True)
        # self.log('polys_class_loss', polys_class_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lines_geom_loss, lines_class_loss, polys_geom_loss, polys_class_loss = self.shared_step(
            batch, batch_idx)
        loss = lines_geom_loss + lines_class_loss + polys_geom_loss + polys_class_loss
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_lines_geom_loss', lines_geom_loss, prog_bar=True)
        # self.log('val_lines_class_loss', lines_class_loss, prog_bar=True)
        # self.log('val_polys_geom_loss', polys_geom_loss, prog_bar=True)
        # self.log('val_polys_class_loss', polys_class_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(),
                                                                 **self.hparams['optimizer_params'])
        return optimizer
