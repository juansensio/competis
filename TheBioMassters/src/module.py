import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import timm
from .attention import PerceiverEncoder


class RGBTemporalModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=3,
        )
        self.encoder = PerceiverEncoder(
            num_latents=self.hparams.num_latents,
            latent_dim=self.hparams.latent_dim,
            input_dim=self.backbone.num_features,
            num_blocks=self.hparams.num_blocks,  # L
        )
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, 12, self.backbone.num_features))

    def forward(self, xs):
        B, L, C, H, W = xs.shape
        features = self.backbone(xs.view(B*L, C, H, W)).view(B, L, -1)
        features += self.pos_embed
        fused_features = self.encoder(features)
        return fused_features

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams['optimizer_params'])
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer


class RGBModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.unet = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return torch.sigmoid(self.unet(x)).squeeze(1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams['optimizer_params'])
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer
