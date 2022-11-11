import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp


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
