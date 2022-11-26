import pytorch_lightning as pl
import torch
from ..transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomTranspose, RandomRotate90

class BaseModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.transforms = torch.nn.ModuleList([
            RandomHorizontalFlip(self.hparams.p),
            RandomVerticalFlip(self.hparams.p),
            RandomTranspose(self.hparams.p),
            RandomRotate90(self.hparams.p)
        ])

    def forward(self, x, y=None):
        raise NotImplementedError

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[0]

    def compute_loss_metrics(self, y_hat, y):
        loss = torch.mean(torch.sqrt(
            torch.mean((y_hat - y)**2, dim=(1, 2))))
        metric = torch.mean(torch.sqrt(
            torch.mean((y_hat * 12905.3 - y * 12905.3)**2, dim=(1, 2))))
        return loss, metric

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, y = self(x, y)
        loss, metric = self.compute_loss_metrics(y_hat, y)
        self.log('loss', loss)
        self.log('metric', metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss, metric = self.compute_loss_metrics(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)

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