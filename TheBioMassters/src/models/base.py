import pytorch_lightning as pl
import torch


class BaseModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, x, y=None):
        raise NotImplementedError

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

    def compute_loss(self, y_hat, y):
        loss = torch.mean(torch.sqrt(torch.mean((y_hat - y)**2, dim=(1, 2))))
        return loss

    def compute_metrics(self, y_hat, y):
        metric = self.compute_loss(y_hat * 12905.3, y * 12905.3)
        return metric

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        metric = self.compute_metrics(y_hat, y)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log('loss', loss)
        self.log('metric', metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
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


class BaseModule2(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)

    def forward(self, s1s, s2s):
        raise NotImplementedError

    def predict(self, s1s, s2s):
        self.eval()
        with torch.no_grad():
            return self(s1s, s2s)

    def shared_step(self, batch):
        x1, x2, y = batch
        y_hat = self(x1, x2)
        loss = self.compute_loss(y_hat, y)
        metric = self.compute_metrics(y_hat, y)
        return loss, metric
