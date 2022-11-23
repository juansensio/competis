import pytorch_lightning as pl
import torch

class BaseModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, s1s, s2s):
        raise NotImplementedError

    def predict(self, s1s, s2s):
        self.eval()
        with torch.no_grad():
            return self(s1s, s2s)

    def shared_step(self, batch):
        s1s, s2s, labels = batch
        y_hat = self(s1s, s2s)
        loss = torch.mean(torch.sqrt(
            torch.mean((y_hat - labels)**2, dim=(1, 2))))
        # loss = F.l1_loss(y_hat, labels)
        # loss = F.mse_loss(y_hat, labels)
        metric = torch.mean(torch.sqrt(
            torch.mean((y_hat * 12905.3 - labels * 12905.3)**2, dim=(1, 2))))
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