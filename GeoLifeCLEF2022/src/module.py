import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch
from .GLC.metrics import top_30_error_rate


class RGBModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=17037
        )

    def forward(self, x):
        x = x.float() / 255
        x = x.permute(0, 3, 1, 2)
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x.to(self.device))
            return torch.softmax(preds, dim=1)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        error = top_30_error_rate(
            y.cpu(), torch.softmax(y_hat, dim=1).cpu().detach())
        return loss, error

    def training_step(self, batch, batch_idx):
        loss, error = self.shared_step(batch, batch_idx)
        self.log('loss', loss)
        self.log('error', error, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, error = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_error', error, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(),
                                                                 **self.hparams['optimizer_params'])
        return optimizer


class RGBNirModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=17037,
            in_chans=4,
        )
