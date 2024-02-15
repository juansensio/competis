import lightning as L
import torchmetrics
import torch
from einops import rearrange
import timm


class Module(L.LightningModule):
    def __init__(
        self,
        hparams={
            "model": "resnet18",
            "pretrained": True,
            "in_chans": 3,
            "optimizer": "Adam",
            "optimizer_params": {},
        },
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(
            hparams["model"],
            pretrained=hparams["pretrained"],
            in_chans=hparams["in_chans"],
            num_classes=1,
        )
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_metric = torchmetrics.F1Score(task="binary")
        self.val_metric = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        return self.model(rearrange(x, "b h w c -> b c h w")).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("loss", loss, prog_bar=True)
        self.train_metric(y_hat, y)
        self.log("f1", self.train_metric, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True)
        self.val_metric(y_hat, y)
        self.log("val_f1", self.val_metric, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams["optimizer_params"]
        )
        if "scheduler" in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer
