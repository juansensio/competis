import lightning as L
import torchmetrics
import torch
import segmentation_models_pytorch as smp
from einops import rearrange


class Module(L.LightningModule):
    def __init__(
        self,
        hparams={
            "encoder": "resnet18",
            "pretrained": True,
            "in_chans": 3,
            "loss": "dice",
            "optimizer": "Adam",
            "optimizer_params": {},
            "padding": 1,
        },
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = smp.Unet(
            self.hparams.encoder,
            encoder_weights="imagenet" if self.hparams.pretrained else None,
            in_channels=self.hparams.in_chans,
            classes=1,
        )
        if not "loss" in hparams or hparams["loss"] == "bce":
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif hparams["loss"] == "dice":
            self.loss = smp.losses.DiceLoss(mode="binary")
        elif hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(mode="binary")
        else:
            raise ValueError(f'Loss {hparams["loss"]} not implemented')
        self.train_metric = (
            torchmetrics.Dice()
        )  # si uso la clase tengo que separar train y val para que lo calcule bien
        self.val_metric = torchmetrics.Dice()

    def forward(self, x):
        # channels first
        x = rearrange(x, "b h w c -> b c h w")
        # paading
        x = torch.nn.functional.pad(
            x,
            (
                self.hparams.padding,
                self.hparams.padding,
                self.hparams.padding,
                self.hparams.padding,
            ),
            "constant",
            0,
        )
        # model output
        y_hat = self.model(x)
        # remove padding
        y_hat = y_hat[
            ...,
            self.hparams.padding : -self.hparams.padding,
            self.hparams.padding : -self.hparams.padding,
        ]
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1).long()
        # print(y.dtype, y_hat.dtype)
        # print(x.dtype, y.dtype)
        loss = self.loss(y_hat, y)
        self.train_metric(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        self.log(
            "metric", self.train_metric, prog_bar=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1).long()
        loss = self.loss(y_hat, y)
        self.val_metric(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_metric", self.val_metric, prog_bar=True, on_step=False, on_epoch=True
        )

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
