import lightning as L
import torchmetrics
import torch
import segmentation_models_pytorch as smp
from einops import rearrange
from .models.unet import Unet as MyUnet
from .loss import ModifiedLovaszLoss, LovaszHingeLoss


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
            "mask_loss": False,
            "architecture": "Unet",
            "upsample": False,
        },
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = (
            getattr(smp, self.hparams.architecture)(
                self.hparams.encoder,
                encoder_weights="imagenet" if self.hparams.pretrained else None,
                in_channels=self.hparams.in_chans,
                classes=1,
            )
            if self.hparams.architecture != "MyUnet"
            else MyUnet(
                self.hparams.encoder,
                self.hparams.pretrained,
                self.hparams.in_chans,
            )
        )

        if not "loss" in hparams or hparams["loss"] == "dice":
            self.loss = smp.losses.DiceLoss(mode="binary")
        elif hparams["loss"] == "mylovasz":
            self.loss = LovaszHingeLoss()  # va ok
        elif (
            hparams["loss"] == "bce+mylovasz"
        ):  # va bien, pero no se si la BCE puede estar jodiendo
            self.loss = None
            self.loss1 = smp.losses.SoftBCEWithLogitsLoss()
            self.loss2 = LovaszHingeLoss()
        elif hparams["loss"] == "bce":
            # self.loss = smp.losses.SoftBCEWithLogitsLoss() # no va bien
            self.loss = torch.nn.BCEWithLogitsLoss()  # no va bien
        elif hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(mode="binary")  # no va bien
        elif hparams["loss"] == "lovasz":
            self.loss = smp.losses.LovaszLoss(mode="binary")  # no va bien
        elif hparams["loss"] == "bce+lovasz":  # no va bien
            self.loss = None
            self.loss1 = smp.losses.SoftBCEWithLogitsLoss()
            self.loss2 = smp.losses.LovaszLoss(mode="binary")
        elif hparams["loss"] == "mylovasz2":
            self.loss = (
                ModifiedLovaszLoss()
            )  # va pero muy lento, probablemente la implementacion está mal
        elif hparams["loss"] == "bce+mylovasz2":
            self.loss = None
            self.loss1 = smp.losses.SoftBCEWithLogitsLoss()
            self.loss2 = ModifiedLovaszLoss()

        else:
            raise ValueError(f'Loss {hparams["loss"]} not implemented')
        self.train_metric = (
            torchmetrics.Dice()
        )  # si uso la clase tengo que separar train y val para que lo calcule bien
        self.val_metric = torchmetrics.Dice()
        # self.val_precision = torchmetrics.Precision(task="binary")
        # self.val_recall = torchmetrics.Recall(task="binary")
        # self.val_iou = torchmetrics.JaccardIndex(task="binary")

    def forward(self, x):
        # channels first
        x = rearrange(x, "b h w c -> b c h w")
        if self.hparams.upsample:
            x = torch.nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
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
        if self.hparams.upsample:
            y_hat = torch.nn.functional.interpolate(
                y_hat, scale_factor=0.5, mode="bilinear", align_corners=True
            )
        return y_hat

    def shared_step(self, batch):
        x, y, _ = batch
        y = y.unsqueeze(1).long()
        mask = torch.ones_like(y)
        if self.hparams.mask_loss:
            x = x[..., :-1]
            mask = x[..., -1]
        y_hat = self(x)
        loss = (
            self.loss(y_hat * mask, y * mask)
            if self.loss is not None
            else self.loss1(y_hat * mask, y * mask) + self.loss2(y_hat * mask, y * mask)
        )
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self.shared_step(batch)
        self.train_metric(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        self.log(
            "metric", self.train_metric, prog_bar=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self.shared_step(batch)
        self.val_metric(y_hat, y)
        # self.val_precision(y_hat, y)
        # self.val_recall(y_hat, y)
        # self.val_iou(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_metric", self.val_metric, prog_bar=True, on_step=False, on_epoch=True
        )
        # self.log(
        #     "precision",
        #     self.val_precision,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        # )
        # self.log("recall", self.val_recall, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("iou", self.val_iou, prog_bar=True, on_step=False, on_epoch=True)

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
