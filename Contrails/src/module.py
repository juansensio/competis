import lightning as L
import torchmetrics
import torch
from .models.unet import Unet
import segmentation_models_pytorch as smp
from einops import rearrange
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2


def norm_rho(rho, shape):
    diag = np.sqrt(shape[0] ** 2 + shape[1] ** 2)
    max, min = -diag, diag
    return (rho - min) / (max - min)


def norm_theta(theta):
    max, min = np.pi, 0.0
    return (theta - min) / (max - min)


class Module(L.LightningModule):
    def __init__(
        self,
        hparams={
            "t": 1,
            "encoder": "resnet18",
            "pretrained": True,
            "in_chans": 3,
            "loss": "dice",
            "optimizer": "Adam",
            "optimizer_params": {},
            "freeze": False,
            # 'scale_factor': 2,
        },
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = Unet(
            self.hparams.encoder,
            self.hparams.pretrained,
            self.hparams.in_chans,
            self.hparams.t,
            self.hparams.freeze,
        )
        # self.model = smp.Unet(self.hparams.encoder, encoder_weights='imagenet', in_channels=self.hparams.in_chans*self.hparams.t, classes=1)
        if not "loss" in hparams or hparams["loss"] == "dice":
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
        # x = rearrange(x, 'b h w t c -> b (t c) h w')
        return self.model(x)

    def chamfer_distance(
        self, masks, pred_lines, rho_res=1, theta_res=np.pi / 180, threshold=40
    ):
        # print(masks.shape, pred_lines.shape)
        pred_lines = torch.sigmoid(pred_lines)  # B 2 L
        pred_lines = rearrange(pred_lines, "b n l -> b l n")
        loss = 0.0
        for ix, mask in enumerate(masks):
            _lines = cv2.HoughLines(
                mask.cpu().numpy().astype(np.uint8), rho_res, theta_res, threshold
            )
            if _lines is None:
                continue
            _lines = _lines.reshape(-1, 2)
            _lines = [
                [norm_rho(rho, masks.shape[-2:]), norm_theta(theta)]
                for rho, theta in _lines
            ]
            lines = torch.tensor(_lines, device=masks.device)
            # print(pred_lines[ix].shape, lines.shape)
            # assert lines.max() <= 1 and lines.min() >= 0, "Lines not normalized"
            diff = pred_lines[ix][:, None, :] - lines[None, :, :]
            dists = torch.sqrt(torch.sum(diff**2, dim=-1))  # shape: (n, m)
            min_dists_1, _ = dists.min(dim=-1)
            min_dists_2, _ = dists.min(dim=-2)
            chamfer_dist = min_dists_1.mean() + min_dists_2.mean()
            loss += chamfer_dist
        return loss / len(masks)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y_hat, y_hat2 = self(x)
        y_hat = self(x)
        y_hat = torch.nn.functional.interpolate(
            y_hat, size=y.shape[-2:], mode="bilinear"
        )
        self.train_metric(y_hat, y)
        loss = self.loss(y_hat, y)
        # dice = self.loss(y_hat, y)
        # chamfer = self.chamfer_distance(y, y_hat2)
        # loss = 0.5 * dice + 0.5 * chamfer
        # loss = dice + 10.0 * chamfer
        # loss = dice + chamfer
        # loss = dice
        # self.log("dice", dice, prog_bar=True)
        # self.log("chamfer", chamfer, prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        self.log(
            "metric", self.train_metric, prog_bar=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y_hat, y_hat2 = self(x)
        y_hat = self(x)
        y_hat = torch.nn.functional.interpolate(
            y_hat, size=y.shape[-2:], mode="bilinear"
        )
        self.val_metric(y_hat, y)
        loss = self.loss(y_hat, y)
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
