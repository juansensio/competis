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
            "freeze": False,
            "backbone_lr_mult": 1.0,
        },
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = timm.create_model(
            hparams["model"],
            pretrained=hparams["pretrained"],
            in_chans=hparams["in_chans"],
        )
        if self.hparams.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.backbone.feature_info[-1]["num_chs"], 1),
        )
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_metric = torchmetrics.F1Score(task="binary")
        self.val_metric = torchmetrics.F1Score(task="binary")
        self.automatic_optimization = False

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        if self.hparams.freeze:
            with torch.no_grad():
                x = self.backbone.forward_features(x)
        else:
            x = self.backbone.forward_features(x)
        return self.fc(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt = [opt] if not isinstance(opt, list) else opt
        for o in opt:
            o.zero_grad()
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.manual_backward(loss)
        for o in opt:
            o.step()
        if self.hparams.scheduler and self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch = [sch] if not isinstance(sch, list) else sch
            for s in sch:
                s.step()
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
        optimizer = [
            getattr(torch.optim, self.hparams.optimizer)(
                self.fc.parameters(), **self.hparams["optimizer_params"]
            )
        ]
        if not self.hparams.freeze and (
            "backbone_optimizer_params" in self.hparams
            or "backbone_scheduler_params" in self.hparams
        ):
            optimizer += [
                getattr(torch.optim, self.hparams.optimizer)(
                    self.backbone.parameters(),
                    **(
                        self.hparams["backbone_optimizer_params"]
                        if "backbone_optimizer_params" in self.hparams
                        else self.hparams["optimizer_params"]
                    )
                )
            ]
        if "scheduler" in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                    optimizer[0], **self.hparams["scheduler_params"]
                )
            ]
            if not self.hparams.freeze and "backbone_scheduler_params" in self.hparams:
                schedulers += [
                    getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                        optimizer[1], **self.hparams["backbone_scheduler_params"]
                    )
                ]
            return optimizer, schedulers
        return optimizer
