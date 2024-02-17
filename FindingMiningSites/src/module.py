import lightning as L
import torchmetrics
import torch
from einops import rearrange
import timm
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT
import numpy as np


class BaseModule(L.LightningModule):
    def __init__(
        self,
        hparams,
        backbone,
        fc,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.backbone = backbone
        self.fc = fc
        if self.hparams.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_metric = torchmetrics.F1Score(task="binary")
        self.val_metric = torchmetrics.F1Score(task="binary")
        self.automatic_optimization = False

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
        if "scheduler" in self.hparams and self.trainer.is_last_batch:
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


class Module(BaseModule):
    def __init__(
        self,
        hparams={
            "model": "resnet18",
            "pretrained": True,
            "in_chans": 3,
            "optimizer": "Adam",
            "optimizer_params": {},
            "freeze": False,
        },
    ):
        backbone = timm.create_model(
            hparams["model"],
            pretrained=hparams["pretrained"],
            in_chans=hparams["in_chans"],
        )
        fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(backbone.feature_info[-1]["num_chs"], 1),
        )
        super().__init__(hparams, backbone, fc)

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c h w")
        if self.hparams.freeze:
            self.backbone.eval()
            with torch.no_grad():
                x = self.backbone.forward_features(x)
        else:
            x = self.backbone.forward_features(x)
        return self.fc(x).squeeze(-1)


class PrivthiModule(BaseModule):
    def __init__(
        self,
        hparams={
            "weights_path": "./prithvi/Prithvi_100M.pt",
            "model_cfg_path": "./prithvi/Prithvi_100M_config.yaml",
            "optimizer": "Adam",
            "optimizer_params": {},
            "freeze": False,
        },
    ):
        with open(hparams["model_cfg_path"]) as f:
            model_config = yaml.safe_load(f)
        model_config["model_args"]["num_frames"] = 1
        hparams["model_config"] = model_config
        backbone = MaskedAutoencoderViT(**model_config["model_args"])
        checkpoint = torch.load(hparams["weights_path"], map_location="cpu")
        del checkpoint["pos_embed"]
        del checkpoint["decoder_pos_embed"]
        _ = backbone.load_state_dict(checkpoint, strict=False)
        fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(hparams["model_config"]["model_args"]["embed_dim"], 1),
        )
        super().__init__(hparams, backbone, fc)

    def forward(self, x):
        x = rearrange(x, "b h w c -> b c 1 h w")
        if self.hparams.freeze:
            self.backbone.eval()
            with torch.no_grad():
                x, _, _ = self.backbone.forward_encoder(x, mask_ratio=0)
        else:
            x, _, _ = self.backbone.forward_encoder(x, mask_ratio=0)
        reshaped_features = x[:, 1:, :]
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.view(
            -1,
            feature_img_side_length,
            feature_img_side_length,
            self.hparams["model_config"]["model_args"]["embed_dim"],
        )
        x = reshaped_features.permute(0, 3, 1, 2)
        return self.fc(x).squeeze(-1)
