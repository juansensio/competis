import lightning as L
from .model.src.model import ClayMAEModule
import torch
import torchmetrics


class Module(L.LightningModule):
    def __init__(
        self,
        hparams={"freeze": True, "optimizer": "Adam", "optimizer_params": {"lr": 3e-4}},
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        ckpt = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"
        self.model = ClayMAEModule.load_from_checkpoint(
            ckpt,
            metadata_path="src/model/configs/metadata.yaml",
            shuffle=False,
            mask_ratio=0,
            map_location="cpu",
        )
        # self.model.cuda()
        self.fc = torch.nn.Linear(768, 1)
        if self.hparams.freeze:
            for param in self.model.model.encoder.parameters():
                param.requires_grad = False
            self.model.eval()
        self.loss = torch.nn.MSELoss()
        self.train_metric = torchmetrics.PearsonCorrCoef()
        self.val_metric = torchmetrics.PearsonCorrCoef()

    def forward(self, x):
        if self.hparams.freeze:
            self.model.eval()
            with torch.no_grad():
                unmsk_patch, _, _, _ = self.model.model.encoder(x)
        else:
            unmsk_patch, _, _, _ = self.model.model.encoder(x)
        embeddings = unmsk_patch[:, 0, :]
        return self.fc(embeddings).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("loss", loss, prog_bar=True, batch_size=y.shape[0])
        self.train_metric(y_hat, y)
        self.log(
            "metric",
            self.train_metric,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=y.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=y.shape[0])
        self.val_metric(y_hat, y)
        self.log(
            "val_metric",
            self.val_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=y.shape[0],
        )

    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.hparams.optimizer)(
                self.fc.parameters(), **self.hparams["optimizer_params"]
            )
        ]
        if "scheduler" in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                    optimizer[0], **self.hparams["scheduler_params"]
                )
            ]
            return optimizer, schedulers
        return optimizer

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Custom batch transfer to GPU
        dict_of_tensors, tensor = batch
        # Move each tensor in the dictionary to the device
        dict_of_tensors = {k: v.to(device) for k, v in dict_of_tensors.items()}
        # Move the tensor to the device
        tensor = tensor.to(device)
        return dict_of_tensors, tensor
