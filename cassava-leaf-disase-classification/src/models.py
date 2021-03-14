import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy
import torch
from torchvision import transforms
import timm

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = accuracy(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer
        

class ViT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = timm.create_model(self.hparams.backbone,pretrained=self.hparams.pretrained, num_classes=5)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss


class Model(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.backbone = timm.create_model(
            self.hparams.backbone, 
            pretrained=self.hparams.pretrained, 
            features_only=True
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.backbone.feature_info.channels(-1), 5)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features[-1])

    def extract_features(self, x):
        if self.trainer.current_epoch < self.hparams.unfreeze:
            with torch.no_grad():
                features = self.backbone(x)
        else: 
            features = self.backbone(x)
        return features

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.extract_features(x)
        y_hat = self.head(features[-1])
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss
