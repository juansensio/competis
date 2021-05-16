import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import src.losses as losses
import segmentation_models_pytorch as smp

class SMP(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.num_classes = 4
        self.loss = getattr(losses, self.hparams['loss'])
        self.model = getattr(smp, self.hparams['model'])(
            encoder_name=self.hparams['backbone'],        
            encoder_weights=self.hparams['pretrained'],    
            in_channels=1,                  
            classes=self.num_classes,                      
        )

    def forward(self, x):
        return self.model(x)

    def iou(self, pr, gt, th=0.5, eps=1e-7):
        pr = torch.sigmoid(pr) > th
        gt = gt > th
        intersection = torch.sum(gt * pr, axis=(-2,-1))
        union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
        ious = (intersection + eps) / union
        return torch.mean(ious)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        iou = self.iou(y_hat, y)
        self.log('loss', loss)
        self.log('iou', iou, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        iou = self.iou(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer