import segmentation_models_pytorch as smp
import torch.nn.functional as F 
import torch 
import pytorch_lightning as pl
from torchmetrics import MatthewsCorrcoef

class UNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.unet = smp.Unet(
            encoder_name=self.hparams.encoder,        
            in_channels=3,                  
            classes=1,                     
        )
        self.metric = MatthewsCorrcoef(num_classes=2)

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        upsampled = F.interpolate(decoder_output, scale_factor=4, mode="nearest")
        masks = self.unet.segmentation_head(upsampled)
        return masks

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_hat = self(x)
            return torch.sigmoid(y_hat)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return y_hat, loss
        
    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat, loss = self.shared_step(batch)
        metric = self.metric(y_hat, y.long())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mathCorrCoef', metric, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)