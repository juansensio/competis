import torch.nn.functional as F 
import torch 
import pytorch_lightning as pl
from torchmetrics import MatthewsCorrcoef
from .ConvLSTM import ConvLSTM
import torch.nn as nn

class ConvLSTMModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.convlstm = ConvLSTM(
            input_dim=3,
            hidden_dim=self.hparams.hidden_dim,
            kernel_size=(3, 3),
            num_layers=len(self.hparams.hidden_dim),
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.segmentation_head = nn.Conv2d(self.hparams.hidden_dim[-1], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.metric = MatthewsCorrcoef(num_classes=2)

    def forward(self, x):
        outputs = self.convlstm(x)[0][0] # b, t, c, h, w
        output = outputs[:,-1,...] # b, c, h, w
        upsampled = F.interpolate(output, scale_factor=4, mode="nearest") # b, c, 4*h, 4*w
        masks = self.segmentation_head(upsampled)
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
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y = batch
        y_hat, loss = self.shared_step(batch)
        metric = self.metric(torch.sigmoid(y_hat), y.long())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mathCorrCoef', metric, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)