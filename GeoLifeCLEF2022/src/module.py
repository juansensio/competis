import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch
from .GLC.metrics import top_30_error_rate
import torch.nn as nn

class RGBModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=17037
        )

    def forward(self, x):
        x = x.float() / 255
        x = x.permute(0, 3, 1, 2)
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x.to(self.device))
            return torch.softmax(preds, dim=1)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        error = top_30_error_rate(
            y.cpu(), torch.softmax(y_hat, dim=1).cpu().detach())
        return loss, error

    def training_step(self, batch, batch_idx):
        loss, error = self.shared_step(batch, batch_idx)
        self.log('loss', loss)
        self.log('error', error, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, error = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_error', error, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(),
                                                                 **self.hparams['optimizer_params'])
        return optimizer



class NirGBModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=17037,
            in_chans=3,
        )


class RGBNirModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=17037,
            in_chans=4,
        )


class RGBNirBioModule(RGBModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=4,
        )
        layer = lambda h: nn.Sequential(
            nn.ReLU(), 
            nn.Dropout(self.hparams.bio_dropout),
            nn.Linear(self.hparams.bio_layers[h], self.hparams.bio_layers[h+1])
        )
        self.bio_mlp = nn.Sequential(
            nn.Linear(self.hparams.num_bio, self.hparams.bio_layers[0]),
            *[layer(h) for h in range(len(self.hparams.bio_layers)-1)],
        )
        self.classifier = nn.Linear(self.hparams.bio_layers[-1] + self.model.feature_info[-1]['num_chs'], 17037)

    def forward(self, rgb, nir, bio):
        # rgb, nir, bio = x['rgb'], x['nir'], x['bio']
        img = torch.cat((rgb, nir.unsqueeze(-1)), dim=-1)
        img = img.float() / 255
        img = img.permute(0, 3, 1, 2)
        fi = self.model(img)
        fb = self.bio_mlp(bio)
        f = torch.cat((fi, fb), dim=-1)
        return  self.classifier(f)

    def shared_step(self, batch, batch_idx):
        rgb, nir, bio, y = batch
        # y = batch['label']
        y_hat = self(rgb, nir, bio)
        loss = F.cross_entropy(y_hat, y)
        error = top_30_error_rate(
            y.cpu(), torch.softmax(y_hat, dim=1).cpu().detach())
        return loss, error
