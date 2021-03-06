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

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.bio_layers[h],
                      self.hparams.bio_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.bio_dropout)
        )
        self.bio_mlp = nn.Sequential(
            nn.Linear(self.hparams.num_bio, self.hparams.bio_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.bio_dropout),
            *[layer(h) for h in range(len(self.hparams.bio_layers)-1)],
        )
        self.classifier = nn.Linear(
            self.hparams.bio_layers[-1] + self.model.feature_info[-1]['num_chs'], 17037)

    def forward(self, x):
        rgb, nir, bio = x['rgb'], x['nir'], x['bio']
        img = torch.cat((rgb, nir.unsqueeze(-1)), dim=-1)
        img = img.float() / 255
        img = img.permute(0, 3, 1, 2)
        fi = self.model(img)
        fb = self.bio_mlp(bio)
        f = torch.cat((fi, fb), dim=-1)
        return self.classifier(f)

    def shared_step(self, batch, batch_idx):
        y = batch['label']
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, y)
        error = top_30_error_rate(
            y.cpu(), torch.softmax(y_hat, dim=1).cpu().detach())
        return loss, error

    def predict(self, x):
        x['rgb'] = x['rgb'].to(self.device)
        x['nir'] = x['nir'].to(self.device)
        x['bio'] = x['bio'].to(self.device)
        self.eval()
        with torch.no_grad():
            preds = self(x)
            return torch.softmax(preds, dim=1)


class RGBNirBioCountryModule(RGBNirBioModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def forward(self, x):
        rgb, nir, bio, country = x['rgb'], x['nir'], x['bio'], x['country']
        img = torch.cat((rgb, nir.unsqueeze(-1)), dim=-1)
        img = img.float() / 255
        img = img.permute(0, 3, 1, 2)
        fi = self.model(img)
        fb = self.bio_mlp(
            torch.cat((bio, country.float().unsqueeze(-1)), dim=-1))
        f = torch.cat((fi, fb), dim=-1)
        return self.classifier(f)

    def predict(self, x):
        x['rgb'] = x['rgb'].to(self.device)
        x['nir'] = x['nir'].to(self.device)
        x['bio'] = x['bio'].to(self.device)
        x['country'] = x['country'].to(self.device)
        self.eval()
        with torch.no_grad():
            preds = self(x)
            return torch.softmax(preds, dim=1)


class AllModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.image_backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=4,  # rgbnir
        )
        self.alt_backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=1,
        )
        self.lc_backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=34,
        )

        def layer(h): return nn.Sequential(
            nn.Linear(self.hparams.mlp_layers[h],
                      self.hparams.mlp_layers[h+1]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout)
        )
        self.mlp = nn.Sequential(
            # bio + latlng + country
            nn.Linear(27 + 2 + 1, self.hparams.mlp_layers[0]),
            nn.ReLU(),
            nn.Dropout(self.hparams.mlp_dropout),
            *[layer(h) for h in range(len(self.hparams.mlp_layers)-1)],
        )
        # self.classifier = nn.Linear(
        #     self.hparams.mlp_layers[-1] + self.image_backbone.feature_info[-1]['num_chs'] + self.alt_backbone.feature_info[-1]['num_chs'] + self.lc_backbone.feature_info[-1]['num_chs'], 17037)
        self.classifier = nn.Linear(
            self.hparams.mlp_layers[-1] + + self.lc_backbone.feature_info[-1]['num_chs'], 17037)
        # self.classifier = nn.Linear(
        #     self.hparams.mlp_layers[-1] + 1280, 17037)

    def forward(self, x):
        rgb, nir, alt, lc, latlng, bio, country = x['rgb'], x['nir'], x[
            'alt'], x['lc'], x['latlng'], x['bio'], x['country']
        img = torch.cat((rgb, nir.unsqueeze(-1)), dim=-1)
        img = img.float() / 255
        img = img.permute(0, 3, 1, 2)
        fi = self.image_backbone(img)
        alt = (alt.float() + 85) / (4396 + 85)
        alt = alt.unsqueeze(1)
        fa = self.alt_backbone(alt)
        lc = F.one_hot(lc.long(), num_classes=34).float().permute(0, 3, 1, 2)
        fc = self.lc_backbone(lc)
        fb = self.mlp(
            torch.cat((bio, latlng, country.float().unsqueeze(-1)), dim=-1))
        # print(fi.shape, fa.shape, fc.shape, fb.shape)
        # f = torch.cat((fi, fa, fc, fb), dim=-1)
        f = torch.cat((fi+fa+fc, fb), dim=-1)
        return self.classifier(f)

    def predict(self, x):
        for k, v in x.items():
            x[k] = v.to(self.device)
        self.eval()
        with torch.no_grad():
            preds = self(x)
            return torch.softmax(preds, dim=1)

    def shared_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch['label']
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
