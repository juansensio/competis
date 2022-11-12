import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import timm
from .attention import PerceiverEncoder
from einops import rearrange
import segmentation_models_pytorch as smp


class BaseModule(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, x):
        raise NotImplementedError

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams['optimizer_params'])
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer


class RGBTemporalModule(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.backbone = timm.create_model(
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            num_classes=0,
            in_chans=3,
        )
        self.encoder = PerceiverEncoder(
            num_latents=self.hparams.num_latents,
            latent_dim=self.hparams.latent_dim,
            input_dim=self.backbone.num_features,
            num_blocks=self.hparams.num_blocks,  # L
            n_heads=self.hparams.n_heads,
        )
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.hparams.num_months, self.backbone.num_features))
        # self.apply(self._init_weights)
        # freeze backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, xs):
        B, L, C, H, W = xs.shape
        xs = rearrange(xs, 'b l c h w -> (b l) c h w')
        # with torch.no_grad():
        features = self.backbone(xs)
        features = rearrange(features, '(b l) f -> b l f', b=B)
        features2 = features + self.pos_embed
        fused_features = self.encoder(features2)
        # return torch.sigmoid(fused_features)
        return fused_features

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module,  torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


class RGBModule(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.unet = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        # return torch.sigmoid(self.unet(x)).squeeze(1)
        return self.unet(x).squeeze(1)

  