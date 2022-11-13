import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import timm
from .attention import PerceiverEncoder
from einops import rearrange
import segmentation_models_pytorch as smp
from .unet_decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init

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

    def shared_step(self, batch):
        images, labels = batch
        y_hat = self(images)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))        
        # loss = F.l1_loss(y_hat, labels)
        # loss = F.mse_loss(y_hat, labels)
        metric = torch.mean(torch.sqrt(
            torch.sum((y_hat* 12905.3 - labels* 12905.3)**2, dim=(1, 2))))
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log('loss', loss)
        self.log('metric', metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric = self.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)

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

class UNet(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.unet = smp.Unet(
            encoder_name=self.hparams.encoder,
            encoder_weights=self.hparams.pretrained,
            in_channels=self.hparams.in_channels,
            classes=1,
        )

    def forward(self, x):
        return torch.sigmoid(self.unet(x)).squeeze(1)
        # return self.unet(x).squeeze(1)

class UNetDF(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        encoder_depth = 5
        self.encoder1 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s1,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        self.encoder2 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s2,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        # default values for unet
        decoder_use_batchnorm = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        # double decoder channels for feature fusion
        # assuming encoder1 and encoder2 have the same number of out_channels
        self.decoder = UnetDecoder(
            encoder_channels=[2*c for c in self.encoder1.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if self.hparams.encoder.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        activation = None
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )
        self.name = "u-{}".format(self.hparams.encoder)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x1, x2):
        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)
        features = []
        for f1, f2 in zip(features1, features2):
            features.append(torch.cat([f1, f2], dim=1))
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)

    def shared_step(self, batch):
        s1, s2, labels = batch
        y_hat = self(s1, s2)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        metric = torch.mean(torch.sqrt(
            torch.sum((y_hat* 12905.3 - labels* 12905.3)**2, dim=(1, 2))))
        return loss, metric

    def predict(self, x1, x2):
        self.eval()
        with torch.no_grad():
            return self(x1, x2)

class UnetTemporal(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        encoder_depth = 5
        self.encoder = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        decoder_use_batchnorm = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        self.decoder = UnetDecoder(
            encoder_channels=[self.hparams.seq_len*c for c in self.encoder.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if self.hparams.encoder.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        activation = None
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )
        self.name = "u-{}".format(self.hparams.encoder)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, xs):
        B, L, C, H, W = xs.shape
        xs = rearrange(xs, 'b l c h w -> (b l) c h w')
        features = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L) for f in self.encoder(xs)]
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)

class UnetTemporalDF(BaseModule):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        encoder_depth = 5
        self.encoder1 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s1,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        self.encoder2 = get_encoder(
            self.hparams.encoder,
            in_channels=self.hparams.in_channels_s2,
            depth=encoder_depth,
            weights=self.hparams.pretrained,
        )
        decoder_use_batchnorm = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        self.decoder = UnetDecoder(
            encoder_channels=[self.hparams.seq_len*c*2 for c in self.encoder1.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if self.hparams.encoder.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        activation = None
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )
        self.name = "u-{}".format(self.hparams.encoder)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, xs1, xs2):
        B, L, C, H, W = xs1.shape
        xs1 = rearrange(xs1, 'b l c h w -> (b l) c h w')
        features1 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L) for f in self.encoder1(xs1)]
        xs2 = rearrange(xs2, 'b l c h w -> (b l) c h w')
        features2 = [rearrange(f, '(b l) c h w -> b (l c) h w', b=B, l=L) for f in self.encoder2(xs2)]
        features = []
        for f1, f2 in zip(features1, features2):
            features.append(torch.cat([f1, f2], dim=1))
        decoder_output = self.decoder(*features)
        outputs = self.segmentation_head(decoder_output)
        return torch.sigmoid(outputs).squeeze(1)

    def shared_step(self, batch):
        s1, s2, labels = batch
        y_hat = self(s1, s2)
        loss = torch.mean(torch.sqrt(
            torch.sum((y_hat - labels)**2, dim=(1, 2))))
        metric = torch.mean(torch.sqrt(
            torch.sum((y_hat* 12905.3 - labels* 12905.3)**2, dim=(1, 2))))
        return loss, metric

    def predict(self, x1, x2):
        self.eval()
        with torch.no_grad():
            return self(x1, x2)

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




  