import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.SaTformerPixelClassifier import SaTformerPixelClassifier, LOG_NORM_DIVISOR
from .model.SaTformer.SaTformer import SaTformer

class Module(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        num_frames = self.hparams.get('num_frames', 4)
        satformer_model = SaTformer(
            dim=self.hparams.get('dim', 512),
            num_frames=num_frames,
            num_classes=self.hparams.get('num_classes', 64),
            image_size=self.hparams.get('image_size', 32),
            patch_size=self.hparams.get('patch_size', 4),
            channels=self.hparams.get('channels', 11),
            depth=self.hparams.get('depth', 12),
            heads=self.hparams.get('heads', 8),
            dim_head=self.hparams.get('dim_head', 64),
            attn_dropout=self.hparams.get('attn_dropout', 0.1),
            ff_dropout=self.hparams.get('ff_dropout', 0.1),
            rotary_emb=self.hparams.get('rotary_emb', False),
            attn=self.hparams.get('attn', "ST^2"),
        )

        pretrained_path = self.hparams.get('pretrained_path', None)
        if pretrained_path is not None:
            print(f"Loading pretrained SaTformer weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model_dict = satformer_model.state_dict()
            compatible = {}
            skipped = []
            for key, value in state_dict.items():
                if key not in model_dict:
                    continue
                if model_dict[key].shape != value.shape:
                    skipped.append(
                        f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_dict[key].shape)}"
                    )
                    continue
                compatible[key] = value
            if skipped:
                print(f"Skipping {len(skipped)} mismatched pretrained keys:")
                for item in skipped:
                    print(f"  - {item}")
            satformer_model.load_state_dict(compatible, strict=False)

        self.model = SaTformerPixelClassifier(
            satformer_model,
            num_classes=self.hparams.get('num_classes', 64),
            decoder_dim=self.hparams.get('decoder_dim', 128),
        )

        class_weights = self.hparams.get('class_weights', None)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        if self.hparams.get('freeze_encoder_epochs', 0) > 0:
            self._set_encoder_requires_grad(False)

    def _set_encoder_requires_grad(self, requires_grad):
        for param in self.model.encoder.parameters():
            param.requires_grad = requires_grad

    def on_train_epoch_start(self):
        freeze_epochs = self.hparams.get('freeze_encoder_epochs', 0)
        frozen = self.current_epoch < freeze_epochs
        was_frozen = getattr(self, '_encoder_was_frozen', None)

        self._set_encoder_requires_grad(not frozen)

        if frozen:
            if was_frozen is not True:
                print(
                    f"Epoch {self.current_epoch}: encoder FROZEN "
                    f"(epochs 0–{freeze_epochs - 1})"
                )
            self.log('encoder_frozen', 1.0, prog_bar=False)
        else:
            if was_frozen is not False:
                print(f"Epoch {self.current_epoch}: encoder UNFROZEN")
            self.log('encoder_frozen', 0.0, prog_bar=False)

        self._encoder_was_frozen = frozen

    def forward(self, x):
        return self.model(x)

    def _bin_targets(self, y, num_classes):
        y_norm = torch.log1p(y) / LOG_NORM_DIVISOR
        y_clamped = torch.clamp(y_norm, 0.0, 1.0)
        return (y_clamped * (num_classes - 1)).round().long()

    def _predict_physical(self, logits):
        return self.model.predict_physical(logits)

    def _shared_step(self, batch):
        x = batch['inputs']
        y = batch['target']

        logits = self(x)
        y_binned = self._bin_targets(y, self.model.num_classes)

        ce_loss = F.cross_entropy(logits, y_binned, weight=self.class_weights)
        y_pred_physical = self._predict_physical(logits)
        mse_loss = F.mse_loss(y_pred_physical, y)
        mse_weight = self.hparams.get('mse_loss_weight', 0.1)
        loss = ce_loss + mse_weight * mse_loss
        rmse = torch.sqrt(torch.mean((y_pred_physical - y) ** 2))

        return loss, ce_loss, mse_loss, rmse

    def training_step(self, batch, batch_idx):
        loss, ce_loss, mse_loss, rmse = self._shared_step(batch)
        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('ce_loss', ce_loss, on_step=False, on_epoch=True)
        self.log('mse_loss', mse_loss, on_step=False, on_epoch=True)
        self.log('rmse', rmse, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ce_loss, mse_loss, rmse = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_ce_loss', ce_loss, on_epoch=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True)
        self.log('val_rmse', rmse, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        encoder_lr = self.hparams.get('encoder_lr', self.hparams.lr)
        decoder_lr = self.hparams.get('decoder_lr', self.hparams.lr)

        optimizer = torch.optim.AdamW(
            [
                {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
                {'params': self.model.decoder.parameters(), 'lr': decoder_lr},
            ],
            weight_decay=self.hparams.get('weight_decay', 0.),
        )

        warmup_epochs = self.hparams.get('warmup_epochs', 2)
        max_epochs = self.hparams.get('max_epochs', 50)

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / warmup_epochs
            progress = float(current_epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
