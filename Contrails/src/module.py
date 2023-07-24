import lightning as L
import torchmetrics 
import torch 
from .models.unet import Unet
import segmentation_models_pytorch as smp
from einops import rearrange

class Module(L.LightningModule):
	def __init__(self, hparams={
		't': 1, 
		'encoder': 'resnet18', 
		'pretrained': True,
		'in_chans': 3,
		'loss': 'dice', 
		'optimizer': 'Adam', 
		'optimizer_params': {},
		# 'scale_factor': 2,
	}):
		super().__init__()
		self.save_hyperparameters(hparams)
		self.model = Unet(self.hparams.encoder, self.hparams.pretrained, self.hparams.in_chans, self.hparams.t)
		# self.model = smp.Unet(self.hparams.encoder, encoder_weights='imagenet', in_channels=self.hparams.in_chans*self.hparams.t, classes=1)
		if not 'loss' in hparams or hparams['loss'] == 'dice':
			self.loss = smp.losses.DiceLoss(mode="binary")
		elif hparams['loss'] == 'focal':
			self.loss = smp.losses.FocalLoss(mode="binary")
		else:
			raise ValueError(f'Loss {hparams["loss"]} not implemented')
		self.metric = torchmetrics.Dice()

	def forward(self, x):
		# x = rearrange(x, 'b h w t c -> b (t c) h w')
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		y_hat = torch.nn.functional.interpolate(y_hat, size=y.shape[-2:], mode='bilinear')
		loss = self.loss(y_hat, y)
		metric = self.metric(y_hat, y)
		self.log('loss', loss, prog_bar=True,)
		self.log('metric', metric, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		y_hat = torch.nn.functional.interpolate(y_hat, size=y.shape[-2:], mode='bilinear')
		loss = self.loss(y_hat, y)
		metric = self.metric(y_hat, y)
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