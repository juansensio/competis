import lightning as L
import torchmetrics 
import torch 
from .models.unet import Unet
import segmentation_models_pytorch as smp

class Module(L.LightningModule):
	def __init__(self, hparams={
		't': 1, 
		'encoder': 'resnet18', 
		'pretrained': True,
		'in_chans': 3,
		'loss': 'dice', 
		'optimizer': 'Adam', 
		'optimizer_params': {},
		'scale_factor': 2,
	}):
		super().__init__()
		self.save_hyperparameters(hparams)
		self.model = Unet(self.hparams.encoder, self.hparams.pretrained, self.hparams.in_chans, self.hparams.t, self.hparams.scale_factor)
		if hparams['loss'] == 'dice':
			self.loss = smp.losses.DiceLoss(mode="binary")
		elif hparams['loss'] == 'focal':
			self.loss = smp.losses.FocalLoss(mode="binary")
		else:
			raise ValueError(f'Loss {hparams["loss"]} not implemented')
		self.metric = torchmetrics.Dice()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y, _ = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		metric = self.metric(y_hat, y)
		self.log('loss', loss, prog_bar=True,)
		self.log('metric', metric, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y, y0 = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		metric = self.metric(y_hat, y)
		probas = torch.sigmoid(y_hat) > 0.5
		probas = torch.nn.functional.interpolate(probas.float(), size=y0.shape[-2:], mode='bilinear', align_corners=False)
		metric0 = self.metric(probas, y0)
		self.log('val_loss', loss, prog_bar=True) 
		self.log('val_metric', metric0, prog_bar=True)
		self.log('val_metric2', metric, prog_bar=True)

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