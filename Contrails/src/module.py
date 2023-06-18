import lightning as L
import torchmetrics 
import torch 
from .models.unet import Unet
import segmentation_models_pytorch as smp
from .utils import log_cosh_dice

class Module(L.LightningModule):
	def __init__(self, hparams={}):
		super().__init__()
		if not 'encoder' in hparams:
			hparams['encoder'] = 'resnet18'
		if not 'optimizer' in hparams:
			hparams['optimizer'] = 'Adam'
		if not 'optimizer_params' in hparams:
			hparams['optimizer_params'] = {}
		self.save_hyperparameters(hparams)
		self.model = Unet(hparams['encoder'])
		if hparams['loss'] == 'dice':
			self.loss = smp.losses.DiceLoss(mode="binary")
		elif hparams['loss'] == 'focal':
			self.loss = smp.losses.FocalLoss(mode="binary")
		elif hparams['loss'] == 'logcoshdice':
			self.loss = log_cosh_dice  
		else:
			raise ValueError(f'Loss {hparams["loss"]} not implemented')
		self.metric = torchmetrics.Dice()

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		metric = self.metric(y_hat, y)
		self.log('loss', loss, prog_bar=True)
		self.log('metric', metric, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
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