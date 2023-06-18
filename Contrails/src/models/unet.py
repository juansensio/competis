import torch
from .encoder import Encoder
from .decoder import Decoder

class Unet(torch.nn.Module):
	def __init__(self, encoder='resnet18'):
		super().__init__()
		self.encoder = Encoder(encoder)
		channels = [self.encoder.encoder.feature_info.channels(i) for i in range(len(self.encoder.encoder.feature_info))]
		self.decoder = Decoder(channels[::-1])

	def forward(self, x):
		features = self.encoder(x)
		return self.decoder(features)