import torch
from .encoder import Encoder
from .decoder import Decoder
from einops import rearrange

class Unet(torch.nn.Module):
	def __init__(self, encoder='resnet18', t=1):
		super().__init__()
		self.encoder = Encoder(encoder)
		self.channels = [self.encoder.encoder.feature_info.channels(i) for i in range(len(self.encoder.encoder.feature_info))]
		self.decoder = Decoder(self.channels[::-1], t)

	def forward(self, x):
		features = self.encoder(x)
		return self.decoder(features)

class UnetTemp(Unet):
	def __init__(self, encoder='resnet18', t=3):
		super().__init__(encoder, t)
	
	def forward(self, x):
		B = x.size(0)
		x = rearrange(x, 'b h w t c -> (b t) c h w')
		features = self.encoder(x)
		features = [rearrange(f, '(b t) c h w -> b (t c) h w', b=B) for f in features]
		return self.decoder(features)
