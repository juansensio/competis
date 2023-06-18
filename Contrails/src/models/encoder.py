import torch 
import timm 

class Encoder(torch.nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = timm.create_model(
			encoder, 
			pretrained=True,
			features_only=True,
			in_chans=9,
		)

	def forward(self, x):
		return self.encoder(x)