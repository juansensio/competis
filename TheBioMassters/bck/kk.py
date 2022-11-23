from src.module import UNetA
import torch 

hparams = {
	'encoder': 'resnet18',
	'pretrained': "imagenet",
	'in_channels_s1': 2,
	'in_channels_s2': 3,
	'optimizer': 'Adam',
	'optimizer_params': {
		'lr': 1e-3
	},
}

module = UNetA(hparams)

# module
inputs = torch.randn(4, 2, 256, 256), torch.randn(4, 3, 256, 256)
out = module(*inputs)
print(out.shape)