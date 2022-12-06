import torch
torch.set_float32_matmul_precision('high')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
opt_model = torch.compile(model, backend="inductor").cuda()
out = model(torch.randn(1, 3, 64, 64).cuda())
print(out.shape)
