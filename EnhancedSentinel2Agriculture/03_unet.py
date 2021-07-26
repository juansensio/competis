from src.unet.dm import UNetDataModule
from src.unet.model import UNet
import pytorch_lightning as pl

dm = UNetDataModule(
    batch_size=32
)

model = UNet({
    'encoder': 'resnet18',
    'lr': 0.0003
})

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    max_epochs = 10
)

trainer.fit(model, dm)