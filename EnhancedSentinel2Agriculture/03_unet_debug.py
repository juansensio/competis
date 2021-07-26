from src.unet.dm import UNetDataModule
from src.unet.model import UNet
import pytorch_lightning as pl

dm = UNetDataModule(
    batch_size=1, 
    shuffle=False, 
    val_with_train=True
)

model = UNet({
    'encoder': 'resnet18',
    'lr': 0.001
})

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    max_epochs = 10,
    limit_train_batches = 1,
    limit_val_batches = 1
)

trainer.fit(model, dm)