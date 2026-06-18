import albumentations as A
import lightning as L

from src.dm import DataModule
from src.module import Module
from src.util.class_weights import compute_class_weights

NUM_FRAMES = 4

trans = A.Compose([
    A.Resize(32, 32)
], additional_targets={f'image{i}': 'image' for i in range(1, NUM_FRAMES)})

dm = DataModule(
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    train_trans=trans,
    val_trans=trans,
    min_obs=3,
    num_obs=3,
    num_frames=4,
)

class_weights = compute_class_weights(
    'data/train_split.csv',
    data_dir='data',
    max_samples=500,
    cache_dir='data/cache',
)

model = Module(hparams={
    'lr': 1e-3,
    'encoder_lr': 1e-5,
    'decoder_lr': 1e-3,
    'pretrained_path': 'weights/sf-64-cls.pt',
    'class_weights': class_weights,
    'num_frames': NUM_FRAMES,
    'max_epochs': 100,
    'freeze_encoder_epochs': 0,
    'mse_loss_weight': 0.1,
})

trainer = L.Trainer(
    max_epochs=100,
    accelerator='cuda',
    devices=1,
    precision='16-mixed',
    overfit_batches=1,
    logger=None,
    enable_checkpointing=False,
)

trainer.fit(model, dm)
