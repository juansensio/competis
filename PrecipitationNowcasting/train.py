from dotenv import load_dotenv

load_dotenv()

import albumentations as A
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import torch

from src.dm import DataModule
from src.module import Module
from src.util.class_weights import compute_class_weights

MAX_EPOCHS = 50
NUM_FRAMES = 3  # pretrained weights expect 4 frames (pad last obs if only 3 available)
IMAGE_SIZE = 64  # pretrained weights expect 32x32 inputs
BATCH_SIZE = 16
WEIGHTS_CACHE_DIR = 'data/cache'
LOAD_FROM_CHECKPOINT = 'checkpoints/epoch=19-val_rmse=0.7176.ckpt'
# NAME = f'satformer-da-flips'
NAME = f'satformer-scratch-{IMAGE_SIZE}-da-flips'

trans = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE)
], additional_targets={f'image{i}': 'image' for i in range(1, NUM_FRAMES)})

trans2 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
], additional_targets={f'image{i}': 'image' for i in range(1, NUM_FRAMES)}, is_check_shapes=False)

# trans = A.Compose([
#     A.Resize(41, 41),
# ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'})

# val_trans = A.Compose([
#     A.Resize(IMAGE_SIZE, IMAGE_SIZE),
# ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'})

# trans2 = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.Transpose(p=0.5),
#     A.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
# ], is_check_shapes=False, additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'})

dm = DataModule(
    batch_size=BATCH_SIZE,
    num_workers=20,
    pin_memory=True,
    train_trans=trans,
    val_trans=trans,
    train_trans2=trans2,
    min_obs=3,
    num_obs=3,
    num_frames=NUM_FRAMES,
    importance_sampling=True,
    rain_boost=5.0,
    weights_cache_dir=WEIGHTS_CACHE_DIR,
    resize_target=True
)

print('Loading or computing class weights...')
class_weights = compute_class_weights(
    'data/train_split.csv',
    data_dir='data',
    min_obs=3,
    cache_dir=WEIGHTS_CACHE_DIR,
)

model = Module(hparams={
    'lr': 5e-5,
    'encoder_lr': 5e-5,
    'decoder_lr': 5e-5,
    # 'pretrained_path': 'weights/sf-64-cls.pt',
    'class_weights': class_weights,
    'num_frames': NUM_FRAMES,
    'image_size': IMAGE_SIZE,
    'rotary_emb': False,
    'max_epochs': MAX_EPOCHS,
    'warmup_epochs': 2,
    # 'freeze_encoder_epochs': 3,
    'mse_loss_weight': 0.1,
})

if LOAD_FROM_CHECKPOINT:
    print(f'Loading pretrained weights from {LOAD_FROM_CHECKPOINT}')
    state_dict = torch.load(LOAD_FROM_CHECKPOINT)['state_dict']
    model.load_state_dict(state_dict)

torch.set_float32_matmul_precision('medium')

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='cuda',
    devices=1,
    precision='16-mixed',
    logger=MLFlowLogger(
        experiment_name='PrecipitationNowcasting',
        tracking_uri='https://mlflow.earthpulse.es',
        run_name=NAME,
    ),
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=10, leave=True),
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch}-{val_rmse:.4f}',
            monitor='val_rmse',
            mode='min',
            save_top_k=1,
        ),
    ],
    enable_checkpointing=True,
)

trainer.fit(model, dm)
