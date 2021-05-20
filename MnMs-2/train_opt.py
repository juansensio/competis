import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.dm import DataModule
from src.utils import parse_config
from src.models import SMP
import optuna
from src.cbs import PyTorchLightningPruningCallback

def objective(trial):
    
    trans = {'Resize': {'width': 224, 'height' :224}}
    dm = DataModule(
        batch_size=64,
        num_workers=24,
        pin_memory=True,
        train_trans=trans,
        val_trans=trans,
        shuffle_train=False
    )

    net = trial.suggest_categorical("model", ["Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus"])
    backbone = trial.suggest_categorical("backbone", ["resnet18", "se_resnext50_32x4d", "efficientnet-b3", "efficientnet-b5", "mobilenet_v2"])
    
    model = SMP({
        'optimizer': 'Adam',
        'lr': 0.0003,
        'loss': 'bce',
        'model': 'unet',
        'backbone': 'renset18',
        'pretrained': 'imagenet'
    })

    wandb_logger = WandbLogger(project="MnMs2-opt", name=f'{net}-{backbone}')
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        logger=wandb_logger,
        max_epochs=10,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_iou")],
        checkpoint_callback=False,
        limit_train_batches=10,
        limit_val_batches=10
    )

    trainer.fit(model, dm)

sampler = optuna.samplers.TPESampler(seed=17)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100)
print(study.best_params)
