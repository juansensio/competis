import pytorch_lightning as pl
from src.dm import DataModule
from src.models import SMP
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.utils import parse_config
import sys

def get_cbs(config):
    cbs = []
    if config['log']:
        checkpoint = ModelCheckpoint(
            dirpath='./', 
            filename=f"{config['model']}-{config['backbone']}-{{val_iou:.4f}}",
            save_top_k=1, 
            monitor='val_iou', 
            mode='max'
        )
        #lr_monitor = LearningRateMonitor(logging_interval='step')
        cbs = [checkpoint]#, lr_monitor]
    return cbs

def train(config):
    dm = DataModule(**config)
    model = SMP(config)
    wandb_logger = WandbLogger(project="MnMs2_yt", config=config)
    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        logger=wandb_logger if config['log'] else None,
        max_epochs=config['max_epochs'],
        callbacks=get_cbs(config),
        limit_train_batches=config['train_batches'],
        limit_val_batches=config['val_batches']
    )        
    trainer.fit(model, dm)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = parse_config(config_file)
    train(config)