import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src import DataModule, MyEarlyStopping, parse_config_file
from src import models
import sys

def train(config):
    dm = DataModule(file = config['data'], **config)
    model = getattr(models, config['model'])(config)
    wandb_logger = WandbLogger(project="cassava", name=config['name'], config=config)
    es = MyEarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
    checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        logger= wandb_logger,
        max_epochs=config['max_epochs'],
        callbacks=[es, checkpoint, lr_monitor],
        limit_val_batches=config['val_batches']
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = parse_config_file(config_file)
    train(config)
