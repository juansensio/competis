from src.unet.datamodules.dm_fast import UNetDataModule
from src.unet.model import UNet
import pytorch_lightning as pl
import yaml
import sys 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'encoder': 'resnet18',
    'lr': 0.0003,
    'gpus': 1,
    'precision': 16,
    'max_epochs': 30,
    'log': False,
    'path': "data/eopatches", 
    'batch_size': 32, 
    'num_workers': 20, 
    'shuffle': True,
    'val_with_train': False,
    'train_batches': 1.,
    'val_batches': 1.
}

def train(config):
    pl.seed_everything(42, workers=True)
    dm = UNetDataModule(**config)
    model = UNet(config)
    wandb_logger = WandbLogger(project="ES2A-UNET", config=config)
    checkpoint = ModelCheckpoint(
        dirpath='./', 
        filename=f"unet-{config['encoder']}-{{val_mathCorrCoef:.4f}}",
        save_top_k=1, 
        monitor='val_mathCorrCoef', 
        mode='max'
    )
    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        max_epochs=config['max_epochs'],
        logger=wandb_logger if config['log'] else None,
        deterministic=True,
        callbacks=[checkpoint],
        limit_train_batches=config['train_batches'],
        limit_val_batches=config['val_batches'],
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    config_file = sys.argv[1]
    if config_file:
        with open(config_file, 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        config = config.update(loaded_config)
    train(config)