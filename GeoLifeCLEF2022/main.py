from src import RGBModule, RGBDataModule
import pytorch_lightning as pl
import sys
import yaml
from src import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'backbone': 'resnet18',
    'pretrained': True,
    'optimizer': 'Adam',
    'optimizer_params': {
        'lr': 1e-3
    },
    'trainer': {
        'gpus': 1,
        'max_epochs': 10,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': True
    },
    'datamodule': {
        'batch_size': 512,
        'num_workers': 20,
        'pin_memory': True
    },
}


def train(config, name):
    pl.seed_everything(42, workers=True)
    dm = RGBDataModule(**config['datamodule'])
    module = RGBModule(config)
    if config['trainer']['logger']:
        config['trainer']['logger'] = WandbLogger(
            project="GeoLifeCLEF2022",
            name=name,
            config=config
        )
    if config['trainer']['enable_checkpointing']:
        config['trainer']['callbacks'] = [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{name}-{{val_loss:.5f}}-{{epoch}}',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            )
        ]
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(module, dm)
    trainer.save_checkpoint('final.ckpt')


if __name__ == '__main__':
    name = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        name = config_file[:-4]
        if config_file:
            with open(config_file, 'r') as stream:
                loaded_config = yaml.safe_load(stream)
            deep_update(config, loaded_config)
    print(config)
    train(config, name)
