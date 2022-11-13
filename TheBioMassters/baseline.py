from src.dm import BaseDataModule
from src.module import UNet
import pytorch_lightning as pl
import sys
import yaml
from src.utils import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'encoder': 'resnet18',
    'pretrained': 'imagenet',
    'optimizer': 'Adam',
    'in_channels': 3,
    'optimizer_params': {
        'lr': 1e-3
    },
    'trainer': {
        'gpus': 1,
        'max_epochs': 1000,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': True,
        'precision': 16,
        'log_every_n_steps': 30
    },
    'datamodule': {
        'batch_size': 64,
        'num_workers': 10,
        'pin_memory': True,
        'bands': (2, 1, 0),
        'sensor': 'S2',
        'val_size': 0.2,
        'train_trans': {
            'HorizontalFlip': {'p': 0.5},
            'VerticalFlip': {'p': 0.5},
            'RandomRotate90': {'p': 0.5},
            'Transpose': {'p': 0.5}
        },
    },
}


def train(config, name):
    pl.seed_everything(42, workers=True)
    dm = BaseDataModule(**config['datamodule'])
    module = UNet(config)
    if config['trainer']['logger']:
        config['trainer']['logger'] = WandbLogger(
            project="TheBioMassters",
            name=name,
            config=config
        )
    config['trainer']['callbacks'] = []
    if config['trainer']['enable_checkpointing']:
        config['trainer']['callbacks'] += [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{name}-{{val_metric:.5f}}-{{epoch}}',
                monitor='val_metric',
                mode='min',
                save_top_k=1
            )
        ]
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(module, dm)


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
