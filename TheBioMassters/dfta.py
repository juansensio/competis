from src.dm import DataModule2 as DataModule
from src.models.unet_ltae import UNetLTAE as Module
import pytorch_lightning as pl
import sys
import yaml
from src.utils import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

config = {
    'encoder': 'resnet18',
    'pretrained': 'imagenet',
    'in_channels_s1': 2,
    'in_channels_s2': 6,
    'seq_len': 12,
    'optimizer': 'Adam',
    'n_head': 16,
    'optimizer_params': {
        'lr': 1e-3
    },
    'load_from_checkpoint': None,
    'trainer': {
        'gpus': 1,
        'max_epochs': 500,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': True,
        'precision': 16,
        'log_every_n_steps': 30
    },
    'datamodule': {
        'batch_size': 8,
        # 's1_bands': (0, 1, 2, 3),
        # 's2_bands': (2, 1, 0, 3, 4, 5, 6, 7, 8, 9),
        # 'use_ndvi': True,
        # 'use_ndwi': True,
        # 'use_clouds': True,
        'num_workers': 20,
        'pin_memory': True,
        'val_size': 0.2,
        'subset': 0,
        'train_trans': {
            'HorizontalFlip': {'p': 0.5},
            'VerticalFlip': {'p': 0.5},
            'RandomRotate90': {'p': 0.5},
            'Transpose': {'p': 0.5}
        }
    },
}


def train(config, name):
    pl.seed_everything(42, workers=True)
    dm = DataModule(**config['datamodule'])
    # config['in_channels_s1'] = len(config['datamodule']['s1_bands'])
    # config['in_channels_s2'] = len(config['datamodule']['s2_bands'])
    # if config['datamodule']['use_ndvi']:
    #     config['in_channels_s2'] += 1
    # if config['datamodule']['use_ndwi']:
    #     config['in_channels_s2'] += 1
    # if config['datamodule']['use_clouds']:
    #     config['in_channels_s2'] += 1
    # config['seq_len'] = len(dm.months)
    module = Module(config)
    if config['load_from_checkpoint'] is not None:
        state_dict = torch.load(config['load_from_checkpoint'])['state_dict']
        module.load_state_dict(state_dict)
    config['trainer']['callbacks'] = []
    if config['trainer']['enable_checkpointing']:
        if config['datamodule']['val_size'] > 0:
            config['trainer']['callbacks'] += [
                ModelCheckpoint(
                    dirpath='./checkpoints',
                    filename=f'{name}-{{val_metric:.5f}}-{{epoch}}',
                    monitor='val_metric',
                    mode='min',
                    save_top_k=1
                )
            ]
        config['trainer']['callbacks'] += [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{name}-{{epoch}}',
                monitor='epoch',
                mode='max',
                save_top_k=1
            )
        ]
    if config['trainer']['logger']:
        config['trainer']['logger'] = WandbLogger(
            project="TheBioMassters-yt",
            name=name,
            config=config
        )
        if 'scheduler' in config and config['scheduler']:
            config['trainer']['callbacks'] += [
                LearningRateMonitor(logging_interval='step')]
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
