from src.dm import DFTemporalDataModule as DataModule
from src.module import UnetTemporalDF as Module
import pytorch_lightning as pl
import sys
import yaml
from src.utils import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'encoder': 'resnet18',
    'pretrained': 'imagenet',
    'in_channels_s1': 2,
    'in_channels_s2': 3,
    'seq_len': 12,
    'optimizer': 'Adam',
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
        'batch_size': 16,
        'num_workers': 10,
        'pin_memory': True,
        'val_size': 0.2,
        's1_bands': (0, 1),
        's2_bands': (2, 1, 0),
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
    dm = DataModule(**config['datamodule'])
    module = Module(config)
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
