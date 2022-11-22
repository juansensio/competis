from src.dm import DFTemporalDataModule as DataModule
from src.module import UnetTemporalDF as Module
import pytorch_lightning as pl
import sys
import yaml
from src.utils import deep_update
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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
        'max_epochs': 30,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': True,
        'precision': 16,
    },
    'datamodule': {
        'batch_size': 32,
        'num_workers': 10,
        'pin_memory': True,
        'val_size': 0.2,
        's1_bands': (0, 1),
        's2_bands': (2, 1, 0),
    },
}


def train(config, name):
    pl.seed_everything(42, workers=True)
    dm = DataModule(**config['datamodule'])
    module = Module(config)
    config['trainer']['callbacks'] = []
    if config['trainer']['logger']:
        config['trainer']['logger'] = WandbLogger(
            project="TheBioMassters (models)",
            name=name,
            config=config
        )
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
