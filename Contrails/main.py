from src.dm import DataModule
from src.module import Module
import lightning as L
import sys
import yaml
from src.utils import deep_update
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

config = {
    'encoder': 'resnet18',
    'pretrained': True,
    'optimizer': 'Adam',
    'optimizer_params': {
        'lr': 1e-3
    },
    'loss': 'dice',
    'ckpt_path': None, # resume
    'load_from_checkpoint': None, # load from checkpoint
    'trainer': {
        'accelerator': 'cuda',
        'devices': 1,
        'max_epochs': 100,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0,
        'deterministic': False, # problema con una op del modelo
        'precision': '16-mixed',
    },
    'datamodule': {
        'batch_size': 32,
        'num_workers': 10,
        'pin_memory': True,
        'bands': list(range(8,17)),
        't': tuple(range(8)),
        'norm_mode': 'mean_std',
        'false_color': False,
        'stats_path': 'stats.csv',
        'train_trans': {
            'HorizontalFlip': {'p': 0.5},
            'VerticalFlip': {'p': 0.5},
            'RandomRotate90': {'p': 0.5},
            'Transpose': {'p': 0.5}
        }
    },
}


def train(config, name):
    L.seed_everything(42, workers=True)
    dm = DataModule(**config['datamodule'])
    if config['datamodule']['false_color']: config['in_chans'] = 3
    else: config['in_chans'] = len(config['datamodule']['bands'])
    config['t'] = len(config['datamodule']['t'])
    if config['load_from_checkpoint']:
        module = Module.load_from_checkpoint(config['load_from_checkpoint'])
    else:
        module = Module(config)
    config['trainer']['callbacks'] = []
    if config['trainer']['enable_checkpointing']:
        config['trainer']['callbacks'] += [
            ModelCheckpoint(
                dirpath='./checkpoints',
                filename=f'{name}-{{val_metric:.5f}}-{{epoch}}',
                monitor='val_metric',
                mode='max',
                save_top_k=1
            ),
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
            project="Contrails",
            name=name,
            config=config
        )
        if 'scheduler' in config and config['scheduler']:
            config['trainer']['callbacks'] += [
                LearningRateMonitor(logging_interval='step')]
    trainer = L.Trainer(**config['trainer'])
    torch.set_float32_matmul_precision('medium')
    # module = torch.compile(module)
    trainer.fit(module, dm, ckpt_path=config['ckpt_path'])


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
