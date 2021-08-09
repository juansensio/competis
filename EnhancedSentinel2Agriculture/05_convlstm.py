from src.convlstm.dm import ConvLSTMDataModule
from src.convlstm.model import ConvLSTMModule
import pytorch_lightning as pl
import yaml
import sys 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'lr': 0.0003,
    'gpus': 1,
    'precision': 16,
    'max_epochs': 30,
    'log': False,
    'path': "data/eopatches", 
    'batch_size': 32, 
    'num_workers': 20, 
    'resume': None,
    'accelerator': None,
    'hidden_dim': [64],
    'overfit_batches': 0.0,
    'max_len': 10
}

def train(config):
    pl.seed_everything(42, workers=True)
    dm = ConvLSTMDataModule(**config)
    model = ConvLSTMModule(config)
    wandb_logger = WandbLogger(project="ES2A-CONVLSTM", config=config)
    checkpoint = ModelCheckpoint(
        dirpath='./', 
        filename=f"convlstm-{{val_mathCorrCoef:.4f}}",
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
        resume_from_checkpoint=config['resume'],
        accelerator=config['accelerator'],
        overfit_batches=config['overfit_batches']
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    config_file = sys.argv[1]
    if config_file:
        with open(config_file, 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        config.update(loaded_config)
    train(config)