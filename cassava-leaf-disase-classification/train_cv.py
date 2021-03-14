import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd
import numpy as np

from src import DataModule, Model, MyEarlyStopping

size = 256
config = {
    # optimization
    'lr': 1e-5,
    'optimizer': 'Adam',
    'batch_size': 250,
    'scheduler': {
     'OneCycleLR': {
         'max_lr': 5e-3,
         'total_steps': 10,
         'pct_start': 0.2,
         'verbose': True
     }
    },
    # data
    'extra_data': 1,
    'subset': 0,
    'num_workers': 20,
    'pin_memory': True,
    # model
    'backbone': 'seresnext50_32x4d',
    'pretrained': True,
    'unfreeze': 0,
    # data augmentation
    'size': size,
    'train_trans': {
        'RandomCrop': {
            'height': size, 
            'width': size
        },
        'HorizontalFlip': {},
        'VerticalFlip': {},
        'Normalize': {}
    },
    'val_trans': {
        'CenterCrop': {
            'height': size, 
            'width': size
        },
        'Normalize': {}
    },
    # training params
    'gpus': 1,
    'precision': 16,
    'max_epochs': 10,
    'val_batches': 1.0,
    'es_start_from': 3,
    'patience': 3,
    'folds': 5
}

path = Path('./data')
file = Path('train_extra.csv' if config['extra_data'] else 'train_old.csv')
train = pd.read_csv(path/file)

skf = StratifiedKFold(n_splits=config['folds'], random_state=42, shuffle=True)
for f, (train_index, val_index) in enumerate(skf.split(X=np.zeros(len(train)), y=train['label'])):
    print("Fold: ", f+1)

    train_fold = train.iloc[train_index]
    val_fold = train.iloc[val_index]
    train_fold.to_csv(path/f'fold_{f+1}_train.csv')
    val_fold.to_csv(path/f'fold_{f+1}_val.csv')

    dm = DataModule(
        file = f'fold_{f+1}', 
        **config
    )

    model = Model(config)

    #wandb_logger = WandbLogger(project="cassava", config=config)
    #es = MyEarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
    checkpoint = ModelCheckpoint(
        dirpath='./', 
        filename=f'{config["backbone"]}-{config["size"]}-fold_{f+1}-{{val_acc:.5f}}', 
        save_top_k=1, 
        monitor='val_acc', 
        mode='max')
    #lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        #logger= wandb_logger,
        max_epochs=config['max_epochs'],
        #callbacks=[es, checkpoint, lr_monitor],
        callbacks=[checkpoint],
        limit_val_batches=config['val_batches']
    )

    trainer.fit(model, dm)
