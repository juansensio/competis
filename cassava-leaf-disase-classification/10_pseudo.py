import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import DataModule, Model, MyEarlyStopping

size = 256
config = {
    # optimization
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 256,
    # 'scheduler': {
    #  'OneCycleLR': {
    #      'max_lr': 5e-3,
    #      'total_steps': 10,
    #      'pct_start': 0.2,
    #      'verbose': True
    #  }
    # },
    # data
    'data': 'data_extra_pseudo',
    'subset': 0.1,
    'num_workers': 0,
    'pin_memory': True,
    # model
    'backbone': 'resnet18',
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
    'val_batches': 5,
    'es_start_from': 0,
    'patience': 3
}

dm = DataModule(
    file = config['data'], 
    **config
)

model = Model(config)

#wandb_logger = WandbLogger(project="cassava", config=config)

es = MyEarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-{config["size"]}-pseudo-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    gpus=config['gpus'],
    precision=config['precision'],
    #logger= wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint, lr_monitor],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)
