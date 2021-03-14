import pytorch_lightning as pl
from src.dm import DataModule
from src.utils import parse_config_file
import sys
import matplotlib.pyplot as plt
from src.models import Transformer
from src.vocab import VOCAB

def find_batch_size(model, config):
    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        auto_scale_batch_size='power'
    )
    trainer.tune(model, dm)
    return model


def find_lr(model, config):
    trainer = pl.Trainer(
        gpus=config['gpus'],
        precision=config['precision'],
        auto_lr_find=True
    )
    lr_finder = trainer.tuner.lr_find(model, dm)    
    return lr_finder

if __name__ == '__main__':
    # parse config file
    config_file = sys.argv[1]
    config = parse_config_file(config_file)
    # create model and data
    dm = DataModule(**config)
    model = Transformer(config)
    # find best bs
    #find_batch_size(model, config)
    #config['batch_size'] = model.hparams.batch_size
    # find best lr
    lr_finder = find_lr(model, config)
    new_lr = lr_finder.suggestion()
    # show results
    print('bs:', model.hparams.batch_size, "lr:", new_lr)
    fig = lr_finder.plot(suggest=True)
    plt.show()