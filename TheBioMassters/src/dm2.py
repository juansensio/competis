import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from .dali import Dataloader


class DataModule2(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_size=0, num_workers=0, trans=False, seed=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans = trans
        self.val_size = val_size
        self.seed = seed

    def setup(self, stage=None):
        train = pd.read_csv('data/train_chip_ids.csv')
        val = None
        if self.val_size > 0:
            train, val = train_test_split(
                train, test_size=self.val_size, random_state=self.seed)
        self.dl = {
            'train': Dataloader(train.chip_id.values, self.batch_size, trans=self.trans, seed=self.seed),
            'val': Dataloader(val.chip_id.values, self.batch_size) if val is not None else None
        }
        # print('train:', len(self.dl['train']))
        # if self.dl['val'] is not None:
        #     print('val:', len(self.dl['val']))

    def train_dataloader(self):
        return self.dl['train']

    def val_dataloader(self):
        return self.dl['val']
