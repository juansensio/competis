import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .ds import Dataset, collate_fn
# from pathlib import Path
# from .ds import RGBDataset, S1Dataset, DFDataset, RGBTemporalDataset, DFTemporalDataset, collate_fn
# import albumentations as A
# import numpy as np

class DataModule(pl.LightningDataModule):
    def __init__(self, use_ndvi=False, use_ndwi=False, use_clouds=False, s1_bands=(0, 1), s2_bands=(2, 1, 0), months=None, batch_size=32, num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.months = months or ['September', 'October', 'November', 'December', 'January',
                                 'February', 'March', 'April', 'May', 'June', 'July', 'August']
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.random_state = random_state
        self.use_ndvi = use_ndvi
        self.use_ndwi = use_ndwi
        self.use_clouds = use_clouds

    def setup(self, stage=None):
        # read json files
        train = pd.read_json('data/train.json')
        val = None
        test = pd.read_json('data/test.json')
        # validation split
        if self.val_size > 0:
            train, val = train_test_split(train, test_size=self.val_size, random_state=self.random_state)
        # generate datastes
        self.ds_train = Dataset(train.filename.values, self.s1_bands, self.s2_bands, self.months, train.label.values, use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds)
        self.ds_val = None 
        if val is not None:
            self.ds_val = Dataset(val.filename.values, self.s1_bands, self.s2_bands, self.months, val.label.values, use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds)
        self.ds_test = Dataset(test.filename.values, self.s1_bands, self.s2_bands, self.months, chip_ids=test.index.values, use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds)
        print('train:', len(self.ds_train))
        if self.ds_val is not None:
            print('val:', len(self.ds_val))
        print('test:', len(self.ds_test))

    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle if shuffle is not None else True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)
