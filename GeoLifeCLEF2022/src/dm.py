import pytorch_lightning as pl
import os
from pathlib import Path
import pandas as pd
from .utils import get_patch_rgb
from .ds import RGBDataset
from torch.utils.data import DataLoader


class RGBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        path = Path(self.path)
        obs_fr = pd.read_csv(path / 'observations' /
                             'observations_fr_train.csv', sep=';')
        obs_us = pd.read_csv(path / 'observations' /
                             'observations_us_train.csv', sep=';')
        data = pd.concat([obs_fr, obs_us])
        data['image'] = data['observation_id'].apply(get_patch_rgb)

        self.data_train = data[data['subset'] == 'train']
        self.data_val = data[data['subset'] == 'val']
        obs_fr_test = pd.read_csv(
            path / 'observations' / 'observations_fr_test.csv', sep=';')
        obs_us_test = pd.read_csv(
            path / 'observations' / 'observations_us_test.csv', sep=';')
        data_test = pd.concat([obs_fr_test, obs_us_test])
        data_test['image'] = data_test['observation_id'].apply(get_patch_rgb)
        self.data_test = data_test
        self.ds_train = RGBDataset(
            self.data_train.image.values, self.data_train.species_id.values)
        self.ds_val = RGBDataset(
            self.data_val.image.values, self.data_val.species_id.values)
        self.ds_test = RGBDataset(self.data_test.image.values)
        print('train:', len(self.ds_train))
        print('val:', len(self.ds_val))
        print('test:', len(self.ds_test))

    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle if shuffle is not None else True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)
