import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from .ds import RGBDataset
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np


class RGBTemporalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', temporal=True, num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.temporal = temporal

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv(self.path / 'train.csv')
        val = None
        test = pd.read_csv(self.path / 'test.csv')
        # keep only s2 data
        train = train[train.satellite == 'S2']
        test = test[test.satellite == 'S2']
        # groupby chip_id
        train = train.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        test = test.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        # keep unique label
        train_labels, test_labels = [], []
        for chip_id, chip in train.iterrows():
            assert len(set(chip.corresponding_agbm)) == 1
            train_labels.append(chip['corresponding_agbm'][0])
        train['corresponding_agbm'] = train_labels
        for chip_id, chip in test.iterrows():
            assert len(set(chip.corresponding_agbm)) == 1
            test_labels.append(chip['corresponding_agbm'][0])
        test['corresponding_agbm'] = test_labels
        # inpute missing months
        max_len, max_len_chip_id = 0, None
        for chip_id, group in train.iterrows():
            if len(group.month) > max_len:
                max_len = len(group.month)
                max_len_chip_id = chip_id
        months = train.month.loc[max_len_chip_id]
        train_filenames, test_filenames = [], []
        for chip_id, group in train.iterrows():
            train_filenames.append([None]*len(months))
            for i, m in enumerate(months):
                if m in group.month:
                    train_filenames[-1][i] = train.filename[train.month.index(
                        m)]
        for chip_id, group in test.iterrows():
            test_filenames.append([None]*len(months))
            for i, m in enumerate(months):
                if m in group.month:
                    test_filenames[-1][i] = test.filename[test.month.index(m)]
        train = pd.DataFrame(
            {'filename': train_filenames, 'corresponding_agbm': train_labels})
        test = pd.DataFrame({'filename': test_filenames,
                            'corresponding_agbm': test_labels})
        # generate image paths
        train['image'] = train.filename.apply(
            lambda x: [self.path / 'train_features' / _x for _x in x if _x is not None])
        train['label'] = train.corresponding_agbm.apply(
            lambda x: self.path / 'train_agbm' / x)
        test['image'] = test.filename.apply(
            lambda x: [self.path / 'test_features' / _x for _x in x if _x is not None])
        # validation split
        if self.val_size > 0:
            ixs = np.random.choice(train.chip_id.values,
                                   int(self.val_size*len(train)), replace=False)
            val = train[train.chip_id.isin(ixs)]
            train = train[~train.chip_id.isin(ixs)]
        # generate datastes
        self.ds_train = RGBDataset(
            train.image.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBDataset(
            val.image.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ])
            if self.val_trans is not None else None
        ) if val is not None else None
        self.ds_test = RGBDataset(
            test.image.values, test.chip_id.values, False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ])
            if self.test_trans is not None else None
        )
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
            pin_memory=self.pin_memory
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)


class RGBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.temporal = temporal

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv(self.path / 'train.csv')
        val = None
        test = pd.read_csv(self.path / 'test.csv')
        # keep only s2 data
        train = train[train.satellite == 'S2']
        test = test[test.satellite == 'S2']
        # keep one image per chip
        train = train.drop_duplicates(subset='chip_id')
        test = test.drop_duplicates(subset='chip_id')
        # generat image paths
        train['image'] = train.filename.apply(
            lambda x: self.path / 'train_features' / x)
        train['label'] = train.corresponding_agbm.apply(
            lambda x: self.path / 'train_agbm' / x)
        test['image'] = test.filename.apply(
            lambda x: self.path / 'test_features' / x)
        # validation split
        if self.val_size > 0:
            ixs = np.random.choice(train.chip_id.values,
                                   int(self.val_size*len(train)), replace=False)
            val = train[train.chip_id.isin(ixs)]
            train = train[~train.chip_id.isin(ixs)]
        # generate datastes
        self.ds_train = RGBDataset(
            train.image.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None
        )
        self.ds_val = RGBDataset(
            val.image.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ])
            if self.val_trans is not None else None
        ) if val is not None else None
        self.ds_test = RGBDataset(
            test.image.values, test.chip_id.values, False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ])
            if self.test_trans is not None else None
        )
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
            pin_memory=self.pin_memory
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)
