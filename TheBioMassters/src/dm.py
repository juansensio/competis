import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from .ds import RGBDataset, S1Dataset, DFDataset, RGBTemporalDataset, DFTemporalDataset, collate_fn
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None, sensor='S2', bands=(2, 1, 0)):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.sensor = sensor
        if self.sensor == 'S1':
            self.Dataset = S1Dataset
        elif self.sensor == 'S2':
            self.Dataset = RGBDataset
        else:
            raise ValueError('sensor must be S1 or S2')
        self.bands = bands

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv(self.path / 'train.csv')
        val = None
        test = pd.read_csv(self.path / 'test.csv')
        # keep only one sensor data
        if self.sensor is not None:
            train = train[train.satellite == self.sensor]
            test = test[test.satellite == self.sensor]
        # keep one image per chip
        train = train.drop_duplicates(subset='chip_id')
        test = test.drop_duplicates(subset='chip_id')
        # generate image paths
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
        self.ds_train = self.Dataset(
            train.image.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ])
            if self.train_trans is not None else None, bands=self.bands
        )
        self.ds_val = self.Dataset(
            val.image.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ])
            if self.val_trans is not None else None, bands=self.bands
        ) if val is not None else None
        self.ds_test = self.Dataset(
            test.image.values, test.chip_id.values, False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ])
            if self.test_trans is not None else None, bands=self.bands
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


class DFModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None, s1_bands=(0, 1), s2_bands=(2, 1, 0)):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv(self.path / 'train.csv')
        val = None
        test = pd.read_csv(self.path / 'test.csv')
        # group by chip_id
        train = train.groupby('chip_id').agg(
            list)[['filename', 'satellite', 'corresponding_agbm']]
        test = test.groupby('chip_id').agg(list)[['filename', 'satellite']]
        # generate columns
        filename, corresponding_agbm = [], []
        for chip_id, row in train.iterrows():
            ix1 = row['satellite'].index('S1')
            ix2 = row['satellite'].index('S2')
            filename.append([
                self.path / 'train_features' / row['filename'][ix1],
                self.path / 'train_features' / row['filename'][ix2]
            ])
            corresponding_agbm.append(
                self.path / 'train_agbm' / row['corresponding_agbm'][0])
        train = pd.DataFrame(
            {'image': filename, 'label': corresponding_agbm}, index=train.index)
        filename = []
        for chip_id, row in test.iterrows():
            ix1 = row['satellite'].index('S1')
            ix2 = row['satellite'].index('S2')
            filename.append([
                self.path / 'test_features' / row['filename'][ix1],
                self.path / 'test_features' / row['filename'][ix2]
            ])
        test = pd.DataFrame({'image': filename}, index=test.index)
        # validation split
        if self.val_size > 0:
            ixs = np.random.choice(train.index.values,
                                   int(self.val_size*len(train)), replace=False)
            val = train[train.index.isin(ixs)]
            train = train[~train.index.isin(ixs)]
        # generate datastes
        additional_targets = {'image2': 'image'}
        self.ds_train = DFDataset(
            train.image.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ], additional_targets=additional_targets)
            if self.train_trans is not None else None, s1_bands=self.s1_bands, s2_bands=self.s2_bands
        )
        self.ds_val = DFDataset(
            val.image.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ], additional_targets=additional_targets)
            if self.val_trans is not None else None, s1_bands=self.s1_bands, s2_bands=self.s2_bands
        ) if val is not None and val is not None else None
        self.ds_test = DFDataset(
            test.image.values, test.index.values, train=False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ], additional_targets=additional_targets)
            if self.test_trans is not None else None, s1_bands=self.s1_bands, s2_bands=self.s2_bands
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


class RGBTemporalDataModule(pl.LightningDataModule):
    def __init__(self, months=None, batch_size=32, path='data', temporal=True, num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None):
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
        self.months = months or ['September', 'October', 'November', 'December', 'January',
                                 'February', 'March', 'April', 'May', 'June', 'July', 'August']

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
        train_labels = []
        for chip_id, chip in train.iterrows():
            assert len(set(chip.corresponding_agbm)) == 1
            train_labels.append(self.path / 'train_agbm' /
                                chip['corresponding_agbm'][0])
        train['corresponding_agbm'] = train_labels
        # inpute missing months

        train_filenames, test_filenames = [], []
        for chip_id, group in train.iterrows():
            train_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    train_filenames[-1][i] = self.path / \
                        'train_features' / group.filename[group.month.index(m)]
        for chip_id, group in test.iterrows():
            test_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    test_filenames[-1][i] = self.path / 'test_features' / group.filename[group.month.index(
                        m)]
        train = pd.DataFrame(
            {'filename': train_filenames, 'corresponding_agbm': train_labels}, index=train.index)
        test = pd.DataFrame({'filename': test_filenames}, index=test.index)
        # generate image paths
        train['label'] = train.corresponding_agbm.apply(
            lambda x:  x)
        # validation split
        if self.val_size > 0:
            ixs = np.random.choice(train.index.values,
                                   int(self.val_size*len(train)), replace=False)
            val = train[train.index.isin(ixs)]
            train = train[~train.index.isin(ixs)]
        # generate datastes
        additional_targets = {}
        for i in range(len(self.months)-1):
            additional_targets[f'image{i}'] = 'image'
        self.ds_train = RGBTemporalDataset(
            train.filename.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ], additional_targets=additional_targets)
            if self.train_trans is not None else None, num_months=len(self.months)
        )
        self.ds_val = RGBTemporalDataset(
            val.filename.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ], additional_targets=additional_targets)
            if self.val_trans is not None else None, num_months=len(self.months)
        ) if val is not None else None
        self.ds_test = RGBTemporalDataset(
            test.filename.values, test.index.values, False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ], additional_targets=additional_targets)
            if self.test_trans is not None else None, num_months=len(self.months)
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


class DFTemporalDataModule(pl.LightningDataModule):
    def __init__(self, use_ndvi=False, s1_bands=(0, 1), s2_bands=(2, 1, 0), months=None, batch_size=32, path='data', temporal=True, num_workers=0, pin_memory=False, train_trans=None, val_size=0, val_trans=None, test_trans=None):
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
        self.months = months or ['September', 'October', 'November', 'December', 'January',
                                 'February', 'March', 'April', 'May', 'June', 'July', 'August']
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.use_ndvi = use_ndvi

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv(self.path / 'train.csv')
        val = None
        test = pd.read_csv(self.path / 'test.csv')
        # split satellites
        train1 = train[train.satellite == 'S1']
        test1 = test[test.satellite == 'S1']
        train2 = train[train.satellite == 'S2']
        test2 = test[test.satellite == 'S2']
        # groupby chip_id
        train1 = train1.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        test1 = test1.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        train2 = train2.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        test2 = test2.groupby('chip_id').agg(
            list)[['filename', 'month', 'corresponding_agbm']]
        # inpute missing months
        train1_filenames, test1_filenames = [], []
        for chip_id, group in train1.iterrows():
            train1_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    train1_filenames[-1][i] = self.path / \
                        'train_features' / group.filename[group.month.index(m)]
        for chip_id, group in test1.iterrows():
            test1_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    test1_filenames[-1][i] = self.path / 'test_features' / group.filename[group.month.index(
                        m)]
        train2_filenames, test2_filenames = [], []
        for chip_id, group in train2.iterrows():
            train2_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    train2_filenames[-1][i] = self.path / \
                        'train_features' / group.filename[group.month.index(m)]
        for chip_id, group in test2.iterrows():
            test2_filenames.append([None]*len(self.months))
            for i, m in enumerate(self.months):
                if m in group.month:
                    test2_filenames[-1][i] = self.path / 'test_features' / group.filename[group.month.index(
                        m)]
        train_filenames = [[f1, f2] for f1, f2 in zip(
            train1_filenames, train2_filenames)]
        test_filenames = [[f1, f2]
                          for f1, f2 in zip(test1_filenames, test2_filenames)]
        train = pd.DataFrame(
            {'image': train_filenames, 'label': train1.corresponding_agbm.apply(lambda x: self.path / 'train_agbm' / x[0])}, index=train1.index)
        test = pd.DataFrame({'image': test_filenames}, index=test1.index)
        # validation split
        if self.val_size > 0:
            ixs = np.random.choice(train.index.values,
                                   int(self.val_size*len(train)), replace=False)
            val = train[train.index.isin(ixs)]
            train = train[~train.index.isin(ixs)]
        # generate datastes
        additional_targets = {'image_s2_0': 'image'}
        for i in range(len(self.months)-1):
            additional_targets[f'image_s1_{i}'] = 'image'
            additional_targets[f'image_s2_{i+1}'] = 'image'
        self.ds_train = DFTemporalDataset(
            train.image.values, train.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ], additional_targets=additional_targets)
            if self.train_trans is not None else None, num_months=len(self.months), s1_bands=self.s1_bands, s2_bands=self.s2_bands, use_ndvi=self.use_ndvi
        )
        self.ds_val = DFTemporalDataset(
            val.image.values, val.label.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ], additional_targets=additional_targets)
            if self.val_trans is not None else None, num_months=len(self.months), s1_bands=self.s1_bands, s2_bands=self.s2_bands, use_ndvi=self.use_ndvi
        ) if val is not None else None
        self.ds_test = DFTemporalDataset(
            test.image.values, test.index.values, train=False, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.test_trans.items()
            ], additional_targets=additional_targets)
            if self.test_trans is not None else None, num_months=len(self.months), s1_bands=self.s1_bands, s2_bands=self.s2_bands, use_ndvi=self.use_ndvi
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
            pin_memory=self.pin_memory,
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)
