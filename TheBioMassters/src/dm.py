import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .ds import Dataset, collate_fn, Dataset2
import albumentations as A


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
            train, val = train_test_split(
                train, test_size=self.val_size, random_state=self.random_state)
        # generate datastes
        additional_targets = {}
        for i in range(len(self.months)):
            additional_targets[f'image_s1_{i}'] = 'image'
            additional_targets[f'image_s2_{i}'] = 'image'
        self.ds_train = Dataset(
            train.filename.values, self.s1_bands, self.s2_bands, self.months, train.label.values,
            use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds,
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ], additional_targets=additional_targets)
            if self.train_trans is not None else None
        )
        self.ds_val = None
        if val is not None:
            self.ds_val = Dataset(val.filename.values, self.s1_bands, self.s2_bands, self.months,
                                  val.label.values, use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds,
                                  trans=A.Compose([
                                      getattr(A, trans)(**params) for trans, params in self.val_trans.items()
                                  ], additional_targets=additional_targets)
                                  if self.val_trans is not None else None)
        self.ds_test = Dataset(test.filename.values, self.s1_bands, self.s2_bands, self.months,
                               chip_ids=test.index.values, use_ndvi=self.use_ndvi, use_ndwi=self.use_ndwi, use_clouds=self.use_clouds,
                               trans=A.Compose([
                                   getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                               ], additional_targets=additional_targets)
                               if self.train_trans is not None else None
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
            collate_fn=collate_fn
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)


class DataModule2(DataModule):
    def __init__(self, batch_size=32, num_workers=0, pin_memory=False, train_trans=None, val_size=0, random_state=42, subset=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         train_trans=train_trans, val_size=val_size, random_state=random_state)
        self.subset = subset

    def setup(self, stage=None):
        # read csv files
        train = pd.read_csv('data/train_chip_ids.csv')
        val = None
        test = pd.read_csv('data/test_chip_ids.csv')
        # validation split
        if self.val_size > 0:
            train, val = train_test_split(
                train, test_size=self.val_size, random_state=self.random_state)
        # subset
        if self.subset > 0:
            train = train.sample(int(self.subset*len(train)),
                                 random_state=self.random_state)
        # generate datastes
        self.ds_train = Dataset2(
            train.chip_id.values,
            trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ], additional_targets={'image2': 'image'})
            if self.train_trans is not None else None
        )
        self.ds_val = None
        if val is not None:
            self.ds_val = Dataset2(val.chip_id.values)
        self.ds_test = Dataset2(test.chip_id.values, test=True)
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
            # persistent_workers=True,
        ) if ds is not None else None
