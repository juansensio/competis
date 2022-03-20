import pytorch_lightning as pl
import os
from pathlib import Path
import pandas as pd
from .utils import get_patch_rgb
from .ds import RGBDataset, RGBNirDataset, NirGBDataset, RGBNirBioDataset
from torch.utils.data import DataLoader
import albumentations as A
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class RGBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans

    def read_data(self, mode="train"):
        obs_fr = pd.read_csv(self.path / 'observations' /
                             f'observations_fr_{mode}.csv', sep=';')
        obs_us = pd.read_csv(self.path / 'observations' /
                             f'observations_us_{mode}.csv', sep=';')
        return pd.concat([obs_fr, obs_us])

    def split_data(self):
        self.data_train = self.data[self.data['subset'] == 'train']
        self.data_val = self.data[self.data['subset'] == 'val']

    def generate_datasets(self):
        self.ds_train = RGBDataset(
            self.data_train.image.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBDataset(
            self.data_val.image.values, self.data_val.species_id.values)
        self.ds_test = RGBDataset(self.data_test.image.values)

    def print_dataset_info(self):
        print('train:', len(self.ds_train))
        print('val:', len(self.ds_val))
        print('test:', len(self.ds_test))

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data['image'] = self.data['observation_id'].apply(get_patch_rgb)
        self.data_test = self.read_data('test')
        self.data_test['image'] = self.data_test['observation_id'].apply(
            get_patch_rgb)
        self.split_data()
        self.generate_datasets()
        self.print_dataset_info()

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

class NirGBDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = NirGBDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = NirGBDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = NirGBDataset(self.data_test.observation_id.values)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data('test')
        self.split_data()
        self.generate_datasets()
        self.print_dataset_info()


class RGBNirDataModule(NirGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def generate_datasets(self):
        self.ds_train = RGBNirDataset(
            self.data_train.observation_id.values, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNirDataset(
            self.data_val.observation_id.values, self.data_val.species_id.values)
        self.ds_test = RGBNirDataset(self.data_test.observation_id.values)


class RGBNirBioDataModule(RGBDataModule):
    def __init__(self, batch_size=32, path='data', num_workers=0, pin_memory=False, train_trans=None):
        super().__init__(batch_size, path, num_workers, pin_memory, train_trans)

    def setup(self, stage=None):
        self.data = self.read_data()
        self.data_test = self.read_data('test')
        self.split_data()
        # read bioclimatic data
        df_env = pd.read_csv(self.path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        # get train, val, test bioclimatic data
        X_train = df_env.loc[self.data_train.observation_id.values]
        X_val = df_env.loc[self.data_val.observation_id.values]
        X_test = df_env.loc[self.data_test.observation_id.values]
        # inputer and normalizer
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])
        self.X_train = pipeline.fit_transform(X_train.values)
        self.X_val = pipeline.transform(X_val.values)
        self.X_test = pipeline.transform(X_test.values)
        self.generate_datasets()
        self.print_dataset_info()

    def generate_datasets(self):
        self.ds_train = RGBNirBioDataset(
            self.data_train.observation_id.values, self.X_train, self.data_train.species_id.values, trans=A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
                ]) 
                if self.train_trans is not None else None
            )
        self.ds_val = RGBNirBioDataset(
            self.data_val.observation_id.values, self.X_val, self.data_val.species_id.values)
        self.ds_test = RGBNirBioDataset(self.data_test.observation_id.values, self.X_test)
