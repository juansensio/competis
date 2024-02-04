import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split, KFold
import src
import albumentations as A
import numpy as np
import pandas as pd


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path="data",
        batch_size=32,
        train_trans={},
        val_trans={},
        num_workers=20,
        pin_memory=True,
        val_size=0.2,
        seed=42,
        image_folder="train_satellite",
        Dataset="DatasetRGB",
        filter_train=None,
    ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_folder = image_folder
        self.val_size = val_size
        self.seed = seed
        self.Dataset = getattr(src, Dataset)
        self.filter_train = (
            pd.read_csv(self.path / filter_train).id.values.tolist()
            if filter_train
            else []
        )

    def setup(self, stage=None):
        train_images = os.listdir(self.path / self.image_folder)
        image_ids = [image.split("_")[0] for image in train_images]
        train_image_ids, val_image_ids = (
            train_test_split(image_ids, test_size=self.val_size, random_state=self.seed)
            if self.val_size > 0
            else (image_ids, [])
        )
        train_image_ids = [
            image_id
            for image_id in train_image_ids
            if image_id not in self.filter_train
        ]
        self.train_ds = self.Dataset(
            train_image_ids,
            mode="train",
            trans=A.Compose(
                [getattr(A, t)(**params) for t, params in self.train_trans.items()]
            ),
        )
        self.val_ds = (
            self.Dataset(
                val_image_ids,
                mode="train",
                trans=A.Compose(
                    [getattr(A, t)(**params) for t, params in self.val_trans.items()]
                ),
            )
            if self.val_size > 0
            else []
        )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.val_ds,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class DataModuleCV(L.LightningDataModule):
    def __init__(
        self,
        path="data",
        batch_size=32,
        train_trans={},
        val_trans={},
        num_workers=20,
        pin_memory=True,
        n_folds=5,
        seed=42,
        image_folder="train_satellite",
        Dataset="DatasetRGB",
        filter_train=None,
    ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_folder = image_folder
        self.n_folds = n_folds
        self.seed = seed
        self.Dataset = getattr(src, Dataset)
        self.filter_train = (
            pd.read_csv(self.path / filter_train).id.values.tolist()
            if filter_train
            else []
        )

    def setup(self, stage=None):
        train_images = os.listdir(self.path / self.image_folder)
        image_ids = [image.split("_")[0] for image in train_images]
        image_ids = [
            image_id for image_id in image_ids if image_id not in self.filter_train
        ]
        kf = KFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True)
        self.train_ds, self.val_ds = [], []
        for i, (train_image_ixs, val_image_ixs) in enumerate(kf.split(image_ids)):
            self.train_ds.append(
                self.Dataset(
                    np.array(image_ids)[train_image_ixs],
                    mode="train",
                    trans=A.Compose(
                        [
                            getattr(A, t)(**params)
                            for t, params in self.train_trans.items()
                        ]
                    ),
                )
            )
            self.val_ds.append(
                self.Dataset(
                    np.array(image_ids)[val_image_ixs],
                    mode="train",
                    trans=A.Compose(
                        [
                            getattr(A, t)(**params)
                            for t, params in self.val_trans.items()
                        ]
                    ),
                )
            )

    def train_dataloader(self, fold, shuffle=True):
        return DataLoader(
            self.train_ds[fold],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, fold, shuffle=False):
        return DataLoader(
            self.val_ds[fold],
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
