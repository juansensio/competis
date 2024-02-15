import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd
from .ds import Dataset
import os


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path="data",
        batch_size=8,
        train_trans={},
        val_trans={},
        test_trans={},
        num_workers=20,
        pin_memory=True,
        val_size=0.2,
        seed=42,
        metadata_file="answer.csv",
        train_folder="train",
        test_folder="evaluation",
        bands=(
            3,
            2,
            1,
        ),  # 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12' https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
    ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_size = val_size
        self.seed = seed
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.metadata_file = metadata_file
        self.bands = bands
        self.test_trans = test_trans

    def setup(self, stage=None):
        metadata = pd.read_csv(
            self.path / self.metadata_file, header=None, names=["image", "label"]
        )
        train, val = (
            train_test_split(
                metadata,
                test_size=self.val_size,
                random_state=self.seed,
                stratify=metadata.label,
            )
            if self.val_size > 0
            else (metadata, None)
        )
        self.train_ds = Dataset(
            train.image.values,
            train.label.values,
            trans=A.Compose(
                [getattr(A, t)(**params) for t, params in self.train_trans.items()]
            ),
            path=self.path / self.train_folder,
            bands=self.bands,
        )
        self.val_ds = (
            Dataset(
                val.image.values,
                val.label.values,
                trans=A.Compose(
                    [getattr(A, t)(**params) for t, params in self.val_trans.items()]
                ),
                bands=self.bands,
                path=self.path / self.train_folder,
            )
            if self.val_size > 0
            else []
        )
        test = os.listdir(self.path / self.test_folder)
        self.test_ds = Dataset(
            test,
            mode="test",
            path=self.path / self.test_folder,
            trans=A.Compose(
                [getattr(A, t)(**params) for t, params in self.test_trans.items()]
            ),
            bands=self.bands,
        )

    def train_dataloader(self, shuffle=True, batch_size=None):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, shuffle=False, batch_size=None):
        return DataLoader(
            self.val_ds,
            shuffle=shuffle,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, shuffle=False, batch_size=None):
        return DataLoader(
            self.test_ds,
            shuffle=shuffle,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
