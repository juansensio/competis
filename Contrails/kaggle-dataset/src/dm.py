import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from .ds import Dataset
import albumentations as A
import pandas as pd


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path="/fastdata/contrails",
        stats_path=None, #"/fastdata/contrails/stats.csv",
        bands=list(range(8, 17)),
        t=tuple(range(8)),
        norm_mode="mean_std",
        false_color=True,
        batch_size=16,
        train_trans=None,
        val_trans=None,
        num_workers=20,
        pin_memory=True,
        fold=None,
        input_size=(256, 256),
    ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.stats_path = stats_path
        self.bands = bands
        self.t = t
        self.norm_mode = norm_mode
        self.false_color = false_color
        self.fold = fold
        self.input_size = input_size

    def get_dataset(self, split, trans, records=None):
        return Dataset(
            split,
            path=self.path,
            stats_path=self.stats_path,
            bands=self.bands,
            t=self.t,
            norm_mode=self.norm_mode,
            false_color=self.false_color,
            records=records,
            input_size=self.input_size,
            trans=A.Compose([getattr(A, t)(**params) for t, params in trans.items()])
            if trans is not None
            else None,
        )

    def setup(self, stage=None):
        train_records, val_records = None, None
        if self.fold is not None:
            train_records = pd.read_csv(self.path / f"fold_{self.fold}.csv")["0"].values
            val_records = pd.read_csv(self.path / f"fold_{self.fold}_val.csv")[
                "0"
            ].values
        self.train_ds = self.get_dataset("train", self.train_trans, train_records)
        self.val_ds = self.get_dataset(
            "validation" if self.fold is None else "train", self.val_trans, val_records
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
