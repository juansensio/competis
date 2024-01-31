import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from .ds import Dataset
import os
from sklearn.model_selection import train_test_split
from .ds import Dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path="data",
        batch_size=32,
        train_trans=None,
        val_trans=None,
        num_workers=20,
        pin_memory=True,
        val_size=0.2,
        seed=42,
        image_folder="train_satellite",
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

    def setup(self, stage=None):
        train_images = os.listdir(self.path / self.image_folder)
        image_ids = [image.split("_")[0] for image in train_images]
        train_image_ids, val_image_ids = train_test_split(
            image_ids, test_size=self.val_size, random_state=self.seed
        )
        self.train_ds = Dataset(train_image_ids, mode="train")
        self.val_ds = Dataset(val_image_ids, mode="train")

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
