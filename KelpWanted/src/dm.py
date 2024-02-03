import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import src
import albumentations as A


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

    def setup(self, stage=None):
        train_images = os.listdir(self.path / self.image_folder)
        image_ids = [image.split("_")[0] for image in train_images]
        train_image_ids, val_image_ids = (
            train_test_split(image_ids, test_size=self.val_size, random_state=self.seed)
            if self.val_size > 0
            else (image_ids, [])
        )
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
