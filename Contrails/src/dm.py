import lightning as L
from pathlib import Path
import os 
from torch.utils.data import DataLoader
from .ds import Dataset
import albumentations as A

class DataModule(L.LightningDataModule):
    def __init__(
            self, 
            path='/fastdata/contrails', 
            image_name="all_bands_t5.npy", 
            batch_size=16, 
            train_trans=None, 
            val_trans=None,
            num_workers=20,
            pin_memory=True
        ):
        super().__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.image_name = image_name
        self.train_trans = train_trans 
        self.val_trans = val_trans
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def get_dataset(self, split, trans):
        records = os.listdir(self.path / split)
        images = [self.path / split / record / self.image_name for record in records]
        masks = [self.path / split / record / 'human_pixel_masks.npy' for record in records]
        return Dataset(images, masks, trans=A.Compose([
                getattr(A, t)(**params) for t, params in trans.items()
            ]) if trans is not None else None)

    def setup(self, stage=None):
        self.train_ds = self.get_dataset('train', self.train_trans)
        self.val_ds = self.get_dataset('validation', self.val_trans)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )