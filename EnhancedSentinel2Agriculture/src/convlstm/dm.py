import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from .ds import Dataset
import glob
#warnings.simplefilter("ignore")

class ConvLSTMDataModule(pl.LightningDataModule):

    def __init__(self, path = "data/eopatches", batch_size = 256, pin_memory=False, num_workers=20, max_len=10, **kwargs):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_len = max_len

    def setup(self, stage = None):
       
        # generar lista de imágenes y máscaras
        paths = glob.glob(f'{self.path}/train/*')
        train_paths, val_paths = random_split(paths, [80, 20])
        train_images = [f'{path}/data/BANDS.npy.gz' for path in train_paths]
        train_masks = [f'{path}/mask_timeless/CULTIVATED.npy.gz' for path in train_paths]
        val_images = [f'{path}/data/BANDS.npy.gz' for path in val_paths]
        val_masks = [f'{path}/mask_timeless/CULTIVATED.npy.gz' for path in val_paths]

        # generar datasets
        self.train_ds = Dataset(
            train_images,
            train_masks,
            self.max_len
        )
        self.val_ds = Dataset(
            val_images,
            val_masks,
            self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
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

