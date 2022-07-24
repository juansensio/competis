import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from .ds import Dataset


class BaselineDM(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        path='/fastdata/SN8/tarballs',
        train_locations=['Germany_Training_Public',
                         'Louisiana-East_Training_Public'],
        test_locations=['Louisiana-West_Test_Public'],
        train_trans={
            # 'center_crop': {'size': (1000, 1000), 'p': 1},
            # 'random_crop': {'size': (512, 512), 'p': 1.},
        },
        val_size=0,
        val_trans={}

    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.path = Path(path)
        self.train_locations = train_locations
        self.test_locations = test_locations
        self.train_trans = train_trans
        self.val_size = val_size
        self.val_trans = val_trans

    def setup(self, stage=None):
        images, locations, labels, date = [], [], [], []
        for location in self.train_locations:
            mapping = pd.read_csv(self.path / location /
                                  f'{location}_label_image_mapping.csv')
            paths = mapping['pre-event image'].apply(
                lambda x: self.path / location / 'PRE-event' / x)
            images += list(paths)
            date += ['pre']*len(mapping)
            paths = mapping['post-event image 1'].apply(
                lambda x: self.path / location / 'POST-event' / x)
            images += list(paths)
            date += ['post']*len(mapping)
            paths = mapping['label'].apply(
                lambda x: self.path / location / 'annotations' / x)
            labels += list(paths)*2
            locations += [location]*len(mapping)*2
        self.df = pd.DataFrame({
            'image': images,
            'location': locations,
            'label': labels,
            'date': date
        })
        if self.val_size:
            self.df_train, self.df_val = train_test_split(
                self.df, test_size=self.val_size, random_state=42, stratify=self.df.location
            )
            self.ds_train = Dataset(self.df_train, self.train_trans)
            self.ds_val = Dataset(self.df_train, self.val_trans)
        else:
            self.ds_train = Dataset(self.df, self.train_trans)
            self.ds_val = None

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
        return self.get_dataloader(self.ds_val, batch_size, shuffle) if self.ds_val else None
