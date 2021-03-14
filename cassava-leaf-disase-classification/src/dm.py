import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision
import os
from pathlib import Path
import math
import cv2 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, imgs, labels, trans=None):
        self.path = path
        self.imgs = imgs
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        #img = torchvision.io.read_image(
        #    f'{self.path}/{self.imgs[ix]}').float() / 255.
        img = cv2.imread(f'{self.path}/{self.imgs[ix]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.trans:
            img = self.trans(image=img)['image']
        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label


class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            path='data',
            file='data_extra',
            subset=0,
            batch_size=64,
            train_trans=None,
            val_trans=None,
            num_workers=0,
            pin_memory=False,
            **kwargs):
        super().__init__()
        self.path = path
        self.file = file
        self.train_trans = train_trans
        self.subset = subset
        self.val_trans = val_trans
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # read csv files with imgs names and labels
        train = pd.read_csv(f'{self.path}/{self.file}_train.csv')
        val = pd.read_csv(f'{self.path}/{self.file}_val.csv')
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        if self.subset:
            _, train = train_test_split(
                train,
                test_size=self.subset,
                shuffle=True,
                stratify=train['label'],
                random_state=42
            )
            print("Training only on", len(train), "samples")
    
        # train dataset
        self.train_ds = Dataset(
            self.path,
            train['image_id'].values,
            train['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else A.Normalize()
        )
        # val dataset
        self.val_ds=Dataset(
            self.path,
            val['image_id'].values,
            val['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else A.Normalize()
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory)
