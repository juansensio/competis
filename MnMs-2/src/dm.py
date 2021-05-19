import torch
import nibabel as nib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
import numpy as np
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, data, trans=None):
        self.path = path
        self.data = data
        self.trans = trans
        self.num_classes = 4
        self.max_val = {
            'LA_ED': 4104.,
            'LA_ES': 7875.,
            'SA_ED': 11510.,
            'SA_ES': 9182.
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        patient = self.data.iloc[ix].patient
        image = self.data.iloc[ix].image
        channel = self.data.iloc[ix].channel
        img = nib.load(f'{self.path}/{patient}/{patient}_{image}.nii.gz').get_fdata()[...,channel] / self.max_val[image]
        mask = nib.load(f'{self.path}/{patient}/{patient}_{image}_gt.nii.gz').get_fdata()[...,channel].astype(np.int)
        if self.trans:
            t = self.trans(image=img, mask=mask)
            img = t['image']
            mask = t['mask'] 
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        # mask encoding
        mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), self.num_classes).permute(2,0,1).float()
        return img_t, mask_oh


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        path='data/MnM-2/training',
        file="training_data.csv", 
        val_split=(121, 160), # 120 / 40 / 40
        batch_size=32, 
        num_workers=0, 
        pin_memory=True, 
        shuffle_train=True, 
        val_with_train=False,
        train_trans=None,
        val_trans=None,
        **kwargs
    ):
        super().__init__()
        self.path = path
        self.file = file
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.val_with_train = val_with_train
        self.train_trans = train_trans
        self.val_trans = val_trans
            
    def setup(self, stage=None):
        
        # get list of patients
        data = pd.read_csv(self.file)

        # train / val splits
        train = data[(data.patient < self.val_split[0]) | (data.patient > self.val_split[1])]
        val = data[(data.patient >= self.val_split[0]) & (data.patient <= self.val_split[1])]

        print(train.patient.unique())
        print(val.patient.unique())

        train.patient = train.patient.astype(str).str.zfill(3)
        val.patient = val.patient.astype(str).str.zfill(3)

        if self.val_with_train:
            val = train

        # datasets
        self.train_ds = Dataset(
            self.path, 
            train,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else None
        )
        
        self.val_ds = Dataset(
            self.path, 
            val,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else None
        )     
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=self.shuffle_train, 
            pin_memory=self.pin_memory, 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=self.pin_memory, 
        )