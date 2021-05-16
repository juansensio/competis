import torch
import nibabel as nib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, patients, trans=None, mode='train'):
        self.path = path
        self.patients = patients
        self.trans = trans
        self.mode = mode
        self.max_val = (4104., 7875.) # LA_ED, LA_ES max values

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, ix):
        patient = self.patients[ix]
        ed_img = nib.load(f'{self.path}/{patient}/{patient}_LA_ED.nii.gz').get_fdata() / self.max_val[0]
        es_img = nib.load(f'{self.path}/{patient}/{patient}_LA_ES.nii.gz').get_fdata() / self.max_val[1]
        img = np.stack([ed_img, es_img], axis=2)[...,0]
        if self.mode=='train':
            ed_mask = nib.load(f'{self.path}/{patient}/{patient}_LA_ED_gt.nii.gz').get_fdata()
            es_mask = nib.load(f'{self.path}/{patient}/{patient}_LA_ES_gt.nii.gz').get_fdata()
            mask = np.stack([ed_mask, es_mask], axis=2)[...,0].astype(np.int)
            if self.trans:
                t = self.trans(image=img, mask=mask)
                img = t['image']
                mask = t['mask'] # funciona con varios canales ?
            img_t = torch.from_numpy(img).float().permute(2,0,1)
            # masks encoding
            ed_mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask[...,0]).long())
            es_mask_oh =  torch.nn.functional.one_hot(torch.from_numpy(mask[...,1]).long())
            mask_oh = torch.cat([ed_mask_oh, es_mask_oh], axis=-1)
            return img_t, mask_oh
        if self.trans:
            img = self.trans(image=img)['image']
        return torch.from_numpy(img).float().permute(2,0,1), self.imgs[ix]


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        path='data/MnM-2/training', 
        val_split=40, # 120 / 40 / 40
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
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.val_with_train = val_with_train
        self.train_trans = train_trans
        self.val_trans = val_trans
            
    def setup(self, stage=None):
        
        # get list of patients, sort by number
        patients = os.listdir(self.path)
        patients = sorted(patients)

        # train / val splits
        train = patients[:-self.val_split]
        val = patients[-self.val_split:]

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