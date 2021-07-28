import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from src.utils import get_npy
from ..datasets.ds_slow import Dataset

#warnings.simplefilter("ignore")

class UNetDataModule(pl.LightningDataModule):

    def __init__(self, path = "data/eopatches", batch_size = 256, num_workers=20, shuffle=True, val_with_train=False, **kwargs):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_with_train = val_with_train 
        self.shuffle = shuffle

    def get_steps(self, args):
        patch, mode = args
        bands = get_npy(f'{self.path}/{mode}', patch)
        steps = bands.shape[0]        
        return (patch, steps)

    def generate_dataframes(self, patches, mode='train'):
        num_cores = multiprocessing.cpu_count()
        args = [(patch, mode) for patch in patches]
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=len(patches)) as progress:
                futures = []
                for _args in args:
                    future = pool.submit(self.get_steps, _args)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                steps = []
                for future in futures:
                    result = future.result()
                    steps.append(result)
        train_files, train_steps, train_masks = [], [], []
        for patch, _steps in steps:
            for step in range(_steps):
                train_files.append(f'{self.path}/{mode}/{patch}/data/BANDS.npy.gz')
                train_steps.append(step)
                train_masks.append(f'{self.path}/{mode}/{patch}/mask_timeless/CULTIVATED.npy.gz')
        return pd.DataFrame({
            'file': train_files,
            'step': train_steps,
            'mask': train_masks
        })

    def setup(self, stage = None):
       
        try:
            self.train_df = pd.read_csv('unet_processed_train_patches.csv')
            self.val_df = pd.read_csv('unet_processed_val_patches.csv')
            self.test_df = pd.read_csv('unet_processed_test_patches.csv')
        except:
            print("Generando lista de imágenes")
            # cargar eopatches de training
            train_patches = os.listdir(f'{self.path}/train')
            test_patches = os.listdir(f'{self.path}/test')
            # separar en train / val 
            train_patches, val_patches = random_split(train_patches, [80, 20])
            # separar series temporales
            self.train_df = self.generate_dataframes(train_patches)
            self.train_df.to_csv(f'unet_processed_train_patches.csv', index=False)
            self.val_df = self.generate_dataframes(val_patches)
            self.val_df.to_csv(f'unet_processed_val_patches.csv', index=False)
            self.test_df = self.generate_dataframes(test_patches, mode='test')
            self.test_df.to_csv(f'unet_processed_test_patches.csv', index=False)
        
        # generar datasets
        self.train_ds = Dataset(
            self.train_df
        )
        self.val_ds = Dataset(
            self.val_df
        )
        if self.val_with_train:
            self.val_ds = self.train_ds
        self.test_ds = Dataset(
            self.test_df,
            mode="test"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False 
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False 
        )