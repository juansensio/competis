import torch
from skimage import io
from .vocab import VOCAB
import pytorch_lightning as pl
from pathlib import Path
from .utils import get_image_path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A 
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, inchis=None, max_len=512, trans=None, train=True, tokens=(0, 1, 2)):
        self.images = images
        self.inchis = inchis
        self.trans = trans
        self.train = train
        self.max_len = max_len
        self.PAD, self.SOS, self.EOS = tokens

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image = io.imread(self.images[ix]) 
        if self.trans:
            image = self.trans(image=image)['image']
        image = torch.tensor(image / 255., dtype=torch.float).unsqueeze(0)
        if self.train:
            inchi = torch.tensor([self.SOS] + self.inchis[ix] + [self.EOS], dtype=torch.long)
            #inchi = torch.nn.functional.pad(inchi, (0, self.max_len - len(inchi)), 'constant', self.PAD)
            return image, inchi
        return image

    def collate(self, batch):
        if self.train:
            # calcular longitud máxima en el batch 
            lens = [len(inchi) for _, inchi in batch]
            max_len = max(lens)    
            # añadimos padding a los inchis cortos para que todos tengan la misma longitud
            images, inchis = [], []
            for image, inchi in batch:
                images.append(image)
                inchis.append(torch.nn.functional.pad(inchi, (0, max_len - len(inchi)), 'constant', self.PAD))
            # opcionalmente, podríamos re-ordenar las frases en el batch (algunos modelos lo requieren)
            ixs = torch.argsort(torch.tensor(lens), descending=True)
            return torch.stack(images)[ixs], torch.stack(inchis)[ixs]
            #return torch.stack(images)[ixs], torch.stack(inchis)[ixs]
        return torch.stack([img for img in batch])

class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_file = 'train_labels_tokenized.csv', 
        path=Path('data'), 
        text_column="InChI_text",
        test_size=0.1, 
        random_state=42, 
        batch_size=64, 
        num_workers=0, 
        pin_memory=True, 
        shuffle_train=True, 
        val_with_train=False,
        train_trans=None,
        val_trans=None,
        subset=None,
        max_len=512,
        **kwargs
    ):
        super().__init__()
        self.data_file = data_file
        self.path = path
        self.test_size=test_size
        self.random_state=random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.val_with_train = val_with_train
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.subset = subset
        self.max_len = max_len
        self.text_column = text_column
        self.stoi = {}
        self.itos = {}

    def encode(self, InChI):
        return [self.stoi[token] for token in InChI]

    def decode(self, ixs):
        skip = [self.stoi['PAD'], self.stoi['SOS'], self.stoi['EOS']]
        return ('').join([self.itos[ix.item()] for ix in ixs if ix.item() not in skip])

    def setup(self, stage=None):
        # build indices
        for i, s in enumerate(VOCAB):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        # read csv file with data
        df = pd.read_csv(self.path / self.data_file)
        if self.subset:
            df = df.sample(int(len(df)*self.subset), random_state=self.random_state)
        # build images absolute paths
        df.image_id = df.image_id.map(get_image_path)
        #df.InChI = df.InChI.map(lambda x: x.split('/')[1])
        df.InChI = df[self.text_column].map(lambda x: x.split(' '))
        df.InChI = df.InChI.map(self.encode)
        # train test splits
        train, val = train_test_split(df, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        # datasets
        self.train_ds = Dataset(train.image_id.values, train.InChI.values, self.max_len, 
            tokens=(self.stoi['PAD'], self.stoi['SOS'], self.stoi['EOS']), trans = A.Compose([
            getattr(A, trans)(**params) for trans, params in self.train_trans.items()
        ]) if self.train_trans else None)
        self.val_ds = Dataset(val.image_id.values, val.InChI.values, self.max_len, 
            tokens=(self.stoi['PAD'], self.stoi['SOS'], self.stoi['EOS']), trans = A.Compose([
            getattr(A, trans)(**params) for trans, params in self.val_trans.items()
        ]) if self.val_trans else None)
        if self.val_with_train:
            self.val_ds = self.train_ds           
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=self.shuffle_train, 
            pin_memory=self.pin_memory, 
            collate_fn=self.train_ds.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=self.pin_memory, 
            collate_fn=self.val_ds.collate
        )
