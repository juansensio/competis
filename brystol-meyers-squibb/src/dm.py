import torch
from skimage import io
from .vocab import t2ix, encode
import pytorch_lightning as pl
from pathlib import Path
from .utils import get_image_path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, inchis=None, max_len=512, trans=None, train=True):
        self.images = images
        self.inchis = inchis
        self.trans = trans
        self.train = train
        self.max_len = max_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image = io.imread(self.images[ix]) 
        if self.trans:
            image = self.trans(image=image)['image']
        image = torch.tensor(image / 255., dtype=torch.float).unsqueeze(0)
        if self.train:
            inchi = torch.tensor(self.inchis[ix] + [t2ix('EOS')], dtype=torch.long)
            #inchi = torch.nn.functional.pad(inchi, (0, self.max_len - len(inchi)), 'constant', t2ix('PAD'))
            return image, inchi
        return image

    def collate(self, batch):
        if self.train:
            # calcular longitud máxima en el batch 
            max_len = 0
            for image, inchi in batch:
                max_len = len(inchi) if len(inchi) > max_len else max_len        
            # añadimos padding a los inchis cortos para que todos tengan la misma longitud
            images, inchis = [], []
            for image, inchi in batch:
                images.append(image)
                inchis.append(torch.nn.functional.pad(inchi, (0, max_len - len(inchi)), 'constant', t2ix('PAD')))
            # opcionalmente, podríamos re-ordenar las frases en el batch (algunos modelos lo requieren)
            return torch.stack(images), torch.stack(inchis)
        return torch.stack([img for img in batch])

class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_file = 'train_labels.csv', 
        path=Path('data'), 
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

    def setup(self, stage=None):
        # read csv file with data
        df = pd.read_csv(self.path / self.data_file)
        if self.subset:
            df = df.head(int(len(df)*self.subset))
        # build images absolute paths
        df.image_id = df.image_id.map(get_image_path)
        df.InChI = df.InChI.map(lambda x: x.split('/')[1])
        #df.InChI = df.InChI.map(lambda x: ('/').join(x.split('/')[1:4]))
        df.InChI = df.InChI.map(encode)
        self.df = df
        # train test splits
        train, val = train_test_split(df, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        # datasets
        self.train_ds = Dataset(train.image_id.values, train.InChI.values, self.max_len, trans = A.Compose([
            getattr(A, trans)(**params) for trans, params in self.train_trans.items()
        ]) if self.train_trans else None)
        self.val_ds = Dataset(val.image_id.values, val.InChI.values, self.max_len, trans = A.Compose([
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
