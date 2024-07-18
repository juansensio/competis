import lightning as L
import albumentations as A
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from .ds import Dataset
from torch.utils.data import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path="data",
        batch_size=64,
        val_size=0,
        num_workers=20,
        pin_memory=True,
        train_trans=None,
        val_trans=None,
        test_trans=None,
        random_state=42,
        bands=[2, 3, 4, 5, 6, 7, 8, 9, 11, 12],  # bands used in clay model
        aod_stats=(0.209845, 0.224224),  # mean, std
        n_folds=1,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.random_state = random_state
        self.bands = bands
        self.aod_stats = aod_stats
        self.n_folds = n_folds

    def setup(self, stage=None):
        train_images = glob.glob(self.path + "/train_images/*.tif")
        test_images = glob.glob(self.path + "/test_images/*.tif")
        assert len(train_images) == 4465
        assert len(test_images) == 1489
        self.train_df = pd.read_csv(
            self.path + "/train_answer.csv", names=["image", "location", "AOD"]
        )
        assert len(self.train_df) == len(train_images)
        self.train_df.image = self.train_df.image.apply(
            lambda x: self.path + "/train_images/" + x
        )
        self.train_df.AOD = (self.train_df.AOD - self.aod_stats[0]) / self.aod_stats[1]
        if self.n_folds > 1:
            kf = KFold(
                n_splits=self.n_folds, random_state=self.random_state, shuffle=True
            )
            self.train_ds, self.val_ds = [], []
            for i, (train_ixs, val_ixs) in enumerate(kf.split(self.train_df)):
                self.train_ds.append(
                    Dataset(
                        self.train_df.image.values[train_ixs],
                        self.bands,
                        labels=self.train_df.AOD.values[train_ixs],
                        trans=(
                            A.Compose(
                                [
                                    getattr(A, trans)(**params)
                                    for trans, params in self.train_trans.items()
                                ]
                            )
                            if self.train_trans is not None
                            else None
                        ),
                    )
                )
                self.val_ds.append(
                    Dataset(
                        self.train_df.image.values[val_ixs],
                        self.bands,
                        labels=self.train_df.AOD.values[val_ixs],
                        trans=(
                            A.Compose(
                                [
                                    getattr(A, trans)(**params)
                                    for trans, params in self.val_trans.items()
                                ]
                            )
                            if self.val_trans is not None
                            else None
                        ),
                    )
                )
            print("iepa", len(self.train_ds), len(self.val_ds))
        else:
            self.val_df, self.val_ds = None, None
            if self.val_size > 0:
                self.train_df, self.val_df = train_test_split(
                    self.train_df,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
                self.val_ds = [
                    Dataset(
                        self.val_df.image.values,
                        self.bands,
                        labels=self.val_df.AOD.values,
                        trans=(
                            A.Compose(
                                [
                                    getattr(A, trans)(**params)
                                    for trans, params in self.val_trans.items()
                                ]
                            )
                            if self.val_trans is not None
                            else None
                        ),
                    )
                ]
            self.train_ds = [
                Dataset(
                    self.train_df.image.values,
                    self.bands,
                    labels=self.train_df.AOD.values,
                    trans=(
                        A.Compose(
                            [
                                getattr(A, trans)(**params)
                                for trans, params in self.train_trans.items()
                            ]
                        )
                        if self.train_trans is not None
                        else None
                    ),
                )
            ]
        self.test_df = pd.read_csv(
            self.path + "/sample_answer.csv", names=["image", "AOD"]
        )
        self.test_df.image = self.test_df.image.apply(
            lambda x: self.path + "/test_images/" + x
        )
        self.test_ds = Dataset(
            self.test_df.image.values,
            self.bands,
            trans=(
                A.Compose(
                    [
                        getattr(A, trans)(**params)
                        for trans, params in self.test_trans.items()
                    ]
                )
                if self.test_trans is not None
                else None
            ),
        )

    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return (
            DataLoader(
                ds,
                batch_size=batch_size if batch_size is not None else self.batch_size,
                shuffle=shuffle if shuffle is not None else True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=ds.collate_fn,
            )
            if ds is not None
            else None
        )

    def train_dataloader(self, fold=0, batch_size=None, shuffle=True):
        return self.get_dataloader(self.train_ds[fold], batch_size, shuffle)

    def val_dataloader(self, fold=0, batch_size=None, shuffle=False):
        return (
            self.get_dataloader(self.val_ds[fold], batch_size, shuffle)
            if self.val_ds is not None
            else None
        )

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.test_ds, batch_size, shuffle)
