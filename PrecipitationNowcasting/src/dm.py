import lightning as L
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
import ast

from .ds import Dataset
from .util.sample_weights import compute_sample_weights

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        path=Path('data'),
        satellite_target=None,
        train_trans=None,
        val_trans=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        band_mapping=True,
        min_obs=3,
        num_obs=3,
        num_frames=3,
        importance_sampling=False,
        rain_boost=5.0,
        weights_cache_dir=None,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.satellite_target = satellite_target
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.pin_memory = pin_memory
        self.band_mapping = band_mapping
        self.min_obs = min_obs
        self.num_obs = num_obs
        self.num_frames = num_frames
        self.importance_sampling = importance_sampling
        self.rain_boost = rain_boost
        self.weights_cache_dir = weights_cache_dir
        self.train_sample_weights = None

    def _parse_last_30_minutes_observation_filename(self, df):
        df['last_30_minutes_observation_filename'] = df['last_30_minutes_observation_filename'].apply(ast.literal_eval)
        return df

    def _filter_observations(self, df):
        return df[
            df['last_30_minutes_observation_filename'].apply(
                lambda x: isinstance(x, list) and len(x) >= self.min_obs
            )
        ].reset_index(drop=True)

    def setup(self, stage=None):
        train = pd.read_csv(self.path / 'train_split.csv')
        train = self._parse_last_30_minutes_observation_filename(train)
        train = self._filter_observations(train)
        val = pd.read_csv(self.path / 'test_split.csv')
        val = self._parse_last_30_minutes_observation_filename(val)
        val = self._filter_observations(val)
        if self.satellite_target is not None:
            train = train[train.satellite_target == self.satellite_target]
            val = val[val.satellite_target == self.satellite_target]
        ds_kwargs = dict(
            num_obs=self.num_obs,
            num_frames=self.num_frames,
            band_mapping=self.band_mapping,
        )
        self.train_ds = Dataset(self.path, train, trans=self.train_trans, **ds_kwargs)
        self.val_ds = Dataset(self.path, val, trans=self.val_trans, **ds_kwargs)

        if self.importance_sampling:
            self.train_sample_weights = compute_sample_weights(
                data_dir=self.path,
                df=train,
                rain_boost=self.rain_boost,
                min_obs=self.min_obs,
                satellite_target=self.satellite_target,
                cache_dir=self.weights_cache_dir,
            )

    def get_dataloader(self, ds, batch_size=None, shuffle=None, sampler=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self, shuffle=True, batch_size=None):
        sampler = None
        if self.importance_sampling and self.train_sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.train_sample_weights.tolist(),
                num_samples=len(self.train_ds),
                replacement=True,
            )
            shuffle = False
        return self.get_dataloader(
            self.train_ds,
            shuffle=shuffle,
            batch_size=batch_size,
            sampler=sampler,
        )

    def val_dataloader(self, shuffle=False, batch_size=None):
        return self.get_dataloader(self.val_ds, shuffle=shuffle, batch_size=batch_size)
