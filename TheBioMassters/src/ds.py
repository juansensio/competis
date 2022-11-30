import torch
from skimage.io import imread
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, s1_bands, s2_bands, months, labels=None, chip_ids=None, use_ndvi=False, use_ndwi=False, use_clouds=False, trans=None):
        self.images = images
        self.labels = labels
        self.max = 12905.3
        self.mean = 63.32611
        self.std = 63.456604
        self.months = months
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.chip_ids = chip_ids
        self.use_ndvi = use_ndvi
        self.use_ndwi = use_ndwi
        self.use_clouds = use_clouds
        assert 10 not in self.s2_bands, 'use_clouds=True para usar band 10'
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        s1s, s2s = [], []
        paths = self.images[ix]
        if self.s1_bands is not None:
            for month in self.months:
                path = paths['S1'][month]
                if path is None:
                    s1s.append(np.zeros((256, 256, len(self.s1_bands))))
                else:
                    s1 = imread(path)[..., self.s1_bands]
                    s1 = np.clip(s1, -30, 0)*(-8.4) / 255.
                    s1s.append(s1)
        if self.s2_bands is not None:
            for month in self.months:
                path = paths['S2'][month]
                if path is None:
                    s2 = np.zeros((256, 256, len(self.s2_bands)))
                    if self.use_ndvi:
                        s2 = np.concatenate([s2, np.zeros((256, 256, 1))], axis=-1)
                    if self.use_ndwi:
                        s2 = np.concatenate([s2, np.zeros((256, 256, 1))], axis=-1)
                    if self.use_clouds:
                        s2 = np.concatenate([s2, np.zeros((256, 256, 1))], axis=-1)
                    s2s.append(s2)
                else:
                    s20 = imread(path)
                    s2 = np.clip(s20[..., self.s2_bands] / 4000, 0, 1)
                    if self.use_ndvi:
                        red = s20[..., 2].astype(np.float32)
                        nir = s20[..., 6].astype(np.float32)
                        ndvi = (nir - red) / (nir + red + 1e-8)
                        ndvi = (ndvi + 1.) / 2.
                        s2 = np.concatenate([s2, ndvi[..., None]], axis=-1)
                    if self.use_ndwi:
                        swir = s20[..., 8].astype(np.float32)
                        nir = s20[..., 6].astype(np.float32)
                        ndwi = (nir - swir) / (nir + swir + 1e-8)
                        ndwi = (ndwi + 1.) / 2.
                        s2 = np.concatenate([s2, ndwi[..., None]], axis=-1)
                    if self.use_clouds:
                        clouds = s20[..., 10]
                        clouds[clouds == 255] = 100
                        clouds = clouds / 100
                        s2 = np.concatenate([s2, clouds[..., None]], axis=-1)
                    s2s.append(s2)
        if self.labels is not None:
            label = imread(self.labels[ix])
            label = label / self.max
            # label = (label - self.mean) / self.std
            s1s, s2s, label = self.apply_transforms(s1s, s2s, label)
            return s1s, s2s, label
        assert self.chip_ids is not None
        s1s, s2s, _ = self.apply_transforms(s1s, s2s)
        return s1s, s2s, self.chip_ids[ix]

    def apply_transforms(self, s1s, s2s, label=None):
        if self.trans is not None:
            params = {
                    'image': s1s[0] if len(s1s) > 0 else s2s[0], 
                    'mask': label
            }
            for i in range(len(s1s)):
                params[f'image_s1_{i}'] = s1s[i]
            for i in range(len(s2s)):
                params[f'image_s2_{i}'] = s2s[i]
            trans = self.trans(**params)
            s1s = np.stack([trans[f'image_s1_{i}'].transpose(2, 0, 1) for i in range(len(s1s))]).astype(np.float32) if len(s1s) > 0 else None
            s2s = np.stack([trans[f'image_s2_{i}'].transpose(2, 0, 1) for i in range(len(s2s))]).astype(np.float32) if len(s2s) > 0 else None
            return s1s, s2s, trans['mask']
        s1s = np.stack([img.transpose(2, 0, 1) for img in s1s]).astype(np.float32) if len(s1s) > 0 else None
        s2s = np.stack([img.transpose(2, 0, 1) for img in s2s]).astype(np.float32) if len(s2s) > 0 else None
        return s1s, s2s, label


def collate_fn(batch):
    s1s, s2s, labels = zip(*batch)
    s1s = torch.from_numpy(np.stack(s1s)) if s1s[0] is not None else None
    s2s = torch.from_numpy(np.stack(s2s)) if s2s[0] is not None else None
    return (s1s, s2s), labels if isinstance(labels[0], str) else torch.from_numpy(np.stack(labels))
    