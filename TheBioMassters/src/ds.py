import torch
from skimage.io import imread
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, s1_bands, s2_bands, months, labels=None, chip_ids=None):
        self.images = images
        self.labels = labels
        self.max = 12905.3
        self.mean = 63.32611
        self.std = 63.456604
        self.months = months
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.chip_ids = chip_ids

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
                    s2s.append(np.zeros((256, 256, len(self.s2_bands))))
                else:
                    s2 = imread(path)[..., self.s2_bands]
                    s2 = np.clip(s2 / 4000, 0, 1)
                    s2s.append(s2)
        s1s = np.stack([img.transpose(2, 0, 1) for img in s1s]).astype(np.float32) if len(s1s) > 0 else None
        s2s = np.stack([img.transpose(2, 0, 1) for img in s2s]).astype(np.float32) if len(s2s) > 0 else None
        if self.labels is not None:
            label = imread(self.labels[ix])
            label = label / self.max
            # label = (label - self.mean) / self.std
            return s1s, s2s, label
        assert self.chip_ids is not None
        return s1s, s2s, self.chip_ids[ix]

def collate_fn(batch):
    s1s, s2s, labels = zip(*batch)
    if s1s[0] is None:
        s1s = None
    else:
        s1s = torch.from_numpy(np.stack(s1s))
    if s2s[0] is None:
        s2s = None
    else:
        s2s = torch.from_numpy(np.stack(s2s))
    if isinstance(labels[0], str):
        return s1s, s2s, labels
    return s1s, s2s, torch.from_numpy(np.stack(labels))
    