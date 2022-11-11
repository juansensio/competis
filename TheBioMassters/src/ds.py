import torch
from skimage.io import imread
import numpy as np


class RGBTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, train=True, trans=None, num_months=12):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
        self.min = 0.
        self.mean = 63.32611
        self.std = 63.456604
        self.train = train
        self.num_months = num_months

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        # read all images in time series
        images = []
        for image in self.images[ix]:
            if image is None:  # 0s if no image
                images.append(np.zeros((256, 256, 3)))
            else:
                img = imread(image)[..., (2, 1, 0)]
                img = np.clip(img / 4000, 0., 1.).astype(np.float32)
                images.append(img)
        if self.train:
            label = imread(self.labels[ix])
            # label = (label - self.mean) / self.std
            label = (label - self.min) / (self.max - self.min)
            if self.trans is not None:
                params = {'image': images[0], 'mask': label}
                for i in range(len(images)-1):
                    params[f'image{i}'] = images[i]
                trans = self.trans(**params)
                return np.stack([trans['image'].transpose(2, 0, 1)]+[trans[f'image{i}'].transpose(2, 0, 1) for i in range(len(images)-1)]).astype(np.float32), trans['mask']
            return np.stack([img.transpose(2, 0, 1) for img in images]).astype(np.float32), label
        if self.trans is not None:
            params = {'image': images[0], 'mask': label}
            for i in range(len(images)-1):
                params[f'image{i}'] = images[i]
            trans = self.trans(**params)
            return np.stack([trans['image'].transpose(2, 0, 1)]+[trans[f'image{i}'].transpose(2, 0, 1) for i in range(len(images)-1)]).astype(np.float32), self.labels[ix]
        return np.stack([img.transpose(2, 0, 1) for img in images]).astype(np.float32), self.labels[ix]


class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, train=True, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
        self.min = 0.
        self.mean = 63.32611
        self.std = 63.456604
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = imread(self.images[ix])[..., (2, 1, 0)]
        img = np.clip(img / 4000, 0., 1.).astype(np.float32)
        if self.train:
            label = imread(self.labels[ix])
            # label = (label - self.mean) / self.std
            label = (label - self.min) / (self.max - self.min)
            if self.trans is not None:
                trans = self.trans(image=img, mask=label)
                return trans['image'].transpose(2, 0, 1), trans['mask']
            return img.transpose(2, 0, 1), label
        if self.trans is not None:
            return self.trans(image=img)['image'].transpose(2, 0, 1), self.labels[ix]
        return img.transpose(2, 0, 1), self.labels[ix]
