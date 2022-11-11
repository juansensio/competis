import torch
from skimage.io import imread
import numpy as np


class RGBTemporalDataset(torch.utils.data.Dataset):
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
                trans = self.trans(
                    image=images[0],
                    mask=label,
                    image2=images[1],
                    image3=images[2],
                    image4=images[3],
                    image5=images[4],
                    image6=images[5],
                    image7=images[6],
                    image8=images[7],
                    image9=images[8],
                    image10=images[9],
                    image11=images[10],
                    image12=images[11],
                )
                return np.stack([
                    trans['image'].transpose(2, 0, 1),
                    trans['image2'].transpose(2, 0, 1),
                    trans['image3'].transpose(2, 0, 1),
                    trans['image4'].transpose(2, 0, 1),
                    trans['image5'].transpose(2, 0, 1),
                    trans['image6'].transpose(2, 0, 1),
                    trans['image7'].transpose(2, 0, 1),
                    trans['image8'].transpose(2, 0, 1),
                    trans['image9'].transpose(2, 0, 1),
                    trans['image10'].transpose(2, 0, 1),
                    trans['image11'].transpose(2, 0, 1),
                    trans['image12'].transpose(2, 0, 1),
                ]).astype(np.float32), trans['mask']
            return np.stack([img.transpose(2, 0, 1) for img in images]).astype(np.float32), label
        if self.trans is not None:
            trans = self.trans(
                image=images[0],
                image2=images[1],
                image3=images[2],
                image4=images[3],
                image5=images[4],
                image6=images[5],
                image7=images[6],
                image8=images[7],
                image9=images[8],
                image10=images[9],
                image11=images[10],
                image12=images[11],
            )
            return np.stack([
                trans['image'].transpose(2, 0, 1),
                trans['image2'].transpose(2, 0, 1),
                trans['image3'].transpose(2, 0, 1),
                trans['image4'].transpose(2, 0, 1),
                trans['image5'].transpose(2, 0, 1),
                trans['image6'].transpose(2, 0, 1),
                trans['image7'].transpose(2, 0, 1),
                trans['image8'].transpose(2, 0, 1),
                trans['image9'].transpose(2, 0, 1),
                trans['image10'].transpose(2, 0, 1),
                trans['image11'].transpose(2, 0, 1),
                trans['image12'].transpose(2, 0, 1),
            ]).astype(np.float32), self.labels[ix]
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
