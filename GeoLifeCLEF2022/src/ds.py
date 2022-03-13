import torch
from skimage.io import imread


class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, trans=None):
        self.images = images
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = imread(self.images[ix])
        if self.trans is not None:
            img = self.trans(image=img)['image']
        if self.labels is not None:
            label = self.labels[ix]
            return img, label
        observation_id = self.images[ix].split('/')[-1].split('_')[0]
        return img, observation_id
