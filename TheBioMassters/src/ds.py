import torch
from skimage.io import imread
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, train, trans, bands=None):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
        # self.min = 0.
        # self.mean = 63.32611
        # self.std = 63.456604
        self.train = train
        self.bands = bands

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = imread(self.images[ix])
        img = self.pick_bands(img)
        img = self.normalize(img)
        if self.train:
            label = imread(self.labels[ix])
            # label = (label - self.mean) / self.std
            # label = (label - self.min) / (self.max - self.min)
            label = label / self.max
            if self.trans is not None:
                trans = self.trans(image=img, mask=label)
                return trans['image'].transpose(2, 0, 1), trans['mask']
            return img.transpose(2, 0, 1), label
        if self.trans is not None:
            return self.trans(image=img)['image'].transpose(2, 0, 1), self.labels[ix]
        return img.transpose(2, 0, 1), self.labels[ix]

    def normalize(self, img):
        raise NotImplementedError

    def pick_bands(self, img):
        if self.bands is not None:
            return img[..., self.bands]
        return img


class S1Dataset(BaseDataset):
    def __init__(self, images, labels, train=True, trans=None, bands=(0, 1)):
        super().__init__(images, labels, train, trans, bands)

    def normalize(self, img):
        return np.clip(img, -30, 0)*(-8.4) / 255.


class RGBDataset(BaseDataset):
    def __init__(self, images, labels, train=True, trans=None, bands=(2, 1, 0)):
        super().__init__(images, labels, train, trans, bands)

    def normalize(self, img):
        return np.clip(img / 4000, 0., 1.).astype(np.float32)


class DFDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, s1_bands=(0, 1), s2_bands=(2, 1, 0), train=True, trans=None, use_ndvi=False, use_ndwi=False, use_clouds=False):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
        self.train = train
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.use_ndvi = use_ndvi
        if self.use_ndvi:
            assert 2 in self.s2_bands, 'NDVI requires band 2'
            assert 7 in self.s2_bands, 'NDVI requires band 7'
        self.use_ndwi = use_ndwi
        if self.use_ndwi: 
            assert 7 in self.s2_bands, 'NDWI requires band 7'
            assert 9 in self.s2_bands, 'NDWI requires band 9'
        self.use_clouds = use_clouds
        assert 10 not in self.s2_bands, 'Do not use band 10, use use_clouds=True instead'
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        s1 = imread(self.images[ix][0])
        s2_0 = imread(self.images[ix][1])
        s1 = s1[..., self.s1_bands]
        s2 = s2_0[..., self.s2_bands]
        s1 = np.clip(s1, -30, 0)*(-8.4) / 255.
        s2 = np.clip(s2 / 4000, 0., 1.).astype(np.float32)
        if self.use_ndvi:
            red_band = self.s2_bands.index(2)
            nir_band = self.s2_bands.index(7)
            red = s2[..., red_band]
            nir = s2[..., nir_band]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = (ndvi + 1.) / 2.
            s2 = np.concatenate([s2, ndvi[..., None]], axis=-1)
        if self.use_ndwi:
            swir_band = self.s2_bands.index(9)
            nir_band = self.s2_bands.index(7)
            swir = s2[..., swir_band]
            nir = s2[..., nir_band]
            ndwi = (nir - swir) / (nir + swir + 1e-8)
            ndwi = (ndwi + 1.) / 2.
            s2 = np.concatenate([s2, ndwi[..., None]], axis=-1)
        if self.use_clouds:
            clouds = s2_0[..., 10]
            clouds[clouds == 255] = 100.
            clouds = clouds.astype(np.float32) / 100.
            s2 = np.concatenate([s2, clouds[..., None]], axis=-1)
        if self.train:
            label = imread(self.labels[ix])
            label = label / self.max
            if self.trans is not None:
                trans = self.trans(image=s1, image2=s2, mask=label)
                return trans['image'].transpose(2, 0, 1), trans['image2'].transpose(2, 0, 1), trans['mask']
            return s1.transpose(2, 0, 1), s2.transpose(2, 0, 1), label
        if self.trans is not None:
            trans = self.trans(image=s1, image2=s2)['image']
            return trans['image'].transpose(2, 0, 1), trans['image2'].transpose(2, 0, 1), self.labels[ix]
        return s1.transpose(2, 0, 1), s2.transpose(2, 0, 1), self.labels[ix]


class RGBTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, train=True, trans=None, num_months=12):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
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
            label = label / self.max
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


class DFTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, s1_bands=(0, 1), s2_bands=(2, 1, 0), train=True, trans=None, num_months=12, use_ndvi=False, use_ndwi=False, use_clouds=False):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.max = 12905.3
        self.train = train
        self.num_months = num_months
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.use_ndvi = use_ndvi
        if self.use_ndvi:
            assert 2 in self.s2_bands, 'NDVI requires band 2'
            assert 7 in self.s2_bands, 'NDVI requires band 7'
        self.use_ndwi = use_ndwi
        if self.use_ndwi: 
            assert 7 in self.s2_bands, 'NDWI requires band 7'
            assert 9 in self.s2_bands, 'NDWI requires band 9'
        self.use_clouds = use_clouds
        assert 10 not in self.s2_bands, 'Do not use band 10, use use_clouds=True instead'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        # read all images in time series
        s1s, s2s = [], []
        s1_paths, s2_paths = self.images[ix]
        for s1_path in s1_paths:
            if s1_path is None:  # 0s if no image
                s1s.append(np.zeros((256, 256, len(self.s1_bands))))
            else:
                s1 = imread(s1_path)[..., self.s1_bands]
                s1 = np.clip(s1, -30, 0)*(-8.4) / 255.
                s1s.append(s1)
        for s2_path in s2_paths:
            if s2_path is None:
                channels = len(self.s2_bands)
                if self.use_ndvi:
                    channels += 1
                if self.use_ndwi:
                    channels += 1
                if self.use_clouds:
                    channels += 1
                s2s.append(np.zeros((256, 256, channels)))
            else:
                s2_0 = imread(s2_path)
                s2 = s2_0[..., self.s2_bands]
                s2 = np.clip(s2 / 4000, 0., 1.).astype(np.float32)
                if self.use_ndvi:
                    red_band = self.s2_bands.index(2)
                    nir_band = self.s2_bands.index(7)
                    red = s2[..., red_band]
                    nir = s2[..., nir_band]
                    ndvi = (nir - red) / (nir + red + 1e-8)
                    ndvi = (ndvi + 1.) / 2.
                    s2 = np.concatenate([s2, ndvi[..., None]], axis=-1)
                if self.use_ndwi:
                    swir_band = self.s2_bands.index(9)
                    nir_band = self.s2_bands.index(7)
                    swir = s2[..., swir_band]
                    nir = s2[..., nir_band]
                    ndwi = (nir - swir) / (nir + swir + 1e-8)
                    ndwi = (ndwi + 1.) / 2.
                    s2 = np.concatenate([s2, ndwi[..., None]], axis=-1)
                if self.use_clouds:
                    clouds = s2_0[..., 10]
                    clouds[clouds == 255] = 100.
                    clouds = clouds.astype(np.float32) / 100.
                    s2 = np.concatenate([s2, clouds[..., None]], axis=-1)
                s2s.append(s2)
        if self.train:
            label = imread(self.labels[ix])
            label = label / self.max
            if self.trans is not None:
                params = {'image': s1s[0], 'mask': label}
                for i in range(len(s1s)-1):
                    params[f'image_s1_{i}'] = s1s[i]
                for i in range(len(s2s)):
                    params[f'image_s2_{i}'] = s2s[i]
                trans = self.trans(**params)
                return np.stack(
                    [trans['image'].transpose(2, 0, 1)] +
                    [trans[f'image_s1_{i}'].transpose(
                        2, 0, 1) for i in range(len(s1s)-1)]
                ).astype(np.float32), np.stack([trans[f'image_s2_{i}'].transpose(2, 0, 1) for i in range(len(s2s))]).astype(np.float32), trans['mask']
            return np.stack([img.transpose(2, 0, 1) for img in s1s]).astype(np.float32), np.stack([img.transpose(2, 0, 1) for img in s2s]).astype(np.float32), label
        if self.trans is not None:
            params = {'image': s1s[0], 'mask': label}
            for i in range(len(s1s)-1):
                params[f'image_s1_{i}'] = s1s[i]
            for i in range(len(s2s)):
                params[f'image_s2_{i}'] = s2s[i]
            trans = self.trans(**params)
            return np.stack(
                [trans['image'].transpose(2, 0, 1)] +
                [trans[f'image_s1_{i}'].transpose(
                    2, 0, 1) for i in range(len(s1s)-1)]
            ).astype(np.float32), np.stack([trans[f'image_s2_{i}'].transpose(2, 0, 1) for i in range(len(s2s))]).astype(np.float32), self.labels[ix]
        return np.stack([img.transpose(2, 0, 1) for img in s1s]).astype(np.float32), np.stack([img.transpose(2, 0, 1) for img in s2s]).astype(np.float32), self.labels[ix]

def collate_fn(batch):
    s1s, s2s, labels = zip(*batch)
    s1s = np.stack(s1s)
    s2s = np.stack(s2s)
    labels = np.stack(labels)
    return torch.from_numpy(s1s), torch.from_numpy(s2s), torch.from_numpy(labels)