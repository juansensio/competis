import torch
from skimage import io
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_ids,
        trans=None,
        mode="train",
        path="data",
        image_folder="train_satellite",
        mask_folder="train_kelp",
        image_suffix="_satellite.tif",
        mask_suffix="_kelp.tif",
    ):
        self.image_ids = image_ids
        self.trans = trans
        assert mode in ["train", "test"], "mode must be train or test"
        self.mode = mode
        self.path = path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = f"{self.path}/{self.image_folder}/{image_id}{self.image_suffix}"
        image = io.imread(image_path)  # swir1, nir, r, g, b, clouds, dem
        image = self.read_image(image)
        if self.mode == "train":
            mask = io.imread(
                f"{self.path}/{self.mask_folder}/{image_id}{self.mask_suffix}"
            )
            if self.trans:
                trans = self.trans(image=image, mask=mask)
                image, mask = trans["image"], trans["mask"]
            return image, mask, image_id
        if self.trans:
            image = self.trans(image=image)["image"]
        return image, image_id

    def read_image(self, image):
        raise NotImplementedError


class DatasetRGB(Dataset):
    def __init__(
        self,
        image_ids,
        trans=None,
        mode="train",
    ):
        super().__init__(image_ids, trans, mode)

    def read_image(self, image):
        rgb = image[..., 2:5]
        return np.clip(rgb / 3e4, 0, 1).astype(np.float32)


class DatasetFC(Dataset):
    def __init__(
        self,
        image_ids,
        trans=None,
        mode="train",
        image_folder="train_satellite",
    ):
        super().__init__(image_ids, trans, mode, image_folder=image_folder)

    def read_image(self, image):
        fc = image[..., (1, 3, 4)]
        return np.clip(fc / 2e4, 0, 1).astype(np.float32)


class DatasetFCI(Dataset):
    def __init__(
        self,
        image_ids,
        trans=None,
        mode="train",
        image_folder="train_satellite",
    ):
        super().__init__(image_ids, trans, mode, image_folder=image_folder)

    def read_image(self, image):
        fc = image[..., (1, 3, 4)]
        fc = np.clip(fc / 2e4, 0, 1)
        nir = image[..., 1]
        red = image[..., 2]
        ndvi = (nir - red) / (nir + red)
        ndvi = (np.clip(ndvi, -1, 1) + 1) / 2.0
        swir = image[..., 0]
        ndwi = (nir - swir) / (swir + nir)
        ndwi = (np.clip(ndwi, -1, 1) + 1) / 2.0
        image = np.concatenate([fc, ndvi[..., None], ndwi[..., None]], axis=-1).astype(
            np.float32
        )
        return image


class DatasetFCIm(Dataset):
    def __init__(
        self,
        image_ids,
        trans=None,
        mode="train",
        image_folder="train_satellite",
    ):
        super().__init__(image_ids, trans, mode, image_folder=image_folder)

    def read_image(self, image):
        fc = image[..., (1, 3, 4)]
        fc = np.clip(fc / 2e4, 0, 1)
        nir = image[..., 1]
        red = image[..., 2]
        ndvi = (nir - red) / (nir + red)
        ndvi = (np.clip(ndvi, -1, 1) + 1) / 2.0
        swir = image[..., 0]
        ndwi = (nir - swir) / (swir + nir)
        ndwi = (np.clip(ndwi, -1, 1) + 1) / 2.0
        dem = image[..., 6] > 0
        clouds = image[..., 5] > 0
        mask = dem + clouds
        # no data ?
        image = np.concatenate(
            [
                fc,
                ndvi[..., None],
                ndwi[..., None],
                mask[..., None],
            ],
            axis=-1,
        ).astype(np.float32)
        return image
