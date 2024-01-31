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
        image = io.imread(image_path)
        # rgb
        rgb = image[..., 2:5]
        # normalization
        rgb_norm = np.clip(rgb / 3e4, 0, 1).astype(np.float32)
        if self.mode == "train":
            mask = io.imread(
                f"{self.path}/{self.mask_folder}/{image_id}{self.mask_suffix}"
            )
            return rgb_norm, mask
        return rgb_norm, image_id
