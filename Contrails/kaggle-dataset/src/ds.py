import torch
import numpy as np
from einops import rearrange
import pandas as pd
import os
from skimage.transform import resize
import albumentations as A
from .augmentations import coin, cutmix, mosaic
import random


def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode="train",
        records=None,
        path="/fastdata/contrails",
        stats_path=None,  #'/fastdata/contrails/stats.csv', # min, max, mean, std for each band
        bands=list(range(8, 17)),
        t=tuple(range(8)),
        norm_mode="mean_std",
        false_color=False,
        trans=None,  # NO PONER RESIZE AQUI !!!
        input_size=(256, 256),
        cutmix={"p": 0.0, "min_h": 64, "min_w": 64, "max_h": 128, "max_w": 128},
        mosaic={"p": 0.0, "min_h": 64, "min_w": 64, "max_h": 192, "max_w": 192},
    ):
        if mode not in ["train", "validation"]:
            raise ValueError(f"Invalid mode {mode}")
        if norm_mode not in ["mean_std", "min_max"]:
            raise ValueError(f"Invalid norm_mode {norm_mode}")
        for b in bands:
            assert b in range(8, 17), f"Invalid band {b}"
        for t_ in t:
            assert t_ in range(8), f"Invalid time index {t_}"
        self.mode = mode
        self.records = os.listdir(f"{path}/{mode}") if records is None else records
        self.records = sorted(self.records)
        self.path = path
        self.bands = bands
        self.trans = trans
        self.t = t
        self.stats = (
            pd.read_csv(stats_path, index_col=0) if stats_path is not None else None
        )
        self.norm_mode = norm_mode
        self.false_color = false_color
        self.input_size = (
            (input_size, input_size) if isinstance(input_size, int) else input_size
        )
        self.cutmix = cutmix
        self.mosaic = mosaic

    def __len__(self):
        return len(self.records)

    def get_data(self, ix):
        data = []
        for band in self.bands:
            band_data = np.load(
                f"{self.path}/{self.mode}/{self.records[ix]}/band_{band:02d}.npy"
            )[..., self.t]
            if self.norm_mode == "mean_std":
                band_data = (band_data - self.stats.loc[band]["mean"]) / self.stats.loc[
                    band
                ]["std"]
            elif self.norm_mode == "min_max":
                band_data = (band_data - self.stats.loc[band]["min"]) / (
                    self.stats.loc[band]["max"] - self.stats.loc[band]["min"]
                )
            data.append(band_data)
        return np.stack(data, axis=-1)  # H, W, T, C

    def get_false_color(self, ix):
        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
        b11 = np.load(f"{self.path}/{self.mode}/{self.records[ix]}/band_11.npy")[
            ..., self.t
        ]
        b14 = np.load(f"{self.path}/{self.mode}/{self.records[ix]}/band_14.npy")[
            ..., self.t
        ]
        b15 = np.load(f"{self.path}/{self.mode}/{self.records[ix]}/band_15.npy")[
            ..., self.t
        ]
        r = normalize_range(b15 - b14, _TDIFF_BOUNDS)
        g = normalize_range(b14 - b11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(b14, _T11_BOUNDS)
        return np.clip(np.stack([r, g, b], axis=-1), 0, 1)

    def __getitem__(self, ix):
        if self.false_color:
            image = self.get_false_color(ix)
        else:
            image = self.get_data(ix)
        mask = np.load(
            f"{self.path}/{self.mode}/{self.records[ix]}/human_pixel_masks.npy"
        )
        if self.trans is not None:
            H, W, T, C = image.shape
            image = rearrange(image, "h w t c -> h w (t c)")
            trans = self.trans(image=image, mask=mask)
            image, mask = trans["image"], trans["mask"]
            image = rearrange(image, "h w (t c) -> h w t c", t=T, c=C)
        if self.input_size != (256, 256):
            H, W, T, C = image.shape
            image = rearrange(image, "h w t c -> h w (t c)")
            image = resize(image, self.input_size, anti_aliasing=True)
            image = rearrange(image, "h w (t c) -> h w t c", t=T, c=C)
        return torch.from_numpy(image), torch.from_numpy(mask).long().squeeze(-1)

    def collate_fn(self, batch):
        images, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        if coin(self.cutmix["p"]):
            B, H, W, T, C = images.shape
            h = random.randint(self.cutmix["min_h"], self.cutmix["max_h"])
            w = random.randint(self.cutmix["min_w"], self.cutmix["max_w"])
            y0 = random.randint(0, H - h)
            x0 = random.randint(0, W - w)
            y1 = y0 + h
            x1 = x0 + w
            bb = (y0, x0, y1, x1)
            images = cutmix(images, bb)
            masks = cutmix(masks, bb)
        elif coin(self.mosaic["p"]):
            B, H, W, T, C = images.shape
            py = random.randint(self.mosaic["min_h"], self.mosaic["max_h"])
            px = random.randint(self.mosaic["min_w"], self.mosaic["max_w"])
            p = (py, px)
            images = mosaic(images, p)
            masks = mosaic(masks, p)
        return images, masks
