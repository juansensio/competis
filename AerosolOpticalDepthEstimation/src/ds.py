import torch
from skimage import io
import rasterio as rio
import numpy as np
import math
import yaml
from box import Box
from datetime import datetime


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        bands=[2, 3, 4, 5, 6, 7, 8, 9, 11, 12],  # bands used in clay model
        labels=None,
        trans=None,
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.trans = trans
        self.bands = bands

        platform = "sentinel-2-l2a"
        metadata = Box(yaml.safe_load(open("src/model/configs/metadata.yaml")))
        self.waves = [
            metadata[platform].bands.wavelength[band]
            for band in metadata[platform]["band_order"]
        ]
        self.gsd = metadata[platform]["gsd"]
        self.platform = platform

    def __len__(self):
        return len(self.images)

    def normalize_latlon(self, lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    def normalize_timestamp(self, date):
        # week = date.isocalendar().week * 2 * np.pi / 52
        week = date.isocalendar()[1] * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def __getitem__(self, index):
        image_path = self.images[index]
        ds = rio.open(image_path)
        bbox = ds.bounds  # long, lat, long, lat
        # bounds = [bbox.left, bbox.bottom, bbox.right, bbox.top]
        lat, lon = bbox.left + bbox.right / 2, bbox.bottom + bbox.top / 2
        lat_norm, lon_norm = self.normalize_latlon(lat, lon)
        # image = io.imread(image_path)[self.bands, ...] # si uso esto hay que restar 1 a las bandas
        image = ds.read(self.bands)
        image = np.clip(image, 0, 3)
        if self.trans:
            image = self.trans(image=image.transpose(1, 2, 0))["image"].transpose(
                2, 0, 1
            )
        # date = ds.tags()["TIFFTAG_DATETIME"]
        date = datetime.strptime(
            "2023-05-01 10:01:20", "%Y-%m-%d %H:%M:%S"
        )  # me lo invento porque no lo tengo
        week_norm, hour_norm = self.normalize_timestamp(date)
        if self.labels is not None:
            return image, lat_norm, lon_norm, week_norm, hour_norm, self.labels[index]
        return image, lat_norm, lon_norm, week_norm, hour_norm, image_path

    def collate_fn(self, batch):
        images, lats_norm, lons_norm, weeks_norm, hours_norm, labels = zip(*batch)
        images = torch.from_numpy(np.stack(images)).float()
        labels = (
            torch.tensor(labels, dtype=torch.float32)
            if self.labels is not None
            else None
        )
        return {
            # "platform": [self.platform],  # 1
            "pixels": images,  # [B C H W]
            "time": torch.from_numpy(
                np.hstack((weeks_norm, hours_norm))
            ).float(),  # según comment en model [B 2], pero es [B 4]
            "latlon": torch.tensor(
                np.hstack((lats_norm, lons_norm))
            ).float(),  # según comment en model [B 2], pero es [B 4]
            "gsd": torch.tensor(self.gsd),  # 1
            "waves": torch.tensor(self.waves),  # [N]
        }, labels
