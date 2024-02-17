import torch
from skimage import io
from pathlib import Path
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels=None,
        trans=None,
        mode="train",
        path="data/train",
        bands=(3, 2, 1),
        indices=[],
    ):
        self.images = images
        self.labels = labels
        self.trans = trans
        assert mode in ["train", "test"], "mode must be train or test"
        self.mode = mode
        self.path = Path(path)
        self.bands = bands
        self.indices = indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image_name = self.images[ix]
        image0 = io.imread(self.path / image_name)
        if self.bands:
            image = image0[..., self.bands]
        else:
            image = image0
        if (
            "ndvi" in self.indices
        ):  # NDVI helps in distinguishing between vegetation and non-vegetation areas. Mining sites typically show low NDVI values due to the absence of vegetation.
            nir = image0[..., 8]
            red = image0[..., 3]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = ndvi * 2.0 - 1.0
            image = np.dstack((image, ndvi))
        if (
            "ndwi" in self.indices
        ):  # se utiliza como una medida de la cantidad de agua que posee la vegetación o el nivel de saturación de humedad que posee el suelo.
            nir = image0[..., 8]
            swir = image0[..., 10]
            ndwi = (nir - swir) / (nir + swir + 1e-8)
            ndwi = ndwi * 2.0 - 1.0
            image = np.dstack((image, ndwi))
        if (
            "ndwbi" in self.indices
        ):  # NDBI helps in highlighting built-up areas, including infrastructure associated with mining activities.
            swir = image0[..., 10]
            nir = image0[..., 8]
            ndbi = (swir - nir) / (swir + nir + 1e-8)
            ndbi = ndbi * 2.0 - 1.0
            image = np.dstack((image, ndbi))
        if (
            "mndwi" in self.indices
        ):  # MNDWI is useful for identifying water bodies, which are common features around mining sites.
            swir = image0[..., 10]
            green = image0[..., 2]
            mndwi = (green - swir) / (swir + green + 1e-8)
            mndwi = mndwi * 2.0 - 1.0
            image = np.dstack((image, mndwi))
        if (
            "ioi" in self.indices
        ):  # IOI is particularly useful for detecting iron oxide-rich areas, which are often indicative of mining sites.
            red = image0[..., 3]
            swir = image0[..., 10]
            ioi = (red - swir) / (swir + red + 1e-8)
            ioi = ioi * 2.0 - 1.0
            image = np.dstack((image, ioi))
        if self.trans:
            image = self.trans(image=image)["image"]
        if self.mode == "train":
            return image, self.labels[ix]
        return image, image_name
