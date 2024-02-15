import torch
from skimage import io
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        labels=None,
        trans=None,
        mode="train",
        path="data/train",
        bands=(3, 2, 1),
    ):
        self.images = images
        self.labels = labels
        self.trans = trans
        assert mode in ["train", "test"], "mode must be train or test"
        self.mode = mode
        self.path = Path(path)
        self.bands = bands

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image_name = self.images[ix]
        image = io.imread(self.path / image_name)
        if self.bands:
            image = image[..., self.bands]
        if self.trans:
            image = self.trans(image=image)["image"]
        if self.mode == "train":
            return image, self.labels[ix]
        return image, image_name
