import torch 
import numpy as np
from einops import rearrange

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, trans=None):
        self.images = images
        self.masks = masks
        self.trans = trans

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ix):
        image = np.load(self.images[ix])
        mask = np.load(self.masks[ix])
        if self.trans is not None:
            trans = self.trans(image=image, mask=mask)
            image, mask = trans['image'], trans['mask']
        return torch.from_numpy(image).permute(2,0,1), torch.from_numpy(mask).squeeze(-1)

class DatasetTemp(Dataset):
    def __init__(self, images, masks=None, trans=None):
        super().__init__(images, masks, trans)

    def __getitem__(self, ix):
        image = np.load(self.images[ix])
        mask = np.load(self.masks[ix])
        if self.trans is not None:
            H, W, T, C = image.shape
            image = rearrange(image, 'h w t c -> h w (t c)')
            trans = self.trans(image=image, mask=mask)
            image, mask = trans['image'], trans['mask']
            image = rearrange(image, 'h w (t c) -> h w t c', t=T, c=C)
        return torch.from_numpy(image), torch.from_numpy(mask).squeeze(-1)