import torch 
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks=None, trans=None):
        self.images = images
        self.masks = masks
        self.trans = trans

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ix):
        image = np.load(self.images[ix])
        if self.masks is not None:
            mask = np.load(self.masks[ix])
            if self.trans is not None:
                trans = self.trans(image=image, mask=mask)
                image, mask = trans['image'], trans['mask']
            return torch.from_numpy(image).permute(2,0,1), torch.from_numpy(mask).permute(2,0,1)
        image = self.trans(image=image)['image']
        return torch.from_numpy(image)
