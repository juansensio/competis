import torch 
import numpy as np
import gzip 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, max_len):
        self.images = images
        self.masks = masks
        self.max_len = max_len

    def __len__(self):
        return len(self.images)

    def get_npy(self, img):
        f = gzip.GzipFile(img, "r")
        return np.load(f)

    def __getitem__(self, ix):
        rgb_ts = self.get_npy(self.images[ix])[-self.max_len:,:,:,(3,2,1)]  # t, 500, 500, 3
        rgb_ts = torch.from_numpy(rgb_ts / 4000).clip(0, 1).float()
        rgb_ts = rgb_ts.permute(0, 3, 1, 2) # t, 3, 500, 500
        mask = self.get_npy(self.masks[ix])  # 2000, 2000, 1
        mask =  torch.from_numpy(mask).float().permute(2, 0, 1) # 1, 2000, 2000
        return rgb_ts, mask