import torch 
import gzip 
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, mode="train"):
        self.images = df['file'].values
        self.steps = df['step'].values
        self.masks = df['mask'].values
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        bands_f = gzip.GzipFile(self.images[ix], "r")
        bands = np.load(bands_f)  # steps, 500, 500, 12
        #all_bands = bands[self.step[ix]] # 500, 500, 12
        rgb = bands[...,(3,2,1)][self.steps[ix]] # 500, 500, 3
        rgb_t = torch.from_numpy(rgb / 4000).clip(0, 1).float()
        rgb_t_pad = torch.nn.functional.pad(rgb_t, (0, 0, 6, 6, 6, 6), "constant", 0) # 512, 512, 3
        rgb_t_pad = rgb_t_pad.permute(2,0,1) # 3, 512, 512
        if self.mode == "test":
            return rgb_t_pad
        mask_f = gzip.GzipFile(self.masks[ix], "r")
        mask = np.load(mask_f)  # 2000, 2000, 1
        mask_t =  torch.from_numpy(mask).float().permute(2,0,1) # 1, 2048, 2048
        mask_t_pad = torch.nn.functional.pad(mask_t, (24, 24, 24, 24, 0, 0), "constant", 0) # 1, 2048, 2048
        return rgb_t_pad, mask_t_pad