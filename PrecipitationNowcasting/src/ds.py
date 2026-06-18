from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import einops

BAND_MAPPING = {
    'himawari': [4, 6, 10, 11, 13, 14, 15, 2, 3, 7, 9],
    'goes':     [4, 6, 10, 11, 13, 14, 15, 1, 2, 7, 9],
    'meteosat': [6, 8, 11, 12, 13, 14, 15, 2, 3, 9, 10],
}

class Dataset(Dataset):
    def __init__(
        self,
        path,
        data,
        trans=None,
        band_mapping=True,
        num_obs=3,
        num_frames=3,
        return_target=True,
    ):
        self.path = path
        self.data = data
        self.trans = trans
        self.num_obs = num_obs
        self.num_frames = num_frames
        self.band_mapping = band_mapping
        self.return_target = return_target

    def __len__(self):
        return len(self.data)

    def _apply_transform(self, imgs):
        if self.trans is None:
            return imgs
        kwargs = {'image': imgs[0]}
        for i in range(1, len(imgs)):
            kwargs[f'image{i}'] = imgs[i]
        result = self.trans(**kwargs)
        return [result['image']] + [result[f'image{i}'] for i in range(1, len(imgs))]

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        obss = sample['last_30_minutes_observation_filename']
        sat = sample['satellite_target']
        gpm_imerg_filename = sample['gpm_imerg_filename']
        band_indices = BAND_MAPPING[sat]
        imgs = []
        for i, obs in enumerate(obss):
            img = imread(self.path / sat / obs)
            if img.shape[-1] != 16:
                continue
            if self.band_mapping:
                img = img[..., band_indices]
            imgs.append((img / 255.0).astype(np.float32))
            if i + 1 >= self.num_obs:
                break
        while len(imgs) < self.num_frames:
            if len(imgs) > 0:
                imgs.append(np.zeros_like(imgs[0]))
            else:
                imgs.append(np.zeros((32, 32, len(band_indices)), dtype=np.float32))
        if self.return_target:
            target_img = imread(self.path / 'gpm_imerg' / gpm_imerg_filename)
            target_img = target_img.astype(np.float32)
        else:
            target_img = []
        imgs = self._apply_transform(imgs)
        imgs = np.stack(imgs, axis=0)
        imgs = einops.rearrange(imgs, 'n h w c -> n c h w')
        return {
            'inputs': imgs,
            'target': target_img,
            'satellite_target': sat,
            'gpm_imerg_filename': gpm_imerg_filename,
        }
