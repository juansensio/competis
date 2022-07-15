import torch
import rasterio as rio
import geopandas as gpd
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, trans):
        self.df = df
        self.trans = trans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        sample = self.df.iloc[ix]
        img = sample['image']
        ds = rio.open(img)
        transform = ds.transform
        geom = gpd.read_file(sample['label'])
        if 'center_crop' in self.trans:
            trans = self.trans['center_crop']
            size, p = trans['size'], trans['p']
            if random.random() < p:
                x, y = ds.width // 2, ds.height // 2
                x1, y1 = x - size[1] // 2, y - size[0] // 2
                x2, y2 = x1 + size[1], y1 + size[0]
                print(x1, y1, x2, y2)
                minx, miny = rio.transform.xy(ds.transform, x1, y1)
                maxx, maxy = rio.transform.xy(ds.transform, x2, y2)
                window = Window(y1, x1, size[0], size[1])
                img = ds.read((1, 2, 3), window=window)
                geom = gpd.clip(geom, box(minx, miny, maxx, maxy))
                transform = ds.window_transform(window)
            else:
                img = ds.read((1, 2, 3))
        elif 'random_crop' in self.trans:
            trans = self.trans['random_crop']
            size, p = trans['size'], trans['p']
            if random.random() < p:
                x1 = random.randint(0, ds.width - size[1])
                y1 = random.randint(0, ds.height - size[0])
                x2, y2 = x1 + size[1], y1 + size[0]
                minx, miny = rio.transform.xy(ds.transform, x1, y1)
                maxx, maxy = rio.transform.xy(ds.transform, x2, y2)
                window = Window(y1, x1, size[0], size[1])
                img = ds.read((1, 2, 3), window=window)
                geom = gpd.clip(geom, box(minx, miny, maxx, maxy))
                transform = ds.window_transform(window)
            else:
                img = ds.read((1, 2, 3))
        else:
            img = ds.read((1, 2, 3))
        return img, geom, transform, sample['date']
