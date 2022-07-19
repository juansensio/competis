import torch
import rasterio as rio
import geopandas as gpd
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import random
from shapely.geometry import LineString, Polygon


def latlon_to_xy(x, transform):
    if x.type == 'LineString':
        coords = []
        for x1, y1 in x.coords:
            x, y = rio.transform.rowcol(transform, x1, y1)
            coords.append((y, x))
        return LineString(coords)
    elif x.type == 'Polygon':
        coords = []
        for x1, y1 in x.exterior.coords:
            x, y = rio.transform.rowcol(transform, x1, y1)
            coords.append((y, x))
        return Polygon(coords)
    else:
        raise ValueError(f'Unknown geometry type: {x.type}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, trans):
        self.df = df
        self.trans = trans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        sample = self.df.iloc[ix]
        # read geojson
        geom = gpd.read_file(sample['label'])
        # open image
        img = sample['image']
        ds = rio.open(img)
        transform = ds.transform
        size = ds.shape
        # apply transforms
        if 'center_crop' in self.trans:
            trans = self.trans['center_crop']
            size, p = trans['size'], trans['p']
            if random.random() < p:
                x, y = ds.width // 2, ds.height // 2
                x1, y1 = x - size[1] // 2, y - size[0] // 2
                x2, y2 = x1 + size[1], y1 + size[0]
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
        # convert multilines and multipolygons to single lines and polygons
        geom = geom.explode()
        # convert geometry to pixel coordinates
        wkt_geom = [latlon_to_xy(x, transform)
                    for x in geom.geometry]
        geom.geometry = wkt_geom
        geom.crs = None
        # generate targets
        y1 = -1.*torch.ones(29, 256)  # lines
        y2 = -1.*torch.ones(149, 256)  # polygons
        i, j = 0, 0
        for ix, g in enumerate(geom.geometry):
            if g.type == 'LineString':
                for ixx, (y, x) in enumerate(g.coords):
                    y1[i, ixx*2:(ixx+1)*2] = torch.tensor([y /
                                                           size[0], x / size[1]])
                y1[i, -1] = 1 if geom.flooded.iloc[ix] == 'yes' else 0
                i += 1
            elif g.type == 'Polygon':
                # remove last point in polygons (will be added manually)
                assert g.exterior.coords[0] == g.exterior.coords[-1]
                for ixx, (y, x) in enumerate(g.exterior.coords[:-1]):
                    y2[j, ixx*2:(ixx+1)*2] = torch.tensor([y /
                                                           size[0], x / size[1]])
                y2[j, -1] = 1 if geom.flooded.iloc[ix] == 'yes' else 0
                j += 1
            else:
                raise ValueError(f'Unknown geometry type: {g.type}')
        # TODO: permute valid geometries for data augmentation
        return img, geom, transform, sample['date'], y1, y2
        # return img, y1, y2
