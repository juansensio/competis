import gzip 
import numpy as np
from skimage.transform import resize

def get_npy(path, patch, name='data/BANDS'):
    file_path = f'{path}/{patch}/{name}.npy.gz'
    f = gzip.GzipFile(file_path, "r")
    return np.load(f)

def generate_mask_with_mean_ndvi(path, patch, thresholds = (0.4, 0.6)):
    
    # load bands
    file_path = f'{path}/{patch}/data/BANDS.npy.gz'
    f = gzip.GzipFile(file_path, "r")
    bands = np.load(f)

    # compute mean ndvi
    ndvis = []
    for step in range(bands.shape[0]):
        b8 = bands[...,7][step].astype(float)
        b4 = bands[...,3][step].astype(float)
        ndvi = (b8 - b4) / (b8 + b4)
        ndvis.append(ndvi)
    mean_ndvi = np.mean(ndvis, axis=0)

    # compute target mask
    target = (mean_ndvi >= thresholds[0]) & (ndvi <= thresholds[1])

    # resize
    big_target = resize(target, (2000, 2000)).astype(bool)

    return big_target