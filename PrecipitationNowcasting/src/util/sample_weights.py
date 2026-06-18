from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm

from .weight_cache import cache_is_valid, write_cache_meta


def _cache_name(min_obs, rain_boost, satellite_target, num_rows):
    sat_tag = 'all' if satellite_target is None else satellite_target
    min_obs_tag = 'none' if min_obs is None else str(min_obs)
    return f'sample_weights_minobs{min_obs_tag}_rain{rain_boost}_sat{sat_tag}_n{num_rows}'


def compute_sample_weights(
    data_dir='data',
    df=None,
    csv_path=None,
    rain_boost=5.0,
    max_samples=None,
    min_obs=None,
    satellite_target=None,
    cache_dir=None,
):
    """
    Compute per-sample weights for importance sampling.
    Samples with higher peak rain rates are sampled more often.
    Loads from cache_dir when available and still valid.
    """
    source_paths = []
    if df is None:
        if csv_path is None:
            raise ValueError('Either df or csv_path must be provided')
        csv_path = Path(csv_path)
        source_paths.append(csv_path)
        df = pd.read_csv(csv_path)
    else:
        source_paths.append(Path(data_dir) / 'train_split.csv')

    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    params = {
        'rain_boost': rain_boost,
        'min_obs': min_obs,
        'satellite_target': satellite_target,
        'num_rows': len(df),
    }

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_name = _cache_name(min_obs, rain_boost, satellite_target, len(df))
        cache_path = cache_dir / f'{cache_name}.npy'
        meta_path = cache_dir / f'{cache_name}.meta.json'
        if cache_is_valid(cache_path, meta_path, source_paths, params):
            print(f'Loading sample weights from {cache_path}')
            return np.load(cache_path)

    data_dir = Path(data_dir)
    max_rain_values = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing sample weights'):
        target_path = data_dir / 'gpm_imerg' / row['gpm_imerg_filename']
        target = imread(target_path)
        if target.ndim == 3:
            target = target[..., 0]
        max_rain_values.append(float(target.max()))

    max_rain_values = np.array(max_rain_values, dtype=np.float64)
    weights = 1.0 + rain_boost * np.log1p(max_rain_values)
    weights = weights / weights.mean()
    weights = weights.astype(np.float64)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_name = _cache_name(min_obs, rain_boost, satellite_target, len(df))
        cache_path = cache_dir / f'{cache_name}.npy'
        meta_path = cache_dir / f'{cache_name}.meta.json'
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, weights)
        write_cache_meta(meta_path, source_paths, params)
        print(f'Saved sample weights to {cache_path}')

    return weights
