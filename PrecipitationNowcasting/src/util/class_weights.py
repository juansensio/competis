import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from tqdm import tqdm

from .weight_cache import cache_is_valid, cache_paths, write_cache_meta

LOG_NORM_DIVISOR = 4.58


def target_to_bins(target, num_classes=64):
    y_norm = np.log1p(target.astype(np.float32)) / LOG_NORM_DIVISOR
    y_clamped = np.clip(y_norm, 0.0, 1.0)
    return (y_clamped * (num_classes - 1)).round().astype(np.int64)


def _filter_observations(df, min_obs):
    df = df.copy()
    df['last_30_minutes_observation_filename'] = df['last_30_minutes_observation_filename'].apply(ast.literal_eval)
    return df[
        df['last_30_minutes_observation_filename'].apply(
            lambda x: isinstance(x, list) and len(x) >= min_obs
        )
    ]


def _cache_name(num_classes, min_obs, max_samples, num_rows):
    min_obs_tag = 'none' if min_obs is None else str(min_obs)
    max_samples_tag = 'all' if max_samples is None else str(max_samples)
    return f'class_weights_nc{num_classes}_minobs{min_obs_tag}_ns{max_samples_tag}_n{num_rows}'


def compute_class_weights(
    csv_path,
    data_dir='data',
    num_classes=64,
    max_samples=None,
    min_obs=None,
    cache_dir=None,
):
    """
    Compute SaTformer-style class weights: w_i = -log(|D_i| / |D_total|).
    Loads from cache_dir when available and still valid.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if min_obs is not None:
        df = _filter_observations(df, min_obs)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    params = {
        'num_classes': num_classes,
        'min_obs': min_obs,
        'max_samples': max_samples,
        'num_rows': len(df),
    }

    if cache_dir is not None:
        cache_name = _cache_name(num_classes, min_obs, max_samples, len(df))
        cache_path, meta_path = cache_paths(cache_dir, cache_name)
        if cache_is_valid(cache_path, meta_path, [csv_path], params):
            print(f'Loading class weights from {cache_path}')
            return torch.load(cache_path, weights_only=True)

    counts = np.zeros(num_classes, dtype=np.float64)
    data_dir = Path(data_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing class weights'):
        target_path = data_dir / 'gpm_imerg' / row['gpm_imerg_filename']
        target = imread(target_path)
        if target.ndim == 3:
            target = target[..., 0]
        bins = target_to_bins(target, num_classes=num_classes)
        for b in range(num_classes):
            counts[b] += (bins == b).sum()

    total = counts.sum()
    counts = np.maximum(counts, 1.0)  # avoid log(0) for empty bins
    weights = -np.log(counts / total)
    weights = weights / weights.min()  # normalize so minimum weight is 1.0
    weights = torch.tensor(weights, dtype=torch.float32)

    if cache_dir is not None:
        cache_name = _cache_name(num_classes, min_obs, max_samples, len(df))
        cache_path, meta_path = cache_paths(cache_dir, cache_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights, cache_path)
        write_cache_meta(meta_path, [csv_path], params)
        print(f'Saved class weights to {cache_path}')

    return weights
