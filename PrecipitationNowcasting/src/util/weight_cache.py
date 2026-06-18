import json
from pathlib import Path


def cache_is_valid(cache_path, meta_path, source_paths, params=None):
    if not cache_path.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    if params is not None and meta.get('params') != params:
        return False
    for source_path in source_paths:
        source_path = Path(source_path)
        if not source_path.exists():
            return False
        if source_path.stat().st_mtime > meta['source_mtime']:
            return False
    return True


def write_cache_meta(meta_path, source_paths, params):
    source_mtime = max(Path(p).stat().st_mtime for p in source_paths)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        'source_mtime': source_mtime,
        'params': params,
    }))


def read_cache_meta(meta_path):
    return json.loads(meta_path.read_text())


def cache_paths(cache_dir, name):
    cache_dir = Path(cache_dir)
    return cache_dir / f'{name}.pt', cache_dir / f'{name}.meta.json'
