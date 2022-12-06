import cupy as cp
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class ExternalInputIterator(object):
    def __init__(self, chip_ids, batch_size):
        self.batch_size = batch_size
        self.chip_ids = chip_ids
        self.sensors = ['S1', 'S2']

    def __iter__(self):
        self.i = 0
        self.n = len(self.chip_ids)
        return self

    def __next__(self):
        batch1, batch2, labels = [], [], []
        for _ in range(self.batch_size):
            chip_id = self.chip_ids[self.i]
            x1 = cp.load(
                f'data/train_features_npy/{chip_id}_S1.npy')
            x2 = cp.load(
                f'data/train_features_npy/{chip_id}_S2.npy')
            label = cp.load(f'data/train_agbm_npy/{chip_id}.npy')
            label = label[..., None]
            batch1.append(x1)
            batch2.append(x2)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch1, batch2, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__


def Dataloader(chip_ids, batch_size, num_threads=10, trans=False, seed=42):
    eii = ExternalInputIterator(
        chip_ids,
        batch_size=batch_size
    )
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads, device_id=0)
    with pipe:
        x1, x2, labels = fn.external_source(
            source=eii, num_outputs=3, device="gpu", dtype=types.FLOAT)
        if trans:
            x1 = fn.flip(x1, horizontal=fn.random.coin_flip(
                seed=seed), vertical=fn.random.coin_flip(seed=seed+42))
            x2 = fn.flip(x2, horizontal=fn.random.coin_flip(
                seed=seed), vertical=fn.random.coin_flip(seed=seed+42))
            labels = fn.flip(labels, horizontal=fn.random.coin_flip(
                seed=seed), vertical=fn.random.coin_flip(seed=seed+42))
            x1 = fn.rotate(
                x1, angle=90*fn.random.uniform(values=[0, 1, 2, 3], seed=seed))
            x2 = fn.rotate(
                x2, angle=90*fn.random.uniform(values=[0, 1, 2, 3], seed=seed))
            labels = fn.rotate(
                labels, angle=90*fn.random.uniform(values=[0, 1, 2, 3], seed=seed))
        pipe.set_outputs(x1, x2, labels)
    pipe.build()
    return DALIGenericIterator([pipe], ['x1', 'x2', 'labels'])
