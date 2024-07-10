import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data import transform as A
import logging
from src.data.datasets.base_leash import BaseLeashDataset
import polars as pl
import os
import hydra
import json
from src.utils.misc import padding_collater
from functools import partial

logger = logging.getLogger(__name__)

TRAIN_NUM_MOLECULES = 98_415_610
TEST_NUM_MOLECULES = 878_022


class FingerprintDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD TARGETS
        if stage != 'predict':
            self.targets = cache['train_targets']

        # LOAD FEATURES
        self.features = cache[f'{self._stage}_features']

        self.format = cfg.data.format
        self.fp_dim = self.features.shape[-1] * 8
        if self.format == 'seq_v2':
            self.input_id_offsets = torch.arange(0, self.fp_dim) * 2

    def get_collater(self):
        if self.format == 'set':
            max_token_id = self.fp_dim - 1
        elif self.format == 'seq':
            max_token_id = 1
        elif self.format == 'seq_v2':
            max_token_id = self.fp_dim * 2 - 1
        return partial(padding_collater,
                       max_len=4096,
                       keys=['input_ids', 'padding_mask'],
                       pad_values=[max_token_id + 1, 0],
                       padding=True)

    @property
    def getitem_as_batch(self):
        return False

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        data = {}
        ridx = self.idxs[idx]
        fp = np.unpackbits(self.features[ridx], axis=-1)
        if self.format == 'set':
            # combine with nn.Embedding <=> a nn.Linear layer
            # binary fingerprint
            token_ids = fp.nonzero()[0]
        elif self.format == 'seq':
            token_ids = fp
        elif self.format == 'seq_v2':
            # to combine with nn.Embedding of size x2
            token_ids = self.input_id_offsets + fp
        else:
            raise ValueError

        token_ids = torch.from_numpy(token_ids).long()
        padding_mask = torch.ones_like(token_ids, dtype=torch.bool)
        data = {
            'idx': idx,
            'input_ids': token_ids,
            'padding_mask': padding_mask,
        }
        if self.stage != 'predict':
            target = self.targets[ridx]
            data['target'] = torch.from_numpy(target).float()

        return data

    @classmethod
    def load_cache(cls, cfg):
        cache = {}
        data_dir = cfg.env.data_dir
        feature_name = cfg.data.feature_name
        feature_dir = os.path.join(data_dir, 'processed', 'features',
                                   feature_name)
        # with open(os.path.join(feature_dir, 'meta.json'), 'r') as f:
        #     meta = json.load(f)

        stages = []
        if cfg.train or cfg.test:
            stages.append('train')
        if cfg.predict:
            stages.append('test')

        # LOAD FEATURES
        for stage in stages:
            mmap_mode = 'r' if cfg.data.backend == 'mmap' else None
            features = np.load(os.path.join(feature_dir, f'{stage}.npy'),
                               mmap_mode=mmap_mode)
            cache[f'{stage}_features'] = features

        # LOAD TARGET
        if 'train' in stages:
            cache['train_targets'] = pl.scan_csv(
                os.path.join(data_dir, 'processed', 'train_v2.csv')).select(
                    # pl.col('molecule'),
                    pl.col('BRD4', 'HSA',
                           'sEH').cast(pl.UInt8), ).collect().to_numpy()

        return cache
