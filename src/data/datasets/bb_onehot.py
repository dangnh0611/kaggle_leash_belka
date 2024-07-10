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
import random

logger = logging.getLogger(__name__)

TRAIN_NUM_MOLECULES = 98_415_610
TEST_NUM_MOLECULES = 878_022


class BuildingBlockOnehotDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD FEATURES
        self.features = cache[f'{self._stage}_bbs']

        # LOAD TARGETS
        if stage != 'predict':
            self.targets = cache['train_targets']

        self.swap_prob = cfg.data.swap_prob

    def compute_sampling_weights(self):
        return [1.0] * len(self)

    def get_collater(self):
        return lambda x: x

    @property
    def getitem_as_batch(self):
        return True

    def __getitem__(self, idxs):
        """
        Generate one batch of data.
        """
        data = {}
        ridxs = self.idxs[idxs]
        if random.random() < self.swap_prob:
            # swap bb2 and bb3
            token_ids = self.features[ridxs][:, [0, 2, 1]]
        else:
            token_ids = self.features[ridxs]
        token_ids = torch.from_numpy(token_ids).long()
        padding_mask = torch.ones_like(token_ids, dtype=torch.bool)
        data = {
            'idx': torch.tensor(idxs),
            'input_ids': token_ids,
            'padding_mask': padding_mask,
        }
        if self.stage != 'predict':
            target = self.targets[ridxs]
            data['target'] = torch.from_numpy(target).float()

        return data

    @classmethod
    def load_cache(cls, cfg):
        cache = {}
        data_dir = cfg.env.data_dir

        stages = []
        if cfg.train or cfg.test:
            stages.append('train')
        if cfg.predict:
            stages.append('test')

        # LOAD TARGET
        if 'train' in stages:
            cache['train_bbs'] = pl.scan_csv(
                os.path.join(data_dir, 'processed', 'train_v2.csv')).select(
                    # pl.col('molecule'),
                    pl.col('bb1', 'bb2',
                           'bb3').cast(pl.Int16), ).collect().to_numpy()

            cache['train_targets'] = pl.scan_csv(
                os.path.join(data_dir, 'processed', 'train_v2.csv')
            ).select(
                # pl.col('molecule'),
                # pl.col('bb1', 'bb2', 'bb3').cast(pl.UInt16),
                pl.col('BRD4', 'HSA',
                       'sEH').cast(pl.UInt8), ).collect().to_numpy()
        if 'test' in stages:
            cache['test_bbs'] = pl.scan_csv(
                os.path.join(data_dir, 'processed', 'test_v4.csv')).select(
                    # pl.col('molecule'),
                    pl.col('bb1', 'bb2',
                           'bb3').cast(pl.Int16), ).collect().to_numpy()

        return cache
