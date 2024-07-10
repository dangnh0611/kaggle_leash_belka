import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data import transform as A
import logging
from src.data.datasets.base import BaseDataset, BaseDataModule
import polars as pl
import os
import hydra
import json
from src.utils.misc import padding_collater
from functools import partial

TRAIN_LEN = 98_415_610
TEST_LEN = 878_022


logger = logging.getLogger(__name__)


class BaseLeashDataset(BaseDataset):

    PROTEINS = ['BRD4', 'HSA', 'sEH']

    def __init__(self, cfg, stage="train", cache={}):
        self.cfg = cfg
        self.data_dir = cfg.env.data_dir
        stage = 'val' if stage == 'test' else stage
        assert stage in ['train', 'val', 'predict']
        self.stage = stage
        self._stage = 'test' if stage == 'predict' else 'train'
        self.targets = None

        # LOAD STAGE IDXS
        self.idxs, val_df, test_df = self._get_stage_idxs(cfg, stage)
        if stage == 'val':
            self.val_df = val_df
        if stage == 'predict':
            self.test_df = test_df
        logger.info('%s dataset with %d samples', stage, len(self.idxs))

        self.label_cols = getattr(cfg.task, 'target_cols', ['BRD4', 'HSA', 'sEH'])
        self.label_idxs = [
            self.PROTEINS.index(e)
            for e in self.label_cols
        ]

        # BUILD TRANSFORM FUNC
        self.transform = self._build_transform_fn()

    def __len__(self):
        return len(self.idxs)

    def _get_stage_idxs(self, cfg, stage):
        data_dir = cfg.env.data_dir
        idxs = None
        test_df = None
        val_df = None

        if stage == 'predict':
            idxs = np.arange(0, TEST_LEN, 1)
            test_df = pl.scan_csv(
                os.path.join(data_dir, 'processed', 'test_v4.csv')).collect()
        else:
            cv_strategy = cfg.cv.strategy
            if 'bb_grid-' in cv_strategy:
                sub_strategy = cv_strategy.replace('bb_grid-', '')
                meta_path = os.path.join(self.data_dir, 'processed', 'cv',
                                         'bb_grid', f'{sub_strategy}.json')
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                fold_meta = meta['grid'][self.cfg.cv.fold_idx]
                bb1_idxs = fold_meta['bb1_idxs']
                bb23_idxs = fold_meta['bb23_idxs']
                val_df = pl.scan_csv(
                    os.path.join(
                        data_dir, 'processed',
                        'train_v2.csv')).with_row_index('index').filter(
                            (pl.col('bb1').is_in(bb1_idxs))
                            & (pl.col('bb2').is_in(bb23_idxs))
                            & (pl.col('bb3').is_in(bb23_idxs))).select(
                                pl.col('index'),
                                subset=pl.lit(3).cast(
                                    pl.UInt8)).collect().to_pandas()
                idxs = val_df['index'].to_numpy()
            elif 'nonshare-' in cv_strategy:
                split_strategy = cv_strategy.replace('nonshare-', '').replace(
                    'overfit-', '')
                with open(
                        os.path.join(data_dir, 'processed', 'cv',
                                     'nonshare_bb_split',
                                     f'{split_strategy}.json'), 'r') as f:
                    bb_split_meta = json.load(f)
                val_bb1s = bb_split_meta['val_bb1s']
                val_bb23s = bb_split_meta['val_bb23s']
                if 'overfit-' in cv_strategy:
                    val_df = pl.scan_csv(
                        os.path.join(
                            data_dir, 'processed',
                            'train_v2.csv')).with_row_index('index').filter(
                                (pl.col('bb1').is_in(val_bb1s))
                                & (pl.col('bb2').is_in(val_bb23s))
                                & (pl.col('bb3').is_in(val_bb23s))).select(
                                    pl.col('index'),
                                    subset=pl.lit(3).cast(
                                        pl.UInt8)).collect().to_pandas()
                    idxs = val_df['index'].to_numpy()
                else:
                    if stage == 'train':
                        idxs = pl.scan_csv(
                            os.path.join(data_dir, 'processed', 'train_v2.csv')
                        ).with_row_index('index').filter(~(
                            (pl.col('bb1').is_in(val_bb1s))
                            | (pl.col('bb2').is_in(val_bb23s))
                            | (pl.col('bb3').is_in(val_bb23s)))).select(
                                pl.col('index')).collect()['index'].to_numpy()
                    elif stage == 'val':
                        val_df = pl.scan_csv(
                            os.path.join(data_dir, 'processed', 'train_v2.csv')
                        ).with_row_index('index').filter(
                            (pl.col('bb1').is_in(val_bb1s))
                            & (pl.col('bb2').is_in(val_bb23s))
                            & (pl.col('bb3').is_in(val_bb23s))).select(
                                pl.col('index'),
                                subset=pl.lit(3).cast(
                                    pl.UInt8)).collect().to_pandas()
                        idxs = val_df['index'].to_numpy()
                    else:
                        raise AssertionError
            elif 'kf' in cv_strategy:
                if stage == 'train':
                    if not cfg.cv.train_on_all:
                        filter_cond = pl.col('fold_idx') != cfg.cv.fold_idx
                    else:
                        filter_cond = pl.col('fold_idx') != -9999
                elif stage == 'val':
                    filter_cond = pl.col('fold_idx') == cfg.cv.fold_idx
                val_df = pl.scan_csv(
                    os.path.join(data_dir, 'processed', 'cv', cv_strategy,
                                 'cv.csv')).filter(filter_cond).select(
                                     pl.col('index'),
                                     subset=pl.lit(3).cast(
                                         pl.UInt8)).collect().to_pandas()
                idxs = val_df['index'].to_numpy()
            else:
                if stage == 'train':
                    if cfg.cv.fold_idx == 611:
                        raise NotImplementedError
                        idxs = np.arange(0, len(features))
                    else:
                        idxs = pl.scan_csv(
                            os.path.join(
                                data_dir, 'processed', 'cv', cv_strategy,
                                'train.csv')).select(pl.col(
                                    'index')).collect().to_numpy().flatten()
                elif stage == 'val':
                    val_df = pl.scan_csv(
                        os.path.join(
                            data_dir, 'processed', 'cv', cv_strategy,
                            'val.csv')).filter(pl.col('subset') != 1).select(
                                pl.col('index'),
                                pl.col('subset').cast(
                                    pl.UInt8)).collect().to_pandas()
                    idxs = val_df['index'].to_numpy()
                else:
                    raise AssertionError

        if stage == 'train':
            if cfg.data.subsample is not None and cfg.data.subsample < len(
                    idxs):
                logger.info('Subsample train dataset from %d to %d samples',
                            len(idxs), cfg.data.subsample)
                idxs = np.random.choice(idxs,
                                        size=cfg.data.subsample,
                                        replace=False)
        return idxs, val_df, test_df

    def compute_sampling_weights(self):
        assert self.stage == 'train'
        sample_weights_name = self.cfg.data.sample_weights

        if sample_weights_name is not None:
            logger.info('Using sample weights %s', sample_weights_name)
            sample_weights = pl.scan_csv(
                os.path.join(
                    self.data_dir, 'processed', 'sample_weights',
                    f'{sample_weights_name}.csv')).select(
                        'sample_weight').collect()['sample_weight'].to_numpy()
            sample_weights = sample_weights[self.idxs]
            return sample_weights
        else:
            return np.ones((len(self), ), dtype=np.uint8)

    def get_labels(self):
        targets = self.targets[self.idxs]
        labels = np.any(targets, axis=-1).astype(np.uint8)
        return labels

    def _build_transform_fn(self):

        def _transform_fn(data):
            return data

        return _transform_fn

    @property
    def getitem_as_batch(self):
        return False

    def __getitem__(self, idxs):
        raise NotImplementedError

    @classmethod
    def load_cache(cls, cfg):
        return {}
