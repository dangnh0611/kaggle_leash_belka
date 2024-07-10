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
import gc

logger = logging.getLogger(__name__)


class Pipeline:

    @staticmethod
    def unpackbits(arr, axis=-1):
        return np.unpackbits(arr, axis=-1)

    @staticmethod
    def gather_first(arr, n):
        return arr[:, :n]

    @staticmethod
    def to_fp32(arr, n):
        return arr.astype(np.float32)

    @staticmethod
    def _pregen_fp(arr, **kwargs):
        if kwargs.get('unpackbits', True):
            arr = np.unpackbits(arr, axis=-1)
        top_k_imp = kwargs.get('top_k_imp', None)
        if top_k_imp is not None and top_k_imp > 0:
            label_idxs = kwargs['label_idxs']
            if len(label_idxs) == 3:
                feat_idxs = kwargs['feat_imp']
            elif len(label_idxs) == 1:
                feat_idxs = kwargs[f'feat_imp_{label_idxs[0]}']
            else:
                raise ValueError
            feat_idxs = feat_idxs[:top_k_imp]
            arr = arr[:, feat_idxs]
        input_ids = torch.from_numpy(arr).float()
        return {
            'input_ids': input_ids,
            'padding_mask': torch.ones_like(input_ids, dtype=torch.bool)
        }

    @staticmethod
    def ecfp6(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def maccs(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def lingo(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def map(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def mhfp(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def physio_chemical(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def topological_torsion(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def rdkit(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def ecfp_2_1024(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def ecfp_2_512(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def _(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def _(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def _(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def _(arr, **kwargs):
        return Pipeline._pregen_fp(arr, **kwargs)

    @staticmethod
    def trim_padding(arr, pad_idx=0):
        num_samples, num_cols = arr.shape
        first_non_zero_col_idx = num_cols - 1
        while not np.any(arr[:, first_non_zero_col_idx] != pad_idx):
            first_non_zero_col_idx -= 1
        # assert np.any(arr[:, first_non_zero_col_idx])
        # assert not np.any(arr[:, first_non_zero_col_idx + 1])
        return arr[:, :first_non_zero_col_idx + 1]

    @staticmethod
    def _pregen_tokenize(arr, **kwargs):
        pad_idx = kwargs.get('pad_idx', 0)
        if not kwargs.get('fixed_len', True):
            arr = Pipeline.trim_padding(arr, pad_idx=pad_idx)
        fmt = kwargs.get('fmt', 'seq')
        if fmt == 'seq':
            input_ids = torch.from_numpy(arr).long()
            padding_mask = (input_ids != pad_idx).bool()
        elif fmt == 'onehot':
            vocab_size = kwargs['vocab_size']
            input_ids = torch.zeros((arr.shape[0], vocab_size),
                                    dtype=torch.float32)
            for row_idx, row in enumerate(arr):
                try:
                    input_ids[row_idx, row.tolist()] = 1
                except:
                    print(row_idx, row, input_ids, sep='\n')
                    raise AssertionError
            input_ids = input_ids[:, 1:]
            padding_mask = torch.ones_like(input_ids)
        else:
            raise ValueError
        return {'input_ids': input_ids, 'padding_mask': padding_mask}

    @staticmethod
    def smiles_char(arr, **kwargs):
        return Pipeline._pregen_tokenize(arr, **kwargs)

    @staticmethod
    def ais(arr, **kwargs):
        batch = Pipeline._pregen_tokenize(arr, **kwargs)
        if kwargs.get('rm_unk', False):
            # AIS specific behavior
            # The pregenerated data's vocab is built on train + test data
            # To prevent OOV, we simply ignore those non-trained tokens
            NON_TRAINED_TOKEN_IDS = torch.tensor([105, 111, 146, 151, 155, 161, 162, 163, 194, 195, 197, 198, 219, 220])
            input_ids = batch['input_ids']
            padding_mask = batch['padding_mask']
            old_shapes = [input_ids.shape, padding_mask.shape]

            new_input_ids = torch.zeros_like(input_ids)
            new_padding_mask = torch.zeros_like(padding_mask)

            for i in range(input_ids.shape[0]):
                row_mask = torch.logical_not(torch.isin(input_ids[i], NON_TRAINED_TOKEN_IDS))
                n_keep = row_mask.sum()
                new_input_ids[i][:n_keep] = input_ids[i][row_mask]
                new_padding_mask[i][:n_keep] = padding_mask[i][row_mask]
                # if n_keep < input_ids.shape[1]:
                #     print('INPUT_IDS:', input_ids[i], new_input_ids[i], sep = '\n')
                #     print('PADDING MASK:', padding_mask[i], new_padding_mask[i], sep = '\n')


            # # ignore redundant [PAD] at the end
            # num_samples, num_cols = new_input_ids.shape
            # first_non_zero_col_idx = num_cols - 1
            # while not torch.any(new_input_ids[:, first_non_zero_col_idx] != kwargs['pad_idx']):
            #     first_non_zero_col_idx -= 1
            # assert torch.any(new_input_ids[:, first_non_zero_col_idx])
            # if first_non_zero_col_idx + 1 < num_cols:
            #     assert not torch.any(new_input_ids[:, first_non_zero_col_idx + 1])
            # new_input_ids = new_input_ids[:, :first_non_zero_col_idx + 1]
            # new_padding_mask = new_padding_mask[:, :first_non_zero_col_idx + 1]
            
            new_shapes = [new_input_ids.shape, new_padding_mask.shape]
            logger.warning('Remove UNK: %s -> %s', old_shapes, new_shapes)
            return {
                'input_ids': new_input_ids,
                'padding_mask': new_padding_mask,
            }
        return batch
            



    @staticmethod
    def selfies(arr, **kwargs):
        return Pipeline._pregen_tokenize(arr, **kwargs)

    @staticmethod
    def deepsmiles(arr, **kwargs):
        return Pipeline._pregen_tokenize(arr, **kwargs)


class PregenDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD TARGETS
        if stage != 'predict':
            self.targets = cache['train_targets']

        # LOAD FEATURES
        self.features = []
        for feature_name in cfg.data.features:
            feature_arr = cache[f'{self._stage}_{feature_name}']
            process_func = getattr(Pipeline, feature_name)
            process_func_kwargs = dict(
                cfg.data.all_features.get(feature_name).kwargs)
            process_func_kwargs['label_idxs'] = self.label_idxs
            process_func = partial(process_func, **process_func_kwargs)
            self.features.append([feature_name, feature_arr, process_func])

    def get_collater(self):
        # return partial(padding_collater,
        #                max_len=4096,
        #                keys=['input_ids', 'padding_mask'],
        #                pad_values=[0, 0],
        #                padding=True)
        return lambda x: x

    @property
    def getitem_as_batch(self):
        return True

    def __getitem__(self, idxs):
        """
        Generate one batch of data.
        """
        ridxs = self.idxs[idxs]
        batch = {'idx': torch.tensor(idxs)}

        features = []
        for feature_name, feature_arr, feature_func in self.features:
            feature_arr = feature_arr[ridxs]
            feature_dict = feature_func(feature_arr)
            features.append([feature_name, feature_dict])
        if self.cfg.data.return_as == 'dict':
            features = dict(features)
            batch.update(features)
        elif self.cfg.data.return_as == 'concat':
            if len(features) == 1:
                feature_dict = features[0][1]
            else:
                # raise NotImplementedError
                input_ids = torch.cat([d[1]['input_ids'] for d in features],
                                      dim=-1)
                padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
                feature_dict = {
                    'input_ids': input_ids,
                    "padding_mask": padding_mask
                }
            batch.update(feature_dict)
        else:
            raise ValueError

        if self.stage != 'predict':
            batch['target'] = torch.from_numpy(
                self.targets[ridxs][:, self.label_idxs]).float()
        return batch

    @classmethod
    def load_cache(cls, cfg):
        cache = {}
        data_dir = cfg.env.data_dir

        stages = []
        if cfg.train or cfg.test:
            stages.append('train')
        if cfg.predict:
            stages.append('test')

        # LOAD FEATURES
        for feature_name in cfg.data.features:
            feature_dir = os.path.join(data_dir, 'processed', 'features',
                                       feature_name)
            for stage in stages:
                mmap_mode = 'r' if cfg.data.all_features.get(
                    feature_name).backend == 'mmap' else None
                features = np.load(os.path.join(feature_dir, f'{stage}.npy'),
                                   mmap_mode=mmap_mode)
                cache[f'{stage}_{feature_name}'] = features

        # LOAD TARGET
        if 'train' in stages:
            if cfg.data.target.transform is not None:
                logger.info('Using target transform %s', cfg.data.target.transform)
                train_df = pl.scan_csv(
                    os.path.join(data_dir, 'processed',
                                 'train_v3.csv')).select(
                                     pl.col('BRD4', 'HSA',
                                            'sEH').cast(pl.UInt8),
                                     pl.col('BRD4_baseline', 'HSA_baseline',
                                            'sEH_baseline'))

                def _label_smooth_polars(x,
                                         smooth_base=0.05,
                                         smooth_mul=0.15,
                                         label=0,
                                         cutoff=1.0):
                    # cutoff is selected as 95/99% percentile
                    assert smooth_base + smooth_mul < 0.5
                    if label == 0:
                        return smooth_base + pl.min_horizontal(
                            x, cutoff) * (smooth_mul / cutoff)
                    else:
                        return (1.0 -
                                smooth_base - smooth_mul) + pl.min_horizontal(
                                    x, cutoff) * (smooth_mul / cutoff)

                # 99% percentile
                # CUTOFF_DICT = {}
                # for protein in cls.PROTEINS:
                #     CUTOFF_DICT[protein] = {}
                #     for label in [0, 1]:
                #         tmp = train_df.filter(pl.col(protein) == label)[f'{protein}_baseline']
                #         CUTOFF_DICT[protein][label] = tmp.quantile(0.99)
                CUTOFF_DICT = {
                    'BRD4': {
                        0: 0.00017149973831406243,
                        1: 0.012792630279480384
                    },
                    'HSA': {
                        0: 0.0009380243193755984,
                        1: 0.033118138349205455
                    },
                    'sEH': {
                        0: 8.436829650460108e-05,
                        1: 0.011706071790051883
                    }
                }
                train_df = train_df.with_columns(*[
                    pl.when(pl.col(protein) == 0).then(
                        _label_smooth_polars(pl.col(f'{protein}_baseline'),
                                             smooth_base=cfg.data.target.smooth_base,
                                             smooth_mul=cfg.data.target.smooth_mul,
                                             label=0,
                                             cutoff=CUTOFF_DICT[protein][0])).
                    otherwise(
                        _label_smooth_polars(pl.col(f'{protein}_baseline'),
                                             smooth_base=cfg.data.target.smooth_base,
                                             smooth_mul=cfg.data.target.smooth_mul,
                                             label=1,
                                             cutoff=CUTOFF_DICT[protein]
                                             [1])).alias(f'_smooth_label_{protein}')
                    for protein in cls.PROTEINS
                ])
                train_targets = train_df.select(*[f'_smooth_label_{protein}' for protein in cls.PROTEINS]).collect()
                print(train_targets.columns)
                print(train_targets.describe())
                cache['train_targets']  = train_targets.to_numpy()
                del train_df
                gc.collect()

            else:
                cache['train_targets'] = pl.scan_csv(
                    os.path.join(data_dir, 'processed',
                                 'train_v2.csv')).select(
                                     # pl.col('molecule'),
                                     pl.col('BRD4', 'HSA', 'sEH').cast(
                                         pl.UInt8), ).collect().to_numpy()

        return cache
