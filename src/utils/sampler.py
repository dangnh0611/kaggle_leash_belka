# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/samplers.py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import WeightedRandomSampler
import itertools
from typing import Optional
import logging
import numpy as np
from src.utils.misc import local_numpy_temp_seed
import math
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will prepend a dimension, whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """This function randomly changes the the input dimension of the dataset."""
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None,
                                    self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


class WeightedInfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        weights,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self.weights = weights
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None,
                                    self._world_size)

    def _infinite_indices(self):
        g = None
        # g = torch.Generator()
        # logger.info('SAMPLER MANUAL SEED %d', self._seed)
        # g.manual_seed(self._seed)

        sampler = WeightedRandomSampler(self.weights,
                                        self._size,
                                        replacement=True,
                                        generator=g)

        while True:
            yield from sampler

    def __len__(self):
        return self._size // self._world_size


class BalanceSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        max_epochs,
        num_linear_epochs,
        num_constant_epochs,
        start_ratio=None,
        end_ratio=None,
        constant_ratio=None,
        one_pos_mode=True,
        random_seed=42,
    ):
        """
        BalanceSampler based on targets, useful in a strong classes imbalance scenario.
        Ensure each batch contains `ratio` of positive samples.
        `ratio` is linearly increase/decrease from `start_ratio` to `end_ratio` in `num_linear_epochs`,
        then keep constant/fixed at `constant_ratio` for `num_constant_epochs`,
        finally, use Pytorch's default RandomSampler in the remaining epochs

                          \ <-- start_ratio=0.25
                           \
                            \
         end_ratio=0.05 -->  \
              
                              __________ <-- constant_ratio=0.02

                                        ________________ <-- original_ratio=0.0015 (RandomSampler)
        Args:
            dataset: a dataset object which implement self.get_labels() -> List[int],
                where label=0 indicate negatives, >0 indicate positives
            batch_size: fixed batch size used for training
            max_epochs: expected total number of epochs
            num_linear_epochs: number of epoch to linearly increase/decrease positive ratio from `start_ratio` to `end_ratio`
            num_constant_epochs: number of epoch to keep positive ratio constant at `constant ratio`
            start_ratio: start positive ratio, if is None, use the dataset positive ratio
            end_ratio: end positive ratio, if is None, use the dataset positive ratio
            constant_ratio: constant ratio used for `num_constant_epochs` (constant phase) after linear phase
            one_pos_mode: ensure at least on positive sample each batch
            seed: random seed
        """
        assert min(num_linear_epochs, num_constant_epochs) >= 0
        self.random_seed = random_seed
        self.one_pos_mode = one_pos_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.constant_ratio = constant_ratio
        self.num_linear_epochs = num_linear_epochs
        self.num_constant_epochs = num_constant_epochs
        self.max_epochs = max_epochs
        self.num_cooldown_epochs = max_epochs - num_linear_epochs - num_constant_epochs
        assert self.num_cooldown_epochs >= 0
        labels = dataset.get_labels()

        self.pos_idxs = np.where(labels > 0)[0]
        self.neg_idxs = np.where(labels == 0)[0]
        self.num_pos = len(self.pos_idxs)
        self.num_neg = len(self.neg_idxs)
        self.ori_ratio = self.num_pos / (self.num_neg + self.num_pos)
        logger.info('[SAMPLER] num_pos=%d num_neg=%d pos_ratio=%f',
                    self.num_pos, self.num_neg, round(self.ori_ratio, 6))

        # percentage of pos samples per epoch
        ratios = []
        if self.num_linear_epochs > 0:
            assert start_ratio is not None or end_ratio is not None
            start_ratio = start_ratio if start_ratio is not None else self.ori_ratio
            end_ratio = end_ratio if end_ratio is not None else self.ori_ratio
            _ratios = np.linspace(start_ratio, end_ratio,
                                  self.num_linear_epochs)
            ratios.extend(_ratios)

        if self.num_constant_epochs > 0:
            constant_ratio = constant_ratio if constant_ratio is not None else self.ori_ratio
            ratios.extend([constant_ratio] * self.num_constant_epochs)

        if self.num_cooldown_epochs > 0:
            ratios.extend([-1] * self.num_cooldown_epochs)
        assert len(ratios) == self.max_epochs
        self.ratios = ratios
        logger.info('[SAMPLER] positive ratio per epochs: %s',
                    [(ep, ratio) for ep, ratio in enumerate(self.ratios)])

        self.cur_epoch = None
        self._num_samples = None
        self.set_epoch(0)

    def set_epoch(self, ep):
        assert ep < self.max_epochs, f'[SAMPLER] Invalid set_epoch() with ep={ep} while max_epoch={self.max_epochs}'
        print(f'Set epoch to {ep} with sampler ratio = {self.ratios[ep]}')
        self.cur_epoch = ep
        ratio = self.ratios[self.cur_epoch]
        seed = self.random_seed + self.cur_epoch
        with local_numpy_temp_seed(seed):
            epoch_idxs = self._compute_epoch_idxs(
                ratio, one_pos_mode=self.one_pos_mode)
        logger.info('[SAMPLER] epoch=%d ratio=%f seed=%d samples=%d iters=%d',
                    self.cur_epoch, ratio, seed, len(epoch_idxs),
                    math.ceil(len(epoch_idxs) / self.batch_size))
        self.cur_epoch_idxs = epoch_idxs

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime by setting this _num_samples attribute
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def _compute_epoch_idxs(self, ratio, one_pos_mode=True):
        if ratio < 0:
            # Use default Random Sampler
            logger.info('[SAMPLER] ratio=%f <0, use default RandomSampler..',
                        ratio)
            n = len(self.dataset)
            num_samples = self.num_samples
            epoch_idxs = []
            for _ in range(num_samples // n):
                epoch_idxs.extend(np.random.permutation(n))
            if num_samples % n > 0:
                epoch_idxs.extend(np.random.permutation(n)[:num_samples % n])
            return epoch_idxs

        ratio = ratio / (1 - ratio)
        epoch_num_pos = int(ratio * self.num_neg)
        epoch_num_iters = math.ceil(
            (epoch_num_pos + self.num_neg) / self.batch_size)
        epoch_num_total = epoch_num_iters * self.batch_size
        # never downsampling neg samples if possible
        epoch_num_pos = epoch_num_total - self.num_neg

        # sampling pos idxs
        min_pos_per_iter = epoch_num_pos // epoch_num_iters
        if min_pos_per_iter < 1:
            if one_pos_mode:
                logger.info(
                    '[SAMPLER] one_pos_mode=True, switch num_pos_samples from %d to %d',
                    epoch_num_pos, epoch_num_iters)
                min_pos_per_iter = 1
                epoch_num_pos = epoch_num_iters
            else:
                logger.info(
                    "[SAMPLER] one_pos_mode=False, at least one batch which has no positive sample: num_pos=%d, num_iters=%d",
                    epoch_num_pos, epoch_num_iters)

        ret_idxs = []
        pool_pos_idxs = []
        _count = 0
        while _count < epoch_num_pos:
            temp_pos_idxs = self.pos_idxs.copy()
            np.random.shuffle(temp_pos_idxs)
            pool_pos_idxs.append(temp_pos_idxs)
            _count += len(temp_pos_idxs)
        pool_pos_idxs = np.concatenate(pool_pos_idxs, axis=0)
        assert len(pool_pos_idxs) >= epoch_num_pos

        _start = 0
        _end = 0
        for i in range(epoch_num_iters):
            _start = i * min_pos_per_iter
            _end = (i + 1) * min_pos_per_iter
            ret_idxs.append(pool_pos_idxs[_start:_end].tolist())
        num_pos_remain = epoch_num_pos - _end
        assert num_pos_remain == epoch_num_pos % epoch_num_iters
        pool_remain_pos_idxs = pool_pos_idxs[_end:epoch_num_pos]
        assert len(pool_remain_pos_idxs) == num_pos_remain
        for i, j in enumerate(
                np.random.choice(np.arange(0, epoch_num_iters, 1),
                                 num_pos_remain,
                                 replace=False)):
            ret_idxs[j].append(pool_remain_pos_idxs[i])

        # sampling neg idxs
        pool_neg_idxs = self.neg_idxs.copy()
        np.random.shuffle(pool_neg_idxs)

        _cur = 0
        for i in range(epoch_num_iters):
            iter_idxs = ret_idxs[i]
            assert len(iter_idxs) - min_pos_per_iter <= 1
            _end = _cur + self.batch_size - len(iter_idxs)
            iter_idxs.extend(pool_neg_idxs[_cur:_end].tolist())
            _cur = _end
        if not one_pos_mode:
            assert _cur == len(pool_neg_idxs)

        ret_idxs = np.array(ret_idxs)
        assert ret_idxs.shape[0] == epoch_num_iters and ret_idxs.shape[
            1] == self.batch_size
        ret_idxs = ret_idxs.reshape(-1)
        return ret_idxs

    def __iter__(self):
        return iter(self.cur_epoch_idxs)

    def __len__(self):
        return len(self.cur_epoch_idxs)


# from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() /
                                       torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        logger.info('Done generating rand tensor of shape %s', rand_tensor.shape)
        return iter(rand_tensor)


class WeightedRandomSamplerWrapper(CustomWeightedRandomSampler):

    def __init__(self, dataset, replacement=True):
        super().__init__(weights=dataset.compute_sampling_weights(),
                         num_samples=len(dataset),
                         replacement=replacement,
                         generator=None)
