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
from rdkit import Chem
from rdkit.Chem import AllChem
import atomInSmiles
import selfies
from deepsmiles.encode import encode as deepsmiles_encode
from transformers import AutoTokenizer
from src.utils.chem import normalize_smiles, SMILES_CONVERTERS
import random

logger = logging.getLogger(__name__)

TOKENIZER2FMT = {
    'smiles_char': 'smiles',
    'ais_train': 'ais',
    'ais_test': 'ais',
    'selfies': 'selfies',
    'deepsmiles': 'deepsmiles',
}


class TokenizeDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD DATAFRAME
        self.ds = cache[f'{self._stage}_ds']

        # BUILD TOKENIZER
        tokenizer_path = os.path.join(self.data_dir, 'processed',
                                      'tokenizer_v2',
                                      self.cfg.data.tokenizer.name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True)
        tokenizer_input_fmt = TOKENIZER2FMT[self.cfg.data.tokenizer.name]
        self.smiles_to_tokenizer_input_converter = SMILES_CONVERTERS[
            tokenizer_input_fmt]
        logger.info('TOKENIZER:\n%s', self.tokenizer)
        logger.info('SMILES TO TOKENIZER INPUT (%s) CONVERTER: %s',
                    tokenizer_input_fmt,
                    self.smiles_to_tokenizer_input_converter)
        self.template = cfg.data.template
        logger.info('Using sequence template: `%s`', self.template)

    def get_labels(self):
        targets = np.stack(
            [self.ds[self.idxs][protein] for protein in self.PROTEINS],
            axis=-1)
        labels = np.any(targets, axis=-1).astype(np.uint8)
        return labels

    def get_collater(self):
        return lambda x: x

    @property
    def getitem_as_batch(self):
        return True

    def _get_sample_labels(self, samples):
        labels = np.stack(
            [samples[label_name] for label_name in self.label_cols], axis=-1)
        return labels

    def apply_fmt(self, s):
        return self.template.replace('{0}', s)

    def __getitem__(self, idxs):
        """
        Generate one batch of data.
        """
        ridxs = self.idxs[idxs]
        samples = self.ds[ridxs]

        # ChemBERTa & Molformer accept SMILES with isomeric = False
        _smiles_list = samples['smiles']

        smiles_list = []
        for smiles in _smiles_list:
            if self.stage == 'train':
                do_random = random.random() < self.cfg.data.do_random
            else:
                do_random = False
            smiles_list.append(
                normalize_smiles(smiles,
                                 canonical=not do_random,
                                 do_random=do_random,
                                 isomeric=self.cfg.data.isomeric,
                                 kekulize = self.cfg.data.kekulize,
                                 replace_dy=self.cfg.data.replace_dy,
                                 return_mol=False))

        tok_inputs = [
            self.apply_fmt(self.smiles_to_tokenizer_input_converter(smiles))
            for smiles in smiles_list
        ]

        # assert all([s==self.apply_fmt(_s) for s, _s in zip(tok_inputs, _smiles_list)])

        ret = self.tokenizer(
            tok_inputs,
            add_special_tokens=True,
            padding=self.cfg.data.tokenizer.padding,
            truncation=False,
            max_length=self.cfg.data.tokenizer.max_length,
            is_split_into_words=self.cfg.data.tokenizer.is_split_into_words,
            pad_to_multiple_of=self.cfg.data.tokenizer.pad_to_multiple_of,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_special_tokens_mask=False,
            return_length=True,
            verbose=True)

        batch = {
            'idx': torch.tensor(idxs),
            'input_ids': ret['input_ids'].long(),
            'padding_mask': ret['attention_mask'].bool(),
            'length': ret['length']
        }

        if self.stage != 'predict':
            # this cause deadlock when num_workers > 0
            # target = self.df[ridxs, ['BRD4', 'HSA', 'sEH']].to_numpy()
            # but this did not, why Polars?
            # target = np.stack([
            #     self.df[ridxs, 'BRD4'],
            #     self.df[ridxs, 'HSA'],
            #     self.df[ridxs, 'sEH']
            # ], axis=-1)
            batch['target'] = torch.from_numpy(
                self._get_sample_labels(samples)).float()
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

        # LOAD TARGET
        for stage in stages:
            from datasets import load_from_disk
            cache[f'{stage}_ds'] = load_from_disk(
                os.path.join(data_dir, 'processed', 'hf', 'datasets', stage),
                keep_in_memory=cfg.data.ram_cache)

        return cache
