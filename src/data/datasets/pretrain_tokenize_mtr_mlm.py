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
import math
import random

logger = logging.getLogger(__name__)

TOKENIZER2FMT = {
    'smiles_char': 'smiles',
    'ais_train': 'ais',
    'ais_test': 'ais',
    'selfies': 'selfies',
    'deepsmiles': 'deepsmiles',
}
TRAIN_LEN = 98_415_610
TEST_LEN = 878_022
RDKIT_DESCRIPTORS = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex',
    'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
    'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI',
    'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ',
    'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
    'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
    'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10',
    'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
    'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3',
    'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
    'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
    'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
    'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',
    'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
    'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
    'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
    'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
    'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
    'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH',
    'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',
    'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
    'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
    'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
    'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid',
    'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',
    'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
    'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
    'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
    'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
    'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
    'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
    'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
]
KEEP_RDKIT_DESCRIPTORS = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex',
    'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
    'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
    'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
    'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0',
    'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',
    'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2',
    'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11',
    'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3',
    'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
    'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1',
    'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3',
    'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
    'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
    'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
    'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',
    'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount',
    'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
    'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'RingCount', 'MolLogP', 'MolMR', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
    'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
    'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
    'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
    'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
    'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_benzene', 'fr_bicyclic',
    'fr_ester', 'fr_ether', 'fr_furan', 'fr_halogen', 'fr_hdrzone',
    'fr_imidazole', 'fr_imide', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone',
    'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
    'fr_thiazole', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
]
KEEP_RDKIT_DESCRIPTOR_IDXS = [
    RDKIT_DESCRIPTORS.index(e) for e in KEEP_RDKIT_DESCRIPTORS
]
KEEP_RDKIT_DESCRIPTOR_MINMAXS = [[11.2, 15.98], [11.2, 15.98], [0.0, 0.9526],
                                 [-6.152, 0.1746], [0.01066, 0.9424],
                                 [8.96, 60.84], [273.2,
                                                 1241.0], [254.1, 1209.0],
                                 [273.2, 1241.0], [106.0, 370.0],
                                 [0.2249, 0.586], [-0.6187, -0.3538],
                                 [0.3538, 0.6187], [0.2249, 0.5728],
                                 [0.4, 1.619], [0.7256, 2.4], [0.976, 3.172],
                                 [16.16, 126.94], [9.49, 10.57],
                                 [2.064, 2.889], [-2.566, -2.068],
                                 [2.082, 2.861], [-2.668, -2.156],
                                 [5.754, 14.12], [-0.6055, 0.09674],
                                 [2.406, 4.242], [0.7295, 3.455],
                                 [460.0, 3634.0], [14.27,
                                                   51.0], [11.836, 40.28],
                                 [11.836, 42.4], [9.44, 35.4], [6.254, 24.14],
                                 [6.598, 25.52], [3.818, 20.02], [4.17, 25.23],
                                 [2.197, 18.6], [2.436, 20.19], [1.211, 18.16],
                                 [1.329, 20.73],
                                 [-9.88, 0.16], [10.516, 38.56], [13.15, 49.3],
                                 [4.86, 23.16], [2.033, 17.84], [116.5, 410.0],
                                 [5.316, 60.8], [0.0, 64.6], [0.0, 66.9],
                                 [0.0, 65.4], [0.0, 43.47], [0.0, 36.6],
                                 [4.793, 78.7], [0.0, 51.84], [0.0, 80.8],
                                 [0.0, 60.22], [0.0, 138.4], [0.0, 191.1],
                                 [7.047, 134.6], [0.0, 95.94], [4.793, 79.44],
                                 [11.81, 189.6], [0.0, 21.19], [10.22, 81.44],
                                 [0.0, 106.5], [0.0, 161.8], [13.59, 136.5],
                                 [0.0, 198.5], [0.0, 53.2], [5.316, 60.56],
                                 [0.0, 103.9], [0.0, 46.0], [0.0, 154.1],
                                 [27.9, 177.4], [0.0, 84.0], [0.0, 123.75],
                                 [0.0, 145.2], [0.0, 176.8], [0.0, 54.47],
                                 [0.0, 61.44], [49.4, 356.8], [0.0, 141.0],
                                 [4.793, 79.44], [0.0, 21.95], [0.0, 122.8],
                                 [0.0, 130.8], [0.0, 150.0], [0.0, 113.7],
                                 [0.0, 139.9], [0.0, 175.8], [5.316, 166.8],
                                 [0.0, 90.7], [-0.03653, 232.6], [0.0, 46.38],
                                 [21.02, 160.4], [1.85, 68.8], [-12.29, 22.3],
                                 [-28.36, 15.38], [-4.316, 51.22],
                                 [-26.8, 28.19], [-6.96,
                                                  26.66], [-10.51, 13.96],
                                 [0.02272, 0.8945], [20.0, 72.0], [1.0, 12.0],
                                 [4.0, 26.0], [0.0, 11.0], [0.0, 8.0],
                                 [0.0, 12.0], [0.0, 6.0], [0.0, 7.0],
                                 [1.0, 10.0], [2.0, 22.0], [1.0, 10.0],
                                 [4.0, 26.0], [3.0, 23.0], [0.0, 11.0],
                                 [0.0, 8.0], [0.0, 12.0], [1.0, 14.0],
                                 [-4.406, 12.57], [70.75, 270.2], [0.0, 3.0],
                                 [0.0, 3.0], [0.0, 1.0], [0.0,
                                                          1.0], [0.0, 15.0],
                                 [0.0, 5.0], [0.0, 4.0], [0.0,
                                                          1.0], [0.0, 1.0],
                                 [1.0, 10.0], [1.0, 10.0], [0.0, 2.0],
                                 [0.0, 3.0], [0.0, 2.0], [1.0, 15.0],
                                 [1.0, 10.0], [0.0, 3.0], [0.0,
                                                           4.0], [0.0, 4.0],
                                 [0.0, 3.0], [0.0, 5.0], [0.0,
                                                          2.0], [0.0, 2.0],
                                 [0.0, 2.0], [0.0, 15.0], [0.0, 5.0],
                                 [1.0, 10.0], [0.0, 2.0], [0.0,
                                                           9.0], [0.0, 8.0],
                                 [0.0, 1.0], [0.0, 6.0], [0.0,
                                                          11.0], [0.0, 4.0],
                                 [0.0, 8.0], [0.0, 3.0], [0.0,
                                                          17.0], [0.0, 1.0],
                                 [0.0, 5.0], [0.0, 4.0], [0.0,
                                                          3.0], [0.0, 3.0],
                                 [0.0, 2.0], [0.0, 8.0], [0.0,
                                                          3.0], [0.0, 4.0],
                                 [0.0, 3.0], [0.0, 3.0], [0.0,
                                                          3.0], [0.0, 2.0],
                                 [0.0, 2.0], [0.0, 5.0], [0.0,
                                                          3.0], [0.0, 3.0],
                                 [0.0, 6.0], [0.0, 6.0], [0.0,
                                                          2.0], [0.0, 5.0],
                                 [0.0, 5.0], [0.0, 2.0], [0.0,
                                                          3.0], [0.0, 3.0],
                                 [0.0, 2.0], [0.0, 3.0], [0.0,
                                                          3.0], [0.0, 9.0],
                                 [0.0, 2.0]]


class PretrainTokenizeJoinMTRMLMDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD DATAFRAME
        self.ds = cache['ds']
        self.train_mtr_labels = cache['train_mtr_labels']
        self.test_mtr_labels = cache['test_mtr_labels']

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

        # MTR related
        self.mtr_label_idxs = KEEP_RDKIT_DESCRIPTOR_IDXS
        self.mtr_label_num = len(self.mtr_label_idxs)
        self.mtr_label_mins = np.array(
            [e[0] for e in KEEP_RDKIT_DESCRIPTOR_MINMAXS])
        self.mtr_label_maxs = np.array(
            [e[1] for e in KEEP_RDKIT_DESCRIPTOR_MINMAXS])
        self.mtr_label_divs = self.mtr_label_maxs - self.mtr_label_mins

        # MLM related
        from src.utils.misc import MLMMasker
        self.mlm_masker = MLMMasker(self.tokenizer,
                                    mlm_prob=cfg.data.mlm_prob,
                                    mask_prob=cfg.data.mask_prob,
                                    random_prob=cfg.data.random_prob)
        logger.info('MLM Masker: %s', self.mlm_masker)

    def _get_stage_idxs(self, cfg, stage):
        assert stage == 'train'
        data_dir = cfg.env.data_dir

        train_idxs = np.arange(0, TRAIN_LEN, 1)
        assert train_idxs.shape[0] == TRAIN_LEN
        idxs = [train_idxs]
        test_df = pl.scan_csv(
            os.path.join(data_dir, 'processed',
                         'test_v4.csv')).with_row_index('index').select(
                             (pl.col('index') +
                              len(train_idxs)).alias('index'),
                             pl.col('mol_group')).collect()
        assert test_df.shape[0] == TEST_LEN
        # ┌───────────┬────────┐
        # │ mol_group ┆ count  │
        # │ ---       ┆ ---    │
        # │ i64       ┆ u32    │
        # ╞═══════════╪════════╡
        # │ 0         ┆ 369039 │
        # │ 1         ┆ 486390 │
        # │ 2         ┆ 11271  │
        # │ 3         ┆ 11322  │
        # └───────────┴────────┘
        group_ids = [0, 1, 2, 3]
        group_weights = [1, 25, 280, 280]
        for group_id, group_weight in zip(group_ids, group_weights):
            group_idxs = test_df.filter(
                pl.col('mol_group') == group_id)['index'].to_numpy()
            idxs.extend([group_idxs] * group_weight)

        idxs = np.concatenate(idxs, axis=0)
        return idxs, None, None

    def get_labels(self):
        raise NotImplementedError

    def get_collater(self):
        return lambda x: x

    @property
    def getitem_as_batch(self):
        return True

    def _get_mtr_target(self, idxs):
        labels = np.zeros((len(idxs), self.mtr_label_num), dtype=np.float32)
        # read chunk from same file is more efficient?
        train_rel_idxs = []
        train_idxs = []
        test_rel_idxs = []
        test_idxs = []
        for i, idx in enumerate(idxs):
            if idx < TRAIN_LEN:
                train_rel_idxs.append(i)
                train_idxs.append(idx)
            else:
                test_rel_idxs.append(i)
                test_idxs.append(idx - TRAIN_LEN)

        if train_idxs:
            labels[train_rel_idxs] = self.train_mtr_labels[
                train_idxs][:, self.mtr_label_idxs]
        if test_idxs:
            labels[test_rel_idxs] = self.test_mtr_labels[
                test_idxs][:, self.mtr_label_idxs]

        # normalize to [0, 1] by simple min-max scale
        labels = (labels - self.mtr_label_mins) / self.mtr_label_divs
        return labels

    def apply_fmt(self, s):
        return f'[CLS]{s}'

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
            do_random = random.random() < self.cfg.data.do_random
            smiles_list.append(
                normalize_smiles(smiles,
                                 canonical=not do_random,
                                 do_random=do_random,
                                 isomeric=self.cfg.data.isomeric,
                                 replace_dy=self.cfg.data.replace_dy,
                                 return_mol=False))

        tok_inputs = [
            self.apply_fmt(self.smiles_to_tokenizer_input_converter(smiles))
            for smiles in smiles_list
        ]

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

        mtr_target = torch.from_numpy(self._get_mtr_target(ridxs)).float()
        # we pass special_tokens_mask=None here
        # since tokenizer.__call__(return_special_tokens_mask=True) did not work as expected
        # i.e, return wrong special tokens mask
        # but tokenizer.get_special_tokens_mask() works normally
        input_ids, mlm_target = self.mlm_masker(ret['input_ids'],
                                                special_tokens_mask=None)

        batch = {
            'idx': torch.tensor(idxs),
            'input_ids': input_ids.long(),
            'padding_mask': ret['attention_mask'].bool(),
            'length': ret['length'],
            'mtr_target': mtr_target,
            'mlm_target': mlm_target,
        }
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
        from datasets import load_from_disk, concatenate_datasets
        ds = load_from_disk(os.path.join(data_dir, 'processed', 'hf',
                                         'datasets', 'all'),
                            keep_in_memory=cfg.data.ram_cache)
        assert len(ds) == TRAIN_LEN + TEST_LEN

        cache['ds'] = ds

        # MTR label
        cache['train_mtr_labels'] = np.load(os.path.join(
            data_dir, 'processed', 'features', 'rdkit210', 'train.npy'),
                                            mmap_mode='r')
        cache['test_mtr_labels'] = np.load(os.path.join(
            data_dir, 'processed', 'features', 'rdkit210', 'test.npy'),
                                           mmap_mode='r')

        return cache
