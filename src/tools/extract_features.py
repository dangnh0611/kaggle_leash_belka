from setproctitle import setproctitle

setproctitle("python3 extract_frames.py")

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace, Split, ByteLevel, WhitespaceSplit
from tokenizers.normalizers import Lowercase, NFKC
import os
import polars as pl
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

np.float = np.float64
from tqdm import tqdm
import time
import json
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
import gc
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
# from rdkit.Chem.AllChem import Descriptors
from mordred import Calculator as MordredCalculator
from mordred import descriptors as mordred_descriptors
import math
import mapply
import argparse
from functools import partial
from skfp import fingerprints as skfps

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


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--num-chunks',
                        type=int,
                        default=1,
                        help='Number of chunks')
    parser.add_argument('--chunk-idx',
                        type=int,
                        default=0,
                        help='Current chunk index to process')
    parser.add_argument('--batch-size',
                        type=int,
                        default=5000,
                        help='Batch size for each worker')
    parser.add_argument('--workers',
                        type=int,
                        default=-1,
                        help='Number of parallel workers')
    parser.add_argument('--subset',
                        type=str,
                        default='train',
                        help='Subset to use')
    parser.add_argument('--feature',
                        type=str,
                        default='rdkit210',
                        help='Feature name to extract')

    return parser.parse_args()


def replace_dy(smiles, return_mol=False):
    #Convert your SMILES to a mol object.
    mol = Chem.MolFromSmiles(smiles)

    #Create a mol object to replace the Dy atom with.
    new_attachment = Chem.MolFromSmiles('C')

    #Get the pattern for the Dy atom
    dy_pattern = Chem.MolFromSmiles('[Dy]')

    #This returns a tuple of all possible replacements, but we know there will only be one.
    new_mol = AllChem.ReplaceSubstructs(mol, dy_pattern, new_attachment)[0]

    #Good idea to clean it up
    Chem.SanitizeMol(new_mol)

    # #Since you want 3D mols later, I'd suggest adding hydrogens. Note: this takes up a lot more memory for the obj.
    # Chem.AddHs(new_mol)
    if return_mol:
        return new_mol
    else:
        return Chem.MolToSmiles(new_mol)


def extract_features(func,
                     inputs,
                     save_dir,
                     feature_name,
                     subset,
                     method='batch',
                     num_workers=-1,
                     joblib_backend='loky',
                     batch_size=None,
                     save_formats=['npy'],
                     hook=None):
    if method == 'batch':
        assert batch_size is not None and batch_size > 0
        num_samples = len(inputs)
        starts = np.arange(0, num_samples, batch_size)
        ends = [min(start + batch_size, num_samples) for start in starts]
        ret = Parallel(n_jobs=num_workers, backend=joblib_backend)(
            delayed(func)(inputs[starts[i]:ends[i]])
            for i in tqdm(range(len(starts))))
        ret = np.concatenate(ret, axis=0)
        if hook is not None:
            ret = hook(ret)
        print(f'SHAPE={ret.shape} DTYPE={ret.dtype}')

        for fmt in save_formats:
            save_name = f'{subset}.{fmt}'
            save_path = os.path.join(save_dir, feature_name, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f'Saving {fmt} at {save_path}')

            start = time.time()
            if fmt == 'mmap':
                fp = np.memmap(save_path,
                               dtype=ret.dtype,
                               mode='w+',
                               shape=ret.shape)
                fp[:] = ret[:]
            elif fmt == 'npy':
                np.save(save_path, ret)
            else:
                raise ValueError
            end = time.time()
            print('Take:', end - start, 's')

            meta_json_path = os.path.join(save_dir, feature_name, 'meta.json')
            try:
                with open(meta_json_path, 'r') as f:
                    meta = json.load(f)
            except:
                meta = {}
            with open(meta_json_path, 'w') as f:
                meta[save_name] = {
                    'fmt': fmt,
                    'dtype': str(ret.dtype),
                    'shape': list(ret.shape),
                }
                json.dump(meta, f)
                print(meta)
    elif method == 'element':
        pass


def batch_calc_rdkit_descriptors(smiles_list):
    ret = []
    for smiles in smiles_list:
        desc = Descriptors.CalcMolDescriptors(
            replace_dy(smiles, return_mol=True))
        desc = [desc[col] for col in RDKIT_DESCRIPTORS]

        # # idx=42, name=Ipc, log() to prevent np.float16 Out Of Range casting
        desc[42] = math.log(desc[42])
        ret.append(desc)
    ret = np.array(ret, dtype=np.float16)
    return ret


MORDRED_KEEP_COLS = [
    'TIC1', 'GATS8se', 'Mp', 'n7HRing', 'VR2_Dzv', 'ETA_epsilon_2', 'SdsCH',
    'ATSC3p', 'nG12FRing', 'SpAbs_Dzpe', 'MDEO-22', 'SssNH', 'C2SP3', 'MID',
    'PEOE_VSA4', 'MDEC-44', 'MATS8p', 'Xp-0dv', 'GATS8are', 'n7FaRing',
    'AATS7v', 'GATS7v', 'SpAD_DzZ', 'nG12FHRing', 'NssssPb', 'SMR_VSA2',
    'JGI7', 'VR3_Dzv', 'AATS1are', 'ATSC0v', 'MINssCH2', 'SlogP_VSA8',
    'Xch-3dv', 'GATS8dv', 'PEOE_VSA9', 'MZ', 'NssCH2', 'AATS0d', 'ATSC5dv',
    'JGI2', 'AATS7m', 'SpAbs_Dzv', 'NssSe', 'GATS7i', 'SssssSi', 'GATS8m',
    'n12FAHRing', 'n7FaHRing', 'SlogP_VSA4', 'MAXaaaC', 'SLogP', 'GATS6i',
    'SpMAD_Dzpe', 'ATSC7s', 'NdCH2', 'ATS8d', 'GATS8Z', 'AATSC7Z', 'MATS1c',
    'VSA_EState5', 'ATSC8s', 'VE3_Dzp', 'ATSC0dv', 'AATS3are', 'MINddssS',
    'MATS2are', 'MATS7i', 'PEOE_VSA5', 'MATS5pe', 'n5Ring', 'apol', 'AATS7se',
    'ATSC6Z', 'GATS3i', 'GATS4c', 'ATS2v', 'AATSC4i', 'SpAD_Dzv', 'AATS3dv',
    'ATS1Z', 'SsssSiH', 'AATSC0pe', 'SsSnH3', 'GATS3v', 'Xc-5d', 'MAXsCH3',
    'VE2_DzZ', 'SsPbH3', 'VR2_Dzi', 'TMWC10', 'GATS3c', 'GATS3p', 'TIC2', 'nI',
    'AMID_C', 'ATSC5d', 'ATS6are', 'VE2_Dzi', 'nG12Ring', 'Xp-4dv', 'VR3_Dzi',
    'NsssSnH', 'AATS7dv', 'GATS5d', 'ATS0se', 'AATSC1Z', 'JGI8', 'ATS6Z',
    'NssNH2', 'AATSC7i', 'AATSC0s', 'GATS5se', 'GATS7s', 'VR3_Dzp', 'SpMAD_D',
    'ATSC3s', 'ATSC8c', 'ATS4dv', 'AATSC2Z', 'SdS', 'n7FHRing', 'MATS5i',
    'n7FARing', 'nG12ARing', 'SaaNH', 'VE1_Dzp', 'ATS7se', 'GATS3Z', 'MATS8c',
    'n12ARing', 'NsssNH', 'MATS1p', 'ATSC6se', 'MATS6pe', 'MINssNH', 'SdO',
    'CIC3', 'ATS3m', 'SZ', 'VR1_Dzp', 'AATS5p', 'Xch-5dv', 'nS', 'SsI',
    'MAXsssB', 'AATS8are', 'ATS6se', 'ATS5Z', 'ATS2p', 'AMID_N', 'SMR',
    'ATS7dv', 'BCUTse-1h', 'SRW02', 'SsssPbH', 'GGI4', 'MWC03', 'AATS0s',
    'AATS3Z', 'ATS8p', 'ATSC0p', 'ZMIC3', 'MIC3', 'MWC02', 'AATS2se',
    'SlogP_VSA1', 'Xpc-5dv', 'TopoPSA', 'ZMIC4', 'AATSC8c', 'GGI10', 'GATS3m',
    'BalabanJ', 'C3SP2', 'piPC5', 'n4aHRing', 'GATS8d', 'MAXdssS', 'AATS1m',
    'MDEN-33', 'MPC2', 'ATS7p', 'GATS4are', 'ATS3are', 'NsssGeH', 'MWC07',
    'ABC', 'ATS6d', 'n11aRing', 'ATSC7c', 'AATS0Z', 'n6ARing', 'ATS3d',
    'GATS8s', 'ATSC4pe', 'MINsssCH', 'NsCH3', 'bpol', 'MINtsC', 'SIC2',
    'MATS2i', 'Xch-6d', 'AATSC3are', 'AATSC8s', 'nSpiro', 'Mv', 'GATS7se',
    'MATS1are', 'SssSiH2', 'ATSC0are', 'SssssPb', 'naHRing', 'AATSC6s',
    'mZagreb2', 'AATSC4dv', 'NssssSn', 'GATS5dv', 'IC0', 'TSRW10',
    'SpMAD_Dzare', 'AATSC0v', 'AATSC7se', 'n5HRing', 'VSA_EState7', 'MINaaO',
    'n12HRing', 'n8AHRing', 'SsBr', 'SRW03', 'SpMAD_DzZ', 'AATS2pe', 'AATSC0i',
    'AATS8se', 'MINsSH', 'GATS3se', 'ATSC3pe', 'VR2_Dzse', 'SssCH2', 'ATSC0pe',
    'AETA_eta_F', 'AATS1d', 'SsssB', 'ATSC5se', 'Xp-6dv', 'MWC08', 'NddC',
    'MAXdssC', 'MATS1v', 'AATSC4are', 'AATSC4v', 'MATS1i', 'ATS7i', 'AATS6se',
    'AATS5m', 'VSA_EState9', 'n9FHRing', 'GATS7are', 'MAXdNH', 'AETA_eta_R',
    'n9FARing', 'AATSC7d', 'ZMIC5', 'GATS4pe', 'nFaHRing', 'AXp-0d', 'ATS0i',
    'Xch-6dv', 'ATSC3v', 'n10FRing', 'MINtCH', 'MATS3i', 'NsLi', 'n10aRing',
    'nBr', 'AATSC8se', 'StsC', 'VR3_Dzpe', 'AATSC7m', 'VR1_Dt', 'Xc-6dv',
    'ATSC1c', 'SsLi', 'ATSC4se', 'AATSC1c', 'MATS6se', 'WPol', 'ATS5p',
    'ATS6dv', 'VE2_A', 'SpAD_Dzp', 'SpMAD_Dzp', 'n5aHRing', 'ATS7Z', 'MATS4i',
    'SpAD_Dzm', 'SlogP_VSA5', 'SM1_Dzi', 'VE2_D', 'NssNH', 'AETA_eta_L',
    'AATSC3Z', 'n11AHRing', 'piPC4', 'nAHRing', 'MATS4m', 'Si', 'VSA_EState2',
    'ATS2dv', 'MDEN-13', 'MINsI', 'BCUTd-1h', 'VE1_DzZ', 'MATS4s',
    'SpMax_Dzse', 'NtsC', 'nHetero', 'MATS4are', 'nX', 'AATS1dv', 'RPCG',
    'ATSC4c', 'ATSC4i', 'CIC4', 'MATS4v', 'AATS5Z', 'piPC7', 'AATS4v',
    'MATS5s', 'NssPbH2', 'n4FaHRing', 'n6Ring', 'SRW07', 'ATS2pe', 'Kier1',
    'Xp-3d', 'AETA_beta', 'AATS6pe', 'MIC5', 'ATSC7m', 'SpDiam_Dzp', 'MAXsI',
    'BCUTare-1l', 'NssSnH2', 'Diameter', 'MAXddsN', 'MATS3m', 'Xp-5d',
    'ATSC8pe', 'n6FaHRing', 'ATSC7i', 'GATS3pe', 'MINssO', 'nG12FARing',
    'BCUTd-1l', 'AATS8d', 'ATSC2d', 'SpAbs_DzZ', 'n11HRing', 'ATS0v',
    'MAXssNH', 'VR1_Dzm', 'VR3_Dzm', 'LogEE_Dt', 'AATS5dv', 'AATS1v',
    'AATS4are', 'MATS1dv', 'SpMAD_A', 'BIC3', 'SlogP_VSA9', 'VR2_Dzp',
    'ATSC6v', 'ETA_epsilon_5', 'AETA_beta_ns_d', 'ATS7d', 'VE3_Dzse',
    'PEOE_VSA7', 'Zagreb1', 'SIC4', 'MPC4', 'ATS8Z', 'MINsCl', 'LogEE_Dzse',
    'AATSC8p', 'GATS2d', 'NsPbH3', 'ATS4i', 'ETA_dBeta', 'n12FARing', 'piPC6',
    'SpMAD_Dzi', 'ETA_dPsi_A', 'GATS2i', 'GATS4m', 'nRing', 'n4AHRing',
    'ATSC2s', 'Xch-7dv', 'AATS1pe', 'ATSC0m', 'ETA_shape_x', 'piPC8',
    'n9aHRing', 'SlogP_VSA11', 'ATS3Z', 'AATSC1are', 'MPC6', 'AATSC6are',
    'ETA_eta_RL', 'MATS6d', 'ATS6v', 'VE2_Dzare', 'ATS0are', 'AATSC5se',
    'GATS1p', 'AETA_beta_s', 'ATS1are', 'NtCH', 'LogEE_Dzpe', 'n9aRing',
    'nG12aRing', 'MATS7s', 'n6HRing', 'SpMAD_Dt', 'AATS5i', 'ATSC2i',
    'MATS1se', 'Sp', 'NssBe', 'n9FAHRing', 'MINssssC', 'MINaaNH', 'AATSC7c',
    'NssssC', 'SsSiH3', 'SdssSe', 'MAXdS', 'AATSC6d', 'NsNH3', 'GATS5p',
    'AATS6m', 'NsssP', 'SIC0', 'GATS6d', 'ATSC7v', 'nB', 'ATS2m', 'AATS0se',
    'SssNH2', 'ATSC2are', 'SssPbH2', 'mZagreb1', 'nAromAtom', 'GATS2pe',
    'VSA_EState8', 'SpDiam_Dzm', 'MATS3pe', 'SddsN', 'GATS1s', 'n8Ring',
    'ATS4pe', 'ATS2i', 'AXp-0dv', 'MATS7se', 'AATSC6dv', 'C1SP3', 'n5aRing',
    'Xp-3dv', 'ETA_dEpsilon_B', 'Lipinski', 'n10FAHRing', 'AATS8v', 'GATS2are',
    'MATS6dv', 'GGI2', 'ATS4s', 'ATSC8m', 'AXp-6d', 'AATSC6i', 'GGI7',
    'VE1_Dzpe', 'ATSC8p', 'MAXdO', 'AATS4i', 'nAcid', 'SpMax_Dzpe', 'C4SP3',
    'nRot', 'MDEC-11', 'ETA_eta_FL', 'SlogP_VSA7', 'MATS7v', 'AATS6s', 'MINtN',
    'MATS1d', 'ATSC8i', 'NssBH', 'ATSC7se', 'AATS6d', 'AXp-3d', 'GGI9',
    'VR3_D', 'VSA_EState1', 'MDEC-34', 'SRW10', 'ATS7m', 'n8FAHRing',
    'PEOE_VSA11', 'ATSC6i', 'nBonds', 'NssO', 'ETA_dPsi_B', 'n8FARing',
    'SpMax_A', 'MIC1', 'NssssSi', 'BCUTm-1l', 'AATS7d', 'AATS5pe', 'MDEN-22',
    'GATS2s', 'GATS3dv', 'n10FARing', 'ATS2s', 'SpAD_A', 'BCUTZ-1l', 'ATS8pe',
    'nFRing', 'Xpc-5d', 'VR2_A', 'RNCG', 'nBondsM', 'ATS7are', 'n10Ring',
    'ATS1v', 'ETA_psi_1', 'AATS5d', 'EState_VSA3', 'nARing', 'ATSC8d',
    'GATS5s', 'AATSC5are', 'Xc-5dv', 'SIC5', 'AETA_beta_ns', 'GATS5m',
    'NssGeH2', 'n6FHRing', 'MINaasC', 'ETA_beta_ns', 'ATSC8Z', 'ATSC1are',
    'MAXssssC', 'NaasN', 'ATS8v', 'SsssCH', 'GATS4p', 'GGI1', 'VE3_Dt',
    'AATS0are', 'MATS4dv', 'Xpc-6d', 'ETA_alpha', 'MATS4se', 'VSA_EState4',
    'GATS4s', 'AATS4d', 'ETA_eta_R', 'n10ARing', 'ATS3pe', 'BCUTpe-1h',
    'SdsssP', 'SaaCH', 'NaaSe', 'AATSC8m', 'SsNH2', 'AATSC2d', 'MINaasN',
    'VE2_Dzse', 'GATS8c', 'ATSC5p', 'ATS8are', 'AATS2dv', 'n12AHRing',
    'LogEE_Dzv', 'SssssBe', 'VR1_DzZ', 'n5FaRing', 'AMID_h', 'AATS4dv',
    'MATS1m', 'MAXaaO', 'AATS2s', 'VR2_Dzare', 'AATS7s', 'MATS3v', 'VR1_Dzv',
    'AATSC3i', 'SMR_VSA7', 'NaaaC', 'VE3_Dzpe', 'ETA_dEpsilon_D', 'BCUTare-1h',
    'SdssC', 'n5FHRing', 'nFAHRing', 'Xch-4dv', 'MATS8d', 'MATS5c', 'SssBH',
    'EState_VSA5', 'MDEC-14', 'n4FHRing', 'AATS5se', 'n11FaRing', 'IC1',
    'ETA_dEpsilon_A', 'Xp-1d', 'MIC0', 'MAXtCH', 'nCl', 'ETA_epsilon_4',
    'MATS6c', 'VE1_Dzm', 'ATS3p', 'C1SP1', 'NsCl', 'PetitjeanIndex', 'AATSC1v',
    'VE2_Dzv', 'Xp-4d', 'MINdssS', 'GATS1are', 'AATSC3d', 'MATS5se', 'AATS1p',
    'MWC10', 'MATS7d', 'VE1_Dt', 'NsPH2', 'fragCpx', 'VR2_D', 'ATSC5i',
    'SpMax_Dt', 'NsssSiH', 'ABCGG', 'SMR_VSA9', 'SpAbs_Dzp', 'SlogP_VSA6',
    'nN', 'NsssssAs', 'ATSC2Z', 'VR3_Dzare', 'LogEE_Dzare', 'n6aRing',
    'GATS6s', 'MWC09', 'AATSC6p', 'MAXtsC', 'GGI6', 'n6FRing', 'CIC2', 'NdssC',
    'JGI10', 'RotRatio', 'ATSC5v', 'ATS8se', 'piPC10', 'MAXsBr', 'VR1_Dzpe',
    'n8FHRing', 'SaaSe', 'EState_VSA6', 'NsBr', 'SsssP', 'SaasN',
    'SpDiam_Dzse', 'LogEE_Dzm', 'nC', 'SaasC', 'MID_O', 'AATS1se', 'SpAD_Dzpe',
    'AATS0v', 'nBondsA', 'BCUTm-1h', 'AATS0dv', 'BCUTp-1h', 'ATSC3are',
    'MATS2p', 'BIC2', 'MPC10', 'MATS4d', 'AATS7are', 'SssSnH2', 'n4ARing',
    'AATSC7dv', 'Spe', 'GATS3d', 'NsssPbH', 'VR1_D', 'MDEC-12', 'NdSe',
    'ATS3i', 'MID_N', 'AATS3se', 'IC5', 'MINsBr', 'AATS8i', 'AATS4Z',
    'AATSC5d', 'SM1_Dzse', 'SpMAD_Dzv', 'GATS4v', 'AATSC5m', 'n10aHRing',
    'AATSC3pe', 'Xc-4dv', 'JGI6', 'AATSC0se', 'NsSH', 'SsAsH2', 'CIC1',
    'ATSC3Z', 'fMF', 'AATS2i', 'MW', 'TIC3', 'AATSC4Z', 'GATS1v', 'MATS8are',
    'GATS8i', 'GATS6Z', 'AATS4pe', 'nFHRing', 'n5FRing', 'nAtom', 'SMR_VSA4',
    'n9ARing', 'ATSC4s', 'AATSC8pe', 'AATSC4se', 'BCUTs-1h', 'ATSC1dv',
    'AATS1i', 'SssPH', 'ATSC3m', 'n11aHRing', 'LogEE_Dzp', 'SaaaC',
    'SpDiam_Dzi', 'AXp-7dv', 'MAXsssN', 'SpDiam_A', 'MATS2se', 'SddssS',
    'JGT10', 'ATS5dv', 'MATS7pe', 'Sm', 'GATS7Z', 'GATS6se', 'SssBe', 'AATS7Z',
    'VE1_A', 'ATSC3d', 'NssssGe', 'MINdNH', 'n4HRing', 'MDEC-22', 'PEOE_VSA13',
    'ATSC2v', 'nHBDon', 'FilterItLogS', 'BCUTc-1h', 'Xp-7d', 'BIC4', 'MPC8',
    'ZMIC1', 'nHBAcc', 'ATS2se', 'ATSC7dv', 'MATS3p', 'GATS1d', 'AETA_alpha',
    'AATSC5dv', 'AATS6v', 'n12FHRing', 'n9FaHRing', 'GATS7dv', 'AATSC2dv',
    'SRW09', 'SRW08', 'AATS6p', 'nG12AHRing', 'MWC04', 'AATSC0dv', 'n9HRing',
    'ATS7s', 'MATS8Z', 'n7ARing', 'AATS6Z', 'ATS4v', 'BIC1', 'VR3_DzZ',
    'BCUTc-1l', 'n7aRing', 'NsssAs', 'GATS1Z', 'n3aHRing', 'AXp-2d', 'GATS2Z',
    'AATSC1m', 'GATS1m', 'AXp-1dv', 'SsPH2', 'ETA_eta_L', 'NssssN', 'GATS7d',
    'PEOE_VSA12', 'n11FaHRing', 'ETA_eta_B', 'GhoseFilter', 'n5ARing', 'NsNH2',
    'AATS6dv', 'ATSC8are', 'ATSC0c', 'n12Ring', 'nAromBond', 'SssssSn',
    'AATSC4d', 'ATS6p', 'SMR_VSA8', 'MWC01', 'GATS6v', 'ATS5pe', 'ATS3v',
    'ATSC1i', 'MAXaasN', 'n9FRing', 'GATS2se', 'BCUTi-1l', 'n4aRing',
    'SM1_Dzp', 'SdssS', 'n3AHRing', 'AXp-4dv', 'FCSP3', 'GATS8pe', 'AATS4se',
    'nBondsO', 'n9FaRing', 'nG12HRing', 'ATSC0se', 'SpAD_Dt', 'MATS1s',
    'MINdO', 'MATS5m', 'MAXaaS', 'VE1_Dzv', 'ATS5d', 'n3ARing', 'AETA_eta',
    'SaaO', 'MATS3are', 'AATSC5c', 'MINaaS', 'MID_X', 'ETA_beta_ns_d',
    'GATS3s', 'nHRing', 'AATS5s', 'MATS8pe', 'MATS3d', 'ATS4m', 'GATS4dv',
    'NaaNH', 'GATS2p', 'AATSC4c', 'MATS6i', 'MATS4Z', 'ATS4are', 'NdssSe',
    'VE1_Dzse', 'NssssB', 'SsssN', 'MATS2s', 'n8aRing', 'SpMAD_Dzse', 'MID_h',
    'ATS5i', 'BCUTs-1l', 'SpAD_Dzse', 'n11Ring', 'TMPC10', 'AMID', 'AATS2p',
    'SpMax_Dzm', 'nBondsD', 'SRW06', 'AATSC3c', 'SpMAD_Dzm', 'TIC5',
    'MAXsssCH', 'ATSC2pe', 'n11FRing', 'MINsF', 'VR2_Dt', 'ATS2are', 'SM1_DzZ',
    'ATS1i', 'AATS8Z', 'ETA_epsilon_3', 'GGI8', 'MAXaasC', 'AATS2m', 'C2SP2',
    'TIC0', 'piPC9', 'GATS5i', 'ATSC4are', 'ECIndex', 'ATSC5c', 'AATSC2i',
    'AATSC1d', 'PEOE_VSA10', 'piPC2', 'GATS6pe', 'ATSC1p', 'VR3_Dzse', 'ATS4Z',
    'AATS2are', 'GATS1c', 'HybRatio', 'MATS7Z', 'AATS4m', 'GATS4se', 'SpAbs_A',
    'MATS4pe', 'Xpc-4d', 'nFaRing', 'AATS0m', 'AATSC6Z', 'ATSC7are',
    'VE1_Dzare', 'nBase', 'MATS7are', 'MAXddssS', 'AATSC1p', 'MWC05', 'NdssS',
    'MATS7c', 'JGI9', 'NdsCH', 'ATSC3dv', 'ATSC1pe', 'ATSC4dv', 'AXp-6dv',
    'SRW05', 'ATS2Z', 'AATSC1se', 'ATS1dv', 'GATS2m', 'SsssssP', 'VR2_Dzpe',
    'AATSC5s', 'SpAbs_D', 'SsssssAs', 'MDEO-12', 'ATS0dv', 'SsSH', 'ATSC3c',
    'AATSC2s', 'SRW04', 'GATS6dv', 'AATSC3p', 'MWC06', 'ETA_dAlpha_A',
    'NddssSe', 'AATS4p', 'VE3_Dzi', 'ATS4p', 'Zagreb2', 'MINdsN', 'NssSiH2',
    'SdSe', 'VE2_Dzp', 'AATSC4p', 'n4Ring', 'AATSC0d', 'n3aRing', 'NsssN',
    'AXp-1d', 'StN', 'BCUTdv-1h', 'n12FRing', 'AXp-2dv', 'ATS3dv', 'MATS2Z',
    'SpMax_Dzp', 'MATS5dv', 'SpAbs_Dt', 'MINssS', 'SssO', 'AATS0pe', 'MINdssC',
    'nF', 'AATS3m', 'n11ARing', 'n10FaHRing', 'nG12aHRing', 'Radius',
    'MATS2dv', 'GATS4Z', 'MAXaaN', 'ATS0m', 'VE3_D', 'GATS5Z', 'AATS8m',
    'JGI1', 'Xc-3dv', 'ATS6s', 'SMR_VSA1', 'MPC7', 'AETA_eta_RL',
    'EState_VSA2', 'MINsCH3', 'nBondsKD', 'n3Ring', 'MATS3s', 'MATS6s',
    'AATSC1pe', 'LogEE_D', 'n11FHRing', 'ETA_eta', 'MINsssN', 'MINddsN',
    'GATS5are', 'SsssSnH', 'PEOE_VSA6', 'ATSC0s', 'ATS4se', 'Xch-7d', 'ATS0Z',
    'VE3_Dzv', 'GATS2v', 'AATSC3m', 'Vabc', 'SsCH3', 'MATS7m', 'MPC5', 'SaaS',
    'AATS8dv', 'DetourIndex', 'Xp-6d', 'VR1_Dzse', 'StCH', 'VE1_Dzi', 'MAXtN',
    'AATSC1i', 'MDEO-11', 'AATSC4s', 'EState_VSA8', 'MATS8v', 'MINaaN',
    'MINaaaC', 'GGI3', 'MATS3se', 'ATS6pe', 'ATSC3se', 'nBondsS', 'ATS1d',
    'VE2_Dzpe', 'SpAbs_Dzm', 'Mpe', 'GATS4i', 'NsGeH3', 'AETA_eta_BR',
    'GATS6are', 'SdCH2', 'AATS0p', 'ATSC8dv', 'n12aRing', 'BCUTp-1l', 'MINsOH',
    'AATS3i', 'MINssssSi', 'AATS1s', 'AXp-5dv', 'n4FARing', 'C3SP3', 'AATS3p',
    'NdsN', 'MATS2m', 'NddsN', 'ATS0s', 'GATS1i', 'AETA_dBeta', 'VE3_Dzare',
    'SssssC', 'MATS2v', 'SssssB', 'GATS5c', 'ETA_beta_s', 'AATS0i', 'AATS6are',
    'MATS1pe', 'MATS6Z', 'Xch-3d', 'naRing', 'nG12FaHRing', 'MINsssB',
    'NssAsH', 'SpMax_D', 'ATS6m', 'NsssdAs', 'ATS5s', 'n10FaRing', 'SdNH',
    'AATSC8i', 'VR1_Dzare', 'n4FRing', 'n6aHRing', 'MATS5v', 'PEOE_VSA2',
    'ATSC0Z', 'n7AHRing', 'LogEE_A', 'NsssCH', 'NddssS', 'MPC9', 'Mm',
    'ATSC7d', 'MAXsCl', 'MAXsNH2', 'AATSC7are', 'ATSC0i', 'ATS2d', 'n8FRing',
    'MAXssO', 'NaaS', 'SsSeH', 'MATS1Z', 'VE3_Dzm', 'AATS8s', 'Xp-7dv',
    'ATS5are', 'AATSC0m', 'ATS5m', 'BIC5', 'GATS8v', 'Mare', 'n12aHRing',
    'NdsssP', 'n8ARing', 'MAXdsN', 'SM1_Dt', 'MINdS', 'AATSC8v', 'WPath',
    'AATS2v', 'MDEC-23', 'AATSC4m', 'AATSC7p', 'NdO', 'MIC2', 'MATS4c',
    'BCUTv-1h', 'TIC4', 'n9AHRing', 'MDEC-33', 'MAXssssSi', 'GATS2c',
    'AETA_eta_FL', 'MDEN-12', 'n10AHRing', 'MINdCH2', 'ETA_eta_BR', 'ATS7v',
    'AATS3pe', 'MATS6are', 'Xp-2d', 'AATSC2v', 'nHeavyAtom', 'VR3_Dt',
    'MATS8i', 'SsssdAs', 'MAXdCH2', 'ATS5se', 'nBondsT', 'Xc-6d', 'n8aHRing',
    'VSA_EState6', 'GGI5', 'SssGeH2', 'ATS8i', 'SIC3', 'n7Ring', 'n7FAHRing',
    'GATS5pe', 'nFARing', 'GATS3are', 'ATSC6pe', 'ETA_shape_p', 'NsI', 'SaaN',
    'AATS3s', 'AATSC8Z', 'BIC0', 'MATS8m', 'Sv', 'n5AHRing', 'NaasC',
    'EState_VSA4', 'AATS4s', 'n4FAHRing', 'AATSC5p', 'AATS7i', 'TopoPSA(NO)',
    'ATSC1m', 'AATS5are', 'AATSC2m', 'BCUTpe-1l', 'ATSC5Z', 'SsGeH3', 'ATS1s',
    'NssS', 'n8FaRing', 'MATS6p', 'ATS1pe', 'TpiPC10', 'ATS7pe', 'SpAbs_Dzare',
    'ATSC7pe', 'nG12FaRing', 'IC3', 'BCUTi-1h', 'MINdsCH', 'VE2_Dt', 'ATSC2c',
    'SpMax_Dzare', 'NaaO', 'MAXsSH', 'ATS4d', 'AATSC0are', 'AMID_X', 'NsOH',
    'piPC3', 'MAXaaCH', 'PEOE_VSA3', 'VE3_A', 'AATSC3s', 'AATS3v', 'MATS6m',
    'AATSC3v', 'MID_C', 'ATS1se', 'IC4', 'CIC0', 'ATSC5are', 'MATS7p',
    'AATSC6se', 'GATS7pe', 'VE3_DzZ', 'GATS6c', 'SpDiam_D', 'EState_VSA1',
    'NsssssP', 'ATSC6dv', 'SMR_VSA5', 'NdNH', 'nG12FAHRing', 'AATS8p',
    'ATSC1s', 'SlogP_VSA10', 'SpMax_Dzv', 'SsssAs', 'AATSC0c', 'AATSC2c',
    'nBridgehead', 'n7FRing', 'SpMax_Dzi', 'SdsN', 'n8FaHRing', 'NsAsH2', 'Mi',
    'ATSC6p', 'AATSC2se', 'Xc-4d', 'Xp-5dv', 'AATSC2pe', 'ZMIC0', 'SMR_VSA6',
    'MDEN-23', 'ATS5v', 'ATSC2p', 'SsssNH', 'IC2', 'MINaaCH', 'AATS7p',
    'MATS8dv', 'ATS1m', 'ATSC2se', 'AATSC1s', 'SpDiam_DzZ', 'ATSC6s',
    'AATSC1dv', 'SddC', 'ATSC2m', 'JGI3', 'MIC4', 'AATSC5Z', 'LogEE_Dzi',
    'SssAsH', 'AATSC8dv', 'Xch-5d', 'piPC1', 'AATSC8are', 'BCUTse-1l', 'ATS8m',
    'AATS2Z', 'MATS3c', 'VE1_D', 'ATSC8se', 'Xpc-4dv', 'BCUTZ-1h', 'NsSiH3',
    'ATS0d', 'ATSC1d', 'n11FAHRing', 'GATS7m', 'MAXsOH', 'AATSC6c', 'MAXssS',
    'VE2_Dzm', 'SM1_Dzm', 'SpAbs_Dzse', 'SsOH', 'ATS3s', 'AATSC3se', 'SsNH3',
    'VAdjMat', 'SpDiam_Dzv', 'ATS0pe', 'SIC1', 'ATSC6are', 'GATS8p', 'ZMIC2',
    'n12FaRing', 'GATS6m', 'SlogP_VSA2', 'SddssSe', 'NaaN', 'ETA_dAlpha_B',
    'CIC5', 'ATSC4v', 'SsCl', 'ATSC3i', 'NtN', 'ATSC4d', 'NdS', 'AATS8pe',
    'ATSC1Z', 'AATSC6m', 'SssSe', 'GATS2dv', 'SsssGeH', 'n12FaHRing',
    'AATSC6v', 'SMR_VSA3', 'SssssGe', 'Xpc-6dv', 'MDEN-11', 'MATS8s',
    'SpAD_Dzare', 'MATS3dv', 'EState_VSA10', 'NssssBe', 'MATS5are',
    'EState_VSA7', 'AETA_eta_B', 'SM1_Dzv', 'EState_VSA9', 'n8HRing',
    'SpDiam_Dzpe', 'n5FAHRing', 'VR2_DzZ', 'VR3_A', 'ATS0p', 'JGI4', 'NssPH',
    'GATS1pe', 'ATSC6d', 'NsF', 'Mse', 'ATSC5s', 'MATS5Z', 'SpAbs_Dzi',
    'SpDiam_Dzare', 'MINsNH2', 'MATS2pe', 'n11FARing', 'VR2_Dzm', 'AATS1Z',
    'MAXdsCH', 'C2SP1', 'Kier3', 'VMcGowan', 'MAXaaNH', 'ATSC6c', 'PEOE_VSA8',
    'nH', 'ATSC4Z', 'GATS7c', 'ATS8dv', 'AATSC5pe', 'ETA_shape_y', 'GATS1se',
    'MAXssCH2', 'VSA_EState3', 'Sare', 'Kier2', 'ATS3se', 'ATSC6m', 'nP',
    'n6FaRing', 'SsF', 'C1SP2', 'MATS3Z', 'AATS3d', 'Xp-1dv', 'SM1_Dzpe',
    'AXp-3dv', 'VR1_Dzi', 'PEOE_VSA1', 'AATSC5i', 'AATS7pe', 'ETA_epsilon_1',
    'MATS6v', 'ETA_eta_F', 'n4FaRing', 'AATS6i', 'AATSC8d', 'AXp-7d',
    'nBondsKS', 'SM1_Dzare', 'BCUTdv-1l', 'MATS5d', 'MATS7dv', 'ATSC4m',
    'AATSC0Z', 'MDEC-13', 'Sse', 'SpAD_D', 'NaaCH', 'n10FHRing', 'ATSC5m',
    'n6FARing', 'GATS5v', 'MAXsF', 'AATSC0p', 'ATS1p', 'NsSnH3', 'AXp-4d',
    'GATS1dv', 'n7aHRing', 'LabuteASA', 'AMW', 'MATS5p', 'GATS4d', 'ATSC0d',
    'n5FaHRing', 'ATS6i', 'ATSC4p', 'LogEE_DzZ', 'ATSC8v', 'MATS2d', 'GATS7p',
    'MATS4p', 'MATS8se', 'AATSC6pe', 'NsssB', 'ETA_dEpsilon_C', 'n5FARing',
    'Xch-4d', 'n6FAHRing', 'MPC3', 'n3HRing', 'ATSC7p', 'AXp-5d', 'JGI5',
    'SlogP_VSA3', 'Xp-2dv', 'AATSC7s', 'ATSC2dv', 'ATSC1se', 'MDEC-24',
    'SpMax_DzZ', 'n6AHRing', 'ATS8s', 'AATSC4pe', 'MATS2c', 'ATSC1v',
    'ETA_beta', 'nO', 'TopoShapeIndex', 'BCUTv-1l', 'AATS5v', 'GATS6p',
    'SpDiam_Dt', 'SssS', 'AATSC7pe', 'AATSC2are', 'n9Ring', 'SssssN',
    'SpAD_Dzi', 'n10HRing', 'NsSeH', 'BertzCT', 'Xc-3d', 'AMID_O', 'AATSC5v',
    'AATSC3dv', 'Xp-0d', 'ATSC5pe', 'VR1_A', 'AATS2d', 'AATSC2p', 'AATSC7v',
    'ATSC7Z'
]


def batch_calc_mordred_descriptors(smiles_list):
    NAN = float('nan')
    calc = MordredCalculator(mordred_descriptors,
                             ignore_3D=True)  # register all descriptors
    calc.descriptors = [
        d for d in calc.descriptors if str(d) in MORDRED_KEEP_COLS
    ]
    assert len(calc.descriptors) == 1531

    mols = [replace_dy(smiles, return_mol=True) for smiles in smiles_list]
    ret = calc.pandas(mols, quiet=True).fill_missing(NAN)
    # FP16 overflow columns
    for col in ['VR1_A', 'VR2_A', 'ATS0m', 'ATS4m', 'ATS7i']:
        ret.loc[:, col] = np.log(ret.loc[:, col])
    ret = ret.to_numpy().astype(np.float16)
    return ret


def batch_calc_ecfp_descriptors(smiles_list, radius=3, bits=2048):
    ret = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius,
                                                  nBits=bits).ToList())
        fp = np.packbits(fp, axis=None)
        ret.append(fp)
    ret = np.array(ret, dtype=np.uint8)
    return ret


SKFP_FPS = {
    'atom_pair': skfps.AtomPairFingerprint,  # uint8, 2048
    'avalon': skfps.AvalonFingerprint,  # uint8, 
    'fcfp': partial(skfps.ECFPFingerprint, use_fcfp=True),  # uint8, 2048
    'functional_group': skfps.FunctionalGroupsFingerprint,  # uint8, 85
    'ghose_crippen': skfps.GhoseCrippenFingerprint,  # uint8, 110
    'laggner': skfps.LaggnerFingerprint,  # uint8, 307
    'layered': skfps.LayeredFingerprint,  # uint8, 2048 (quite slow)
    'lingo': skfps.LingoFingerprint,  # uint8, 1024, 52
    'map': skfps.MAPFingerprint,  # uint8, 1024 (quite slow), 52
    'mhfp': skfps.MHFPFingerprint,  # uint8, 2048 (quite slow), 55
    'mqns': skfps.MQNsFingerprint,  # uint32, 42
    'pattern': skfps.PatternFingerprint,  # uint8, 2048
    'physio_chemical':
    skfps.PhysiochemicalPropertiesFingerprint,  # uint8, 2048, 50
    'pubchem': skfps.PubChemFingerprint,  # uint8, 881, slow
    'rdkit': skfps.RDKitFingerprint,  # uint8, 2048, 55-ing
    'secfp': skfps.SECFPFingerprint,  # uint8, 2048, slow
    'topological_torsion': skfps.TopologicalTorsionFingerprint,  # uint8, 2048, 55
}


def batch_calc_skfp_descriptors(smiles_list, feature_name='fcfp'):
    calculator = SKFP_FPS[feature_name]()
    fp = calculator.transform(smiles_list)
    fp = np.packbits(fp, axis=-1)
    return fp


if __name__ == '__main__':
    args = parse_args()
    NUM_CHUNKS = args.num_chunks
    CHUNK_IDX = args.chunk_idx
    SUBSET = args.subset

    if 'train' in SUBSET:
        df = pl.scan_csv(
            '/home/dangnh36/datasets/competitions/leash_belka/processed/train_v2.csv'
        ).select(
            pl.col('molecule'),
            # pl.col('bb1', 'bb2', 'bb3').cast(pl.UInt16),
            # pl.col('BRD4', 'HSA', 'sEH').cast(pl.UInt8),
        ).collect()
        print(df.estimated_size('gb'), 'GB')
        print(df.head())

        if SUBSET == 'train_7.8M':
            all_idxs = pl.scan_csv(
                '/home/dangnh36/datasets/competitions/leash_belka/processed/cv/train_5_7.8M_idxs.csv'
            ).select(pl.col('index')).collect()['index'].to_numpy()
        elif SUBSET == 'train':
            all_idxs = np.arange(0, len(df), 1)
        elif SUBSET == 'train_val02':
            all_idxs = pl.scan_csv(
                '/home/dangnh36/datasets/competitions/leash_belka/processed/cv/16_19_0_2/val.csv'
            ).filter(pl.col('subset') != 1).select(
                pl.col('index')).collect().to_numpy()[:, 0]
        else:
            raise ValueError
    elif 'test' in SUBSET:
        df = pl.scan_csv(
            '/home/dangnh36/datasets/competitions/leash_belka/processed/test_v4.csv'
        ).select(
            pl.col('molecule'),
            #         pl.col('bb1', 'bb2', 'bb3').cast(pl.UInt16),
            # pl.col('BRD4', 'HSA', 'sEH').cast(pl.UInt8),
        ).collect()
        print(df.estimated_size('gb'), 'GB')
        print(df.head())
        all_idxs = np.arange(0, len(df), 1)
    else:
        raise ValueError

    if NUM_CHUNKS == 1:
        chunk_idxs = all_idxs
    else:
        num_samples = len(all_idxs)
        num_samples_per_chunk = num_samples // NUM_CHUNKS + 1
        chunk_idxs = np.arange(
            num_samples_per_chunk * CHUNK_IDX,
            min(num_samples, num_samples_per_chunk * (CHUNK_IDX + 1)))
        print(
            f'num_samples={num_samples} per_chunk={num_samples_per_chunk} start={chunk_idxs.min()} end={chunk_idxs.max()}'
        )
        chunk_idxs = all_idxs[chunk_idxs]

    if args.feature == 'rdkit210':
        feature_calc_func = batch_calc_rdkit_descriptors
    elif args.feature == 'mordred':
        feature_calc_func = batch_calc_mordred_descriptors
    elif args.feature == 'ecfp6':
        feature_calc_func = partial(batch_calc_ecfp_descriptors,
                                    radius=3,
                                    bits=2048)
    elif args.feature == 'ecfp_2_1024':
        feature_calc_func = partial(batch_calc_ecfp_descriptors,
                                    radius=2,
                                    bits=1024)
    elif args.feature == 'ecfp_3_1024':
        feature_calc_func = partial(batch_calc_ecfp_descriptors,
                                    radius=3,
                                    bits=1024)
    elif args.feature == 'ecfp_2_512':
        feature_calc_func = partial(batch_calc_ecfp_descriptors,
                                    radius=2,
                                    bits=512)
    elif args.feature == 'ecfp_3_512':
        feature_calc_func = partial(batch_calc_ecfp_descriptors,
                                    radius=3,
                                    bits=512)
    else:
        feature_calc_func = partial(batch_calc_skfp_descriptors,
                                    feature_name=args.feature)

    extract_features(
        func=feature_calc_func,
        inputs=df[chunk_idxs, 'molecule'].to_list(),
        save_dir=
        '/home/dangnh36/datasets/competitions/leash_belka/processed/features/',
        feature_name=args.feature,
        subset=f'{SUBSET}_{NUM_CHUNKS}_{CHUNK_IDX}' if NUM_CHUNKS > 1 else f'{SUBSET}',
        method='batch',
        num_workers=multiprocessing.cpu_count()
        if args.workers <= 0 else args.workers,
        joblib_backend='loky',
        batch_size=args.batch_size,
        save_formats=['npy'],
        hook=None)
