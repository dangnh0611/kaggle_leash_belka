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
from src.utils.misc import dotdict

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


class MolToGraphV1:
    # allowable node and edge features
    ALLOWABLE_FEATURES = {
        'possible_atomic_num_list':
        list(range(1, 119)),
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        'possible_chirality_list': [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list': [
            Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'possible_bonds': [
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
        ],
        'possible_bond_dirs': [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ]
    }

    def __call__(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        num_nodes = mol.GetNumAtoms()
        NUM_ATOM_FEATURES = 2  # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                self.ALLOWABLE_FEATURES['possible_atomic_num_list'].index(
                    atom.GetAtomicNum())
            ] + [
                self.ALLOWABLE_FEATURES['possible_chirality_list'].index(
                    atom.GetChiralTag())
            ]
            atom_features_list.append(atom_feature)
        node_features = torch.tensor(np.array(atom_features_list),
                                     dtype=torch.long)

        # bonds
        NUM_BOND_FEATURES = 2  # bond type, bond direction
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [
                    self.ALLOWABLE_FEATURES['possible_bonds'].index(
                        bond.GetBondType())
                ] + [
                    self.ALLOWABLE_FEATURES['possible_bond_dirs'].index(
                        bond.GetBondDir())
                ]
                edges_list.append((i, j))
                edge_features.append(edge_feature)
                edges_list.append((j, i))
                edge_features.append(edge_feature)

            # data.edge_index: Graph connectivity with shape [num_edges, 2]
            edges = torch.tensor(np.array(edges_list), dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_features = torch.tensor(np.array(edge_features),
                                         dtype=torch.long)
        else:  # mol has no bonds
            edges = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((0, NUM_BOND_FEATURES),
                                        dtype=torch.long)

    #     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return num_nodes, edges, node_features, edge_features


class MolToGraphV2:

    def one_of_k_encoding(self, x, allowable_set, allow_unk=False):
        if x not in allowable_set:
            if allow_unk:
                x = allowable_set[-1]
            else:
                raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
        return list(map(lambda s: x == s, allowable_set))

    ATOM_SYMBOL = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
        'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
        'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
        'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
        'Pt', 'Hg', 'Pb', 'Dy',
        #'Unknown'
    ]
    #print('ATOM_SYMBOL', len(ATOM_SYMBOL))44
    HYBRIDIZATION_TYPE = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D
    ]

    def get_atom_feature(self, atom):
        """
        1.atom element: 44+1 dimensions    
        2.the atom's hybridization: 5 dimensions
        3.degree of atom: 6 dimensions                        
        4.total number of H bound to atom: 6 dimensions
        5.number of implicit H bound to atom: 6 dimensions    
        6.whether the atom is on ring: 1 dimension
        7.whether the atom is aromatic: 1 dimension           
        Total: 70 dimensions
        """
        feature = (
            self.one_of_k_encoding(atom.GetSymbol(), self.ATOM_SYMBOL)
        + self.one_of_k_encoding(atom.GetHybridization(), self.HYBRIDIZATION_TYPE)
        + self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
        + self.one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + [atom.IsInRing()]
        + [atom.GetIsAromatic()]
        )
        feature = np.array(feature, dtype=np.uint8)
        # feature = np.packbits(feature)
        return feature

    def get_bond_feature(self, bond):
        """
        1.single/double/triple/aromatic: 4 dimensions       
        2.the atom's hybridization: 1 dimensions
        3.whether the bond is on ring: 1 dimension          
        Total: 6 dimensions
        """
        bond_type = bond.GetBondType()
        feature = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        feature = np.array(feature, dtype=np.uint8)
        # feature = np.packbits(feature)
        return feature

    def __call__(self, mol):
        num_nodes = mol.GetNumAtoms()
        node_features = []
        edge_features = []
        edges = []
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            atom_i_features = self.get_atom_feature(atom_i)
            node_features.append(atom_i_features)

            for j in range(mol.GetNumAtoms()):
                bond_ij = mol.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    edges.append([i, j])
                    bond_features_ij = self.get_bond_feature(bond_ij)
                    edge_features.append(bond_features_ij)
        node_features=torch.from_numpy(np.stack(node_features)).float()
        edge_features=torch.from_numpy(np.stack(edge_features)).float()
        edges = torch.from_numpy(np.array(edges)).long()
        return num_nodes, edges, node_features, edge_features
        

class Graph2DDataset(BaseLeashDataset):

    def __init__(self, cfg, stage="train", cache={}):
        super().__init__(cfg, stage, cache)

        # LOAD DATAFRAME
        self.ds = cache[f'{self._stage}_ds']

        if cfg.data.mol2graph == 'v1':
            self.mol2graph = MolToGraphV1()
        elif cfg.data.mol2graph == 'v2':
            self.mol2graph = MolToGraphV2()
        else:
            raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def _get_sample_labels(self, samples):
        labels = np.stack(
            [samples[label_name] for label_name in self.label_cols], axis=-1)
        return labels

    def get_collater(self):
        return lambda x: x

    @property
    def getitem_as_batch(self):
        return True

    def __getitem__(self, idxs):
        """
        Generate one batch of data.
        """
        ridxs = self.idxs[idxs]
        samples = self.ds[ridxs]

        graphs = []
        for smiles in samples['smiles']:
            mol = normalize_smiles(smiles,
                                   canonical=True,
                                   do_random=False,
                                   isomeric=self.cfg.data.isomeric,
                                   replace_dy=self.cfg.data.replace_dy,
                                   return_mol=True)
            graphs.append(self.mol2graph(mol))

        # collate to a batch
        batch = dotdict(x=[],
                        edge_index=[],
                        edge_attr=[],
                        batch=[],
                        idx=torch.tensor(idxs, dtype=torch.int32))
        offset = 0
        for j, idx in enumerate(idxs):
            num_nodes, edges, node_features, edge_features = graphs[j]
            batch.edge_index.append(edges.long() + offset)
            batch.x.append(node_features)
            batch.edge_attr.append(edge_features)
            batch.batch += [j] * num_nodes
            offset += num_nodes
        batch.x = torch.cat(batch.x)
        batch.edge_attr = torch.cat(batch.edge_attr)
        batch.edge_index = torch.cat(batch.edge_index).T
        batch.batch = torch.LongTensor(batch.batch)

        if self.stage != 'predict':
            batch.target = torch.from_numpy(
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
        # LOAD TARGET
        for stage in stages:
            from datasets import load_from_disk
            cache[f'{stage}_ds'] = load_from_disk(
                os.path.join(data_dir, 'processed', 'hf', 'datasets', stage),
                keep_in_memory=cfg.data.ram_cache)

        return cache
