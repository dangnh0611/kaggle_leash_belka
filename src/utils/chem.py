from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import atomInSmiles
import selfies
from deepsmiles.encode import encode as deepsmiles_encode
import os
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PROTEINS = ['BRD4', 'HSA', 'sEH']


def smiles_to_ais(smiles, keep_order=False):
    if not keep_order:
        # By default, it first canonicalizes the input SMILES
        ais = atomInSmiles.encode(smiles)
    else:
        # example smiles: 'C(C(=O)O)N'
        # mapping atomID into SMILES string
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        smiles = Chem.MolToSmiles(mol)  # 'C([C:1](=[O:2])[OH:3])[NH2:4]'
        # SMILES -> atom-in-SMILES
        ais = atomInSmiles.encode(smiles, with_atomMap=True)
    return ais


def smiles_to_selfies(smiles, keep_order=True):
    assert keep_order
    sf = ' '.join(list(selfies.split_selfies(selfies.encoder(smiles))))
    return sf


def smiles_to_deepsmiles(smiles, keep_order=True, rings=True, branches=True):
    assert keep_order
    return deepsmiles_encode(smiles, rings, branches)


SMILES_CONVERTERS = {
    'smiles': lambda x: x,
    'ais': smiles_to_ais,
    'selfies': smiles_to_selfies,
    'deepsmiles': smiles_to_deepsmiles
}


def normalize_smiles(smiles,
                     canonical=True,
                     do_random = False,
                     isomeric=True,
                     kekulize = False,
                     replace_dy=False,
                     return_mol=False):
    if do_random:
        assert not canonical
    if isomeric and not replace_dy and not do_random and not kekulize:
        if return_mol:
            return Chem.MolFromSmiles(smiles)
        else:
            return smiles

    # Convert your SMILES to a mol object.
    mol = Chem.MolFromSmiles(smiles)

    if replace_dy:
        #Create a mol object to replace the Dy atom with.
        new_attachment = Chem.MolFromSmiles('C')
        #Get the pattern for the Dy atom
        dy_pattern = Chem.MolFromSmiles('[Dy]')
        #This returns a tuple of all possible replacements, but we know there will only be one.
        new_mol = AllChem.ReplaceSubstructs(mol, dy_pattern, new_attachment)[0]
        #Good idea to clean it up
        Chem.SanitizeMol(new_mol)
        # Since you want 3D mols later, I'd suggest adding hydrogens. Note: this takes up a lot more memory for the obj.
        # Chem.AddHs(new_mol)
    else:
        new_mol = mol
    if return_mol:
        return new_mol
    else:
        return Chem.MolToSmiles(new_mol,
                                canonical=canonical,
                                doRandom = do_random,
                                isomericSmiles=isomeric,
                                kekuleSmiles=kekulize)


def make_submissions(
    test_df,
    preds,
    output_dir,
    target_cols=PROTEINS,
    submit_name='submission',
    submit_subsets=['all', 'share', 'public-nonshare'],
):
    test_ids = test_df[[f'id_{protein}' for protein in PROTEINS]].to_numpy()
    group_ids = test_df[[f'group_{protein}'
                         for protein in PROTEINS]].to_numpy()
    # in case that model only predict bindding for < 3 proteins
    # make dummy predictions of 0 for other proteins
    submit_preds = np.zeros((preds.shape[0], len(target_cols)),
                            dtype=preds.dtype)
    target_col_idxs = [PROTEINS.index(protein) for protein in target_cols]
    submit_preds[:, target_col_idxs] = preds
    preds = submit_preds

    assert (preds.shape == test_ids.shape)
    test_ids = test_ids.reshape(-1)
    preds = preds.reshape(-1)
    group_ids = group_ids.reshape(-1)
    mask = (test_ids != 0)
    assert (mask.sum() == 1674896)

    df = pl.DataFrame({
        'id': test_ids[mask],
        'binds': preds[mask],
        'group_id': group_ids[mask],
    })

    for submit_subset in submit_subsets:
        if submit_subset == 'all':
            subsets = list(range(12))
        elif submit_subset == 'share':
            # mol_group = 0
            subsets = [0, 1, 2]
        elif submit_subset == 'public-nonshare':
            # mol_group = 2
            subsets = [6, 7, 8]
        else:
            raise ValueError
        subset_df = df.with_columns(
            pl.when(pl.col('group_id').is_in(subsets)).then(
                pl.col('binds')).otherwise(pl.lit(0.0)).alias('binds'))
        save_name = f'{submit_name}_{submit_subset}.csv'
        save_path = os.path.join(output_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        subset_df.select('id', 'binds').write_csv(save_path)
        logger.info('Writting submission %s (%s) to %s', submit_subset,
                    df.shape, save_path)
    
    
