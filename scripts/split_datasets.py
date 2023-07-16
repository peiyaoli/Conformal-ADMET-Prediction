import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.rdBase import LogToPythonStderr
from tqdm import tqdm

LogToPythonStderr()
RDLogger.DisableLog("rdApp.*")

IN_DIR = Path("./data/curated/")
OUT_DIR = Path("./data/split_idxs/")



def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)

    return scaffold

def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def scaffold_split_indices(mols, num_folds: int):
    scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)
    index_sets = sorted(list(scaffold_to_indices.values()),
                    key=lambda index_set: len(index_set),
                    reverse=True)
    fold_indices = [[] for _ in range(num_folds)]
    for s in index_sets:
        length_array = [len(fi) for fi in fold_indices]
        min_index = length_array.index(min(length_array))
        fold_indices[min_index] += s
    return fold_indices

def random_split_indices(all_indices: List[int], num_folds: int, shuffle: bool = True):
    num_data = len(all_indices)
    if shuffle:
        random.shuffle(all_indices)
    fold_indices = []
    for i in range(num_folds):
        begin, end = int(i * num_data / num_folds), int((i + 1) * num_data / num_folds)
        fold_indices.append(np.array(all_indices[begin:end]))
    return fold_indices

    
def create_IVIT_splits(data: pd.DataFrame, num_folds: int = 10, test_folds_to_test: int = 10, val_folds_per_test: int = 1):
    num_data = len(data)
    all_indices = list(range(num_data))
    fold_indices = random_split_indices(all_indices, num_folds)
    random.shuffle(fold_indices)
    all_splits = []
    for i in range(test_folds_to_test):
        for j in range(1, val_folds_per_test + 1):
            val_idx = (i + j) % num_folds
            val = fold_indices[val_idx]
            test = fold_indices[i]
            train = []
            for k in range(num_folds):
                if k != i and k != val_idx:
                    train.append(fold_indices[k])
            train = np.concatenate(train)
            all_splits.append([train, val, test])
    return all_splits

def create_OVOT_splits(data: pd.DataFrame, num_folds: int = 10, test_folds_to_test: int = 10, val_folds_per_test: int = 1):
    mols = [Chem.MolFromSmiles(x) for x in data["X"]]
    fold_indices = scaffold_split_indices(mols=mols, num_folds=10)
    random.shuffle(fold_indices)
    all_splits = []
    for i in range(test_folds_to_test):
        for j in range(1, val_folds_per_test + 1):
            val_idx = (i + j) % num_folds
            val = fold_indices[val_idx]
            test = fold_indices[i]
            train = []
            for k in range(num_folds):
                if k != i and k != val_idx:
                    train.append(fold_indices[k])
            train = np.concatenate(train)
            
            all_splits.append([train, val, test])
    return all_splits
    

def create_IVOT_splits(data: pd.DataFrame, num_folds: int = 10, test_folds_to_test: int = 10, val_folds_per_test: int = 1):
    mols = [Chem.MolFromSmiles(x) for x in data["X"]]
    fold_indices = scaffold_split_indices(mols=mols, num_folds=10)
    random.shuffle(fold_indices)
    all_splits = []
    for i in range(test_folds_to_test):
        for j in range(1, val_folds_per_test + 1):
            val_idx = (i + j) % num_folds
            val = fold_indices[val_idx]
            test = fold_indices[i]
            train = []
            for k in range(num_folds):
                if k != i and k != val_idx:
                    train.append(fold_indices[k])
            train = np.concatenate(train)
            n_train = len(train)
            train_val = np.concatenate([train, val])
            np.random.shuffle(train_val)
            train = train_val[:n_train]
            val = train_val[n_train:]
            all_splits.append([train, val, test])
    return all_splits

def main():
    split_strategies = ["IVIT", "OVOT", "IVOT"]
    split_config = {
        "num_folds": 10,
        "test_folds_to_test": 10,
        "val_folds_per_test": 1
    }
    for dataset in IN_DIR.glob("*.csv"):
        df = pd.read_csv(dataset)
        for strategy in split_strategies:
            if strategy == "IVIT":
                split_idxs = create_IVIT_splits(df, **split_config)
            elif strategy == "OVOT":
                split_idxs = create_OVOT_splits(df, **split_config)
            elif strategy == "IVOT":
                split_idxs = create_IVOT_splits(df, **split_config)
            else:
                raise ValueError("Wrong split strategy.")
            with open(OUT_DIR / f"{dataset.stem}_{strategy}.pkl", "wb") as handle:
                pickle.dump(split_idxs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Split {dataset.stem} with {strategy}.")
                
if __name__ == "__main__":
    random.seed(2023)
    main()