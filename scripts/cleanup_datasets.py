"""
This script cleans up raw SMILES
"""
from pathlib import Path

import datamol as dm
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools

RDLogger.DisableLog('rdApp.*')

IN_DIR = Path("data/raw")
OUT_DIR = Path("data/curated")
TAB_DIR = Path("tables")
DATASETS = {
    "Solubility_Wang2020": {
        "filename": "LogS_Wang2020.xls",
        "X": "SMILES",
        "Y": "LogS Value",
    },
    "Lipophilicity_Wang2020": {
        "filename": "LogD_Wang2020.xls",
        "X": "SMILES",
        "Y": "LogD value",
    },
    "Permeability_Caco2_Wang2020": {
        "filename": "LogPapp_Wang2020.xls",
        "X": "SMILES",
        "Y": "LogPapp Value",
    },
    "Permeability_MDCK_Fang2023": {
        "filename": "Fang2023.csv",
        "X": "SMILES",
        "Y": "LOG MDR1-MDCK ER (B-A/A-B)",
    },
    "VDss_Liu2022": {
        "filename": "Liu2022.xlsx",
        "X": "SMILES",
        "Y": "logVDss",
    },
    "hPPB_Lou2022": {
        "filename": "PPB_Lou2022.xlsx",
        "X": "cano_smiles",
        "Y": "PPB Fractions",
        "sheetname": "Table S2"
    },
    "HLM_Fang2023": {
        "filename": "Fang2023.csv",
        "X": "SMILES",
        "Y": "LOG HLM_CLint (mL/min/kg)",
    },
    "RLM_Fang2023": {
        "filename": "Fang2023.csv",
        "X": "SMILES",
        "Y": "LOG RLM_CLint (mL/min/kg)",
    },
    "LD50_Lunghini2019": {
        "filename": "LD50_Lunghini2019.xlsx",
        "X": "Canonical SMILES",
        "Y": "pLD50",
    },
}

def caborn_atoms(mol):
    pat = Chem.MolFromSmarts("[#6]")
    return bool(len(mol.GetSubstructMatches(pat)))

def inorganics_atoms(mol):
    """
    Filter for inorganic compounds
    """
    atoms = set([x.GetSymbol() for x in mol.GetAtoms()])
    organic_atoms = set(["H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"])
    return bool(len(atoms.difference(organic_atoms)))

def tidy_dataset(dataset: pd.DataFrame, labelCol: str = "Y", smilesCol: str = "SMILES", molCol: str = "MOL") -> pd.DataFrame:
    n_raw = dataset.shape[0]
    print(f"Total {n_raw} in raw dataset.")
    # add RDKit molecule column
    PandasTools.AddMoleculeColumnToFrame(dataset, smilesCol=smilesCol, molCol=molCol)
    dataset.dropna(subset=[molCol, labelCol], inplace=True)
    # add column filters
    dataset[molCol] = dataset[molCol].apply(lambda x: Chem.AddHs(x))
    dataset["INORGANICS"] = dataset[molCol].apply(lambda x: inorganics_atoms(x))
    dataset["CABRONS"] = dataset[molCol].apply(lambda x: caborn_atoms(x))
    dataset_filtered = dataset[~dataset["INORGANICS"] & dataset["CABRONS"]].reset_index(drop=True).copy()
    #dataset_filtered[molCol] = dataset_filtered[molCol].apply(lambda x: standardize_mol(x))
    dataset_filtered["X"] = dataset_filtered[molCol].apply(lambda x: dm.to_smiles(dm.remove_hs(x)))
    mols = dataset_filtered["MOL"].tolist()
    dataset_descriptors = dm.descriptors.batch_compute_many_descriptors(mols, n_jobs=10, progress=True, batch_size='auto')
    dataset_clean = pd.concat([dataset_filtered[["X", "Y"]], dataset_descriptors], axis=1)
    n_clean = dataset_clean.shape[0]
    print(f"Total {n_clean} in fitlered dataset.")
    stats = {"N_RAW": n_raw,
             "N_INORGANICS": dataset["INORGANICS"].sum(),
             "N_NOCARBONS": (~dataset["CABRONS"]).sum(),
             "N_CLEAN": n_clean}
    return dataset_clean, stats
    
def main():
    stats = []
    for d in DATASETS:
        print(f"Cleaning up dataset {d}")
        filename = DATASETS[d]["filename"]
        x_col = DATASETS[d]["X"]
        y_col = DATASETS[d]["Y"]
        sheetname = DATASETS[d]["sheetname"] if "sheetname" in DATASETS[d] else 0
        if filename.endswith("csv"):
            dataset_in = pd.read_csv(IN_DIR / filename, usecols=[x_col, y_col]).dropna().rename(columns={y_col: "Y"})
        else:
            dataset_in = pd.read_excel(IN_DIR / filename, usecols=[x_col, y_col], sheet_name=sheetname).dropna().rename(columns={y_col: "Y"})
    
        dataset_out, dataset_stats = tidy_dataset(dataset_in, smilesCol=x_col)
        dataset_out.to_csv(OUT_DIR / f"{d}.csv", index=False)
        dataset_stats["NAME"] = d
        stats.append(dataset_stats)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(TAB_DIR / "datasets_curation_report.csv", index=False)

if __name__ == "__main__":
    main()