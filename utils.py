import pickle
import math
import numpy as np
import pandas as pd
from pathlib import Path
from chempropv2 import data, featurizers
from chempropv2.utils import makedirs, find_nearest
from chempropv2.utils import estimate_sample_weights
from chempropv2.utils import quantiles as qtools

def df2dset(df, **kwargs):
    smis = df["X"].tolist()
    targets = df["Y"].tolist()
    targets = np.array(targets).reshape(-1, 1)
    if "data_weight" in kwargs:
        datapoints = [data.MoleculeDatapoint(smi=smi, targets=target, data_weight=data_weight) for smi, target, data_weight in zip(smis, targets, kwargs["data_weight"])]
    else:
        datapoints = [data.MoleculeDatapoint(smi=smi, targets=target) for smi, target in zip(smis, targets)]
    # set molecule featurizer
    featurizer = featurizers.MoleculeFeaturizer()
    dset = data.MoleculeDataset(datapoints, featurizer)        
    return dset

def load_molecule_datasets(dataset_path, split_path, fold_idx: int = 0, lds: bool = True):
    # load dataset CSV file
    df = pd.read_csv(dataset_path)
    # load split's indexes pickle file
    with open(split_path, "rb") as handle:
        splits = pickle.load(handle)
    # load split at fold-i
    split_idxs = splits[fold_idx]
    # load indexs for train/val/test sets
    train_idx, valid_idx, test_idx = split_idxs[0], split_idxs[1], split_idxs[2]
    print(f"Train set size: {len(train_idx)}")
    print(f"Validation set size: {len(valid_idx)}")
    print(f"Test set size: {len(test_idx)}")
    # load train/val/test dataframes
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    # sample size of train set
    n_train = len(train_df)
    if lds:
        # extract targets' values from train & val for sample weights estimation
        targets_train_val = np.concatenate((train_df["Y"].values, valid_df["Y"].values))
        w = estimate_sample_weights(targets=targets_train_val)
        w_train = w[:n_train]
        w_val = w[n_train:]
        # convert dataframes to datasets
        train_dset, val_dset, test_dset = df2dset(train_df, data_weight=w_train), df2dset(valid_df, data_weight=w_val), df2dset(test_df)
    else:
        train_dset, val_dset, test_dset = df2dset(train_df), df2dset(valid_df), df2dset(test_df)
    return (train_dset, val_dset, test_dset)

def load_experiment_dataloaders(dataset: str, split: str, fold_idx: int = 0, num_workers: int = 6, batch_size: int = 50, lds: bool = True):
    dataset_path = Path(f"data/curated/{dataset}.csv")
    split_path = Path(f"data/split_idxs/{dataset}_{split}.pkl")
    train_dset, val_dset, test_dset = load_molecule_datasets(dataset_path=dataset_path,
                                                             split_path=split_path,
                                                             fold_idx=fold_idx,
                                                             lds=lds)
    y_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(y_scaler)
    test_dset.normalize_targets(y_scaler)
    train_loader = data.MolGraphDataLoader(train_dset, num_workers=num_workers, batch_size=batch_size)
    val_loader = data.MolGraphDataLoader(val_dset, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    test_loader = data.MolGraphDataLoader(test_dset, num_workers=num_workers, shuffle=False, batch_size=batch_size)
    return (train_loader, val_loader, test_loader), y_scaler

def format_prediction_results(dataloader, y_preds, y_scaler, experiment_folder, **kwargs):
    experiment_setting = str(experiment_folder).split("/")
    ue = experiment_setting[-1]
    dataset = dataloader.dset
    y_true = dataset.targets
    if ue == "JMQR":
        quantiles = kwargs["quantiles"].numpy()
        y_preds_mean = np.array([x[0].detach().numpy() for x in y_preds]).squeeze(-1)
        y_preds_quantiles = np.array([x[1].detach().numpy() for x in y_preds]).squeeze()
        y_posterior_mean = qtools.posterior_mean(y_preds_quantiles, quantiles)
        y_posterior_stddev = qtools.posterior_std_dev(y_preds_quantiles, quantiles)
        y_posterior_stddev = y_posterior_stddev * math.sqrt(y_scaler.var_)
        columns = ["Y_TRUE", "Y_PRED_MEAN"] + [f"Y_PRED_Q{int(i)}" for i in quantiles * 100] + ["Y_POSTERIOR_MEAN", "Y_POSTERIOR_STDDEV"]
        # merge all
        y_preds_all = np.hstack((y_true, y_preds_mean, y_preds_quantiles, y_posterior_mean.reshape(-1, 1)))
        # inverse back
        y_preds_all = y_scaler.inverse_transform(y_preds_all)
        y_preds_all = np.hstack((y_preds_all, y_posterior_stddev.reshape(-1, 1)))
    elif ue == "JQR":
        quantiles = kwargs["quantiles"].numpy()
        y_preds_quantiles = np.array([x[0].detach().numpy() for x in y_preds]).squeeze()
        y_posterior_mean = qtools.posterior_mean(y_preds_quantiles, quantiles)
        y_posterior_stddev = qtools.posterior_std_dev(y_preds_quantiles, quantiles)
        y_posterior_stddev = y_posterior_stddev * math.sqrt(y_scaler.var_)
        columns = ["Y_TRUE"] + [f"Y_PRED_Q{int(i)}" for i in quantiles * 100] + ["Y_POSTERIOR_MEAN", "Y_POSTERIOR_STDDEV"]
        y_preds_all = np.hstack((y_true, y_preds_quantiles, y_posterior_mean.reshape(-1, 1)))
        y_preds_all = y_scaler.inverse_transform(y_preds_all)
        y_preds_all = np.hstack((y_preds_all, y_posterior_stddev.reshape(-1, 1)))
    
    elif ue in ["MVE", "MCD", "DE"]:
        y_means = np.array([x[0].item() for x in y_preds]).flatten().reshape(-1, 1)
        y_var = np.array([x[1].item() for x in y_preds]).flatten()
        y_var = y_var*y_scaler.var_
        y_true = y_scaler.inverse_transform(y_true).flatten()
        y_preds_means = y_scaler.inverse_transform(y_means).flatten()
        y_preds_stddev = np.sqrt(y_var).flatten()
        columns = ["Y_TRUE", "Y_PRED_MEAN", "Y_PRED_STDDEV"]
        y_preds_all = np.vstack((y_true, y_preds_means, y_preds_stddev)).T
    
    elif ue == "EDL":
        columns = ["Y_TRUE", "Y_PRED_MEAN", "Y_PRED_STDDEV_TOTAL", "Y_PRED_STDDEV_MODEL", "Y_PRED_STDDEV_DATA"]
        y_means = np.array([x[0].item() for x in y_preds]).flatten().reshape(-1, 1)
        lambdas = np.array([x[1].item() for x in y_preds]).flatten()
        alphas  = np.array([x[2].item() for x in y_preds]).flatten()
        betas   = np.array([x[3].item() for x in y_preds]).flatten()
        # betas   = betas * y_scaler.var_
        # total uncertainty
        var_total = np.array(betas) * (1 + 1 / np.array(lambdas)) / (np.array(alphas) - 1)
        var_alea =  np.array(betas) / (np.array(alphas) - 1)
        var_epis = np.array(betas) / (np.array(lambdas) * (np.array(alphas) - 1))
        var_total = var_total * y_scaler.var_
        var_aleas = var_alea * y_scaler.var_
        var_epist = var_epis * y_scaler.var_
        
        y_preds_stddev_total = np.sqrt(var_total).flatten()
        y_preds_stddev_epist = np.sqrt(var_epist).flatten()
        y_preds_stddev_aleas = np.sqrt(var_aleas).flatten()
        #y_true = np.array(test_dset.targets).flatten().reshape(-1,1)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        y_preds_means = y_scaler.inverse_transform(y_means).flatten()
        y_preds_all = np.vstack((y_true, y_preds_means, y_preds_stddev_total, y_preds_stddev_epist, y_preds_stddev_aleas)).T
    
    elif ue == "BASE":
        y_preds_mean = np.array([x[0].detach().numpy() for x in y_preds]).squeeze(-1)
        y_preds_all = np.hstack((y_true, y_preds_mean))
        y_preds_all = y_scaler.inverse_transform(y_preds_all)
        columns = ["Y_TRUE", "Y_PRED_MEAN"]
        
    
    y_preds_all_df = pd.DataFrame(data=y_preds_all, columns=columns, index=dataset.smiles)    
    y_preds_all_df = y_preds_all_df.rename_axis('SMILES')
    return y_preds_all_df