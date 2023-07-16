import pickle
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path
from chempropv2 import data
from chempropv2 import featurizers
from chempropv2.models import modules, models
from chempropv2.utils import makedirs
from pytorch_lightning.callbacks import EarlyStopping
import uncertainty_toolbox as uct


# set precision
torch.set_float32_matmul_precision('medium')
# number of cross-validation folds
N_FOLDS: int = 10
NUM_WORKERS: int = 12
OUT_DIR: str = "./experiments"

def df2dset(df):
    smis = df["X"].tolist()
    scores = df["Y"].tolist()
    scores = np.array(scores).reshape(-1, 1)
    datapoints = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, scores)]
    # set molecule featurizer
    featurizer = featurizers.MoleculeFeaturizer()
    dset = data.MoleculeDataset(datapoints, featurizer)        
    return dset

def load_molecule_datasets(dataset_path, split_path, fold_idx: int = 0):
    
    # load dataset CSV file
    df = pd.read_csv(dataset_path)
    # load split's indexes pickle file
    with open(split_path, "rb") as handle:
        splits = pickle.load(handle)
    # load split at fold-i
    split_idxs = splits[fold_idx]
    # load indexs for train/val/test sets
    train_idx, valid_idx, test_idx = split_idxs[0], split_idxs[1], split_idxs[2]
    # load train/val/test dataframes
    train_df, valid_df, test_df = df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]
    # convert dataframes to datasets
    train_dset, val_dset, test_dset = df2dset(train_df), df2dset(valid_df), df2dset(test_df)
    return (train_dset, val_dset, test_dset)

def reliability(y: np.ndarray,
                p_low: np.ndarray,
                p_high: np.ndarray):
    return ((p_low < y) * (y < p_high)).mean()

def inverse_transform_variance(stds, X):
        X = np.array(X).astype(float)
        transformed_with_nan = X * (stds ** 2)
        transformed_with_none = np.where(np.isnan(transformed_with_nan), None , transformed_with_nan)
        return transformed_with_none

def main(args):
    # set experiment name and output directory
    experiment_name = f"{OUT_DIR}/{args.dataset}-{args.split}-{args.ue}/fold_{args.fold}"
    makedirs(experiment_name)
    dataset_path = Path(f"/root/autodl-tmp/Conformal-ADMET-Prediction/data/curated/{args.dataset}.csv")
    split_path = Path(f"/root/autodl-tmp/Conformal-ADMET-Prediction/data/splitted/{args.dataset}_{args.split}.pkl")
    train_dset, val_dset, test_dset = load_molecule_datasets(dataset_path=dataset_path,
                                                             split_path=split_path,
                                                             fold_idx=args.fold)
    # normalize target values
    y_scaler = train_dset.normalize_targets()
    y_scaler_var = y_scaler.var_
    val_dset.normalize_targets(y_scaler)
    test_dset.normalize_targets(y_scaler)
    # create dataloader
    train_loader = data.MolGraphDataLoader(train_dset, num_workers=NUM_WORKERS, batch_size=args.batch_size)
    val_loader = data.MolGraphDataLoader(val_dset, num_workers=NUM_WORKERS, shuffle=False, batch_size=args.batch_size)
    test_loader = data.MolGraphDataLoader(test_dset, num_workers=NUM_WORKERS, shuffle=False, batch_size=args.batch_size)
    # for n_e in range(args.ensembles):
    # define molecular encoder
    molenc = modules.molecule_block()
    # vallina MSE MPNN
    # model 0: DMPNN with MSE loss
    if args.ue == "BASE":
        model = models.RegressionMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2)
        trainer.fit(model, train_loader, val_loader)

    # model 1: DMPNN with MVE loss
    elif args.ue == "MVE":
        model = models.MveRegressionMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")]
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=50,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        trainer.fit(model, train_loader, val_loader)
        #输入得是一维的ndarray
        results = np.array(trainer.predict(model=model, dataloaders=test_loader))
        y_means = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1)
        y_var = np.array([x[1].item() for x in results]).flatten()
        # stds = np.sqrt(y_var).flatten()
        # y_var = inverse_transform_variance(stds, y_var).astype(float)
        y_var = y_var*y_scaler_var
        y_true = np.array(test_dset.targets)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        y_means = y_scaler.inverse_transform(y_means).flatten()
        predictions_std = np.sqrt(y_var)
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true)
        print("MVE Done")
        


    # model 2: DMPNN with evidential loss
    elif args.ue == "EDL":
        model = models.EvidentialMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")]
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=50,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        trainer.fit(model, train_loader, val_loader)
        results = np.array(trainer.predict(model=model, dataloaders=test_loader)) # means, lambdas, alphas, betas
        y_means = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1)
        lambdas = np.array([x[1].item() for x in results]).flatten()
        alphas  = np.array([x[2].item() for x in results]).flatten()
        betas   = np.array([x[3].item() for x in results]).flatten()
        var = np.array(betas) * (1 + 1 / np.array(lambdas)) / (np.array(alphas) - 1)
        stds = np.sqrt(var)
        # y_var = inverse_transform_variance(stds, var).astype(float)
        y_var = var*y_scaler_var
        predictions_std = np.sqrt(y_var).flatten()
        y_true = np.array(test_dset.targets).flatten().reshape(-1,1)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        y_means = y_scaler.inverse_transform(y_means).flatten()
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true)
        print("EDL Done")



    # model
    elif args.ue == "QR":
        # define trainer
        callbacks = [EarlyStopping(monitor="val/mpiw", mode="min")] #QR用的是mpiw,MVE、EDL用rmse才可以，需要修改
        
        trainer = pl.Trainer(
            # logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=50,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        model = models.QuantileRegressionMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2, quantiles=torch.arange(0.01, 1, 0.01))
        trainer.fit(model, train_loader, val_loader)
        results = trainer.predict(model=model, dataloaders=test_loader)
        y_true = test_dset.targets
        y_medians = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1)
        y_intervals = np.array([x[1].flatten().detach().numpy() for x in results])
        y_true = y_scaler.inverse_transform(y_true)
        y_medians = y_scaler.inverse_transform(y_medians)
        y_intervals = y_scaler.inverse_transform(y_intervals)
        coverage = reliability(y_true, y_intervals[:, 0], y_intervals[:, 1])
        efficiency = np.abs(y_intervals[:, 0], y_intervals[:, 1]).mean()
        print(f"Validity: {coverage:.2f} Efficiency: {efficiency:.2f}")
        pd.DataFrame({'y_true': y_true.flatten(), 'y_pred': y_medians.flatten()}).to_csv("test.csv")
        print("QR Done")


    elif args.ue == "MCD":
        model = models.MCDropoutMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2, dropout = 0.2)
        model.add_mc_iteration(5)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")] #QR用的是mpiw,MVE、EDL用rmse才可以，需要修改
        trainer = pl.Trainer(
            # logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=50,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        y_true = np.array(test_dset.targets).flatten().reshape(-1,1)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        trainer.fit(model, train_loader, val_loader)
        results = trainer.predict(model=model, dataloaders=test_loader) #按我的理解，result就是means
        y_means = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1) 
        y_var   = np.array([x[1].item() for x in results]).flatten() 
        # stds = np.sqrt(y_var).flatten()
        # y_var = inverse_transform_variance(stds, y_var).astype(float)
        y_var = y_var*y_scaler_var
        y_means = y_scaler.inverse_transform(y_means).flatten()
        predictions_std = np.sqrt(y_var).flatten()
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true)
        print("MCD Done")



    elif args.ue == "DE":
        model = models.DeepEnsembleMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=2,num_models=5)
        callbacks = [EarlyStopping(monitor="val/loss", mode="min")] 
        trainer = pl.Trainer(
            # logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=50,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        y_true = np.array(test_dset.targets).flatten().reshape(-1,1)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        trainer.fit(model, train_loader, val_loader)
        results = trainer.predict(model=model, dataloaders=test_loader) #按我的理解，result就是means
        y_means = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1) 
        y_var   = np.array([x[1].item() for x in results]).flatten() 
        y_var = y_var*y_scaler_var
        y_means = y_scaler.inverse_transform(y_means).flatten()
        predictions_std = np.sqrt(y_var).flatten()
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true)
        print("DE Done")


    else:
        raise ValueError("Wrong UE method specified!")
    print("Done")
    # method = args.ue
    # smiles = test_dset.smiles
    # y_true = test_dset.targets
    # if method == "MVE":
    #     pred_mean = np.array([x[0].item() for x in results]).reshape(1, -1)
    #     pred_var = np.array([x[1].item() for x in results]).reshape(1, -1)
    #     pred_mean = y_scaler.inverse_transform(pred_mean)
    #     pred_var = pred_var * y_scaler.scale_**2
    #     pred_std = np.sqrt(pred_var)
    #     df = pd.DataFrame(data={"SMILES": smiles, "Y_TRUE": y_true, "Y_PRED": pred_mean, "PRED_VAR": pred_var, "PRED_STD": pred_std})
    # elif method == "EDL":
    #     pred_mean = np.array([x[0].item() for x in results]).reshape(1, -1)
    #     # calculate variance from EDL
        
    # elif method == "QR":
    #     # approximate means with medians
    #     pass    
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Conformalized Molecular ADMET Properties Prediction")
    parser.add_argument("--dataset", type=str, help="Name of dataset", default='VDss_Liu2020')
    parser.add_argument("--split", type=str, help="Name of split strategy, could be 'IVIT', 'IVOT', or 'OVOT'", default='IVIT')
    parser.add_argument("--ensemble", type=int, help="Number of ensembles", default = 3)
    parser.add_argument("--fold", type=int, help="Index of CV fold", default=0)
    parser.add_argument("--ue", type=str, help="Name of uncertainty estimation method")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=100)
    parser.add_argument("--conformal", type=bool, help="Run conformal prediction or not", default=False)
    parser.add_argument("--alpha", type=float, help="Expected coverage rate", default=0.1)
    means = ["MCD","DE","EDL","MVE","QR"]
    for i in means:
        parser.set_defaults(ue=i)
        args = parser.parse_args()
        main(args)
