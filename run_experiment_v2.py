import pickle
from argparse import ArgumentParser
from pathlib import Path

import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
import uncertainty_toolbox as uct
from pytorch_lightning.callbacks import EarlyStopping
from rdkit.rdBase import LogToPythonStderr
from chempropv2 import data, featurizers
from chempropv2.models import models, modules
from chempropv2.utils import makedirs, find_nearest
from chempropv2.conformal import run_icp, evaluate_icp

LogToPythonStderr()
# set precision
torch.set_float32_matmul_precision('medium')
# number of cross-validation folds
N_FOLDS: int = 10
NUM_WORKERS: int = 12
OUT_DIR: str = "/root/autodl-tmp/Conformal-ADMET-Prediction/experiments"

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


def main(args):
    # set experiment name and output directory
    experiment_name = f"{OUT_DIR}/{args.dataset}-{args.split}-{args.ue}/fold_{args.fold}"
    dataset_path = Path(f"/root/autodl-tmp/Conformal-ADMET-Prediction/data/curated/{args.dataset}.csv")
    split_path = Path(f"/root/autodl-tmp/Conformal-ADMET-Prediction/data/split_idxs/{args.dataset}_{args.split}.pkl")
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
        predict = trainer.predict(model=model, dataloaders=test_loader)
        first_column = [row[0] for row in predict]
        predict = np.array(first_column).flatten().reshape(-1, 1)
        predict = y_scaler.inverse_transform(predict).flatten()
        y_true = np.array(test_dset.targets)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        metrics = uct.get_all_accuracy_metrics(y_true=y_true, y_pred=predict)
        df = pd.DataFrame({'y_true': y_true, 'y_predict': predict})
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_PREDICTIONS.csv" 
        df.to_csv(save_path, index=False)
        metrics_index = [f'{i}' for i, value in enumerate(metrics)]
        df = pd.DataFrame(metrics, index = metrics_index)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        


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
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true, verbose=False)
        df = pd.DataFrame({'y_true': y_true, 'y_predict_means': y_means, 'y_predict_std':predictions_std})
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_PREDICTIONS.csv" 
        df.to_csv(save_path, index=False)
        metrics_index = [f'{i}' for i, value in enumerate(metrics)]
        df = pd.DataFrame(metrics, index = metrics_index)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
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
        y_var = var*y_scaler_var
        predictions_std = np.sqrt(y_var).flatten()
        y_true = np.array(test_dset.targets).flatten().reshape(-1,1)
        y_true = y_scaler.inverse_transform(y_true).flatten()
        y_means = y_scaler.inverse_transform(y_means).flatten()
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true,verbose=False)
        df = pd.DataFrame({'y_true': y_true, 'y_predict_means': y_means, 'y_predict_std':predictions_std})
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_PREDICTIONS.csv" 
        df.to_csv(save_path, index=False)
        metrics_index = [f'{i}' for i, value in enumerate(metrics)]
        df = pd.DataFrame(metrics, index = metrics_index)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        print("EDL Done")

    # model
    elif args.ue == "QR":
        # define trainer
        callbacks = [EarlyStopping(monitor="val/loss", mode="min")] #QR用的是mpiw,MVE、EDL用rmse才可以，需要修改
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        quantiles = torch.arange(0.05, 1, 0.05)
        alpha = 0.1
        model = models.QuantileRegressionMPNN(mpn_block=molenc, 
                                              n_tasks=len(quantiles), 
                                              ffn_num_layers=3,
                                              dropout=0.2,
                                              quantiles=quantiles)
        trainer.fit(model, train_loader, val_loader)
        # make prediction
        test_results = trainer.predict(model=model, dataloaders=test_loader)
        y_preds_test = np.array([x[1].detach().numpy() for x in test_results]).squeeze()
        y_tgt_test = test_dset.targets
        x_test = test_dset.smiles
        
        print("Run split-conformal QR")
        val_results = trainer.predict(model=model, dataloaders=test_loader)
        y_preds_val = np.array([x[1].detach().numpy() for x in val_results]).squeeze()
        y_tgt_val = val_dset.targets
        y_pis_test = run_icp(y_tgt_val, y_preds_val, y_preds_test, quantiles, alpha)
        icp_metrics = evaluate_icp(y_pis_test, y_tgt_test)
        print(icp_metrics)
        # # save predictions of QR
        # #prediction_results = np.array([x.detach().numpy() for x in y_preds_test])
        # #y_medians = np.array([x[0].item() for x in predict_results]).flatten().reshape(-1, 1)
        # y_preds = np.array([x[1].detach().numpy() for x in predict_results]).squeeze()
        # y_preds_all = np.hstack((y_test, y_preds))
        # y_preds_all = y_scaler.inverse_transform(y_preds_all)
        # columns = ["Y_TRUE"] + [f"Y_PRED_Q{int(i * 100)}" for i in quantiles]
        # y_preds_df = pd.DataFrame(data=y_preds_all, columns=columns, index=x_test)
        # y_preds_df.to_csv("QR_predictions.csv")
        
        # # # run conformalized QR
        # y_calib_true = val_dset.targets
        # calibration_out = trainer.predict(model=model, dataloaders=val_loader)
        # y_calib_preds = np.array([x[1].detach().numpy() for x in calibration_out]).squeeze()
        
        # # search for lower and upper predictions
        # lower_idx = 0
        # upper_idx = -1
        # y_intervals_calib = y_preds_calib[:, [lower_idx, upper_idx]]
        
        # save predictions of CQR
        
        # evaluate CQR
        
        
        
        # y_medians = np.array([x[0].item() for x in results]).flatten().reshape(-1, 1)
        # y_intervals = np.array([x[1].flatten().detach().numpy() for x in results])
        # y_true = y_scaler.inverse_transform(y_true)
        # y_medians = y_scaler.inverse_transform(y_medians)
        # y_intervals = y_scaler.inverse_transform(y_intervals)
        # coverage = reliability(y_true, y_intervals[:, 0], y_intervals[:, 1])
        # efficiency = np.abs(y_intervals[:, 0], y_intervals[:, 1]).mean()
        # print(f"Validity: {coverage:.2f} Efficiency: {efficiency:.2f}")
        # pd.DataFrame({'y_true': y_true.flatten(), 'y_pred': y_medians.flatten()}).to_csv("test.csv")
        # metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true)
                
    
    elif args.ue == "MCD":
        model = models.MCDropoutMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=3, dropout = 0.5)
        model.add_mc_iteration(20)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")] #QR用的是mpiw,MVE、EDL用rmse才可以，需要修改
        trainer = pl.Trainer(
            # logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
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
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true,verbose=False)
        df = pd.DataFrame({'y_true': y_true, 'y_predict_means': y_means, 'y_predict_std':predictions_std})
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_PREDICTIONS.csv" 
        df.to_csv(save_path, index=False)
        metrics_index = [f'{i}' for i, value in enumerate(metrics)]
        df = pd.DataFrame(metrics, index = metrics_index)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
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
        metrics = uct.metrics.get_all_metrics(y_means, predictions_std, y_true,verbose=False)
        df = pd.DataFrame({'y_true': y_true, 'y_predict_means': y_means, 'y_predict_std':predictions_std})
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}"
        os.makedirs(save_path, exist_ok=True)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_PREDICTIONS.csv" 
        df.to_csv(save_path, index=False)
        metrics_index = [f'{i}' for i, value in enumerate(metrics)]
        df = pd.DataFrame(metrics, index = metrics_index)
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        print("DE Done")
    else:
        raise ValueError("Wrong UE method specified!")
    trainer.save_checkpoint(f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}.ckpt")
    print("Done")



if __name__ == "__main__":
    parser = ArgumentParser(description="Conformalized Molecular ADMET Properties Prediction")
    parser.add_argument("--dataset", type=str, help="Name of dataset", default='VDss_Liu2022')
    parser.add_argument("--split", type=str, help="Name of split strategy, could be 'IVIT', 'IVOT', or 'OVOT'", default='IVIT')
    parser.add_argument("--fold", type=int, help="Index of CV fold", default=0)
    parser.add_argument("--ue", type=str, help="Name of uncertainty estimation method")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=50)
    parser.add_argument("--conformal", type=bool, help="Run conformal prediction or not", default=False)
    parser.add_argument("--alpha", type=float, help="Expected coverage rate", default=0.1)
    methods = ["BASE","MCD","DE","MVE","EDL"]
    DataSets = ["HalfLife_Liu2022","VDss_Liu2022","Solubility_Wang2020","RLM_Fang2023","Permeability_MDCK_Fang2023","Permeability_Caco2_Wang2020","Lipophilicity_Wang2020","LD50_Lunghini2019","hPPB_Lou2022","HLM_Fang2023"]
    SPlit = ['IVIT', 'IVOT', 'OVOT']
    for k in SPlit:
        for j in DataSets:
            for i in methods:
                parser.set_defaults(ue=i, dataset=j, split =k)
                args = parser.parse_args()
                main(args)
