import pickle
from argparse import ArgumentParser
from pathlib import Path
import lightning.pytorch as pl

import os
import numpy as np
import pandas as pd
import torch
import uncertainty_toolbox as uct
from lightning.pytorch.callbacks import EarlyStopping
from rdkit.rdBase import LogToPythonStderr
from chempropv2 import data, featurizers
from chempropv2.models import models, modules
from chempropv2.utils import makedirs, find_nearest
from chempropv2.conformal import run_icp, evaluate_icp
from chempropv2.utils import estimate_sample_weights

LogToPythonStderr()
# set precision
torch.set_float32_matmul_precision('medium')
# number of cross-validation folds
N_FOLDS: int = 10
NUM_WORKERS: int = 12
OUT_DIR: str = Path("experiments")

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
    # sample size of train set
    n_train = len(train_df)
    # extract targets' values from train & val for sample weights estimation
    targets_train_val = np.concatenate((train_df["Y"].values, valid_df["Y"].values))
    w = estimate_sample_weights(targets=targets_train_val)
    w_train = w[:n_train]
    w_val = w[n_train:]
    # convert dataframes to datasets
    train_dset, val_dset, test_dset = df2dset(train_df, data_weight=w_train), df2dset(valid_df, data_weight=w_val), df2dset(test_df)
    return (train_dset, val_dset, test_dset)
    

def main(args):
    # set experiment name and output directory
    experiment_name = f"{args.dataset}-{args.split}-{args.ue}-fold_{args.fold}"
    experiment_folder = OUT_DIR / experiment_name.replace("-", "/")
    makedirs(experiment_folder)
    dataset_path = Path(f"data/curated/{args.dataset}.csv")
    split_path = Path(f"data/split_idxs/{args.dataset}_{args.split}.pkl")
    train_dset, val_dset, test_dset = load_molecule_datasets(dataset_path=dataset_path,
                                                             split_path=split_path,
                                                             fold_idx=args.fold)
    # normalize target values
    y_scaler = train_dset.normalize_targets()
    print(f"Normalize targets with {y_scaler.mean_} (+/- {np.sqrt(y_scaler.var_)})")
    print(y_scaler.mean_)
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
        model = models.RegressionMPNN(mpn_block=molenc, 
                                      n_tasks=1, 
                                      ffn_num_layers=3,
                                      dropout=0.2)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")]
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
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
        df = pd.DataFrame([metrics])
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        


    # model 1: DMPNN with MVE loss
    elif args.ue == "MVE":
        model = models.MveRegressionMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=3)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")]
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
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


        metrics_values = dict()
        for outer_key, inner_dict in metrics.items():
            for inner_key, value in inner_dict.items():
                metrics_values[inner_key]=value
        
        ma_adv_group_cal = metrics_values['ma_adv_group_cal']
        rms_adv_group_cal = metrics_values['rms_adv_group_cal']
        del metrics_values['ma_adv_group_cal']
        del metrics_values['rms_adv_group_cal']
        group_sizes = ma_adv_group_cal['group_sizes']
        ma_adv_group_cali_mean = ma_adv_group_cal['adv_group_cali_mean']
        ma_adv_group_cali_stderr = ma_adv_group_cal['adv_group_cali_stderr']
        rms_adv_group_cali_mean   = rms_adv_group_cal['adv_group_cali_mean']
        rms_adv_group_cali_stderr = rms_adv_group_cal['adv_group_cali_stderr']

        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_mean_group{group_sizes[i]}'] = ma_adv_group_cali_mean[i]    
        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_stderr_group{group_sizes[i]}'] = ma_adv_group_cali_stderr[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_mean_group{group_sizes[i]}'] = rms_adv_group_cali_mean[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_stderr_group{group_sizes[i]}'] = rms_adv_group_cali_stderr[i]        

        df = pd.DataFrame([metrics_values])


        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        print("MVE Done")
        


    # model 2: DMPNN with evidential loss
    elif args.ue == "EDL":
        model = models.EvidentialMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=3)
        callbacks = [EarlyStopping(monitor="val/rmse", mode="min")]
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
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


        metrics_values = dict()
        for outer_key, inner_dict in metrics.items():
            for inner_key, value in inner_dict.items():
                metrics_values[inner_key]=value
        
        ma_adv_group_cal = metrics_values['ma_adv_group_cal']
        rms_adv_group_cal = metrics_values['rms_adv_group_cal']
        del metrics_values['ma_adv_group_cal']
        del metrics_values['rms_adv_group_cal']
        group_sizes = ma_adv_group_cal['group_sizes']
        ma_adv_group_cali_mean = ma_adv_group_cal['adv_group_cali_mean']
        ma_adv_group_cali_stderr = ma_adv_group_cal['adv_group_cali_stderr']
        rms_adv_group_cali_mean   = rms_adv_group_cal['adv_group_cali_mean']
        rms_adv_group_cali_stderr = rms_adv_group_cal['adv_group_cali_stderr']

        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_mean_group{group_sizes[i]}'] = ma_adv_group_cali_mean[i]    
        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_stderr_group{group_sizes[i]}'] = ma_adv_group_cali_stderr[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_mean_group{group_sizes[i]}'] = rms_adv_group_cali_mean[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_stderr_group{group_sizes[i]}'] = rms_adv_group_cali_stderr[i]        

        df = pd.DataFrame([metrics_values])


        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        print("EDL Done")

    # model
    elif args.ue == "JQR":
        callbacks = [EarlyStopping(monitor="val/loss", mode="min", patience=10)]
        
        
        pass
    
    elif args.ue == "JMQR":
        # define trainer
        callbacks = [EarlyStopping(monitor="val/loss", mode="min", patience=10)]
        trainer = pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=100,
            log_every_n_steps=30,
            callbacks=callbacks
        )
        quantiles = torch.arange(0.05, 1, 0.05)
        model = models.JointMeanQuantileRegressionMPNN(mpn_block=molenc, 
                                                       n_tasks=len(quantiles) + 1, 
                                                       ffn_num_layers=3,
                                                       dropout=0.2,
                                                       quantiles=quantiles)
        trainer.fit(model, train_loader, val_loader)
        # inference on validation set
        val_preds = trainer.predict(model=model, dataloaders=val_loader)
        y_true_val = val_dset.targets
        y_preds_mean_val = np.array([x[0].detach().numpy() for x in val_preds]).squeeze(-1)
        y_preds_quantiles_val = np.array([x[1].detach().numpy() for x in val_preds]).squeeze()
        columns = ["Y_TRUE", "Y_PRED_MEAN"] + [f"Y_PRED_Q{int(i * 100)}" for i in quantiles]
        y_preds_valset = np.hstack((y_true_val, y_preds_mean_val, y_preds_quantiles_val))
        y_preds_valset = pd.DataFrame(data=y_preds_valset, columns=columns, index=val_dset.smiles)
        y_preds_valset.to_csv("JMQR_predictions_IVIT_fold0_valset.csv")
        # reverse
        y_preds_valset_inv = y_scaler.inverse_transform(y_preds_valset)
        y_preds_valset_inv_df = pd.DataFrame(data=y_preds_valset_inv, columns=columns, index=val_dset.smiles)
        y_preds_valset_inv_df.to_csv("JMQR_predictions_IVIT_fold0_valset_inv.csv")
        
        # inference on testset
        test_results = trainer.predict(model=model, dataloaders=test_loader)
        y_true_test = test_dset.targets
        y_preds_mean_test = np.array([x[0].detach().numpy() for x in test_results]).squeeze(-1)
        y_preds_quantiles_test = np.array([x[1].detach().numpy() for x in test_results]).squeeze()

        columns = ["Y_TRUE", "Y_PRED_MEAN"] + [f"Y_PRED_Q{int(i * 100)}" for i in quantiles]
        y_preds_testset = np.hstack((y_true_test, y_preds_mean_test, y_preds_quantiles_test))
        y_preds_testset_df = pd.DataFrame(data=y_preds_testset, columns=columns, index=test_dset.smiles)
        y_preds_testset_df.to_csv("JMQR_predictions_IVIT_fold0_testset.csv")
        y_preds_testset_inv = y_scaler.inverse_transform(y_preds_testset)
        y_preds_testset_inv_df = pd.DataFrame(data=y_preds_testset_inv, columns=columns, index=test_dset.smiles)
        y_preds_testset_inv_df.to_csv("JMQR_predictions_IVIT_fold0_testset_inv.csv")
                
    
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


        metrics_values = dict()
        for outer_key, inner_dict in metrics.items():
            for inner_key, value in inner_dict.items():
                metrics_values[inner_key]=value
        
        ma_adv_group_cal = metrics_values['ma_adv_group_cal']
        rms_adv_group_cal = metrics_values['rms_adv_group_cal']
        del metrics_values['ma_adv_group_cal']
        del metrics_values['rms_adv_group_cal']
        group_sizes = ma_adv_group_cal['group_sizes']
        ma_adv_group_cali_mean = ma_adv_group_cal['adv_group_cali_mean']
        ma_adv_group_cali_stderr = ma_adv_group_cal['adv_group_cali_stderr']
        rms_adv_group_cali_mean   = rms_adv_group_cal['adv_group_cali_mean']
        rms_adv_group_cali_stderr = rms_adv_group_cal['adv_group_cali_stderr']

        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_mean_group{group_sizes[i]}'] = ma_adv_group_cali_mean[i]    
        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_stderr_group{group_sizes[i]}'] = ma_adv_group_cali_stderr[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_mean_group{group_sizes[i]}'] = rms_adv_group_cali_mean[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_stderr_group{group_sizes[i]}'] = rms_adv_group_cali_stderr[i]        

        df = pd.DataFrame([metrics_values])
        save_path = f"{OUT_DIR}/{args.dataset}/{args.split}/{args.ue}/{args.fold}/{args.dataset}_{args.split}_{args.ue}_fold_{args.fold}_METRICS.csv"
        df.to_csv(save_path)
        print("MCD Done")



    elif args.ue == "DE":
        model = models.DeepEnsembleMPNN(mpn_block=molenc, n_tasks=1, ffn_num_layers=3, num_models=5)
        callbacks = [EarlyStopping(monitor="val/loss", mode="min")] 
        trainer = pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
            max_epochs=200,
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

        
        metrics_values = dict()
        for outer_key, inner_dict in metrics.items():
            for inner_key, value in inner_dict.items():
                metrics_values[inner_key]=value
        
        ma_adv_group_cal = metrics_values['ma_adv_group_cal']
        rms_adv_group_cal = metrics_values['rms_adv_group_cal']
        del metrics_values['ma_adv_group_cal']
        del metrics_values['rms_adv_group_cal']
        group_sizes = ma_adv_group_cal['group_sizes']
        ma_adv_group_cali_mean = ma_adv_group_cal['adv_group_cali_mean']
        ma_adv_group_cali_stderr = ma_adv_group_cal['adv_group_cali_stderr']
        rms_adv_group_cali_mean   = rms_adv_group_cal['adv_group_cali_mean']
        rms_adv_group_cali_stderr = rms_adv_group_cal['adv_group_cali_stderr']

        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_mean_group{group_sizes[i]}'] = ma_adv_group_cali_mean[i]    
        for i in range(len(group_sizes)):
            metrics_values[f'ma_adv_group_cali_stderr_group{group_sizes[i]}'] = ma_adv_group_cali_stderr[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_mean_group{group_sizes[i]}'] = rms_adv_group_cali_mean[i]
        for i in range(len(group_sizes)):
            metrics_values[f'rms_adv_group_cali_stderr_group{group_sizes[i]}'] = rms_adv_group_cali_stderr[i]        

        df = pd.DataFrame([metrics_values])


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
    parser.add_argument("--ue", type=str, help="Name of uncertainty estimation method", default="JMQR")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=50)    
    methods = ["JMQR"]
    DataSets = ["hPPB_Lou2022"]
    SPlit = ['IVIT']
    for k in SPlit:
        for j in DataSets:
            for i in methods:
                parser.set_defaults(ue=i, dataset=j, split =k)
                args = parser.parse_args()
                main(args)
