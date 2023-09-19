"""Run experiment command line toolkit
"""
import torch
import lightning.pytorch as pl
from argparse import ArgumentParser
from pathlib import Path
from utils import load_experiment_dataloaders, format_prediction_results
from lightning.pytorch.callbacks import EarlyStopping
from chempropv2.models import models, modules

torch.set_float32_matmul_precision("medium")

def main(args):
    out_dir = Path("experiments")
    experiment_name = f"{args.dataset}-{args.split}-fold{args.fold}-{args.ue}"
    experiment_folder = out_dir / experiment_name.replace("-", "/")
    experiment_folder.mkdir(parents=True, exist_ok=True)
    print(f"Running experiments: {experiment_name}.")
    # load data loaders for experiments
    (train_loader, val_loader, test_loader), y_scaler = load_experiment_dataloaders(dataset=args.dataset, 
                                                                                    split=args.split, 
                                                                                    fold_idx=args.fold, 
                                                                                    num_workers=args.num_workers, 
                                                                                    batch_size=args.batch_size)
    # define molecular encoder
    molenc = modules.molecule_block()
    # define early stop callback
    callbacks = [EarlyStopping(monitor="val/loss", mode="min", patience=10)]
    # define trainer
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1,
        num_nodes=1, 
        strategy="ddp",
        max_epochs=100,
        log_every_n_steps=30,
        callbacks=callbacks,
    )
    if args.ue == "BASE":
        # Base experiment
        model = models.RegressionMPNN(mpn_block=molenc,
                                      n_tasks=1,
                                      ffn_num_layers=3,
                                      dropout=0.2)
    elif args.ue == "JMQR":
        # Joint Mean Quantile Regression
        quantiles = torch.arange(0.05, 1, 0.05)
        model = models.JointMeanQuantileRegressionMPNN(mpn_block=molenc, 
                                                       n_tasks=len(quantiles) + 1, 
                                                       ffn_num_layers=3,
                                                       dropout=0.2,
                                                       quantiles=quantiles)

    elif args.ue == "JQR":
        # Joint Quantile Regression
        quantiles = torch.arange(0.05, 1, 0.05)
        model = models.JointQuantileRegressionMPNN(mpn_block=molenc, 
                                                   n_tasks=len(quantiles), 
                                                   ffn_num_layers=3,
                                                   dropout=0.2,
                                                   quantiles=quantiles)
    elif args.ue == "DE":
        model = models.DeepEnsembleMPNN(mpn_block=molenc,
                                        n_tasks=1, 
                                        ffn_num_layers=3,
                                        num_models=5,
                                        dropout=0.2)
    
    elif args.ue == "MCD":
        model = models.MCDropoutMPNN(mpn_block=molenc,
                                     n_tasks=1, 
                                     ffn_num_layers=3,
                                     dropout=0.2)
        model.add_mc_iteration(100)

    elif args.ue == "MVE":
        model = models.MveRegressionMPNN(mpn_block=molenc,
                                         n_tasks=1, 
                                         ffn_num_layers=3,
                                         dropout=0.2)
    
    elif args.ue == "EDL":
        model = models.EvidentialMPNN(mpn_block=molenc,
                                      n_tasks=1, 
                                      ffn_num_layers=3,
                                      dropout=0.2)
        
    # fit on trainset
    trainer.fit(model, train_loader, val_loader)
    # predict on validation set
    val_results = trainer.predict(model=model, dataloaders=val_loader)
    # predict on test set
    test_results = trainer.predict(model=model, dataloaders=test_loader)
    if args.ue in ["JMQR", "JQR"]:
        # save validation results
        preds_val = format_prediction_results(val_loader, val_results, y_scaler, experiment_folder, quantiles=quantiles)
        preds_test = format_prediction_results(test_loader, test_results, y_scaler, experiment_folder, quantiles=quantiles)
    else:
        preds_val = format_prediction_results(val_loader, val_results, y_scaler, experiment_folder)
        preds_test = format_prediction_results(test_loader, test_results, y_scaler, experiment_folder)

    preds_val.to_csv(f"{experiment_folder}/valset_predictions.csv")
    # save test results
    preds_test.to_csv(f"{experiment_folder}/testset_predictions.csv")
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Conformalized Molecular ADMET Properties Prediction")
    parser.add_argument("--dataset", type=str, help="Name of dataset", default='Solubility_Wang2020')
    parser.add_argument("--split", type=str, help="Name of split strategy, could be 'IVIT', 'IVOT', or 'OVOT'", default='OVOT')
    parser.add_argument("--fold", type=int, help="Index of CV fold, 0 to 9", default=6)
    parser.add_argument("--ue", type=str, help="Name of uncertainty estimation method", default="JQR")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=50)
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=6)
    parser.add_argument("--device", type=int, help="Index of CUDA device", default=1)
    parser.add_argument("--lds", type=bool, help="Label Density Smooth", default=False)
    args = parser.parse_args()
    main(args)