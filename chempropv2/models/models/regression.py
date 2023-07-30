import torch
import lightning.pytorch as pl
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from chempropv2.data.dataloader import TrainingBatch
from chempropv2.models.models.base import MPNN
from chempropv2.utils import find_nearest
from typing import Optional
from chempropv2.models.utils import rearrange
from chempropv2.models.metrics import MPIWMetric

class RegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "mse"
    _DEFAULT_METRIC = "rmse"
    def validation_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True)


class MveRegressionMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "mve"

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f=X_f)

        Y_mean, Y_var = Y.split(Y.shape[1] // 2, 1)
        Y_var = F.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, ...]:
        Y = super().predict_step(*args, **kwargs)[0]
        Y_mean, Y_var = Y.split(Y.shape[1] // 2, dim=1)

        return Y_mean, Y_var
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True)


class EvidentialMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "evidential"

    @property
    def n_targets(self) -> int:
        return 4

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)

        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, dim=1)
        lambdas = F.softplus(lambdas)
        alphas = F.softplus(alphas) + 1
        betas = F.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        Y = super().predict_step(*args, **kwargs)[0]
        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, 1)

        return means, lambdas, alphas, betas
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True)



class DeepEnsembleMPNN(pl.LightningModule):

    _DEFAULT_CRITERION = "mse"

    def __init__(self, num_models: int = 5, **kwargs):
        super().__init__()
        self.models = torch.nn.ModuleList([RegressionMPNN(**kwargs) for _ in range(num_models)])

    def forward(self, inputs, X_f) -> Tensor:
        return torch.stack([model.forward(inputs, X_f) for model in self.models], dim=0)

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        losses = []

        for model in self.models:
            preds = model((bmg, X_vd), X_f=features)
            loss = F.mse_loss(preds, targets)
            losses.append(loss * weights)

        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, prog_bar=True, batch_size=len(targets))

        return loss

    def validation_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        losses = []

        for model in self.models:
            preds = model((bmg, X_vd), X_f=features)
            loss = F.mse_loss(preds, targets)
            losses.append(loss * weights)

        loss = torch.stack(losses).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True,  batch_size=len(targets))

    def predict_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = []

        for model in self.models:
            pred = model((bmg, X_vd), X_f=features)
            preds.append(pred)
            
        preds = torch.vstack(preds)
        pred_mean = preds.mean(dim=0)
        pred_var = preds.var(dim=0)
        return pred_mean , pred_var

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class JointMeanQuantileRegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "jmq"
    _DEFAULT_METRIC = "mpiw"
    
    def __init__(self, quantiles: torch.Tensor = None, n_tasks: int = None, **kwargs):
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.quantiles = quantiles
        
    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)
        return Y

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask = torch.ones((len(targets), 1), device=targets.device)
        targets = targets.nan_to_num(nan=0.0)

        preds = self((bmg, X_vd), X_f=features)
        
        l = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log("train/loss", l, prog_bar=True)

        return l
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True)
        
    def predict_step(self, batch, batch_idx):
        y_preds = super().predict_step(batch, batch_idx)[0]
        y_means = y_preds[:, 0].unsqueeze(1) # n x 1
        y_quantiles = y_preds[:, 1:] # n x q
        return y_means, y_quantiles

class JointQuantileRegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "jq"
    _DEFAULT_METRIC = "mpiw"
    
    def __init__(self, quantiles: torch.Tensor = None, n_tasks: int = None, **kwargs):
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.quantiles = quantiles
    
    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)
        return Y
    
    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask = torch.ones((len(targets), len(self.quantiles)), device=targets.device)
        targets = targets.nan_to_num(nan=0.0)

        preds = self((bmg, X_vd), X_f=features)
        
        l = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log("train/loss", l, prog_bar=True)

        return l
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True, sync_dist=True)
    
    def predict_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        y_preds = super().predict_step(batch, batch_idx)[0]
        return y_preds
        
class MCDropoutMPNN(RegressionMPNN):

    _DEFAULT_CRITERION = "mse"

    def add_mc_iteration(self, mc_iteration):
        self.mc_iteration = mc_iteration
        self.Dropout = torch.nn.Dropout(p = 0.2) 


    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), 1), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, w_d=weights, lt_targets=lt_targets, gt_targets=gt_targets
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True, sync_dist=True)
    

    def predict_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        #enable
        # self.Dropout.train()
        self.ffn.train()
        pred = [self((bmg, X_vd), X_f=features).unsqueeze(0) for _ in range(self.mc_iteration)]
        preds = torch.vstack(pred)
        pred_mean = preds.mean(dim=0)
        pred_var = preds.var(dim=0)
        return pred_mean , pred_var