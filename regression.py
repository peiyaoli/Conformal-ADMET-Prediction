import torch
from torch import Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
from chempropv2.data.dataloader import TrainingBatch
from chempropv2.models.models.base import MPNN
from chempropv2.utils import find_nearest

class RegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "mse"
    _DEFAULT_METRIC = "rmse"



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
    


class MCDropoutMPNN(RegressionMPNN):

    _DEFAULT_CRITERION = "mse"

    def add_mc_iteration(self, mc_iteration):
        self.mc_iteration = mc_iteration
        self.Dropout = torch.nn.Dropout(p = 0.5) 


    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        preds = self((bmg, X_vd), X_f=features)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        losses = [
            metric(preds, targets, mask, lt_targets=lt_targets, gt_targets=gt_targets)
            for metric in self.metrics
        ]
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}
        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets), prog_bar=True)
    

    def predict_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        #enable
        self.Dropout.train()
        pred = [self.Dropout(self((bmg, X_vd), X_f=features)).unsqueeze(0) for _ in range(self.mc_iteration)]
        preds = torch.vstack(pred)
        pred_mean = preds.mean(dim=0)
        pred_var = preds.var(dim=0)
        return pred_mean , pred_var



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
            losses.append(loss)

        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, prog_bar=True)

        return loss


    def validation_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        losses = []

        for model in self.models:
            preds = model((bmg, X_vd), X_f=features)
            loss = F.mse_loss(preds, targets)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        self.log("val/loss", loss, prog_bar=True)


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





class QuantileRegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_QUANTILES = torch.arange(0.01, 1, 0.01)
    _DEFAULT_CRITERION = "smoothpinball"
    _DEFAULT_METRIC = "mpiw"
    
    def __init__(self, quantiles: torch.Tensor = None, no_crossing: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = quantiles
        self.no_crossing = no_crossing
            
    @property
    def n_targets(self) -> int:
        return len(self._DEFAULT_QUANTILES)
    
    
    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)
        if self.no_crossing:
            Y, _ = torch.sort(Y, dim=1)
        return Y

    
    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        # mask = torch.ones((targets.size(0), len(self.quantile_grids)), device=targets.device)
        mask = torch.ones((len(targets), self.n_targets), device=targets.device)
        targets = targets.nan_to_num(nan=0.0)

        preds = self((bmg, X_vd), X_f=features)

        # l = self.criterion(
        #     preds, targets, mask, weights=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantile_grids=self.quantile_grids, alpha=0.01
        # )
        l = self.criterion(preds=preds, targets=targets, alpha=0.1, beta=10**2, epsilon=1e-3, quantiles=self.quantiles, mask=mask)

        self.log("train/loss", l, prog_bar=True)

        return l


    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        *_, targets, _, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        #mask = targets.isfinite()
        #targets = targets.nan_to_num(nan=0.0)
        # alpha=0.1
        # q_lo = alpha / 2
        # q_hi = 1 - q_lo
        # q_lo_idx = find_nearest(self.quantile_grids, q_lo)
        # q_hi_idx = find_nearest(self.quantile_grids, q_hi)
        # pis = preds[:, [q_lo_idx, q_hi_idx]]
        # losses = [
        #     metric(preds, targets, mask, lt_targets=lt_targets, gt_targets=gt_targets)
        #     for metric in self.metrics
        # ]
        #print(self.metrics[0](pis, targets, mask=mask))
        metric2loss = {f"val/mpiw": self.metrics[0](preds, targets, normalize=True)}
        
        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets), prog_bar=True)
        #return super().validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        alpha=0.1
        Y = super().predict_step(batch, batch_idx)[0]
        q_lo = alpha / 2
        q_hi = 1 - q_lo
        q_lo_idx = find_nearest(self.quantiles, q_lo)
        q_hi_idx = find_nearest(self.quantiles, q_hi)
        q_med_idx = find_nearest(self.quantiles, 0.5)
        pred_medians =Y[:, q_med_idx]
        pred_intervals = Y[:, [q_lo_idx, q_hi_idx]]
        return pred_medians, pred_intervals
    
    def calibration(self):
        pass

