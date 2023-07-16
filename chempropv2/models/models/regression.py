import torch
from torch import Tensor
from torch.nn import functional as F
from chempropv2.data.dataloader import TrainingBatch
from chempropv2.models.models.base import MPNN
from chempropv2.utils import find_nearest
from chempropv2.models.fds import FDS
from typing import Optional

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

class QuantileRegressionMPNNvFDS(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "pinball"
    _DEFAULT_METRIC = "mpiw"
    def __init__(self, quantiles: torch.Tensor = None, n_tasks: int = None, fds: bool = True,
                 start_smooth: int = 0, **kwargs):
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.quantiles = quantiles
        #self.no_crossing = no_crossing
        self.start_smooth = start_smooth
        self.fds = fds
        if fds:
            self.FDS = FDS(feature_dim=self.mpn_block.output_dim, start_smooth=self.start_smooth)
    
    def forward(self, inputs, X_f, targets=None) -> Tensor:
        encoding = self.encoding(inputs=inputs, X_f=X_f)
        encoding_s = encoding
        
        if self.training and self.fds:
            if self.current_epoch >= self.start_smooth:
                encoding_s = self.FDS.smooth(encoding_s, targets, self.current_epoch)
        
        Y = self.ffn[-1](encoding_s)
        
        if self.training and self.fds:
            return Y, encoding
        else:
            return Y
        
    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask = torch.ones((len(targets), self.n_tasks), device=targets.device)
        targets = targets.nan_to_num(nan=0.0)

        preds, feature = self(inputs=(bmg, X_vd), X_f=features, targets=targets)
        
        # Y = super().forward(inputs, X_f)
        # if self.no_crossing:
        #     Y, _ = torch.sort(Y, dim=1)
        # return Y


class QuantileRegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "pinball"
    _DEFAULT_METRIC = "mpiw"
    
    def __init__(self, quantiles: torch.Tensor = None, n_tasks: int = None, no_crossing: bool = True, **kwargs):
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.quantiles = quantiles
        self.no_crossing = no_crossing
    
    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)
        if self.no_crossing:
            Y, _ = torch.sort(Y, dim=1)
        return Y
    
    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask = torch.ones((len(targets), self.n_tasks), device=targets.device)
        targets = targets.nan_to_num(nan=0.0)

        preds = self((bmg, X_vd), X_f=features)

        l = self.criterion(
            preds, targets, mask, weights=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log("train/loss", l, prog_bar=True)

        return l
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> tuple[list[Tensor], int]:
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch
        preds = super().predict_step(batch, batch_idx)[0]
        mask = torch.ones((len(targets), self.n_tasks), device=targets.device)
        val_loss = self.criterion(
            preds, targets, mask, weights=weights, lt_targets=lt_targets, gt_targets=gt_targets, quantiles=self.quantiles
        )
        self.log_dict({"val/loss": val_loss}, on_epoch=True, batch_size=len(targets), prog_bar=True)
    
    def predict_step(self, batch, batch_idx, alpha: Optional[float] = None):
        y_preds = super().predict_step(batch, batch_idx)[0]
        q_med_idx = find_nearest(self.quantiles, 0.5)
        y_medians = y_preds[:, q_med_idx]
        if alpha is not None:
            q_lo = alpha / 2
            q_hi = 1 - q_lo
            q_lo_idx = find_nearest(self.quantiles, q_lo)
            q_hi_idx = find_nearest(self.quantiles, q_hi)
            y_preds = y_preds[:, [q_lo_idx, q_hi_idx]]
        return y_medians, y_preds
    
    
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
        pred = [self.Dropout(self.ffn(self.fingerprint((bmg, X_vd), X_f=features))).unsqueeze(0) for _ in range(self.mc_iteration)]
        print(pred)
        preds = torch.vstack(pred)
        pred_mean = preds.mean(dim=0)
        pred_var = preds.var(dim=0)
        return pred_mean , pred_var