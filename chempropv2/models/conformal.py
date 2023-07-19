"""
Conformal prediction functions
"""
import numpy as np
from chempropv2.utils import find_nearest

# CQR error function
class CQR_errfun:
    """Calculates conformalized quantile regression error.
    Conformity scores:
    .. math::
    max{\hat{q}_low - y, y - \hat{q}_high}
    """

    def __init__(self):
        super(CQR_errfun, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, alpha):
        q = np.quantile(
            nc, np.minimum(1.0, (1.0 - alpha) * (nc.shape[0] + 1.0) / nc.shape[0])
        )
        return np.vstack([q, q])


def run_cqr(y_true_calib, y_preds_calib, y_pred_test, quantiles, alpha: float = 0.1):
    # extract indexes of lower and upper 
    idx_lower = find_nearest(quantiles, alpha / 2.0)
    idx_upper = find_nearest(quantiles, 1.0 - alpha / 2.0)
    # extract prediction intervals of calibration set
    y_pis_calib = y_preds_calib[:, [idx_lower, idx_upper]]
    scorer = CQR_errfun()
    scores = scorer.apply(y_pis_calib, y_true_calib)
    score_correction = scorer.apply_inverse(scores, alpha)
    y_pis_test = y_pred_test[:, [idx_lower, idx_upper]]
    y_pis_test[:, 0] -= score_correction[0, 0]
    y_pis_test[:, 1] += score_correction[1, 0]
    return y_pis_test