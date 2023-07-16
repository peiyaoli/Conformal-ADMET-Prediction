import numpy as np
import pandas as pd
from chempropv2.conformal.nonconformist import *
from chempropv2.utils import find_nearest

# CQR error function
class QR_errfun:
    """Calculates conformalized quantile regression error.
    Conformity scores:
    .. math::
    max{\hat{q}_low - y, y - \hat{q}_high}
    """

    def __init__(self):
        super(QR_errfun, self).__init__()

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


def run_icp(y_true_calib, y_pred_calib, y_pred_test, quantiles, alpha):
    idx_lower = find_nearest(quantiles, alpha / 2.0)
    idx_upper = find_nearest(quantiles, 1.0 - alpha / 2.0)
    y_pis_calib = y_pred_calib[:, [idx_lower, idx_upper]]
    scorer = QR_errfun()
    scores = scorer.apply(y_pis_calib, y_true_calib)
    score_correction = scorer.apply_inverse(scores, alpha)
    y_pis_test = y_pred_test[:, [idx_lower, idx_upper]]
    y_pis_test[:, 0] -= score_correction[0, 0]
    y_pis_test[:, 1] += score_correction[1, 0]

    return y_pis_test

def evaluate_icp(pred, Y):
    # Extract lower and upper prediction bands
    pred_l = np.min(pred,1)
    pred_h = np.max(pred,1)
    # Marginal coverage
    cover = (Y>=pred_l)*(Y<=pred_h)
    marg_coverage = np.mean(cover)
    # if X is None:
    #     wsc_coverage = None
    # else:
    #     # Estimated conditional coverage (worse-case slab)
    #     wsc_coverage = coverage.wsc_unbiased(X, Y, pred, M=100)

    # Marginal length
    lengths = pred_h-pred_l
    length = np.mean(lengths)
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Length': [length], 'Length cover': [length_cover]})
    return out