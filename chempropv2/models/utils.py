import numpy as np
from torch import nn


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    Parameters
    ----------
    activation : str
        The name of the activation function.

    Returns
    -------
    nn.Module
        The activation function module.
    """
    activation_ = activation.lower()
    if activation_ == "relu":
        return nn.ReLU()
    if activation_ == "leakyrelu":
        return nn.LeakyReLU(0.1)
    if activation_ == "prelu":
        return nn.PReLU()
    if activation_ == "tanh":
        return nn.Tanh()
    if activation_ == "selu":
        return nn.SELU()
    if activation_ == "elu":
        return nn.ELU()

    raise ValueError(
        f'Invalid activation! got: "{activation}". '
        f'expected one of: ("relu", "leakyrelu", "prelu", "tanh", "selu", "elu")'
    )


def rearrange(all_quantiles, quantile_low, quantile_high, test_preds):
    """Produce monotonic quantiles

    Parameters
    ----------

    all_quantiles : numpy array (q), grid of quantile levels in the range (0,1)
    quantile_low : float, desired low quantile in the range (0,1)
    quantile_high : float, desired high quantile in the range (0,1)
    test_preds : numpy array of predicted quantile (nXq)

    Returns
    -------

    q_fixed : numpy array (nX2), containing the rearranged estimates of the
              desired low and high quantile

    References
    ----------
    .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
            "Quantile and probability curves without crossing."
            Econometrica 78.3 (2010): 1093-1125.

    """
    scaling = all_quantiles[-1] - all_quantiles[0]
    low_val = (quantile_low - all_quantiles[0]) / scaling
    high_val = (quantile_high - all_quantiles[0]) / scaling
    q_fixed = np.quantile(test_preds, (low_val, high_val), interpolation="linear", axis=1)
    return q_fixed.T