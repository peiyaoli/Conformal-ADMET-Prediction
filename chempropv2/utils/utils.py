from __future__ import annotations
import numpy as np
import torch
import pandas as pd
from enum import Enum
from typing import Union
from torch import nn
from rdkit import Chem
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            names = [x.value for x in cls]
            raise ValueError(f"Invalid name! got: '{name}'. expected one of: {tuple(names)}")


def make_mol(smi: str, keep_h: bool, add_h: bool) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    return Chem.AddHs(mol) if add_h else mol

def find_nearest(a, a0):
    "Index of element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return np.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


    
def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def estimate_sample_weights(targets, bins: int = 50, reweight: str = 'sqrt_inv', lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    """Esimate sample weights using kernel density estimation (KDE) method"""
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
    labels = pd.cut(targets, bins=50, include_lowest=True)
    left_bounds = [x.left for x in labels]
    value_dict = {x.left: 0 for x in labels.categories}
    for lb in left_bounds:
        value_dict[lb] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        
        
    num_per_label = [value_dict[lb] for lb in left_bounds]
    
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value_dict = dict(zip(value_dict.keys(),
                               convolve1d(np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')))
        num_per_label = [smoothed_value_dict[x] for x in left_bounds]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights