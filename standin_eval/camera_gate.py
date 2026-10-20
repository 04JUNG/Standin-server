"""Choose expanded cameras from query-only match costs, never answer labels."""
import numpy as np


def use_expanded(coarse_best,expanded_best,min_coarse_cost,min_relative_gain):
    values=(coarse_best,expanded_best,min_coarse_cost,min_relative_gain)
    if not np.isfinite(values).all() or coarse_best<0 or expanded_best<0:
        return False
    if min_coarse_cost<0 or not 0<=min_relative_gain<=1:
        raise ValueError('invalid gate thresholds')
    if coarse_best<=max(min_coarse_cost,1e-8):return False
    gain=max(0.,(coarse_best-expanded_best)/coarse_best)
    return bool(gain>=min_relative_gain)
