"""Autocorrelation helpers."""

import numpy as np


def autocorr(x):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros(x.size)
    mean  = float(np.nanmean(x[valid]))
    delta = np.where(valid, x - mean, 0.0)
    result = np.correlate(delta, delta, mode="full")
    acov   = result[result.size // 2:]
    counts = np.arange(x.size, 0, -1, dtype=float)
    acov  /= counts    # unbiased estimator — do NOT divide by acov[0]
    return acov
