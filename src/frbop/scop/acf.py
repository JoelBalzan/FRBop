"""Autocorrelation helpers."""

import numpy as np


def autocorr(x):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros(x.size)
    # Fill NaN channels with zero deviation (they contribute nothing to the ACF)
    x_filled = np.where(valid, x, 0.0)
    mean = np.nanmean(x_filled[valid])
    delta = x_filled - mean
    delta[~valid] = 0.0           # zero-out excluded channels, not NaN
    result = np.correlate(delta, delta, mode="full")
    acf = result[result.size // 2:]
    if acf[0] != 0:
        acf /= acf[0]
    return acf
