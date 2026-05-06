"""Autocorrelation helpers."""

import numpy as np


def autocorr(x):
    x = np.asarray(x)
    mean = np.nanmean(x)
    if mean == 0:
        return np.zeros_like(x)
    delta = (x - mean) / mean
    result = np.correlate(delta, delta, mode="full")
    acf = result[result.size // 2:]
    if acf[0] != 0:
        acf /= acf[0]
    return acf
