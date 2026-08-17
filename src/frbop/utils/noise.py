"""Shared noise-estimation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def noise_from_first_fraction(
    data: np.ndarray,
    frac: float = 0.05,
    axis: Optional[int] = None,
    floor: float = 1e-15,
) -> np.ndarray:
    """Estimate RMS noise from the first *frac* of samples.

    Parameters
    ----------
    data : array-like
        Input data (1-D or multi-dimensional).
    frac : float
        Fraction of samples (from the start) used for the estimate.
    axis : int, optional
        Axis along which to compute the std.  ``None`` treats the input as
        flattened (for 1-D data).
    floor : float
        Minimum returned value (avoids zeros).

    Returns
    -------
    noise : float or ndarray
        Estimated noise RMS.  Scalar for 1-D input, 1-D array when *axis*
        is given.
    """
    data = np.asarray(data, dtype=float)
    if axis is not None:
        n = data.shape[axis]
        n_edge = max(1, int(n * frac))
        slc = [slice(None)] * data.ndim
        slc[axis] = slice(0, n_edge)
        noise = np.nanstd(data[tuple(slc)], axis=axis)
        return np.maximum(noise, floor)
    arr = data.ravel()
    n_edge = max(1, int(len(arr) * frac))
    noise = float(np.nanstd(arr[:n_edge]))
    if not np.isfinite(noise) or noise < 0.0:
        return floor
    return max(noise, floor)


def robust_noise(
    data: np.ndarray,
    frac: float = 0.05,
    axis: Optional[int] = None,
    floor: float = 1e-10,
) -> np.ndarray:
    """Robust noise estimate using MAD (median absolute deviation).

    Falls back to ``|deviation| / 0.6745`` when the standard deviation is
    zero or non-finite.
    """
    data = np.asarray(data, dtype=float)
    if axis is not None:
        n = data.shape[axis]
        n_edge = max(1, int(n * frac))
        slc = [slice(None)] * data.ndim
        slc[axis] = slice(0, n_edge)
        chunk = data[tuple(slc)]
        sig = np.nanstd(chunk, axis=axis)
        med = np.nanmedian(chunk, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(chunk - med), axis=axis)
        robust = mad / 0.6745
        result = np.where((np.isfinite(sig) & (sig > 0)), sig, robust)
        return np.maximum(np.where(np.isfinite(result), result, floor), floor)
    arr = data.ravel()
    n_edge = max(1, int(len(arr) * frac))
    chunk = arr[:n_edge]
    sig = float(np.nanstd(chunk))
    if np.isfinite(sig) and sig > 0:
        return sig
    med = float(np.nanmedian(chunk))
    mad = float(np.nanmedian(np.abs(chunk - med)))
    if np.isfinite(mad) and mad > 0:
        return mad / 0.6745
    return floor
