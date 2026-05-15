"""
Noise statistics and signal-to-noise utilities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit
    def _max_snr_for_series(
        series: np.ndarray,
        noise_std: float,
        min_window_size: int,
        max_window_size: int,
    ) -> float:
        n = series.shape[0]
        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        for i in range(n):
            csum[i + 1] = csum[i] + series[i]

        max_sn = -1e30
        for start in range(0, n - min_window_size):
            max_len = max_window_size
            if start + max_len > n:
                max_len = n - start
            for length in range(min_window_size, max_len):
                s = csum[start + length] - csum[start]
                sn = s / (noise_std * np.sqrt(length))
                if sn > max_sn:
                    max_sn = sn
        return max_sn


def noise_stats_from_series(series: np.ndarray) -> Tuple[float, float]:
    """
    Estimate noise median and standard deviation from the first 5 % of *series*.

    Returns
    -------
    median, std:
        Both as plain Python floats.
    """
    n_edge = max(1, int(0.05 * len(series)))
    noise_region = series[:n_edge]
    return float(np.median(noise_region)), float(np.std(noise_region))


def robust_vmin_vmax(
    data: np.ndarray,
    low: float = 5.0,
    high: float = 99.0,
) -> Tuple[float, float]:
    """
    Percentile-based colour-scale limits with a finite-value fallback.
    """
    vmin = float(np.percentile(data, low))
    vmax = float(np.percentile(data, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(data))
        vmax = float(np.max(data))
    return vmin, vmax


def snr_metric(
    data: np.ndarray,
    noise_median: float,
    noise_std: float,
) -> float:
    """
    Peak S/N of the frequency-collapsed time series.

    Parameters
    ----------
    data:
        2-D dedispersed array (freq × time).
    noise_median, noise_std:
        Pre-computed noise statistics from the full (unsliced) Stokes-I data.

    Returns
    -------
    snr:
        (peak − noise_median) / noise_std.  Returns 0 if noise_std == 0.
    """
    if noise_std == 0:
        return 0.0
    time_series = np.mean(data, axis=0)
    signal = float(np.max(time_series)) - noise_median
    return signal / noise_std
