"""
Descriptive diagnostics for time-series RM measurements and
posterior summary helpers.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def summarize_posterior(posterior_values: np.ndarray,
                        low_percentile: float = 16.0,
                        high_percentile: float = 84.0) -> Tuple[float, float, float]:
    """Return median and bounds for posterior samples."""
    values = np.asarray(posterior_values)
    median = np.nanmedian(values)
    low = np.nanpercentile(values, low_percentile)
    high = np.nanpercentile(values, high_percentile)
    return median, low, high


def time_series_sigma_rm_diagnostic(rm_time: np.ndarray,
                                    weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute descriptive diagnostics for time-series RM measurements.

    Parameters
    ----------
    rm_time : array-like
        RM values per time bin (rad/m^2). NaNs are ignored.
    weights : array-like, optional
        Per-bin non-negative weights for weighted diagnostics. If provided,
        weighted values are computed over bins where RM and weight are finite
        and weight > 0.

    Returns
    -------
    dict
        Dictionary with unweighted and (when possible) weighted diagnostics.
    """
    rm_arr = np.asarray(rm_time, dtype=float)
    valid = np.isfinite(rm_arr)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            'rm_mean': np.nan,
            'sigma_rm_time': np.nan,
            'rm_min': np.nan,
            'rm_max': np.nan,
            'weighted_rm_mean': np.nan,
            'weighted_std_rm_time': np.nan,
            'weighted_n': 0,
            'weight_sum': 0.0,
            'n_valid': 0,
            'n_total': int(rm_arr.size),
        }

    rm_valid = rm_arr[valid]
    weighted_rm_mean = np.nan
    weighted_std_rm_time = np.nan
    weighted_n = 0
    weight_sum = 0.0

    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.shape == rm_arr.shape:
            w_mask = valid & np.isfinite(w_arr) & (w_arr > 0)
            if np.any(w_mask):
                rm_w = rm_arr[w_mask]
                w = w_arr[w_mask]
                weight_sum = float(np.sum(w))
                weighted_n = int(np.sum(w_mask))
                if weight_sum > 0:
                    weighted_rm_mean = float(np.sum(w * rm_w) / weight_sum)
                    weighted_std_rm_time = float(
                        np.sqrt(np.sum(w * (rm_w - weighted_rm_mean) ** 2) / weight_sum)
                    )

    return {
        'rm_mean': float(np.mean(rm_valid)),
        'std_rm_time': float(np.std(rm_valid)),
        'rm_min': float(np.min(rm_valid)),
        'rm_max': float(np.max(rm_valid)),
        'weighted_rm_mean': weighted_rm_mean,
        'weighted_std_rm_time': weighted_std_rm_time,
        'weighted_n': weighted_n,
        'weight_sum': weight_sum,
        'n_valid': n_valid,
        'n_total': int(rm_arr.size),
    }
