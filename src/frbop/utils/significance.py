"""Centralised significance masking routines.

All functions are pure — they accept pre-computed arrays and return boolean
masks.  Callers handle their own noise estimation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Contiguity helpers
# ---------------------------------------------------------------------------

def apply_min_run(mask: np.ndarray, min_run: int = 1) -> np.ndarray:
    """Keep only contiguous True runs with at least *min_run* samples."""
    valid = np.asarray(mask, dtype=bool).astype(int)
    dv = np.diff(np.concatenate(([0], valid, [0])))
    starts = np.where(dv == 1)[0]
    ends = np.where(dv == -1)[0]
    keep = np.zeros_like(mask, dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) >= min_run:
            keep[s:e] = True
    return keep


def longest_true_run(mask: np.ndarray) -> int:
    """Return length of the longest contiguous True run in *mask*."""
    valid = np.asarray(mask, dtype=bool).astype(int)
    dv = np.diff(np.concatenate(([0], valid, [0])))
    starts = np.where(dv == 1)[0]
    ends = np.where(dv == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


# ---------------------------------------------------------------------------
# Individual mask builders
# ---------------------------------------------------------------------------

def l_significance_mask(
    L_debias: np.ndarray,
    sigma_L: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Boolean mask where debiased L exceeds *threshold* × sigma_L."""
    L_debias = np.asarray(L_debias, dtype=float)
    sigma_L = np.asarray(sigma_L, dtype=float)
    return L_debias >= (threshold * sigma_L)


def stokes_i_significance_mask(
    i_ts: np.ndarray,
    sigma_i: float,
    med_i: float,
    sigma_cut: float = 2.0,
    post_peak_only: bool = False,
    peak_idx: int | None = None,
) -> np.ndarray:
    """Boolean mask where Stokes I exceeds med_i + sigma_cut × sigma_i.

    Optionally restrict to samples at or after *peak_idx*.
    """
    i_ts = np.asarray(i_ts, dtype=float)
    mask = i_ts >= (med_i + sigma_cut * sigma_i)
    if post_peak_only:
        idx = peak_idx if peak_idx is not None else int(np.nanargmax(i_ts))
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[idx:] = True
        mask = mask & peak_mask
    return mask


def stokes_i_snr_mask(
    i_vals: np.ndarray,
    sigma_i: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """Boolean mask where Stokes I SNR (I / sigma_I) ≥ *threshold*."""
    i_vals = np.asarray(i_vals, dtype=float)
    sigma_i = np.asarray(sigma_i, dtype=float)
    return (i_vals / (sigma_i + 1e-10)) >= threshold


def snr_mask_with_fallback(
    snr: np.ndarray,
    primary_threshold: float = 5.0,
    fallback_threshold: float = 2.0,
    min_points: int = 2,
) -> np.ndarray:
    """SNR mask that falls back to *fallback_threshold* if too few points."""
    snr = np.asarray(snr, dtype=float)
    mask = snr > primary_threshold
    if np.nansum(mask) < min_points:
        mask = snr > fallback_threshold
    return mask


# ---------------------------------------------------------------------------
# Combined PA significance mask
# ---------------------------------------------------------------------------

def build_pa_mask(
    L_debias: np.ndarray,
    sigma_L: np.ndarray,
    i_ts: np.ndarray | None = None,
    sigma_i: float | None = None,
    med_i: float | None = None,
    li_i_sigma_cut: float = 2.0,
    post_peak_only: bool = False,
    min_run: int = 3,
    l_threshold: float = 2.0,
) -> np.ndarray:
    """Build the combined PA significance mask (L + optional Stokes I + min-run).

    Parameters
    ----------
    L_debias, sigma_L : array-like
        Debiased linear polarisation and its propagated noise.
    i_ts : array-like, optional
        Stokes I time series.  When provided, *sigma_i* and *med_i* are
        required.
    sigma_i, med_i : float, optional
        Noise standard deviation and median of Stokes I.
    li_i_sigma_cut : float
        Number of sigma above the median for the I threshold.
    post_peak_only : bool
        When True, restrict I mask to post-peak samples.
    min_run : int
        Minimum contiguous True samples required.
    l_threshold : float
        Number of sigma_L for the L significance cut.
    """
    mask = l_significance_mask(L_debias, sigma_L, threshold=l_threshold)

    if i_ts is not None:
        if sigma_i is None or med_i is None:
            raise ValueError("sigma_i and med_i are required when i_ts is provided")
        i_mask = stokes_i_significance_mask(
            i_ts, sigma_i, med_i,
            sigma_cut=li_i_sigma_cut,
            post_peak_only=post_peak_only,
        )
        mask = mask & i_mask

    if np.any(mask):
        mask = apply_min_run(mask, max(1, int(min_run)))

    return mask
