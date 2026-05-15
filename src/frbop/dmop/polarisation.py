"""
Polarisation metrics: position angle (PA) and fractional linear polarisation (L/I).

All functions are stateless and operate on plain NumPy arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Q/U noise estimation
# ---------------------------------------------------------------------------

def qu_noise_rms(
    data_q: np.ndarray,
    data_u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-channel Q/U RMS from the first 5 % of the time axis.

    Works for both 1-D (time-series) and 2-D (freq × time) inputs.
    For 2-D inputs the result shape is ``(n_freq, 1)`` so it broadcasts
    correctly in per-pixel calculations.
    """
    if data_q.ndim == 1:
        n_edge = max(1, int(0.05 * len(data_q)))
        return float(np.std(data_q[:n_edge])), float(np.std(data_u[:n_edge]))

    n_edge = max(1, int(0.05 * data_q.shape[1]))
    q_rms = np.std(data_q[:, :n_edge], axis=1, keepdims=True)
    u_rms = np.std(data_u[:, :n_edge], axis=1, keepdims=True)
    return q_rms, u_rms


# ---------------------------------------------------------------------------
# Linear polarisation debiasing
# ---------------------------------------------------------------------------

def debiased_linear_from_qu(
    data_q: np.ndarray,
    data_u: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    debias: bool = False,
    cutoff: float = 1.57,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (optionally debiased) linear polarisation *L*, its per-pixel
    propagated uncertainty *sigma_L*, and a detection boolean mask *det*.

    When *debias* is False, undetected pixels are set to zero but no
    bias subtraction is applied.  When *debias* is True, the standard
    Rice-distribution correction ``L_debias = sqrt(L² - σ_L²)`` is used.
    """
    L_meas = np.sqrt(data_q ** 2 + data_u ** 2)
    sigma_L = (
        np.sqrt(data_q ** 2 * q_rms ** 2 + data_u ** 2 * u_rms ** 2)
        / np.maximum(L_meas, eps)
    )
    r = L_meas / np.maximum(sigma_L, eps)
    det = r >= cutoff

    if debias:
        L_out = np.zeros_like(L_meas)
        L_out[det] = np.sqrt(np.maximum(L_meas[det] ** 2 - sigma_L[det] ** 2, 0.0))
    else:
        L_out = L_meas.copy()
        L_out[~det] = 0.0

    return L_out, sigma_L, det


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def apply_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Zero-out runs of True shorter than *min_run* samples."""
    valid = mask.astype(int)
    dv = np.diff(np.concatenate(([0], valid, [0])))
    starts = np.where(dv == 1)[0]
    ends = np.where(dv == -1)[0]
    keep = np.zeros_like(mask, dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) >= min_run:
            keep[s:e] = True
    return keep


def longest_true_run(mask: np.ndarray) -> int:
    valid = mask.astype(int)
    dv = np.diff(np.concatenate(([0], valid, [0])))
    starts = np.where(dv == 1)[0]
    ends = np.where(dv == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


# ---------------------------------------------------------------------------
# PA series
# ---------------------------------------------------------------------------

def pa_series_deg(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: Optional[np.ndarray],
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    pa_fit_post_peak_only: bool,
    debias: bool = False,
    min_run: int = 10,
) -> np.ndarray:
    """
    Build a masked PA time series (degrees) from Stokes Q and U.

    Samples with insufficient linear polarisation S/N or low Stokes-I are
    set to NaN.  Short valid runs shorter than *min_run* samples are also
    masked.
    """
    q_ts = np.mean(data_q, axis=0)
    u_ts = np.mean(data_u, axis=0)
    L_debias, sigma_L, _ = debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms, debias=debias)

    pa = 0.5 * np.arctan2(u_ts, q_ts)
    pa = 0.5 * np.unwrap(2.0 * pa)
    pa_deg = np.degrees(pa)
    pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

    mask = L_debias >= (2.0 * sigma_L)

    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        threshold_i = noise_median_i + li_i_sigma_cut * noise_std_i
        i_mask = i_ts >= threshold_i
        if pa_fit_post_peak_only:
            peak_idx = int(np.argmax(i_ts))
            peak_mask = np.zeros_like(mask, dtype=bool)
            peak_mask[peak_idx:] = True
            i_mask = i_mask & peak_mask
        mask = mask & i_mask

    pa_deg = np.where(mask, pa_deg, np.nan)
    if np.any(mask):
        keep_run = apply_min_run(mask, min_run)
        pa_deg = np.where(keep_run, pa_deg, np.nan)

    return pa_deg


# ---------------------------------------------------------------------------
# PA fit weights
# ---------------------------------------------------------------------------

def pa_fit_weights(
    L_debias: np.ndarray,
    sigma_L: np.ndarray,
    data_i: Optional[np.ndarray],
    valid: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    pa_weight_strength: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-sample weights for the PA polynomial fit.

    Weights are proportional to L S/N (and optionally I S/N when *data_i*
    is provided), then normalised to [0, 1] and raised to *pa_weight_strength*.
    """
    w_l = np.maximum(L_debias / np.maximum(sigma_L, eps), 0.0)
    weights = w_l

    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        if noise_std_i > 0:
            w_i = np.maximum((i_ts - noise_median_i) / noise_std_i, 0.0)
            weights = weights * w_i

    weights = np.where(valid, weights, 0.0)
    max_w = float(np.max(weights)) if np.any(valid) else 0.0
    if max_w > 0:
        weights = weights / max_w
        if pa_weight_strength != 1.0:
            weights = np.power(weights, pa_weight_strength)

    return weights


# ---------------------------------------------------------------------------
# PA slope metric
# ---------------------------------------------------------------------------

def pa_slope_metric(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: Optional[np.ndarray],
    time_axis: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    pa_fit_post_peak_only: bool,
    pa_fit_degree: int,
    pa_weight_strength: float,
    debias: bool = False,
) -> float:
    """
    Weighted polynomial fit to the masked PA series; returns |slope| (deg/ms).
    """
    q_ts = np.mean(data_q, axis=0)
    u_ts = np.mean(data_u, axis=0)
    L_debias, sigma_L, _ = debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms, debias=debias)

    pa = 0.5 * np.arctan2(u_ts, q_ts)
    pa = 0.5 * np.unwrap(2.0 * pa)
    pa_deg = np.degrees(pa)
    pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

    mask = L_debias >= (2.0 * sigma_L)
    if np.any(mask):
        mask = apply_min_run(mask, 5)

    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        threshold_i = noise_median_i + li_i_sigma_cut * noise_std_i
        i_mask = i_ts >= threshold_i
        if pa_fit_post_peak_only:
            peak_idx = int(np.argmax(i_ts))
            peak_mask = np.zeros_like(mask, dtype=bool)
            peak_mask[peak_idx:] = True
            i_mask = i_mask & peak_mask
        mask = mask & i_mask

    valid = mask & np.isfinite(pa_deg)
    min_points = pa_fit_degree + 1
    if np.sum(valid) < min_points:
        return 0.0
    if longest_true_run(valid) < max(12, min_points):
        return 0.0

    weights = pa_fit_weights(
        L_debias, sigma_L, data_i, valid,
        noise_median_i, noise_std_i, pa_weight_strength,
    )
    positive = valid & (weights > 0)
    if np.sum(positive) < min_points:
        return 0.0

    try:
        coeffs = np.polyfit(time_axis[positive], pa_deg[positive], pa_fit_degree, w=weights[positive])
    except Exception:
        return 0.0

    slope_magnitude = float(np.abs(coeffs[0]))
    return slope_magnitude if np.isfinite(slope_magnitude) else 0.0


def pa_slope_metric_shrine(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: Optional[np.ndarray],
    time_axis: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    pa_fit_post_peak_only: bool,
    pa_fit_degree: int,
    pa_weight_strength: float,
    apply_kc_lowpass_fn,
    resolve_nonshrine_kc_fn,
    debias: bool = False,
) -> float:
    """
    Variant that runs SHRINE kc low-pass smoothing on the PA series before fitting.

    *apply_kc_lowpass_fn* and *resolve_nonshrine_kc_fn* are callables provided
    by the SHRINE wrapper so this module stays SHRINE-free.
    """
    q_ts = np.mean(data_q, axis=0)
    u_ts = np.mean(data_u, axis=0)
    L_debias, sigma_L, _ = debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms, debias=debias)

    pa = 0.5 * np.arctan2(u_ts, q_ts)
    pa = 0.5 * np.unwrap(2.0 * pa)
    pa_deg = np.degrees(pa)
    pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

    mask = L_debias >= (2.0 * sigma_L)
    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        threshold_i = noise_median_i + li_i_sigma_cut * noise_std_i
        i_mask = i_ts >= threshold_i
        if pa_fit_post_peak_only:
            peak_idx = int(np.argmax(i_ts))
            peak_mask = np.zeros_like(mask, dtype=bool)
            peak_mask[peak_idx:] = True
            i_mask = i_mask & peak_mask
        mask = mask & i_mask

    valid = mask & np.isfinite(pa_deg)
    min_points = pa_fit_degree + 1
    if np.sum(valid) < min_points:
        return 0.0

    # Build interpolated fill for kc smoothing
    valid_idx = np.where(np.isfinite(pa_deg) & (L_debias >= 2.0 * sigma_L))[0]
    if valid_idx.size == 1:
        pa_fill = np.full_like(pa_deg, pa_deg[valid_idx[0]], dtype=float)
    elif valid_idx.size >= 2:
        pa_fill = np.interp(np.arange(len(pa_deg)), valid_idx, pa_deg[valid_idx])
    else:
        return 0.0

    kc = resolve_nonshrine_kc_fn(pa_fill[np.newaxis, :])
    pa_shrine = apply_kc_lowpass_fn(pa_fill[np.newaxis, :], kc)[0]
    pa_deg_masked = np.where(np.isfinite(np.where(mask, pa_deg, np.nan)), pa_shrine, np.nan)

    valid_sm = np.isfinite(pa_deg_masked)
    if np.sum(valid_sm) < min_points:
        return 0.0

    weights = pa_fit_weights(
        L_debias, sigma_L, data_i, valid_sm,
        noise_median_i, noise_std_i, pa_weight_strength,
    )
    if np.sum(weights[valid_sm] > 0) < min_points:
        return 0.0

    try:
        coeffs = np.polyfit(
            time_axis[valid_sm], pa_deg_masked[valid_sm], pa_fit_degree,
            w=weights[valid_sm],
        )
    except Exception:
        return 0.0

    slope_magnitude = float(np.abs(coeffs[0]))
    return slope_magnitude if np.isfinite(slope_magnitude) else 0.0


# ---------------------------------------------------------------------------
# Smoothed PA profile + fit line (for plotting)
# ---------------------------------------------------------------------------

def get_pa_smoothed_and_fit(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: Optional[np.ndarray],
    time_axis: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    pa_fit_post_peak_only: bool,
    pa_fit_degree: int,
    pa_weight_strength: float,
    debias: bool = False,
    min_run: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(pa_masked, fit_line)`` — both on the same time grid.

    PA is masked by L S/N and optionally Stokes I; no Gaussian smoothing is applied.
    """
    q_ts = np.mean(data_q, axis=0)
    u_ts = np.mean(data_u, axis=0)
    L_debias, sigma_L, _ = debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms, debias=debias)

    pa = 0.5 * np.arctan2(u_ts, q_ts)
    pa = 0.5 * np.unwrap(2.0 * pa)
    pa_deg = np.degrees(pa)
    pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

    mask = L_debias >= (2.0 * sigma_L)
    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        threshold_i = noise_median_i + li_i_sigma_cut * noise_std_i
        i_mask = i_ts >= threshold_i
        if pa_fit_post_peak_only:
            peak_idx = int(np.argmax(i_ts))
            peak_mask = np.zeros_like(mask, dtype=bool)
            peak_mask[peak_idx:] = True
            i_mask = i_mask & peak_mask
        mask = mask & i_mask

    if np.any(mask):
        mask = apply_min_run(mask, min_run)

    valid = mask & np.isfinite(pa_deg)
    pa_deg_masked = np.where(mask, pa_deg, np.nan)
    fit_line = np.full_like(pa_deg, np.nan)

    min_points = pa_fit_degree + 1
    if np.sum(valid) >= min_points:
        weights = pa_fit_weights(
            L_debias, sigma_L, data_i, valid,
            noise_median_i, noise_std_i, pa_weight_strength,
        )
        if np.sum(weights[valid] > 0) >= min_points:
            coeffs = np.polyfit(time_axis[valid], pa_deg[valid], pa_fit_degree, w=weights[valid])
            fit_line = np.polyval(coeffs, time_axis)
            fit_line = np.where(mask, fit_line, np.nan)

    return pa_deg_masked, fit_line


def get_pa_shrine_smoothed_and_fit(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: Optional[np.ndarray],
    time_axis: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    pa_fit_post_peak_only: bool,
    pa_fit_degree: int,
    pa_weight_strength: float,
    apply_kc_lowpass_fn,
    resolve_nonshrine_kc_fn,
    debias: bool = False,
    min_run: int = 5,
    force_kc: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SHRINE-smoothed variant of :func:`get_pa_smoothed_and_fit`.

    Returns ``(pa_shrine_smooth, fit_line)``.
    """
    q_ts = np.mean(data_q, axis=0)
    u_ts = np.mean(data_u, axis=0)
    L_debias, sigma_L, _ = debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms, debias=debias)

    pa = 0.5 * np.arctan2(u_ts, q_ts)
    pa = 0.5 * np.unwrap(2.0 * pa)
    pa_deg = np.degrees(pa)
    pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

    mask = L_debias >= (2.0 * sigma_L)
    if data_i is not None:
        i_ts = np.mean(data_i, axis=0)
        threshold_i = noise_median_i + li_i_sigma_cut * noise_std_i
        i_mask = i_ts >= threshold_i
        if pa_fit_post_peak_only:
            peak_idx = int(np.argmax(i_ts))
            peak_mask = np.zeros_like(mask, dtype=bool)
            peak_mask[peak_idx:] = True
            i_mask = i_mask & peak_mask
        mask = mask & i_mask

    if np.any(mask):
        mask = apply_min_run(mask, min_run)

    valid = mask & np.isfinite(pa_deg)
    pa_deg_masked = np.where(mask, pa_deg, np.nan)
    pa_shrine_smooth = np.full_like(pa_deg, np.nan)
    fit_line = np.full_like(pa_deg, np.nan)

    kc_source_mask = (L_debias >= 2.0 * sigma_L) & np.isfinite(pa_deg)
    pa_deg_for_kc = np.where(kc_source_mask, pa_deg, np.nan)
    if np.sum(np.isfinite(pa_deg_for_kc)) < 2:
        pa_deg_for_kc = np.where(np.isfinite(pa_deg), pa_deg, np.nan)

    min_points = pa_fit_degree + 1
    if np.sum(valid) >= min_points:
        valid_idx = np.where(np.isfinite(pa_deg_for_kc))[0]
        if valid_idx.size == 1:
            pa_fill = np.full_like(pa_deg, pa_deg_for_kc[valid_idx[0]], dtype=float)
        elif valid_idx.size >= 2:
            pa_fill = np.interp(np.arange(len(pa_deg)), valid_idx, pa_deg_for_kc[valid_idx])
        else:
            return pa_shrine_smooth, fit_line

        kc = force_kc if force_kc is not None else resolve_nonshrine_kc_fn(pa_fill[np.newaxis, :])
        pa_shrine = apply_kc_lowpass_fn(pa_fill[np.newaxis, :], kc)[0]
        pa_shrine_smooth = np.where(np.isfinite(pa_deg_masked), pa_shrine, np.nan)

        weights = pa_fit_weights(
            L_debias, sigma_L, data_i, valid,
            noise_median_i, noise_std_i, pa_weight_strength,
        )
        if np.sum(weights[valid] > 0) >= min_points:
            coeffs = np.polyfit(
                time_axis[valid], pa_shrine_smooth[valid], pa_fit_degree,
                w=weights[valid],
            )
            fit_line = np.polyval(coeffs, time_axis)
            fit_line = np.where(mask, fit_line, np.nan)

    return pa_shrine_smooth, fit_line


# ---------------------------------------------------------------------------
# L/I metric
# ---------------------------------------------------------------------------

def linear_to_stokes_i_metric(
    data_q: np.ndarray,
    data_u: np.ndarray,
    data_i: np.ndarray,
    q_rms: np.ndarray,
    u_rms: np.ndarray,
    noise_median_i: float,
    noise_std_i: float,
    li_i_sigma_cut: float,
    debias: bool = False,
    mode: str = "peak",
) -> float:
    """
    Fractional linear polarisation metric.

    Parameters
    ----------
    mode:
        ``'peak'``  — L/I at the Stokes-I peak within the pulse window.
        ``'mean'``  — mean L/I across the significant pulse region.
        ``'max'``   — L/I where L is maximum within the pulse window.
    """
    L_debias, _, _ = debiased_linear_from_qu(data_q, data_u, q_rms, u_rms, debias=debias)

    L_over_I_2d = np.where(data_i > 0, L_debias / data_i, 0.0)
    L_over_I_2d = np.clip(L_over_I_2d, 0.0, 1.0)
    L_over_I = np.mean(L_over_I_2d, axis=0)

    I_ts = np.mean(data_i, axis=0)
    threshold = noise_median_i + li_i_sigma_cut * noise_std_i
    mask = I_ts > threshold

    if mode == "peak":
        if np.any(mask):
            masked_i = np.where(mask, I_ts, -np.inf)
            return float(L_over_I[int(np.argmax(masked_i))])
        return 0.0

    if mode == "mean":
        return float(np.mean(L_over_I[mask])) if np.any(mask) else 0.0

    if mode == "max":
        L_ts = np.mean(L_debias, axis=0)
        if np.any(mask):
            masked_l = np.where(mask, L_ts, -np.inf)
            return float(L_over_I[int(np.argmax(masked_l))])
        return 0.0

    raise ValueError(f"Unknown mode '{mode}'. Must be 'peak', 'mean', or 'max'.")
