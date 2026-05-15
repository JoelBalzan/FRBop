"""
DM uncertainty estimation methods.

All functions operate on plain NumPy arrays and have no dependency on the
DMOptimiser class, making them independently testable.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fftpack import dct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_metric_arrays(
    dm_values: np.ndarray,
    metric_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dm = np.asarray(dm_values, dtype=float)
    metric = np.asarray(metric_values, dtype=float)
    valid = np.isfinite(dm) & np.isfinite(metric)
    if np.sum(valid) < 2:
        raise ValueError("Need at least two finite DM/metric points for uncertainty estimation")
    return dm[valid], metric[valid]


def _dm_crossing(
    values: np.ndarray,
    dm_values: np.ndarray,
    start_idx: int,
    threshold: float,
    direction: int,
) -> Optional[float]:
    """Linear-interpolated DM where *values* crosses *threshold*."""
    n = len(values)
    i = int(start_idx)
    if direction < 0:
        while i > 0 and values[i] >= threshold:
            i -= 1
        if values[i] >= threshold:
            return None
        i_below, i_above = i, i + 1
    else:
        while i < n - 1 and values[i] >= threshold:
            i += 1
        if values[i] >= threshold:
            return None
        i_below, i_above = i, i - 1

    x1, x2 = float(dm_values[i_below]), float(dm_values[i_above])
    y1, y2 = float(values[i_below]), float(values[i_above])
    if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
        return float(dm_values[i_above])
    frac = float(np.clip((threshold - y1) / (y2 - y1), 0.0, 1.0))
    return x1 + frac * (x2 - x1)


def _uncertainty_dict(
    best_dm: float,
    low_dm: Optional[float],
    high_dm: Optional[float],
    method: str,
) -> Dict[str, Optional[float]]:
    minus = None if low_dm is None else float(best_dm - low_dm)
    plus = None if high_dm is None else float(high_dm - best_dm)
    return {
        "uncertainty_low_dm": None if low_dm is None else float(low_dm),
        "uncertainty_high_dm": None if high_dm is None else float(high_dm),
        "uncertainty_minus": minus,
        "uncertainty_plus": plus,
        "uncertainty_method": method,
    }


# ---------------------------------------------------------------------------
# Public uncertainty estimators
# ---------------------------------------------------------------------------

def from_half_prominence(
    dm_values: np.ndarray,
    metric_values: np.ndarray,
    best_idx: int,
) -> Dict[str, Optional[float]]:
    """
    Uncertainty defined by the DM range where the metric exceeds its halfway
    point between minimum and maximum (half-prominence level).
    """
    dm, metric = _finite_metric_arrays(dm_values, metric_values)
    best_idx_eff = int(np.argmax(metric))
    best_dm = float(dm[best_idx_eff])
    metric_max = float(np.max(metric))
    metric_min = float(np.min(metric))

    if not np.isfinite(metric_max) or not np.isfinite(metric_min):
        return _uncertainty_dict(best_dm, None, None, "half-prominence")

    if metric_max <= metric_min:
        step = float(np.median(np.diff(dm))) if len(dm) > 1 else 0.0
        return _uncertainty_dict(
            best_dm,
            best_dm - 0.5 * step if step > 0 else None,
            best_dm + 0.5 * step if step > 0 else None,
            "half-prominence",
        )

    threshold = metric_min + 0.5 * (metric_max - metric_min)
    low_dm = _dm_crossing(metric, dm, best_idx_eff, threshold, direction=-1)
    high_dm = _dm_crossing(metric, dm, best_idx_eff, threshold, direction=1)
    return _uncertainty_dict(best_dm, low_dm, high_dm, "half-prominence")


def from_local_quadratic(
    dm_values: np.ndarray,
    metric_values: np.ndarray,
    best_idx: int,
    target_points: int = 11,
) -> Dict[str, Optional[float]]:
    """
    Fit a downward-opening quadratic near the peak and derive the 1-sigma
    width from the curvature and residual scatter.
    """
    dm, metric = _finite_metric_arrays(dm_values, metric_values)
    best_idx_eff = int(np.argmax(metric))
    best_dm = float(dm[best_idx_eff])
    n = len(dm)
    if n < 5:
        return _uncertainty_dict(best_dm, None, None, "local quadratic")

    points = int(max(5, min(target_points, n)))
    half = points // 2
    start = max(0, best_idx_eff - half)
    end = min(n, start + points)
    start = max(0, end - points)
    x, y = dm[start:end], metric[start:end]
    if x.size < 5:
        return _uncertainty_dict(best_dm, None, None, "local quadratic")

    try:
        coeffs = np.polyfit(x, y, 2)
    except Exception:
        return _uncertainty_dict(best_dm, None, None, "local quadratic")

    a = float(coeffs[0])
    if not np.isfinite(a) or a >= 0:
        return _uncertainty_dict(best_dm, None, None, "local quadratic")

    resid = y - np.polyval(coeffs, x)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(resid))

    if not np.isfinite(sigma) or sigma <= 0:
        step = float(np.median(np.diff(dm))) if n > 1 else 0.0
        if step > 0:
            return _uncertainty_dict(best_dm, best_dm - 0.5 * step, best_dm + 0.5 * step, "local quadratic")
        return _uncertainty_dict(best_dm, None, None, "local quadratic")

    width = float(np.sqrt(sigma / -a))
    if not np.isfinite(width) or width <= 0:
        return _uncertainty_dict(best_dm, None, None, "local quadratic")
    if x.size > 1:
        width = min(width, 0.5 * float(x[-1] - x[0]))

    return _uncertainty_dict(best_dm, best_dm - width, best_dm + width, "local quadratic (1-sigma)")


def from_snr_drop(
    dm_values: np.ndarray,
    snr_values: np.ndarray,
    best_idx: int,
    drop: float = 1.0,
) -> Dict[str, Optional[float]]:
    """
    Uncertainty defined by the DM range where S/N falls by *drop* from its peak.
    """
    dm, sn = _finite_metric_arrays(dm_values, snr_values)
    best_idx_eff = int(np.argmax(sn))
    best_dm = float(dm[best_idx_eff])
    threshold = float(sn[best_idx_eff]) - float(drop)
    low_dm = _dm_crossing(sn, dm, best_idx_eff, threshold, direction=-1)
    high_dm = _dm_crossing(sn, dm, best_idx_eff, threshold, direction=1)
    return _uncertainty_dict(best_dm, low_dm, high_dm, f"S/N drop = {drop}")


def from_shrine_relative(
    dm_values: np.ndarray,
    metric_values: np.ndarray,
    reference_profiles: np.ndarray,
    shrine_get_kc,
    shrine_lowpass_smooth,
    shrine_get_ranges_above_max,
    shrine_uncertainty_calc,
    kc: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    SHRINE-style relative-uncertainty estimate using the per-DM time profiles.

    Parameters
    ----------
    reference_profiles:
        Array of shape ``(n_dm, n_time)`` — one collapsed time profile per DM trial.
    shrine_*:
        SHRINE helper callables imported from the SHRINE subpackage.
    kc:
        If given, skip the auto kc determination and use this value directly.
    """
    dm = np.asarray(dm_values, dtype=float)
    metric = np.asarray(metric_values, dtype=float)
    profiles = np.asarray(reference_profiles, dtype=float)

    if profiles.ndim != 2 or profiles.shape[0] != dm.shape[0] or metric.shape[0] != dm.shape[0]:
        best_idx = int(np.argmax(metric))
        return _uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

    # Replace non-finite values row-wise so DCT/filtering remains stable.
    profiles_finite = profiles.copy()
    for row_idx in range(profiles_finite.shape[0]):
        row = profiles_finite[row_idx]
        finite = np.isfinite(row)
        if np.any(finite):
            profiles_finite[row_idx, ~finite] = float(np.nanmean(row[finite]))
        else:
            profiles_finite[row_idx, :] = 0.0

    ci_data = dct(profiles_finite, norm="ortho")
    k_len = ci_data.shape[1]
    if k_len < 2:
        best_idx = int(np.argmax(metric))
        return _uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

    if kc is None:
        with contextlib.redirect_stdout(io.StringIO()):
            kc_use = int(shrine_get_kc(ci_data))
    else:
        kc_use = int(kc)
    kc_use = max(1, min(kc_use, k_len))

    i_smooth, lpf_data, _, f_l = shrine_lowpass_smooth(ci_data, kc_use, order=3)
    k = np.linspace(1, k_len, k_len)
    hp = np.sqrt(2 - 2 * np.cos((k - 1) * np.pi / k_len))
    filter_diag = np.diag(hp * f_l)

    delta_i = profiles_finite - i_smooth
    max_idx = int(np.argmax(metric))
    delta_delta_i = delta_i - delta_i[max_idx]

    relative_uncertainty = np.asarray(
        shrine_uncertainty_calc(delta_delta_i, lpf_data, filter_diag), dtype=float
    )
    relative_uncertainty[~np.isfinite(relative_uncertainty)] = 0.0

    max_metric = float(metric[max_idx])
    adjusted_metrics = metric + (metric * relative_uncertainty)
    possible_max_ranges = shrine_get_ranges_above_max(max_metric, adjusted_metrics)

    if len(possible_max_ranges) < 1:
        return _uncertainty_dict(float(dm[max_idx]), None, None, "SHRINE relative uncertainty")

    low_idx = int(possible_max_ranges[0][0])
    low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
    high_dm = None
    if len(possible_max_ranges[-1]) == 2:
        high_idx = int(possible_max_ranges[-1][1])
        if 0 <= high_idx < len(dm):
            high_dm = float(dm[high_idx])

    return _uncertainty_dict(float(dm[max_idx]), low_dm, high_dm, "SHRINE relative uncertainty")


def from_shrine_outputs(
    dm_values: np.ndarray,
    structure_values: np.ndarray,
    run_dir,
    run_prefix: str,
    best_idx: int,
    shrine_get_ranges_above_max,
) -> Dict[str, Optional[float]]:
    """
    Extract uncertainty from a SHRINE *_Relative_Uncertainties.dat* file that
    was written by ``maximise_structure.py``.
    """
    from pathlib import Path

    rel_path = Path(run_dir) / f"{run_prefix}_Relative_Uncertainties.dat"
    best_dm = float(dm_values[int(best_idx)])
    if not rel_path.exists():
        return _uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

    rel = np.asarray(np.loadtxt(rel_path), dtype=float)
    sp = np.asarray(structure_values, dtype=float)
    dm = np.asarray(dm_values, dtype=float)
    if rel.shape != sp.shape:
        return _uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

    max_index = int(best_idx)
    max_sp = float(sp[max_index])
    adjusted_sps = sp + sp * rel
    possible_max_ranges = shrine_get_ranges_above_max(max_sp, adjusted_sps)

    if len(possible_max_ranges) < 1:
        return _uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

    low_idx = int(possible_max_ranges[0][0])
    low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
    high_dm = None
    if len(possible_max_ranges[-1]) == 2:
        high_idx = int(possible_max_ranges[-1][1])
        if 0 <= high_idx < len(dm):
            high_dm = float(dm[high_idx])

    return _uncertainty_dict(float(dm[max_index]), low_dm, high_dm, "SHRINE relative uncertainty")


def clamp_to_dm_bounds(
    best_dm: float,
    uncertainty: Dict[str, Optional[float]],
    dm_values: np.ndarray,
    fill_missing_with_bounds: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Clamp uncertainty DM bounds to the available DM sample range and ensure
    they are on the correct side of *best_dm*.
    """
    dm = np.asarray(dm_values, dtype=float)
    finite = dm[np.isfinite(dm)]
    if finite.size == 0:
        return uncertainty

    dm_min = float(np.min(finite))
    dm_max = float(np.max(finite))
    best_dm = float(best_dm)

    low_dm = uncertainty.get("uncertainty_low_dm")
    high_dm = uncertainty.get("uncertainty_high_dm")
    if fill_missing_with_bounds:
        low_dm = dm_min if low_dm is None else low_dm
        high_dm = dm_max if high_dm is None else high_dm

    low_clamped = None if low_dm is None else float(np.clip(float(low_dm), dm_min, dm_max))
    high_clamped = None if high_dm is None else float(np.clip(float(high_dm), dm_min, dm_max))
    if low_clamped is not None:
        low_clamped = min(low_clamped, best_dm)
    if high_clamped is not None:
        high_clamped = max(high_clamped, best_dm)

    clamped = dict(uncertainty)
    clamped["uncertainty_low_dm"] = low_clamped
    clamped["uncertainty_high_dm"] = high_clamped
    clamped["uncertainty_minus"] = None if low_clamped is None else float(best_dm - low_clamped)
    clamped["uncertainty_plus"] = None if high_clamped is None else float(high_clamped - best_dm)
    return clamped


# ---------------------------------------------------------------------------
# Formatting helpers (no external dependencies)
# ---------------------------------------------------------------------------

def format_dm(dm: float, precision: int = 6) -> str:
    return f"{dm:.{precision}f}".rstrip("0").rstrip(".")


def format_uncertainty(
    best_dm: float,
    minus: Optional[float],
    plus: Optional[float],
    precision: int = 6,
) -> str:
    best = format_dm(best_dm, precision)
    if minus is None and plus is None:
        return f"{best} (-?/+?)"
    if minus is None:
        return f"{best} (-?/+{format_dm(plus, precision)})"
    if plus is None:
        return f"{best} (-{format_dm(minus, precision)}/+?)"
    return f"{best} (-{format_dm(minus, precision)}/+{format_dm(plus, precision)})"
