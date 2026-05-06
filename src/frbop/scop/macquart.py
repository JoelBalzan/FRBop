"""Macquart (2019) autocovariance estimator helpers."""

import numpy as np


def _powerlaw_mean_spectrum(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    spectral_index: float = -1.5,
) -> np.ndarray:
    """Power-law template scaled to match total flux of *spectrum* (returns ndarray)."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    template = np.power(freq_mhz, spectral_index)
    finite = np.isfinite(template) & np.isfinite(spectrum)
    if not np.any(finite):
        raise ValueError("No finite samples available for mean-spectrum correction")
    template_sum = np.nansum(template[finite])
    if template_sum == 0 or not np.isfinite(template_sum):
        raise ValueError("Power-law mean spectrum template is ill-conditioned")
    scale = float(np.nansum(spectrum[finite]) / template_sum)
    return scale * template


def estimate_macquart_modulation_index(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    corrected: bool = False,
    fit_max_lag_mhz: float | None = None,
    min_fit_points: int = 4,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Estimate m^2 and Delta nu_d from the mean-normalised spectral autocovariance.

    Implements Macquart et al. (2019):
        C(Delta nu) = <[F(nu'+Delta nu) - Fbar(nu')] * [F(nu') - Fbar(nu')]> / Fbar^2

    Returns (m2, delta_nu_d, lags_mhz, acov).
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum)
    if np.count_nonzero(finite) < 4:
        return None, None, np.array([]), np.array([])

    freq_mhz = freq_mhz[finite]
    spectrum = spectrum[finite]

    if corrected:
        mean_model = _powerlaw_mean_spectrum(freq_mhz, spectrum, spectral_index=-1.5)
    else:
        mean_level = float(np.nanmean(spectrum))
        if not np.isfinite(mean_level) or mean_level <= 0:
            return None, None, np.array([]), np.array([])
        mean_model = np.full_like(spectrum, mean_level)

    positive = np.isfinite(mean_model) & (mean_model > 0)
    if np.count_nonzero(positive) < 4:
        return None, None, np.array([]), np.array([])

    freq_mhz = freq_mhz[positive]
    spectrum = spectrum[positive]
    mean_model = mean_model[positive]

    # fractional deviation - do NOT subtract residual mean (would bias m^2)
    frac = (spectrum - mean_model) / mean_model

    if np.count_nonzero(np.isfinite(frac)) < 4:
        return None, None, np.array([]), np.array([])

    # channel spacing - use abs() so ascending/descending both work
    if freq_mhz.size > 1:
        df = float(np.median(np.abs(np.diff(freq_mhz))))
    else:
        df = np.nan

    if not np.isfinite(df) or df <= 0:
        return None, None, np.array([]), np.array([])

    result = np.correlate(frac, frac, mode="full")
    acov = result[result.size // 2:]
    counts = np.arange(frac.size, 0, -1, dtype=float)
    acov = acov / counts
    lags_mhz = np.arange(acov.size, dtype=float) * df

    # build fit mask (positive lags, optional upper bound)
    fit_mask = np.isfinite(acov) & (lags_mhz > 0)
    if fit_max_lag_mhz is not None:
        fit_mask &= lags_mhz <= float(fit_max_lag_mhz)

    m2 = None
    delta_nu_d = None

    if acov.size > 0 and np.isfinite(acov[0]):
        m2 = float(acov[0])
        half_power = 0.5 * m2

        fit_indices = np.where(fit_mask)[0]
        if fit_indices.size > 0:
            rel_idxs = np.where(acov[fit_indices] <= half_power)[0]
            candidate_idxs = fit_indices[rel_idxs] if rel_idxs.size > 0 else np.array([], dtype=int)
            candidate_idxs = candidate_idxs[candidate_idxs > 0]
            if candidate_idxs.size > 0:
                idx = int(candidate_idxs[0])
                prev_idx = idx - 1
                x0, x1 = float(lags_mhz[prev_idx]), float(lags_mhz[idx])
                y0, y1 = float(acov[prev_idx]), float(acov[idx])
                if np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
                    t = (half_power - y0) / (y1 - y0)
                    delta_nu_d = float(x0 + t * (x1 - x0))
                elif np.isfinite(x1):
                    delta_nu_d = x1

    if m2 is None:
        fit_acov = acov[fit_mask]
        near_zero = fit_acov[: min(max(3, min_fit_points), fit_acov.size)]
        near_zero = near_zero[np.isfinite(near_zero)]
        if near_zero.size > 0:
            m2 = float(np.nanmean(np.maximum(near_zero, 0.0)))

    return m2, delta_nu_d, lags_mhz, acov


def macquart_dnu_from_window(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    lag_lo_mhz: float,
    lag_hi_mhz: float,
    corrected: bool = False,
) -> float | None:
    """Extract a Macquart half-power Delta nu_d restricted to a specific lag window.

    Useful for isolating the decorrelation scale of one Lorentzian component
    when multiple components are present in the ACF. The half-power crossing
    is searched only within [lag_lo_mhz, lag_hi_mhz].

    Returns Delta nu_d in MHz, or None if no crossing is found in the window.
    """
    _, _, lags, acov = estimate_macquart_modulation_index(
        freq_mhz, spectrum, corrected=corrected, fit_max_lag_mhz=lag_hi_mhz
    )
    if lags.size == 0 or acov.size == 0 or not np.isfinite(acov[0]):
        return None

    half_power = 0.5 * float(acov[0])
    window_mask = np.isfinite(acov) & (lags >= lag_lo_mhz) & (lags <= lag_hi_mhz) & (lags > 0)
    idxs = np.where(window_mask)[0]
    if idxs.size == 0:
        return None

    rel = np.where(acov[idxs] <= half_power)[0]
    if rel.size == 0:
        return None

    idx = int(idxs[rel[0]])
    prev_idx = idx - 1
    if prev_idx < 0:
        return float(lags[idx])
    x0, x1 = float(lags[prev_idx]), float(lags[idx])
    y0, y1 = float(acov[prev_idx]), float(acov[idx])
    if y1 != y0:
        t = (half_power - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return float(x1)
