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


def fit_powerlaw_spectral_index(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    min_points: int = 6,
) -> float | None:
    """Fit a power-law spectral index to the spectrum (log-log linear fit).

    Returns the fitted spectral index, or None if the fit is ill-conditioned.
    """
    alpha, _ = fit_powerlaw_spectral_index_with_error(
        freq_mhz,
        spectrum,
        min_points=min_points,
    )
    return alpha


def fit_powerlaw_spectral_index_with_error(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    min_points: int = 6,
) -> tuple[float | None, float | None]:
    """Fit a power-law spectral index with uncertainty (log-log linear fit).

    Returns (spectral_index, spectral_index_err).
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum) & (freq_mhz > 0) & (spectrum > 0)
    if np.count_nonzero(finite) < min_points:
        return None, None

    log_freq = np.log(freq_mhz[finite])
    log_spec = np.log(spectrum[finite])
    if log_freq.size < min_points:
        return None, None

    try:
        coeffs, cov = np.polyfit(log_freq, log_spec, 1, cov=True)
    except Exception:
        return None, None

    alpha = float(coeffs[0])
    alpha_err = None
    if cov is not None and np.isfinite(cov[0, 0]):
        alpha_err = float(np.sqrt(cov[0, 0]))
    return alpha, alpha_err


def correct_spectrum_powerlaw(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    off_pulse_rms: np.ndarray | None = None,
    spectral_index: float | None = None,
    min_snr: float = 2.0,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray, float, float | None, float | None]:
    """Remove intrinsic spectral structure, returning the fractional residual.

    Instead of dividing raw flux by the model (which inflates noise in faint
    channels), this returns the SNR-gated fractional deviation:

        corrected[i] = (S[i] - S̄[i]) / S̄[i]   if S̄[i] > min_snr * σ[i]
        corrected[i] = 0.0                        otherwise  (excluded from ACF)

    Returns
    -------
    corrected          : fractional residual spectrum, zero where SNR-gated out
    mean_model         : the power-law mean model S̄(ν)
    spectral_index_used: index actually applied
    spectral_index_fit : fitted index (None if user-supplied)
    spectral_index_err : 1-sigma error on fit (None if user-supplied)
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum  = np.asarray(spectrum,  dtype=float)

    if spectral_index is None:
        spectral_index_fit, spectral_index_err = fit_powerlaw_spectral_index_with_error(
            freq_mhz, spectrum, min_points=min_points,
        )
        spectral_index_used = spectral_index_fit if spectral_index_fit is not None else -1.5
    else:
        spectral_index_fit  = None
        spectral_index_err  = None
        spectral_index_used = float(spectral_index)

    mean_model = _powerlaw_mean_spectrum(
        freq_mhz, spectrum, spectral_index=float(spectral_index_used)
    )

    corrected = np.zeros_like(spectrum)
    good = np.isfinite(mean_model) & np.isfinite(spectrum) & (mean_model > 0)

    # SNR gate: only include channels where the model exceeds min_snr * noise
    if off_pulse_rms is not None:
        rms = np.asarray(off_pulse_rms, dtype=float)
        snr_ok = np.isfinite(rms) & (rms > 0) & (mean_model > min_snr * rms)
        good = good & snr_ok
    # If no off_pulse_rms supplied, fall back to gating on model amplitude
    # (channels below 10% of peak model are excluded)
    else:
        peak_model = float(np.nanmax(mean_model[good])) if np.any(good) else 1.0
        good = good & (mean_model > 0.10 * peak_model)

    corrected[good] = (spectrum[good] - mean_model[good]) / mean_model[good]
    # Channels excluded by SNR gate are left as NaN so autocorr ignores them
    corrected[~good] = np.nan

    return corrected, mean_model, float(spectral_index_used), spectral_index_fit, spectral_index_err


def estimate_macquart_modulation_index(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    corrected: bool = False,
    spectral_index: float | None = -1.5,
    mean_model: np.ndarray | None = None,
    fit_max_lag_mhz: float | None = None,
    min_fit_points: int = 4,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Estimate m^2 and Delta nu_d from the mean-normalised spectral autocovariance.

    Implements Macquart et al. (2019):
        C(Delta nu) = <[F(nu'+Delta nu) - Fbar(nu')] * [F(nu') - Fbar(nu')]> / Fbar^2

    spectral_index: power-law index to remove intrinsic structure when corrected=True.
    Returns (m2, delta_nu_d, lags_mhz, acov).
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mean_model_arr = None
    if mean_model is not None:
        mean_model_arr = np.asarray(mean_model, dtype=float)
        if mean_model_arr.shape != spectrum.shape:
            raise ValueError("mean_model must match spectrum shape")

    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum)
    if np.count_nonzero(finite) < 4:
        return None, None, np.array([]), np.array([])

    freq_mhz = freq_mhz[finite]
    spectrum = spectrum[finite]
    if mean_model_arr is not None:
        mean_model_arr = mean_model_arr[finite]

    if mean_model_arr is not None:
        mean_model = mean_model_arr
    elif corrected:
        if spectral_index is None:
            spectral_index = fit_powerlaw_spectral_index(freq_mhz, spectrum)
        if spectral_index is None:
            spectral_index = -1.5
        mean_model = _powerlaw_mean_spectrum(freq_mhz, spectrum, spectral_index=float(spectral_index))
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
    spectral_index: float | None = None,
    mean_model: np.ndarray | None = None,
) -> float | None:
    """Extract a Macquart half-power Delta nu_d restricted to a specific lag window.

    Useful for isolating the decorrelation scale of one Lorentzian component
    when multiple components are present in the ACF. The half-power crossing
    is searched only within [lag_lo_mhz, lag_hi_mhz].

    Returns Delta nu_d in MHz, or None if no crossing is found in the window.
    """
    _, _, lags, acov = estimate_macquart_modulation_index(
        freq_mhz,
        spectrum,
        corrected=corrected,
        spectral_index=spectral_index,
        mean_model=mean_model,
        fit_max_lag_mhz=lag_hi_mhz,
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
