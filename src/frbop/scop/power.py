"""power law fitting and correction."""

import numpy as np


def _powerlaw_mean_spectrum(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    spectral_index: float = -1.5,
) -> np.ndarray:
    """Power-law template scaled robustly via median of per-channel ratios."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum  = np.asarray(spectrum,  dtype=float)
    template  = np.power(freq_mhz, spectral_index)
    finite    = np.isfinite(template) & np.isfinite(spectrum) & (template > 0) & (spectrum > 0)
    if not np.any(finite):
        raise ValueError("No finite positive samples for mean-spectrum correction")
    scale = float(np.nanmedian(spectrum[finite] / template[finite]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Power-law mean spectrum scale is ill-conditioned")
    return scale * template


def fit_powerlaw_spectral_index_with_error(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    min_points: int = 6,
    smooth_bins: int = 64,
) -> tuple[float | None, float | None]:
    """Fit spectral index by binning first to suppress per-channel noise."""
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum  = np.asarray(spectrum,  dtype=float)

    # Bin to suppress noise before log-log fit
    n = freq_mhz.size
    if n > smooth_bins * 2:
        bin_edges   = np.linspace(0, n, smooth_bins + 1, dtype=int)
        binned_freq = np.array([
            np.nanmean(freq_mhz[bin_edges[i]:bin_edges[i+1]])
            for i in range(smooth_bins)
        ])
        binned_spec = np.array([
            np.nanmean(spectrum[bin_edges[i]:bin_edges[i+1]])
            for i in range(smooth_bins)
        ])
    else:
        binned_freq = freq_mhz
        binned_spec = spectrum

    finite = (np.isfinite(binned_freq) & np.isfinite(binned_spec)
              & (binned_freq > 0) & (binned_spec > 0))
    if np.count_nonzero(finite) < min_points:
        return None, None

    try:
        coeffs, cov = np.polyfit(
            np.log(binned_freq[finite]), np.log(binned_spec[finite]), 1, cov=True
        )
    except Exception:
        return None, None

    alpha     = float(coeffs[0])
    alpha_err = float(np.sqrt(cov[0, 0])) if cov is not None and np.isfinite(cov[0, 0]) else None
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

    Returns (corrected, mean_model, index_used, index_fit, index_err).
    corrected[i] = (S[i] - Sbar[i]) / Sbar[i]  for SNR-passing channels
    corrected[i] = NaN                           for channels failing SNR gate
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

    # Base validity: finite model and spectrum
    good = np.isfinite(mean_model) & np.isfinite(spectrum) & (mean_model > 0)

    # SNR gate: exclude channels where model < min_snr * off-pulse noise
    #if off_pulse_rms is not None:
    #    rms    = np.asarray(off_pulse_rms, dtype=float)
    #    snr_ok = np.isfinite(rms) & (rms > 0) & (mean_model > min_snr * rms)
    #    good   = good & snr_ok
    #else:
    #    # Fallback: exclude channels below 10% of peak model (band edges)
    #    peak_model = float(np.nanmax(mean_model[good])) if np.any(good) else 1.0
    #    good = good & (mean_model > 0.10 * peak_model)

    corrected       = np.full_like(spectrum, np.nan)
    corrected[good] = (spectrum[good] - mean_model[good]) / mean_model[good]

    return corrected, mean_model, float(spectral_index_used), spectral_index_fit, spectral_index_err
