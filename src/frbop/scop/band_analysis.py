"""Frequency-band scintillation analysis helpers."""

import numpy as np

from frbop.scop.acf import autocorr
from frbop.scop.fit_utils import build_fit_diagnostics, fit_with_restarts
from frbop.scop.models import lorentzian
from frbop.utils.peaks import \
    select_frequency_bands_manual as shared_select_frequency_bands_manual


def select_frequency_bands_manual(
    freq_axis, spectrum, *,
    title='Click start/end bounds for each frequency band (close window when done)',
    x_label='Frequency [MHz]', y_label='Flux', exclusive_end=True,
):
    return shared_select_frequency_bands_manual(
        freq_axis,
        spectrum,
        title=title,
        x_label=x_label,
        y_label=y_label,
        exclusive_end=exclusive_end,
    )
def convert_mhz_to_frequency_indices(freq_axis: np.ndarray, mhz_values: list[float], N: int) -> list[tuple[int, int]]:
    """
    Convert MHz frequency pairs to (start_idx, stop_idx) tuples.

    Args:
        freq_axis: 1D array of frequencies in MHz (assumed ascending after normalization)
        mhz_values: Flat list of MHz values, grouped as pairs for each band (order-independent)
        N: Total number of frequency channels

    Returns:
        List of (start_idx, stop_idx) tuples
    """
    if len(mhz_values) % 2 != 0:
        raise ValueError(f"mhz_values must have even length (pairs of frequencies); got {len(mhz_values)}")

    band_regions = []
    for i in range(0, len(mhz_values), 2):
        mhz1 = mhz_values[i]
        mhz2 = mhz_values[i + 1]

        # Handle pairs in any order
        low_mhz = min(mhz1, mhz2)
        high_mhz = max(mhz1, mhz2)

        # Find indices using searchsorted (assumes freq_axis is ascending)
        start_idx = np.searchsorted(freq_axis, low_mhz, side='left')
        stop_idx = np.searchsorted(freq_axis, high_mhz, side='right')

        # Clamp to valid range
        start_idx = max(0, min(start_idx, N - 1))
        stop_idx = max(start_idx + 1, min(stop_idx, N))

        band_regions.append((start_idx, stop_idx))

    return band_regions


def measure_scintillation_bands(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    band_regions: list[tuple[int, int]],
    *,
    fit_max_lag_mhz: float,
    off_pulse: np.ndarray | None = None,
    raw_spectrum: np.ndarray | None = None,
) -> list[dict]:
    """Measure Delta nu_d in each sub-band by fitting a Lorentzian to the normalised ACF.

    Parameters
    ----------
    freq_mhz      : channel frequencies [MHz], ascending
    spectrum      : corrected fractional residual spectrum (output of correct_spectrum_powerlaw)
    band_regions  : list of (start_idx, stop_idx) index pairs
    fit_max_lag_mhz : upper lag limit for Lorentzian fit [MHz]
    off_pulse     : 2D off-pulse data (nfreq, n_offpulse) in raw data units, for noise estimation
    raw_spectrum  : 1D raw burst spectrum (nfreq,) in raw data units, for SNR denominator
    """
    results: list[dict] = []
    freq_mhz     = np.asarray(freq_mhz, dtype=float)
    spectrum     = np.asarray(spectrum,  dtype=float)
    raw_spectrum = np.asarray(raw_spectrum, dtype=float) if raw_spectrum is not None else None

    max_bound_fraction = 0.9
    max_rel_err        = 0.75
    min_r2             = 0.5

    for band_idx, (start_idx, stop_idx) in enumerate(band_regions, start=1):
        sub_freq = freq_mhz[start_idx:stop_idx]
        sub_spec = spectrum[start_idx:stop_idx]

        finite = np.isfinite(sub_freq) & np.isfinite(sub_spec)
        if np.count_nonzero(finite) < 8:
            print(f"  Band {band_idx}: skipped — fewer than 8 finite channels")
            continue

        sub_freq = sub_freq[finite]
        sub_spec = sub_spec[finite]
        band_width_mhz = float(np.nanmax(sub_freq) - np.nanmin(sub_freq))
        df_band        = float(np.median(np.abs(np.diff(sub_freq))))

        band_fit_max_lag = min(float(fit_max_lag_mhz), 0.5 * band_width_mhz)
        if band_fit_max_lag < df_band * 2:
            print(f"  Band {band_idx}: skipped — fit-max-lag ({band_fit_max_lag:.3f} MHz) "
                  f"too small for channel spacing ({df_band:.3f} MHz)")
            continue

        # sub_spec is already a fractional residual from correct_spectrum_powerlaw;
        # just zero-mean it (should already be near zero, but remove any residual offset)
        sub_mean = float(np.nanmean(sub_spec))
        sub_norm = sub_spec - sub_mean

        acf_band = autocorr(sub_norm)
        lags_band = np.arange(len(acf_band)) * df_band

        fit_mask = (lags_band > 0) & (lags_band <= band_fit_max_lag) & np.isfinite(acf_band)
        lags_fit = lags_band[fit_mask]
        acf_fit  = acf_band[fit_mask]

        if lags_fit.size < 4:
            print(f"  Band {band_idx}: skipped — fewer than 4 ACF points in fit range")
            continue

        d_guess   = band_fit_max_lag / 4.0
        amp_guess = max(0.05, float(np.nanmax(acf_fit) - np.nanmin(acf_fit)))
        off_guess = float(np.nanmedian(acf_fit[-max(2, acf_fit.size // 5):]))

        best = fit_with_restarts(
            lorentzian, lags_fit, acf_fit,
            p0_list=[
                [d_guess,                amp_guess, off_guess],
                [d_guess * 0.4,          amp_guess, off_guess],
                [d_guess * 2.0,          amp_guess, off_guess],
                [band_fit_max_lag * 0.1, amp_guess, off_guess],
            ],
            bounds=([df_band * 0.5, 0.0, -1.5], [band_fit_max_lag, 2.5, 1.5]),
            maxfev=20000,
        )

        if best is None:
            print(f"  Band {band_idx}: Lorentzian fit failed (all initialisations diverged)")
            continue

        popt, pcov, ymod   = best
        dnu_fit, A_fit, C_fit = popt

        if dnu_fit <= 0 or not np.isfinite(dnu_fit):
            print(f"  Band {band_idx}: Lorentzian fit returned non-physical "
                  f"Δν_d = {dnu_fit:.4f} MHz; skipping")
            continue

        # Covariance-based uncertainty (unreliable alone — correlated ACF residuals)
        try:
            dnu_err_fit = float(np.sqrt(np.diag(pcov))[0])
        except Exception:
            dnu_err_fit = np.nan

        # ------------------------------------------------------------------
        # Noise-informed uncertainty from off-pulse radiometric noise
        # ------------------------------------------------------------------
        dnu_err_noise   = np.nan
        n_eff           = np.nan
        snr_per_scintle = np.nan
        noise_ratio     = np.nan

        if off_pulse is not None and off_pulse.size > 0 and raw_spectrum is not None:
            off_band = off_pulse[start_idx:stop_idx]

            if off_band.ndim == 2 and off_band.shape[1] > 1:
                # Per-channel thermal noise: MAD over off-pulse time samples
                per_chan_rms = 1.4826 * np.nanmedian(
                    np.abs(off_band - np.nanmedian(off_band, axis=1, keepdims=True)),
                    axis=1,
                )
                valid_rms = per_chan_rms[np.isfinite(per_chan_rms) & (per_chan_rms > 0)]
                sigma_n   = float(np.nanmedian(valid_rms)) if valid_rms.size > 0 else np.nan

                # Mean signal in raw data units — use raw_spectrum, not the corrected residual
                raw_sub  = raw_spectrum[start_idx:stop_idx]
                raw_sub  = raw_sub[np.isfinite(raw_sub)]
                raw_mean = float(np.nanmean(raw_sub)) if raw_sub.size > 0 else np.nan

                if (np.isfinite(sigma_n) and sigma_n > 0
                        and np.isfinite(raw_mean) and raw_mean > 0):

                    snr_chan    = raw_mean / sigma_n
                    noise_ratio = float(sigma_n / raw_mean)

                    # Number of independent scintles across the band
                    n_eff = float(band_width_mhz / max(dnu_fit, df_band))

                    # Noise adds a white pedestal ~1/SNR² to the ACF, which broadens
                    # the apparent Lorentzian and inflates Δν_d uncertainty.
                    # Effective scintle count accounting for this bias:
                    #   N_eff_eff = N_eff / (1 + 1/SNR²)²
                    n_eff_effective = n_eff / (1.0 + 1.0 / snr_chan ** 2) ** 2

                    # Fundamental ACF estimator variance for a Lorentzian:
                    #   sigma(Δν_d) / Δν_d = 1 / sqrt(2 * N_eff_eff)
                    if n_eff_effective > 0:
                        dnu_err_noise   = float(dnu_fit / np.sqrt(2.0 * n_eff_effective))
                        snr_per_scintle = float(snr_chan / np.sqrt(max(n_eff, 1.0)))

        # Use noise-informed error where available; fall back to covariance error
        dnu_err = (dnu_err_noise
                   if np.isfinite(dnu_err_noise) and dnu_err_noise > 0
                   else dnu_err_fit)

        diag    = build_fit_diagnostics(acf_fit, ymod, k=3)
        rel_err = dnu_err_fit / dnu_fit if np.isfinite(dnu_err_fit) and dnu_fit > 0 else np.inf

        if dnu_fit >= max_bound_fraction * band_fit_max_lag:
            print(f"  Band {band_idx}: skipped — Δν_d = {dnu_fit:.4f} MHz is too close to the "
                  f"fit upper bound ({band_fit_max_lag:.4f} MHz)")
            continue
        if not np.isfinite(rel_err) or rel_err > max_rel_err:
            print(f"  Band {band_idx}: skipped — fractional uncertainty {rel_err:.2f} exceeds "
                  f"the limit ({max_rel_err:.2f})")
            continue
        if not np.isfinite(diag["r2"]) or diag["r2"] < min_r2:
            print(f"  Band {band_idx}: skipped — poor fit quality R² = {diag['r2']:.4f} < {min_r2:.2f}")
            continue

        results.append(dict(
            band_idx        = band_idx,
            start_idx       = int(start_idx),
            stop_idx        = int(stop_idx),
            center_mhz      = float(np.nanmean(sub_freq)),
            band_width_mhz  = band_width_mhz,
            dnu_mhz         = float(dnu_fit),
            dnu_err_mhz     = dnu_err,
            dnu_err_fit_mhz = dnu_err_fit,
            dnu_err_noise_mhz = dnu_err_noise,
            n_eff           = n_eff,
            snr_per_scintle = snr_per_scintle,
            noise_ratio     = noise_ratio,
            A_fit           = float(A_fit),
            C_fit           = float(C_fit),
            r2              = diag["r2"],
            rmse            = diag["rmse"],
            _lags           = lags_band,
            _acf            = acf_band,
            _lags_fit       = lags_fit,
            _ymod           = ymod,
            _spec_freq      = sub_freq,
            _spec_flux      = sub_spec,
        ))

    return results


def fit_scintillation_band_power_law(
    band_centers_mhz: np.ndarray,
    band_dnu_mhz: np.ndarray,
    reference_freq_mhz: float,
    band_dnu_err_mhz: np.ndarray | None = None,
    *,
    comparison_alpha: float = 4.4,
) -> dict:
    band_centers_mhz = np.asarray(band_centers_mhz, dtype=float)
    band_dnu_mhz = np.asarray(band_dnu_mhz, dtype=float)
    if band_dnu_err_mhz is not None:
        band_dnu_err_mhz = np.asarray(band_dnu_err_mhz, dtype=float)

    finite = (
        np.isfinite(band_centers_mhz)
        & np.isfinite(band_dnu_mhz)
        & (band_centers_mhz > 0)
        & (band_dnu_mhz > 0)
    )
    if band_dnu_err_mhz is not None:
        finite = finite & np.isfinite(band_dnu_err_mhz) & (band_dnu_err_mhz > 0)

    if np.count_nonzero(finite) < 2:
        raise ValueError("Need at least two valid bands to fit a scintillation bandwidth power law")

    x = np.log(band_centers_mhz[finite])
    y = np.log(band_dnu_mhz[finite])

    popt, pcov = np.polyfit(x, y, 1, cov=True)

    alpha_fit, log_norm_fit = popt
    alpha_err = float(np.sqrt(np.diag(pcov)[0])) if np.isfinite(pcov[0, 0]) else np.nan
    log_norm_44 = float(np.mean(y - comparison_alpha * x))

    ref = float(reference_freq_mhz)
    if ref <= 0:
        raise ValueError("reference_freq_mhz must be > 0")

    dnu_ref_fit = float(np.exp(log_norm_fit + alpha_fit * np.log(ref)))
    dnu_ref_44 = float(np.exp(log_norm_44 + comparison_alpha * np.log(ref)))
    model_fit = np.exp(log_norm_fit + alpha_fit * x)
    model_44 = np.exp(log_norm_44 + comparison_alpha * x)
    residual_rms = float(np.sqrt(np.mean((y - np.log(model_fit)) ** 2)))

    return dict(
        alpha_fit=float(alpha_fit),
        alpha_err=alpha_err,
        log_norm_fit=float(log_norm_fit),
        dnu_ref_fit=dnu_ref_fit,
        comparison_alpha=float(comparison_alpha),
        log_norm_44=log_norm_44,
        dnu_ref_44=dnu_ref_44,
        residual_rms=residual_rms,
        fit_freq_mhz=band_centers_mhz[finite],
        fit_dnu_mhz=band_dnu_mhz[finite],
        fit_model_mhz=model_fit,
        model_44_mhz=model_44,
        reference_freq_mhz=ref,
    )
