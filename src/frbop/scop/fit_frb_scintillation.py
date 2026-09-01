import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

# Model-selection significance threshold for the nested F-test on residual RSS.
# Calibrated on synthetic single-component data: p=0.01 keeps the spurious
# multi-component rate at ~2% while still detecting most genuine two-component ACFs.
F_TEST_P_THRESH = 0.01

from frbop.scop import acf
from frbop.scop.acf import autocorr
from frbop.scop.band_analysis import (convert_mhz_to_frequency_indices,
                                      fit_scintillation_band_power_law,
                                      measure_scintillation_bands,
                                      select_frequency_bands_manual)
from frbop.scop.fit_utils import (_decode_lorentzian_components,
                                  build_fit_diagnostics, fit_with_restarts)
from frbop.scop.gating import (find_burst_window, select_peak_fwhm_manual,
                               select_peaks_manual)
from frbop.scop.models import (lorentzian, lorentzian_2c, lorentzian_3c,
                               scattered_gaussian)
from frbop.scop.ne2025 import (estimate_lg_kpc_from_ne2025,
                               ne2025_scattering_prediction,
                               print_ne2025_scattering_prediction)
from frbop.scop.physics import (estimate_ds_kpc_from_redshift,
                                radec_to_galactic_deg,
                                scale_scintillation_bandwidth)
from frbop.scop.plotting import (compute_modulation_index, plot_acf_fit,
                                 plot_lorentzian_diagnostics,
                                 plot_modulation_index,
                                 plot_scintillation_band_power_law,
                                 plot_spectrum_powerlaw_fit)
from frbop.scop.power import correct_spectrum_powerlaw
from frbop.scop.two_screen import print_two_screen_results, two_screen_estimate
from frbop.utils.peaks import (measure_fwhm_region, parse_peak_index_pairs,
                               split_frequency_bands_equal,
                               split_frequency_bands_equal_snr)
from frbop.utils.plotting import (set_pub_col, set_pub_style)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit FRB scintillation from dynamic spectrum files")
    parser.add_argument("ds",   nargs="?", default="FRB_250607_htr_dsI.npy")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy")
    parser.add_argument("--smooth",            type=int,   default=1)
    parser.add_argument("--manual-peaks",      action="store_true")
    parser.add_argument("--manual-peak-fwhm", action="store_true",
                        help="Click a peak once and gate the burst using its FWHM.")
    parser.add_argument("--auto-peak-fwhm",   action="store_true",
                        help="Use the argmax peak and gate the burst using its FWHM.")
    parser.add_argument("--peak-indices",      nargs='*',  type=int, default=None)
    parser.add_argument("--freq-bands",        type=int,   default=None,
                        help="Automatically divide the spectrum into N equal contiguous frequency bands.")
    parser.add_argument("--freq-bands-snr", "--freq-snr",    type=int,   default=None,
                        help="Divide the spectrum into N contiguous bands with equal total S/N.")
    parser.add_argument("--manual-freq-bands", "--manual-freq", "--manual-freqs",  action="store_true")
    parser.add_argument("--freq-band-indices", "--freq-indices", nargs='*', type=int, default=None)
    parser.add_argument("--freq-band-mhz",    nargs='*', type=float, default=None,
                        help="Specify frequency bands by MHz pairs (e.g. 1300 1350 1350 1400). "
                             "Pairs are (low_mhz, high_mhz) for each band.")
    parser.add_argument("--offpulse-fraction", "--offpulse", type=float, default=0.1,
                        help="Fraction of the start of the observation to use as off-pulse baseline.")
    parser.add_argument("--threshold-sigma", "--threshold", type=float, default=1.0, 
                        help="Minimum per-channel SNR (model/off-pulse RMS) to include "
                         "a channel in the corrected spectrum (default: 1.0).")
    parser.add_argument("--fmin",   type=float, default=None,
                        help="Lower frequency bound [MHz] for channels used in the Lorentzian ACF fit.")
    parser.add_argument("--fmax",   type=float, default=None,
                        help="Upper frequency bound [MHz] for channels used in the Lorentzian ACF fit.")
    parser.add_argument("--pad",               type=int,   default=50)
    parser.add_argument("--fallback-window",   type=int,   default=200)
    parser.add_argument("--fit-max-lag",       type=float, default=8.0)
    parser.add_argument("--lag-zoom",          type=float, default=None,
                        help="Zoom factor for ACF lag axis in diagnostic plots.")
    parser.add_argument("--mod-plot", action="store_true",
                        help="Generate time-resolved modulation index plot.")
    parser.add_argument("--mod-sigma", type=float, default=3.0,
                        help="I significance threshold (sigma) for modulation index (default: 3.0).")
    parser.add_argument("--mod-nbins", type=int, default=None,
                        help="Number of time bins for binned modulation index (default: per-sample).")
    parser.add_argument("--dnu-mhz",           type=float, nargs='+', default=None,
                        help="Provide one or more Δν_d values in MHz (skips Lorentzian fitting). "
                             "E.g. --dnu-mhz 0.68 3.2 for two components.")
    parser.add_argument("--dnu-ref-freq-mhz",  type=float, default=None)
    parser.add_argument("--bline", action="store_true", 
                            help="Baseline correct dynamic spectrum.")
    parser.add_argument("--raw-acf", action="store_true",
                        help="Use raw ACF instead of corrected spectrum ACF for Lorentzian fitting.")
    parser.add_argument("--output",            default=None)
    parser.add_argument("--time-acf-model",    choices=["exp", "gauss"], default="exp")
    parser.add_argument("--fit-max-tau",       type=float, default=100.0)
    parser.add_argument("--tau-ms",            type=float, default=None)
    parser.add_argument("--redshift",          type=float, default=None)
    parser.add_argument("--ds-kpc",            type=float, default=None)
    parser.add_argument("--center-freq-mhz",   type=float, default=None)
    parser.add_argument("--mg",                type=float, default=None)
    parser.add_argument("--lg-kpc",            type=float, default=None)
    parser.add_argument("--estimate-lg-ne2025", "--lg-ne2025", action="store_true")
    parser.add_argument("--gl-deg",            type=float, default=None)
    parser.add_argument("--gb-deg",            type=float, default=None)
    parser.add_argument("--ra-hms",            type=str,   default=None)
    parser.add_argument("--dec-dms",           type=str,   default=None)
    parser.add_argument("--lg-max-dist-kpc",   type=float, default=50.0)
    parser.add_argument("--scatt-ref-freq-mhz", "--scatt-freq", type=float, default=None,
                        help="Frequency [MHz] at which to predict τ_scatt, Δν_d, and t_scint "
                             "from the NE2025 Galactic screen (requires --estimate-lg-ne2025). "
                             "Defaults to the observing centre frequency if not supplied.")
    parser.add_argument("--iss-velocity-km-s",  type=float, default=100.0,
                        help="Assumed ISS transverse velocity in km/s for scintillation "
                             "timescale prediction (default: 100 km/s).")
    parser.add_argument('--pub-col', type=float, default=2, help='Publication figure column count (1, 2, 3, ...). Default: 2')
    args = parser.parse_args()
    set_pub_col(args.pub_col)
    set_pub_style(use_latex=False)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    ds   = np.load(args.ds)
    freq = np.load(args.freq)
    time = np.load(args.time).astype(float)

    if ds.shape[0] != len(freq):
        ds = ds.T

    nfreq, ntime = ds.shape
    df            = np.abs(freq[1] - freq[0])
    nu_center_mhz = float(args.center_freq_mhz) if args.center_freq_mhz else float(np.median(freq))

    print(f"nchan={nfreq}, ntime={ntime}")
    print(f"Channel width       = {df:.4f} MHz")
    print(f"Time resolution     = {time[1] - time[0]:.6e} ms")

# ------------------------------------------------------------------
    # Off-pulse baseline: fixed leading fraction of the observation
    # ------------------------------------------------------------------
    n_offpulse = max(1, int(args.offpulse_fraction * ntime))
    off_pulse  = ds[:, :n_offpulse]                          # shape (nfreq, n_offpulse)
    if args.bline:
        bandpass   = np.nanmean(off_pulse, axis=1)               # per-channel mean, shape (nfreq,)
        ds         = ds - bandpass[:, None]                      # baseline-subtract the full array once

    print(f"Off-pulse baseline: first {n_offpulse} samples "
          f"({args.offpulse_fraction*100:.1f}% of {ntime}), "
          f"t = {time[0]:.4f}–{time[n_offpulse-1]:.4f} ms")

    # ------------------------------------------------------------------
    # Pulse gating (applied after baseline subtraction)
    # ------------------------------------------------------------------
    ts       = np.nanmean(ds, axis=0)
    ts_smooth = np.convolve(ts, np.ones(args.smooth) / args.smooth, mode="same") \
                if args.smooth > 1 else ts
    peak_idx  = int(np.argmax(ts_smooth))

    onpulse_mask = np.zeros(ntime, dtype=bool)
    if args.peak_indices is not None and len(args.peak_indices) > 0:
        peak_pairs = parse_peak_index_pairs(args.peak_indices, ntime)
        if args.auto_peak_fwhm:
            # Auto-measure the FWHM within each user-supplied crop
            for start_idx, end_idx in peak_pairs:
                crop_ts = ts_smooth[start_idx:end_idx]
                if not np.any(np.isfinite(crop_ts)):
                    continue
                local_peak = start_idx + int(np.nanargmax(crop_ts))
                (f_start, f_stop), fwhm_ms, _ = measure_fwhm_region(time, ts, local_peak)
                f_start = max(f_start, start_idx)
                f_stop  = min(f_stop, end_idx)
                if f_stop > f_start:
                    onpulse_mask[f_start:f_stop] = True
                if np.isfinite(fwhm_ms):
                    print(f"Auto FWHM gating in crop {start_idx}–{end_idx}: "
                          f"{f_start}–{f_stop} (FWHM={fwhm_ms:.4f} ms)")
                else:
                    print(f"Auto FWHM gating in crop {start_idx}–{end_idx}: {f_start}–{f_stop}")
            if not np.any(onpulse_mask):
                print("Peak-index + auto-FWHM gating produced no valid samples; "
                      "falling back to automatic window")
        else:
            for start_idx, end_idx in peak_pairs:
                onpulse_mask[start_idx:end_idx] = True
            if np.any(onpulse_mask):
                print(f"Peak-index gating: {list(zip(*[iter(args.peak_indices)]*2))}")
            else:
                print("Peak-index gating produced no valid samples; falling back to automatic window")
    elif args.manual_peak_fwhm:
        (start_idx, end_idx), fwhm_ms = select_peak_fwhm_manual(time, ts)
        onpulse_mask[start_idx:end_idx] = True
        if np.isfinite(fwhm_ms):
            print(f"Manual FWHM gating: {start_idx}–{end_idx} (FWHM={fwhm_ms:.4f} ms)")
        else:
            print(f"Manual FWHM gating: {start_idx}–{end_idx}")
    elif args.auto_peak_fwhm:
        (start_idx, end_idx), fwhm_ms, _ = measure_fwhm_region(time, ts, peak_idx)
        onpulse_mask[start_idx:end_idx] = True
        if np.isfinite(fwhm_ms):
            print(f"Auto FWHM gating: {start_idx}–{end_idx} (FWHM={fwhm_ms:.4f} ms)")
        else:
            print(f"Auto FWHM gating: {start_idx}–{end_idx}")
    elif args.manual_peaks:
        for start_idx, end_idx in select_peaks_manual(time, ts):
            onpulse_mask[start_idx:end_idx] = True

    if not np.any(onpulse_mask):
        tmin, tmax = find_burst_window(ts, peak_idx, smooth_win=args.smooth,
                                       threshold_sigma=args.threshold_sigma,
                                       pad=args.pad, fallback_window=args.fallback_window)
        onpulse_mask[max(0, tmin):min(ntime, tmax)] = True
        print(f"Auto gating: {tmin}–{tmax}")

    burst_ds  = ds[:, onpulse_mask]       # already baseline-subtracted
    off_pulse  = ds[:, :n_offpulse]       # re-slice for noise estimation downstream

    # ------------------------------------------------------------------
    # Pulse-profile scattering fit
    # ------------------------------------------------------------------
    pulse_profile = np.nanmean(burst_ds, axis=0)
    t_burst       = time[onpulse_mask]

    # Frequency mask for ACF fitting (shared by the full-band and time-resolved paths)
    freq_mask = np.ones(nfreq, dtype=bool)
    if args.fmin is not None:
        freq_mask &= freq >= args.fmin
    if args.fmax is not None:
        freq_mask &= freq <= args.fmax

    # Store modulation-index data for later plotting (after Lorentzian fit)
    mi = None
    if args.mod_plot:
        mi = compute_modulation_index(burst_ds, off_pulse, freq,
                                      i_sigma=args.mod_sigma, nbins=args.mod_nbins,
                                      min_snr=args.threshold_sigma,
                                      raw_acf=args.raw_acf, freq_mask=freq_mask)
        if args.mod_nbins is not None and burst_ds.shape[1] > 1:
            dt = float(np.abs(t_burst[1] - t_burst[0]))
            bin_dur = dt * burst_ds.shape[1] / args.mod_nbins
            print(f"  Modulation-index bin duration: {bin_dur:.2f} ms")

    t_scatt_fit_ms     = args.tau_ms
    t_scatt_fit_err_ms = None

    if t_scatt_fit_ms is None and pulse_profile.size >= 5:
        prof_max_idx = int(np.argmax(pulse_profile))
        mu0          = float(t_burst[prof_max_idx])
        p_low        = np.percentile(pulse_profile, 5)
        p_high       = np.percentile(pulse_profile, 95)
        amp0         = max(1e-6, float(p_high - p_low))
        offset0      = float(p_low)
        dt_ms        = float(np.abs(time[1] - time[0])) if time.size > 1 else 1e-3
        burst_dur    = float(t_burst[-1] - t_burst[0])
        sigma0 = tau0 = max(burst_dur / 20.0, dt_ms)
        try:
            popt_t, pcov_t = curve_fit(
                scattered_gaussian, t_burst, pulse_profile,
                p0=[amp0, mu0, sigma0, tau0, offset0],
                bounds=([0.0, float(t_burst[0]), dt_ms*0.5, dt_ms*0.5, -np.inf],
                        [np.inf, float(t_burst[-1]), burst_dur*0.5, burst_dur*0.5, np.inf]),
                maxfev=50000,
            )
            t_scatt_fit_ms = float(popt_t[3])
            try:
                t_scatt_fit_err_ms = float(np.sqrt(np.diag(pcov_t))[3])
            except Exception:
                pass
        except Exception as e:
            print(f"Pulse-profile τ_scatt fit failed: {e}")

    # ------------------------------------------------------------------
    # Spectrum + frequency masking
    # ------------------------------------------------------------------
    off_pulse_rms = np.nanstd(off_pulse, axis=1) if off_pulse.size > 0 else None
    raw_spectrum  = np.nanmean(burst_ds, axis=1)
    corrected_spectrum, mean_model, spec_index_used, spec_index_fit, spec_index_err = \
        correct_spectrum_powerlaw(freq, raw_spectrum, off_pulse_rms, min_snr=args.threshold_sigma)
    if args.raw_acf:
        print("Using raw spectrum for ACF fitting (no power-law correction).")
        corrected_spectrum = raw_spectrum
    else:
        print(f"Power-law spectral index used for correction: {spec_index_used:.3f} +/- {spec_index_err:.3f}" if np.isfinite(spec_index_used) else "Power-law spectral index used for correction: None")

    # Frequency mask for ACF fitting (defined before the modulation-index block)
    freq_acf     = freq[freq_mask]
    spectrum_acf = corrected_spectrum[freq_mask]
    df_acf       = np.abs(freq_acf[1] - freq_acf[0]) if freq_acf.size > 1 else df

    # Per-channel off-pulse RMS restricted to the same frequency mask,
    # used for noise-informed Δν_d uncertainty on the full-band fit
    off_pulse_rms_acf = off_pulse_rms[freq_mask] if off_pulse_rms is not None else None
    raw_spectrum_acf  = raw_spectrum[freq_mask]

    # Mean raw intensity over the ACF range, used to check m = sqrt(A)/<I> <= 1
    # in --raw-acf mode where the ACF amplitude A is in raw intensity² units.
    raw_mean_acf = None
    if args.raw_acf and np.any(np.isfinite(spectrum_acf)):
        raw_mean_acf = float(np.nanmean(spectrum_acf))

    print(f"ACF frequency range: {float(freq_acf[0]):.3f}–{float(freq_acf[-1]):.3f} MHz "
          f"({freq_acf.size} channels)")

    # ------------------------------------------------------------------
    # Frequency-band scintillation (optional)
    # ------------------------------------------------------------------
    band_scintillation_results: list[dict] = []
    band_powerlaw_fit: dict | None = None
    band_freq     = np.asarray(freq, dtype=float)
    band_spectrum = np.asarray(corrected_spectrum, dtype=float)
    band_off_pulse = off_pulse if off_pulse.size > 0 else None
    band_snr_weights = None
    freq_reversed = False
    N = band_freq.size
    if N > 1 and band_freq[0] > band_freq[-1]:
        band_freq     = band_freq[::-1]
        band_spectrum = band_spectrum[::-1]
        freq_reversed = True
    if band_off_pulse is not None and freq_reversed:
        band_off_pulse = band_off_pulse[::-1, :]

    if off_pulse_rms is not None:
        band_snr_weights = np.zeros_like(raw_spectrum, dtype=float)
        valid = np.isfinite(raw_spectrum) & np.isfinite(off_pulse_rms) & (off_pulse_rms > 0)
        band_snr_weights[valid] = np.maximum(raw_spectrum[valid] / off_pulse_rms[valid], 0.0)
        if freq_reversed:
            band_snr_weights = band_snr_weights[::-1]
    elif args.freq_bands_snr is not None:
        print("Equal-SNR banding requested but off-pulse RMS is unavailable; falling back to equal-width bands.")

    if args.freq_bands_snr is not None:
        if args.freq_bands_snr <= 0:
            print(f"Skipping frequency-band analysis: --freq-bands-snr must be > 0 (got {args.freq_bands_snr})")
        else:
            band_regions = split_frequency_bands_equal_snr(
                band_freq,
                band_snr_weights,
                args.freq_bands_snr,
            )
            print(f"Frequency-band gating: auto {len(band_regions)} equal-SNR bands")
            for i, (start, stop) in enumerate(band_regions, start=1):
                print(f"  Band {i}: {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
            band_scintillation_results = measure_scintillation_bands(
                band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
                off_pulse=band_off_pulse, raw_spectrum=raw_spectrum)
    elif args.freq_bands is not None:
        if args.freq_bands <= 0:
            print(f"Skipping frequency-band analysis: --freq-bands must be > 0 (got {args.freq_bands})")
        else:
            band_regions = split_frequency_bands_equal(band_freq, args.freq_bands)
            print(f"Frequency-band gating: auto {len(band_regions)} equal bands")
            for i, (start, stop) in enumerate(band_regions, start=1):
                print(f"  Band {i}: {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
            band_scintillation_results = measure_scintillation_bands(
                band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
                off_pulse=band_off_pulse, raw_spectrum=raw_spectrum)
    elif args.freq_band_indices is not None and len(args.freq_band_indices) > 0:
        band_regions = parse_peak_index_pairs(args.freq_band_indices, N)
        if freq_reversed:
            band_regions = [(N - stop, N - start) for start, stop in band_regions]
        print(f"Frequency-band gating: {list(zip(*[iter(args.freq_band_indices)]*2))}")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse, raw_spectrum=raw_spectrum)
    elif args.freq_band_mhz is not None and len(args.freq_band_mhz) > 0:
        band_regions = convert_mhz_to_frequency_indices(band_freq, args.freq_band_mhz, N)
        mhz_pairs = list(zip(*[iter(args.freq_band_mhz)]*2))
        print(f"Frequency-band gating: MHz {mhz_pairs}")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: indices [{start}, {stop}) = {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse, raw_spectrum=raw_spectrum)
    elif args.manual_freq_bands:
        band_regions = select_frequency_bands_manual(band_freq, band_spectrum, dspec=raw_spectrum if raw_spectrum.ndim == 2 else None)
        print(f"Frequency-band gating: manual {len(band_regions)} bands")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: indices [{start}, {stop}) = {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse, raw_spectrum=raw_spectrum)

    if band_scintillation_results:
        if len(band_scintillation_results) >= 2:
            try:
                band_powerlaw_fit = fit_scintillation_band_power_law(
                    np.array([r["center_mhz"] for r in band_scintillation_results], dtype=float),
                    np.array([r["dnu_mhz"]    for r in band_scintillation_results], dtype=float),
                    nu_center_mhz,
                    band_dnu_err_mhz=np.array([r["dnu_err_mhz"] for r in band_scintillation_results], dtype=float),
                )
            except Exception as e:
                print(f"Band power-law fit failed: {e}")
        else:
            print("Frequency-band selection produced one valid band; "
                  "power-law fit needs at least two bands.")
    elif args.freq_band_indices is not None or args.manual_freq_bands:
        print("Frequency-band selection produced no valid scintillation measurements.")

    spectrum = spectrum_acf.copy()
    n_finite_spec = int(np.count_nonzero(np.isfinite(spectrum)))
    if n_finite_spec < 4:
        print(f"Warning: only {n_finite_spec} finite channels in ACF range; "
              "fits will be poorly conditioned.")
    # spectrum is already a fractional residual (S - S̄)/S̄ from correct_spectrum_powerlaw.
    # autocorr() handles zero-meaning internally; no further normalisation needed.

    # ------------------------------------------------------------------
    #  ACF
    # ------------------------------------------------------------------
    acf  = autocorr(spectrum)
    lags = np.arange(len(acf)) * df_acf

    mask_plot        = (lags >= 0) & (lags <= args.fit_max_lag)
    lags_plot        = lags[mask_plot]
    acf_plot         = acf[mask_plot]
    lags_plot_sym    = np.concatenate((-lags_plot[::-1], lags_plot))
    acf_plot_sym     = np.concatenate(( acf_plot[::-1], acf_plot))

    mask_lorentz_fit = (lags > 0) & (lags < args.fit_max_lag) & np.isfinite(acf)
    lags_lorentz_fit = lags[mask_lorentz_fit]
    acf_lorentz_fit  = acf[mask_lorentz_fit]

    # ------------------------------------------------------------------
    # Lorentzian component fitting (1, 2, 3)
    # ------------------------------------------------------------------
    # user_dnu_components: list of (equal_weight, dnu_mhz) for each user-supplied
    # value, or None when we should fit from the ACF.
    user_dnu_components = None
    if args.dnu_mhz is not None:
        valid_dnus = [v for v in args.dnu_mhz if v > 0]
        if valid_dnus:
            w = 1.0 / len(valid_dnus)
            user_dnu_components = [(w, v) for v in valid_dnus]
        else:
            print("Warning: all --dnu-mhz values are <= 0; falling back to fitting.")

    # primary Δν_d (first/only component) used for display and scaling
    delta_nu_d = user_dnu_components[0][1] if user_dnu_components else None

    def _noise_informed_dnu_err(
        dnu_fit: float,
        band_width_mhz: float,
        df_mhz: float,
        off_pulse_rms_band: np.ndarray | None,
        raw_spectrum_band: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return (dnu_err_noise, n_eff, snr_per_scintle).  NaN on failure."""
        if off_pulse_rms_band is None or off_pulse_rms_band.size == 0:
            print(f"Warning: no valid off-pulse RMS values for noise-informed Δν_d error: ")
            return np.nan, np.nan, np.nan

        valid_rms = off_pulse_rms_band[np.isfinite(off_pulse_rms_band) & (off_pulse_rms_band > 0)]
        if valid_rms.size == 0:
            print(f"Warning: no valid off-pulse RMS values for noise-informed Δν_d error: ")
            return np.nan, np.nan, np.nan
        sigma_n = float(np.nanmedian(valid_rms))

        raw_finite = raw_spectrum_band[np.isfinite(raw_spectrum_band)]
        raw_mean   = float(np.nanmean(raw_finite)) if raw_finite.size > 0 else np.nan

        if not (np.isfinite(sigma_n) and sigma_n > 0
                and np.isfinite(raw_mean) and raw_mean > 0):
            print(f"Warning: invalid noise or signal level for noise-informed Δν_d error: ")
            return np.nan, np.nan, np.nan

        snr_chan        = raw_mean / sigma_n
        n_eff           = float(args.fit_max_lag / max(dnu_fit, df_mhz))
        n_eff_effective = n_eff / (1.0 + 1.0 / snr_chan ** 2) ** 2

        if n_eff_effective <= 0:
            print(f"Warning: invalid effective number of samples for noise-informed Δν_d error: ")
            return np.nan, n_eff, snr_chan / max(n_eff, 1.0) ** 0.5

        dnu_err_noise   = float(dnu_fit / np.sqrt(2.0 * n_eff_effective))
        snr_per_scintle = float(snr_chan / np.sqrt(max(n_eff, 1.0)))
        return dnu_err_noise, n_eff, snr_per_scintle

    if user_dnu_components is not None:
        # user-supplied: skip Lorentzian fitting entirely
        fit_models  = []
        best_n_comp = len(user_dnu_components)
        best_fit    = None
        d_base      = delta_nu_d
    else:
        if lags_lorentz_fit.size < 8:
            raise RuntimeError("Insufficient positive-lag ACF points for Lorentzian fitting")

        d_base    = max(1e-3, args.fit_max_lag / 4.0)
        amp_guess = float(acf[0]) if np.isfinite(acf[0]) else 0.1   # acf[0] = m²
        off_guess = float(np.nanmedian(acf_lorentz_fit[-max(3, int(0.2 * acf_lorentz_fit.size)):]))

        # Scale-aware fit bounds. In the normalised fractional-residual ACF, A = m² <= 1
        # and C ~ 0, so the fixed caps are fine. With --raw-acf the ACF is in raw
        # intensity² units and A = m²<I>² can be arbitrarily large, so the caps must
        # scale with the observed zero-lag value (amp_guess) instead of hard-coded ones.
        if args.raw_acf:
            amp_scale = max(amp_guess, 0.0)
            a_upper = max(2.5, 2.0 * amp_scale)
            c_upper = max(1.5, amp_scale)
        else:
            a_upper = 10.0
            c_upper = 1.5

        best_1c = fit_with_restarts(
            lorentzian, lags_lorentz_fit, acf_lorentz_fit,
            p0_list=[
                [d_base,       amp_guess, off_guess],
                [d_base * 0.5, amp_guess, off_guess],
                [d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([1e-6, 0.0, -c_upper], [np.inf, a_upper, c_upper]),
        )
        best_2c = fit_with_restarts(
            lorentzian_2c, lags_lorentz_fit, acf_lorentz_fit,
            p0_list=[
                [0.5, d_base * 0.3, d_base * 1.5, amp_guess, off_guess],
                [0.7, d_base * 0.2, d_base * 3.0, amp_guess, off_guess],
                [0.3, d_base * 0.6, d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([0.0, 1e-6, 1e-6, 0.0, -c_upper], [1.0, np.inf, np.inf, a_upper, c_upper]),
            maxfev=50000,
        )
        best_3c = fit_with_restarts(
            lorentzian_3c, lags_lorentz_fit, acf_lorentz_fit,
            p0_list=[
                [0.3, 0.5, d_base * 0.15, d_base * 0.7, d_base * 2.0, amp_guess, off_guess],
                [0.5, 0.5, d_base * 0.2,  d_base * 1.0, d_base * 3.0, amp_guess, off_guess],
                [0.7, 0.4, d_base * 0.1,  d_base * 0.8, d_base * 2.5, amp_guess, off_guess],
            ],
            bounds=(
                [0.0, 0.0, 1e-6, 1e-6, 1e-6, 0.0, -c_upper],
                [1.0, 1.0, np.inf, np.inf, np.inf, a_upper, c_upper],
            ),
            maxfev=80000,
        )

        def _make_result(best, k, n_comp):
            if best is None:
                return dict(error="all initialisations failed")
            popt, pcov, ymod = best
            # Effective number of independent ACF lags for model selection:
            # lags separated by ~Delta nu_d are quasi-independent, so
            # n_eff ~ fit_max_lag / Delta nu_d (primary component).
            n_eff = None
            try:
                comps, _, _ = _decode_lorentzian_components(n_comp, popt)
                d_prim = max(comps, key=lambda x: x[0])[1]
                n_eff = float(args.fit_max_lag / max(d_prim, df_acf))
            except Exception:
                pass
            return dict(popt=popt, pcov=pcov, ymod=ymod, n_eff=n_eff,
                        **build_fit_diagnostics(acf_lorentz_fit, ymod, k=k, n_eff=n_eff))

        r1 = _make_result(best_1c, k=3, n_comp=1)
        r2 = _make_result(best_2c, k=5, n_comp=2)
        r3 = _make_result(best_3c, k=7, n_comp=3)

        fit_models = [
            ("1-component", r1, lorentzian),
            ("2-component", r2, lorentzian_2c),
            ("3-component", r3, lorentzian_3c),
        ]


        def _pcov_dnu_err(n_comp, comp_idx, pcov):
            """Return sigma(Δν_d) for component comp_idx from the fit covariance matrix."""
            try:
                diag = np.diag(pcov)
                if n_comp == 1:
                    # popt = [d1, A, C]
                    return float(np.sqrt(diag[0]))
                elif n_comp == 2:
                    # popt = [w1, d1, dd12, A, C]
                    # d1_err = sqrt(var[d1])
                    # d2_err = sqrt(var[d1] + var[dd12] + 2*cov[d1,dd12])
                    if comp_idx == 0:
                        return float(np.sqrt(diag[1]))
                    else:
                        var = diag[1] + diag[2] + 2.0 * pcov[1, 2]
                        return float(np.sqrt(max(var, 0.0)))
                elif n_comp == 3:
                    # popt = [a, b, d1, dd12, dd23, A, C]
                    # d1_err = sqrt(var[d1])
                    # d2_err = sqrt(var[d1] + var[dd12] + 2*cov[d1,dd12])
                    # d3_err = sqrt(var[d1] + var[dd12] + var[dd23]
                    #               + 2*cov[d1,dd12] + 2*cov[d1,dd23] + 2*cov[dd12,dd23])
                    if comp_idx == 0:
                        return float(np.sqrt(diag[2]))
                    elif comp_idx == 1:
                        var = diag[2] + diag[3] + 2.0 * pcov[2, 3]
                        return float(np.sqrt(max(var, 0.0)))
                    else:
                        var = (diag[2] + diag[3] + diag[4]
                               + 2.0 * pcov[2, 3] + 2.0 * pcov[2, 4] + 2.0 * pcov[3, 4])
                        return float(np.sqrt(max(var, 0.0)))
            except Exception:
                return np.nan

        # ------------------------------------------------------------------
        # Physical validation before model selection
        # ------------------------------------------------------------------
        def _validate_fit(result: dict, n_comp: int) -> tuple[bool, str]:
            if "error" in result or "popt" not in result:
                return False, "fit failed"
            components, A, C = _decode_lorentzian_components(n_comp, result["popt"])
            if not np.isfinite(A) or A <= 0:
                return False, f"non-finite/non-positive amplitude A={A:.4f}"
            if args.raw_acf:
                # A is in raw intensity² units: m = sqrt(A)/<I> must be <= 1 for DISS
                m_check = (np.sqrt(max(A, 0.0)) / raw_mean_acf
                           if raw_mean_acf is not None and raw_mean_acf > 0
                           else np.sqrt(max(A, 0.0)))
                if m_check > 1.0:
                    return False, f"m=sqrt(A)/<I>={m_check:.4f} > 1 unphysical for DISS"
            elif A > 1.0:
                return False, f"m=sqrt(A)={np.sqrt(A):.4f} > 1 unphysical for DISS"
            if abs(C) > 0.9 * c_upper:
                return False, f"offset C={C:.4f} hitting bound — degenerate fit"
            for i, (w, d) in enumerate(components):
                if d < df_acf:
                    return False, f"component {i+1} Δν_d={d:.4f} MHz < channel width {df_acf:.4f} MHz"
                if d > args.fit_max_lag:
                    return False, f"component {i+1} Δν_d={d:.4f} MHz > fit_max_lag {args.fit_max_lag:.4f} MHz"
                if w < 0.01:
                    return False, f"component {i+1} weight={w:.4f} negligible"
                try:
                    d_err = _pcov_dnu_err(n_comp, i, result["pcov"])
                    if not np.isfinite(d_err):
                        return False, f"component {i+1} Δν_d uncertainty non-finite"
                    if d_err > d:
                        return False, f"component {i+1} fractional uncertainty {d_err/d:.2f} > 1"
                except Exception:
                    pass
            return True, ""

        validations = {}
        for name, result, _ in fit_models:
            ok, reason = _validate_fit(result, int(name[0]))
            validations[name] = (ok, reason)

        valid_physical = [
            (name, r) for name, r, _ in fit_models
            if "aic" in r and np.isfinite(r["aic"]) and validations[name][0]
        ]

        # Fall back to all models if everything fails validation
        if not valid_physical:
            print("  Warning: all fits failed physical validation; ignoring physical cuts")
            valid_physical = [(name, r) for name, r, _ in fit_models
                              if "aic" in r and np.isfinite(r["aic"])]

        if not valid_physical:
            # All Lorentzian fits failed outright (e.g. degenerate/garbage spectrum):
            # degrade gracefully instead of crashing on min([]).
            print("\n  Error: all Lorentzian ACF fits failed; no ACF model selected.")
            best_n_comp = 1
            best_fit    = None
            delta_nu_d  = None
            dnu_err     = np.nan
            component_noise_errs = []
        else:
            # Forward model selection via nested F-test on residual RSS against the
            # correlated-noise floor.  n_eff (number of quasi-independent lags) gives
            # the residual degrees of freedom, so a more complex model is accepted
            # only if its RSS improvement is significant (p < F_TEST_P_THRESH).
            by_name        = {name: r for name, r in valid_physical}
            best_name      = "1-component"
            best_result    = by_name.get("1-component")
            selection_note = ""
            if best_result is None:
                best_name, best_result = min(valid_physical, key=lambda x: x[1]["aic"])
                selection_note = "no valid 1-component fit; best by AIC"
            else:
                sel_notes = []
                for nxt_name in ("2-component", "3-component"):
                    nxt_result = by_name.get(nxt_name)
                    if nxt_result is None:
                        break
                    k_cur = int(best_name[0]) * 2 + 1
                    k_nxt = int(nxt_name[0]) * 2 + 1
                    n_eff = nxt_result.get("n_eff")
                    if n_eff is None or not np.isfinite(n_eff) or n_eff < (k_nxt + 1):
                        break
                    dof2 = float(n_eff) - k_nxt
                    if dof2 <= 1 or nxt_result["rss"] <= 0:
                        break
                    f_stat = ((best_result["rss"] - nxt_result["rss"]) / (k_nxt - k_cur)
                              / (nxt_result["rss"] / dof2))
                    p_val  = float(1.0 - f_dist.cdf(f_stat, k_nxt - k_cur, dof2))
                    if p_val < F_TEST_P_THRESH:
                        sel_notes.append(f"{nxt_name} accepted (F-test p={p_val:.3f} < {F_TEST_P_THRESH})")
                        best_name, best_result = nxt_name, nxt_result
                    else:
                        sel_notes.append(f"{nxt_name} rejected (F-test p={p_val:.3f} >= {F_TEST_P_THRESH})")
                        break
                selection_note = "; ".join(sel_notes)

            best_n_comp = int(best_name[0])
            best_fit    = best_result

            # Extract Δν_d from best model (primary component = highest-weight one)
            if "popt" in best_result:
                components, _, _ = _decode_lorentzian_components(best_n_comp, best_result["popt"])
                primary_comp = max(components, key=lambda x: x[0])
                delta_nu_d   = primary_comp[1]
                dnu_err      = np.nan

                acf_band_width = float(args.fit_max_lag)
                component_noise_errs = []
                for i, (w_c, d_c) in enumerate(components):
                    dnu_err_noise, n_eff_c, snr_c = _noise_informed_dnu_err(
                        d_c, acf_band_width, df_acf,
                        off_pulse_rms_acf, raw_spectrum_acf,
                    )
                    dnu_err_cov = _pcov_dnu_err(best_n_comp, i, best_result["pcov"])
                    dnu_err = dnu_err_cov
                    #if np.isfinite(dnu_err_noise) and dnu_err_noise > 0 and n_eff_c >= 1.0:
                    #    dnu_err = dnu_err_noise
                    #else:
                    #    dnu_err = dnu_err_cov
                    component_noise_errs.append(
                        dict(dnu_err=dnu_err, dnu_err_noise=dnu_err_noise,
                             dnu_err_cov=dnu_err_cov, n_eff=n_eff_c, snr_per_scintle=snr_c)
                    )
            else:
                delta_nu_d           = None
                dnu_err              = np.nan
                component_noise_errs = []

    # Generate the modulation-index plot
    if mi is not None:
        t_mod = t_burst[mi["t_centers"].astype(int)] if mi["t_centers"] is not None else t_burst
        plot_modulation_index(
            t_mod, t_burst, mi["mod_index"], mi["mod_err"], pulse_profile,
            mi["weighted_mean"], mi["weighted_mean_err"],
            dnu_d=mi["dnu_d"], dnu_d_err=mi["dnu_d_err"],
            i_cut=mi["i_cut"], output=args.output, ncol=args.pub_col,
        )

    # ------------------------------------------------------------------
    # ===== SECTION 1: SCINTILLATION & MODULATION INDEX =====
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCINTILLATION & MODULATION INDEX")
    print("=" * 60)

    if t_scatt_fit_ms is not None:
        if t_scatt_fit_err_ms is not None and np.isfinite(t_scatt_fit_err_ms):
            src = "--tau-ms" if args.tau_ms is not None else "scattered Gaussian fit"
            print(f"  τ_scatt = {t_scatt_fit_ms:.4f} ± {t_scatt_fit_err_ms:.4f} ms  ({src})")
        else:
            src = "--tau-ms" if args.tau_ms is not None else "scattered Gaussian fit"
            print(f"  τ_scatt = {t_scatt_fit_ms:.4f} ms  ({src})")

    if user_dnu_components is not None:
        dnu_strs = ", ".join(f"{d:.4f}" for _, d in user_dnu_components)
        print(f"\n  Δν_d (provided) = [{dnu_strs}] MHz  — Lorentzian fitting skipped")

    # Lorentzian model comparison table
    if fit_models:
        print(f"\n  Lorentzian ACF component fits (lag range 0–{args.fit_max_lag} MHz):")
        print(f"  {'Model':<14} {'AIC':>10} {'BIC':>10} {'AICc':>10} {'RSS':>12} {'RMSE':>10} {'R²':>8}")
        for name, result, _ in fit_models:
            if "error" in result:
                print(f"  {name:<14} failed: {result['error']}")
            else:
                print(f"  {name:<14} {result['aic']:>10.3f} {result['bic']:>10.3f} "
                      f"{result['aicc']:>10.3f} {result['rss']:>12.4e} "
                      f"{result['rmse']:>10.4e} {result['r2']:>8.4f}")

        # Print validation results for all models
        for name, result, _ in fit_models:
            ok, reason = validations[name]
            if not ok:
                print(f"  {name} rejected: {reason}")

        # Print AIC/BIC table over physically valid models only
        if valid_physical:
            best_aic_n, best_aic_r = min(valid_physical, key=lambda x: x[1]["aic"])
            best_bic_n, best_bic_r = min(valid_physical, key=lambda x: x[1]["bic"])
            sorted_aic  = sorted(valid_physical, key=lambda x: x[1]["aic"])
            daic_runner = (sorted_aic[1][1]["aic"] - sorted_aic[0][1]["aic"]
                           if len(sorted_aic) > 1 else None)

            print(f"\n  Best by AIC : {best_aic_n}  (AIC={best_aic_r['aic']:.3f}"
                  + (f", ΔAIC to runner-up={daic_runner:.2f}" if daic_runner else "") + ")")
            print(f"  Best by BIC : {best_bic_n}  (BIC={best_bic_r['bic']:.3f})")
            if selection_note:
                print(f"  Selected    : {best_name}  ({selection_note})")
            else:
                print(f"  Selected    : {best_name}")

        # Component parameters for best model
        m = 0.0
        if best_fit and "popt" in best_fit:
            components, A_fit, C_fit = _decode_lorentzian_components(best_n_comp, best_fit["popt"])
            m = np.sqrt(max(A_fit, 0.0))
            # When using the power-law corrected spectrum, `spectrum` is a fractional residual
            # (S - \bar{S})/\bar{S} with mean ~0, so the modulation index is simply std(spectrum).
            # When `--raw-acf` is used, `spectrum` is in raw intensity units and m = std/mean.
            if args.raw_acf:
                _spec_mean = float(np.nanmean(spectrum))
                m_spec = float(np.nanstd(spectrum) / _spec_mean) if _spec_mean != 0.0 else np.nan
            else:
                m_spec = float(np.nanstd(spectrum))
            print(f"\n  Best model ({best_n_comp}-component) parameters:")
            print(f"    m²  (ACF amplitude A) = {A_fit:.6f}")
            m_err = np.nan
            if best_fit.get("pcov") is not None:
                try:
                    var_A = best_fit["pcov"][-2, -2]
                    m_err = float(np.sqrt(max(var_A, 0.0))) / (2.0 * m) if m > 0 else np.nan
                except Exception:
                    pass
            if np.isfinite(m_err):
                print(f"    m   (modulation index) = {m:.6f} ± {m_err:.6f}")
            else:
                print(f"    m   (modulation index) = {m:.6f}")
            print(f"    m   (from spectrum)     = {m_spec:.6f}")
            print(f"    C   (offset)           = {C_fit:.6f}")
            for i, (w, d) in enumerate(components):
                errs    = component_noise_errs[i] if i < len(component_noise_errs) else {}
                dnu_err = errs.get("dnu_err", np.nan)
                err_str = f" ± {dnu_err:.4f}" if np.isfinite(dnu_err) else ""
                n_eff_str = f", N_eff={errs.get('n_eff', np.nan):.1f}" \
                            if np.isfinite(errs.get("n_eff", np.nan)) else ""
                snr_str   = f", SNR/scintle={errs.get('snr_per_scintle', np.nan):.2f}" \
                            if np.isfinite(errs.get("snr_per_scintle", np.nan)) else ""
                print(f"    Component {i+1}: Δν_d = {d:.4f}{err_str} MHz,  "
                      f"weight = {w:.4f}{n_eff_str}{snr_str}")

    if band_scintillation_results:
        print(f"\n  Frequency-band Lorentzian fits (Δν_d vs ν power law):")
        print(f"    reference ν_c        = {nu_center_mhz:.3f} MHz")
        if band_powerlaw_fit is not None:
            alpha_str = f"{band_powerlaw_fit['alpha_fit']:.4f}"
            if np.isfinite(band_powerlaw_fit.get('alpha_err', np.nan)):
                alpha_str += f" ± {band_powerlaw_fit['alpha_err']:.4f}"
            print(f"    fitted index α       = {alpha_str}")
            print(f"    Δν_d(ν_c) fit        = {band_powerlaw_fit['dnu_ref_fit']:.6f} MHz")
            print(f"    Δν_d(ν_c) Kolmogorov = {band_powerlaw_fit['dnu_ref_44']:.6f} MHz  (α=4.4)")
            print(f"    log-space RMS resid  = {band_powerlaw_fit['residual_rms']:.4e}")
        print(f"    {'Band':<6} {'ν_c[MHz]':>10} {'Δν_d[MHz]':>12} {'err':>10} "
              f"{'Δν_d@ν_c(fit)':>15} {'Δν_d@ν_c(4.4)':>15} {'R²':>8}")
        for row in band_scintillation_results:
            sc_fit = np.nan
            sc_44  = np.nan
            if band_powerlaw_fit is not None and row["center_mhz"] > 0:
                sc_fit = row["dnu_mhz"] * (nu_center_mhz / row["center_mhz"]) ** band_powerlaw_fit["alpha_fit"]
            if row["center_mhz"] > 0:
                sc_44 = row["dnu_mhz"] * (nu_center_mhz / row["center_mhz"]) ** 4.4
            err_str = f"{row['dnu_err_mhz']:.6f}" if np.isfinite(row.get("dnu_err_mhz", np.nan)) else "N/A"
            print(f"    {row['band_idx']:<6d} {row['center_mhz']:>10.3f} "
                  f"{row['dnu_mhz']:>12.6f} {err_str:>10} "
                  f"{sc_fit:>15.6f} {sc_44:>15.6f} {row.get('r2', np.nan):>8.4f}")

    # Δν_d scaling
    delta_nu_d_scaled     = None
    delta_nu_d_scaled_err = None
    if args.dnu_ref_freq_mhz is not None and args.dnu_ref_freq_mhz > 0 and delta_nu_d is not None:
        try:
            scale_factor      = (float(args.dnu_ref_freq_mhz) / nu_center_mhz) ** 4.0
            delta_nu_d_scaled = scale_scintillation_bandwidth(
                delta_nu_d, nu_center_mhz, float(args.dnu_ref_freq_mhz))
            if best_fit and component_noise_errs:
                primary_idx = max(range(len(components)), key=lambda i: components[i][0])
                d_err = component_noise_errs[primary_idx].get("dnu_err", np.nan)
                if np.isfinite(d_err):
                    delta_nu_d_scaled_err = d_err * abs(scale_factor)
            print(f"\n  Δν_d scaling (ν^4): {nu_center_mhz:.3f} → {args.dnu_ref_freq_mhz:.3f} MHz")
            err_str = f" ± {delta_nu_d_scaled_err:.6f}" if delta_nu_d_scaled_err else ""
            print(f"    Scaled Δν_d = {delta_nu_d_scaled:.6f}{err_str} MHz")
        except Exception as e:
            print(f"  Δν_d scaling failed: {e}")

    # ------------------------------------------------------------------
    # ===== SECTION 2: SCREEN DISTANCE ESTIMATES =====
    # ------------------------------------------------------------------

    # Choose which Δν_d / ν to use for two-screen calculation
    delta_nu_d_for_calc  = delta_nu_d
    nu_for_two_screen    = nu_center_mhz
    source_label         = "measured (primary component)"
    if args.dnu_ref_freq_mhz is not None and delta_nu_d_scaled is not None:
        delta_nu_d_for_calc = delta_nu_d_scaled
        nu_for_two_screen   = float(args.dnu_ref_freq_mhz)
        source_label        = "scaled-to-ref"

    t_scatt_for_calc = args.tau_ms if args.tau_ms is not None else t_scatt_fit_ms

    # Resolve Ds
    ds_kpc_for_calc = args.ds_kpc
    if ds_kpc_for_calc is None and args.redshift is not None:
        try:
            ds_kpc_for_calc = estimate_ds_kpc_from_redshift(float(args.redshift))
        except Exception as e:
            print(f"\nCould not estimate D_s from redshift: {e}")

    # Resolve Lg (NE2025 or user-supplied)
    lg_kpc_for_calc = args.lg_kpc
    lg_peak_kpc     = args.lg_kpc
    lg_source       = "--lg-kpc"
    lg_eff_kpc      = None
    if lg_kpc_for_calc is None and args.estimate_lg_ne2025:
        gl_for_lg = args.gl_deg
        gb_for_lg = args.gb_deg
        if (gl_for_lg is None or gb_for_lg is None) and args.ra_hms and args.dec_dms:
            try:
                gl_for_lg, gb_for_lg = radec_to_galactic_deg(args.ra_hms, args.dec_dms)
            except Exception as e:
                print(f"\nRA/Dec → Galactic conversion failed: {e}")
        if gl_for_lg is not None and gb_for_lg is not None:
            try:
                lg_peak_kpc, cn2_peak, lg_eff_kpc = estimate_lg_kpc_from_ne2025(
                    gl_for_lg, gb_for_lg, ds_kpc_for_calc,
                    max_dist_kpc=args.lg_max_dist_kpc,
                    output=args.output
                )
                lg_kpc_for_calc = lg_peak_kpc
                if lg_eff_kpc is not None and lg_eff_kpc > 0:
                    lg_kpc_for_calc = lg_eff_kpc
                lg_source = "NE2025"
            except Exception as e:
                print(f"\nNE2025 L_g estimate failed: {e}")
        else:
            print("\nNE2025 L_g estimate skipped: provide --gl-deg/--gb-deg or --ra-hms/--dec-dms")

    # Modulation index for two-screen
    mg_for_calc = args.mg if args.mg is not None else m

    can_do_two_screen = (
        delta_nu_d_for_calc is not None
        and t_scatt_for_calc is not None
        and args.redshift is not None
        and ds_kpc_for_calc is not None
        and t_scatt_for_calc > 0
        and ds_kpc_for_calc > 0
        and args.redshift >= 0
    )

    if can_do_two_screen:
        # Build one result per Lorentzian component (multi-screen case)
        if fit_models and best_fit and "popt" in best_fit:
            components, _, _ = _decode_lorentzian_components(best_n_comp, best_fit["popt"])
            active_components = [(w, d) for (w, d) in components if w > 0.05]
        else:
            active_components = [(1.0, delta_nu_d_for_calc)]

        # Scale each component's Δν_d if ref-freq was requested
        ts_results = []
        for i, (w, d) in enumerate(active_components):
            d_calc = d
            if args.dnu_ref_freq_mhz is not None and args.dnu_ref_freq_mhz > 0:
                try:
                    d_calc = scale_scintillation_bandwidth(d, nu_center_mhz, float(args.dnu_ref_freq_mhz))
                except Exception:
                    pass
            label = f"Component {i+1} (w={w:.3f}, Δν_d={d:.4f} MHz)"
            ts_results.append(two_screen_estimate(
                d_calc, t_scatt_for_calc, nu_for_two_screen,
                float(args.redshift), ds_kpc_for_calc,
                mg_for_calc, lg_kpc_for_calc, label=label,
            ))

        # NE2025 scattering / scintillation prediction from Galactic screen
        if lg_source == "NE2025" and ds_kpc_for_calc is not None:
            scatt_ref_mhz = float(args.scatt_ref_freq_mhz) if args.scatt_ref_freq_mhz else nu_for_two_screen
            try:
                ne_pred = ne2025_scattering_prediction(
                    gl_for_lg, gb_for_lg,
                    ds_kpc=ds_kpc_for_calc,
                    nu_ref_mhz=scatt_ref_mhz,
                    v_iss_km_s=args.iss_velocity_km_s,
                    lg_eff_kpc=lg_eff_kpc,
                    lg_max_dist_kpc=args.lg_max_dist_kpc,
                )
                lg_peak_for_print = lg_peak_kpc if lg_source == "NE2025" else lg_kpc_for_calc
                print_ne2025_scattering_prediction(ne_pred, lg_peak_for_print, ds_kpc_for_calc)
            except Exception as e:
                print(f"\n  NE2025 scattering prediction failed: {e}")

        print_two_screen_results(
            ts_results, t_scatt_for_calc, nu_for_two_screen,
            float(args.redshift), mg_for_calc, lg_kpc_for_calc, lg_peak_kpc, lg_eff_kpc,
            source_label, lg_source,
        )

        # Supplemental: NE2025 summary (if estimated)
        if lg_kpc_for_calc is not None and lg_source == "NE2025":
            print(f"\n  NE2025 L_g details:")
            print(f"    l={gl_for_lg:.4f} deg, b={gb_for_lg:.4f} deg")
            line = f"    L_g (peak) = {lg_peak_kpc:.4f} kpc,  Cn²_peak = {cn2_peak:.4e} m^{{-20/3}}"
            if lg_eff_kpc is not None:
                line += f",  L_g (weighted) = {lg_eff_kpc:.4f} kpc"
            print(line)
        if ds_kpc_for_calc is not None and args.redshift is not None:
            print(f"  D_s = {ds_kpc_for_calc:.4e} kpc  (z={args.redshift:.6f})")
    else:
        missing = []
        if delta_nu_d_for_calc is None:  missing.append("Δν_d")
        if t_scatt_for_calc    is None:  missing.append("τ_scatt")
        if args.redshift       is None:  missing.append("--redshift")
        if ds_kpc_for_calc     is None:  missing.append("D_s")
        if missing:
            print(f"\nTwo-screen estimate skipped (missing: {', '.join(missing)})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_spectrum_powerlaw_fit(
        freq,
        raw_spectrum,
        mean_model,
        spectral_index=spec_index_used,
        output=args.output,
    )

    lag_zoom = args.lag_zoom if args.lag_zoom is not None else args.fit_max_lag

    plot_acf_fit(
        lags_plot_sym, acf_plot_sym,
        best_fit, best_n_comp, component_noise_errs,
        lag_zoom, delta_nu_d, dnu_err,
        output=args.output,
    )


    # Lorentzian component diagnostics
    if fit_models:
        plot_lorentzian_diagnostics(
            lags_plot_sym, acf_plot_sym,
            lags_lorentz_fit, acf_lorentz_fit,
            fit_models, lag_zoom, output=args.output,
        )

    # Frequency-band scintillation power-law plot
    if band_scintillation_results:
        plot_scintillation_band_power_law(
            band_scintillation_results,
            band_powerlaw_fit,
            output=args.output,
            fit_max_lag_mhz=args.fit_max_lag,
        )


if __name__ == "__main__":
    main()
