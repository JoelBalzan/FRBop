import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from frbop.scop.acf import autocorr
from frbop.scop.band_analysis import (
    convert_mhz_to_frequency_indices,
    fit_scintillation_band_power_law,
    measure_scintillation_bands,
    select_frequency_bands_manual,
    split_frequency_bands_equal,
)
from frbop.scop.fit_utils import build_fit_diagnostics, fit_with_restarts, _decode_lorentzian_components
from frbop.scop.gating import find_burst_window, select_peaks_manual
from frbop.scop.macquart import estimate_macquart_modulation_index, macquart_dnu_from_window
from frbop.scop.models import lorentzian, lorentzian_2c, lorentzian_3c, scattered_gaussian
from frbop.scop.ne2025 import (
    estimate_lg_kpc_from_ne2025,
    get_cn2_profile,
    ne2025_scattering_prediction,
    print_ne2025_scattering_prediction,
)
from frbop.scop.physics import estimate_ds_kpc_from_redshift, radec_to_galactic_deg, scale_scintillation_bandwidth
from frbop.scop.plotting import (
    plot_lorentzian_diagnostics,
    plot_macquart_diagnostics,
    plot_scintillation_band_power_law,
)
from frbop.scop.two_screen import print_two_screen_results, two_screen_estimate
from frbop.utils.peaks import parse_peak_index_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit FRB scintillation from dynamic spectrum files")
    parser.add_argument("ds",   nargs="?", default="FRB_250607_htr_dsI.npy")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy")
    parser.add_argument("--smooth",            type=int,   default=5)
    parser.add_argument("--manual-peaks",      action="store_true")
    parser.add_argument("--peak-indices",      nargs='*',  type=int, default=None)
    parser.add_argument("--freq-bands",        type=int,   default=None,
                        help="Automatically divide the spectrum into N equal contiguous frequency bands.")
    parser.add_argument("--manual-freq-bands", "--manual-freq",  action="store_true")
    parser.add_argument("--freq-band-indices", "--freq-indices",  nargs='*', type=int, default=None)
    parser.add_argument("--freq-band-mhz",    nargs='*', type=float, default=None,
                        help="Specify frequency bands by MHz pairs (e.g. 1300 1350 1350 1400). "
                             "Pairs are (low_mhz, high_mhz) for each band.")
    parser.add_argument("--threshold-sigma",   type=float, default=3.0)
    parser.add_argument("--pad",               type=int,   default=50)
    parser.add_argument("--fallback-window",   type=int,   default=200)
    parser.add_argument("--fit-max-lag",       type=float, default=8.0)
    parser.add_argument("--dnu-mhz",           type=float, nargs='+', default=None,
                        help="Provide one or more Δν_d values in MHz (skips Lorentzian fitting). "
                             "E.g. --dnu-mhz 0.68 3.2 for two components.")
    parser.add_argument("--dnu-ref-freq-mhz",  type=float, default=None)
    parser.add_argument("--output",            default=None)
    parser.add_argument("--time-acf-model",    choices=["exp", "gauss"], default="exp")
    parser.add_argument("--fit-max-tau",       type=float, default=100.0)
    parser.add_argument("--tau-ms",            type=float, default=None)
    parser.add_argument("--redshift",          type=float, default=None)
    parser.add_argument("--ds-kpc",            type=float, default=None)
    parser.add_argument("--center-freq-mhz",   type=float, default=None)
    parser.add_argument("--mg",                type=float, default=None)
    parser.add_argument("--lg-kpc",            type=float, default=None)
    parser.add_argument("--estimate-lg-ne2025",action="store_true")
    parser.add_argument("--gl-deg",            type=float, default=None)
    parser.add_argument("--gb-deg",            type=float, default=None)
    parser.add_argument("--ra-hms",            type=str,   default=None)
    parser.add_argument("--dec-dms",           type=str,   default=None)
    parser.add_argument("--lg-max-dist-kpc",   type=float, default=50.0)
    parser.add_argument("--scatt-ref-freq-mhz", type=float, default=None,
                        help="Frequency (MHz) at which to predict τ_scatt, Δν_d, and t_scint "
                             "from the NE2025 Galactic screen (requires --estimate-lg-ne2025). "
                             "Defaults to the observing centre frequency if not supplied.")
    parser.add_argument("--iss-velocity-km-s",  type=float, default=100.0,
                        help="Assumed ISS transverse velocity in km/s for scintillation "
                             "timescale prediction (default: 100 km/s).")
    args = parser.parse_args()

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
    # Pulse gating
    # ------------------------------------------------------------------
    ts = np.nanmean(ds, axis=0)
    ts_smooth = np.convolve(ts, np.ones(args.smooth) / args.smooth, mode="same") if args.smooth > 1 else ts
    peak_idx  = int(np.argmax(ts_smooth))

    onpulse_mask = np.zeros(ntime, dtype=bool)
    if args.peak_indices is not None and len(args.peak_indices) > 0:
        for start_idx, end_idx in parse_peak_index_pairs(args.peak_indices, ntime):
            onpulse_mask[start_idx:end_idx] = True
        if np.any(onpulse_mask):
            print(f"Peak-index gating: {list(zip(*[iter(args.peak_indices)]*2))}")
        else:
            print("Peak-index gating produced no valid samples; falling back to automatic window")
    elif args.manual_peaks:
        for start_idx, end_idx in select_peaks_manual(time, ts):
            onpulse_mask[start_idx:end_idx] = True

    if not np.any(onpulse_mask):
        tmin, tmax = find_burst_window(ts, peak_idx, smooth_win=args.smooth,
                                       threshold_sigma=args.threshold_sigma,
                                       pad=args.pad, fallback_window=args.fallback_window)
        onpulse_mask[max(0, tmin):min(ntime, tmax)] = True
        print(f"Auto gating: {tmin}–{tmax}")

    burst_ds    = ds[:, onpulse_mask]
    off_pulse   = ds[:, ~onpulse_mask] if np.any(~onpulse_mask) else np.empty((nfreq, 0))
    bandpass    = np.nanmean(off_pulse, axis=1) if off_pulse.size > 0 else np.percentile(ds, 10, axis=1)
    burst_ds   -= bandpass[:, None]

    # ------------------------------------------------------------------
    # Pulse-profile scattering fit
    # ------------------------------------------------------------------
    pulse_profile = np.nanmean(burst_ds, axis=0)
    t_burst       = time[onpulse_mask]
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
    # Spectrum
    # ------------------------------------------------------------------
    raw_spectrum = np.nanmean(burst_ds, axis=1)

    # ------------------------------------------------------------------
    # Frequency-band scintillation (optional)
    # ------------------------------------------------------------------
    band_scintillation_results: list[dict] = []
    band_powerlaw_fit: dict | None = None
    band_freq     = np.asarray(freq, dtype=float)
    band_spectrum = np.asarray(raw_spectrum, dtype=float)
    band_off_pulse = off_pulse if off_pulse.size > 0 else None
    freq_reversed = False
    N = band_freq.size
    if N > 1 and band_freq[0] > band_freq[-1]:
        band_freq     = band_freq[::-1]
        band_spectrum = band_spectrum[::-1]
        freq_reversed = True
    if band_off_pulse is not None and freq_reversed:
        band_off_pulse = band_off_pulse[::-1, :]

    if args.freq_bands is not None:
        if args.freq_bands <= 0:
            print(f"Skipping frequency-band analysis: --freq-bands must be > 0 (got {args.freq_bands})")
        else:
            band_regions = split_frequency_bands_equal(band_freq, args.freq_bands)
            print(f"Frequency-band gating: auto {len(band_regions)} equal bands")
            for i, (start, stop) in enumerate(band_regions, start=1):
                print(f"  Band {i}: {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
            band_scintillation_results = measure_scintillation_bands(
                band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
                off_pulse=band_off_pulse)
    elif args.freq_band_indices is not None and len(args.freq_band_indices) > 0:
        band_regions = parse_peak_index_pairs(args.freq_band_indices, N)
        if freq_reversed:
            band_regions = [(N - stop, N - start) for start, stop in band_regions]
        print(f"Frequency-band gating: {list(zip(*[iter(args.freq_band_indices)]*2))}")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse)
    elif args.freq_band_mhz is not None and len(args.freq_band_mhz) > 0:
        band_regions = convert_mhz_to_frequency_indices(band_freq, args.freq_band_mhz, N)
        mhz_pairs = list(zip(*[iter(args.freq_band_mhz)]*2))
        print(f"Frequency-band gating: MHz {mhz_pairs}")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: indices [{start}, {stop}) = {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse)
    elif args.manual_freq_bands:
        band_regions = select_frequency_bands_manual(band_freq, band_spectrum)
        print(f"Frequency-band gating: manual {len(band_regions)} bands")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: indices [{start}, {stop}) = {band_freq[start]:.3f}–{band_freq[stop - 1]:.3f} MHz")
        band_scintillation_results = measure_scintillation_bands(
            band_freq, band_spectrum, band_regions, fit_max_lag_mhz=args.fit_max_lag,
            off_pulse=band_off_pulse)

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

    spectrum   = raw_spectrum.copy()
    spec_med   = np.median(spectrum)
    spec_mad   = np.median(np.abs(spectrum - spec_med))
    spec_sigma = 1.4826 * spec_mad if spec_mad > 0 else np.std(spectrum)
    spectrum   = (spectrum - spec_med) / spec_sigma if spec_sigma > 0 else spectrum - spec_med

    # ------------------------------------------------------------------
    # Normalised ACF (used for Lorentzian fitting)
    # ------------------------------------------------------------------
    acf  = autocorr(spectrum)
    if acf[0] != 0:
        acf /= acf[0]
    lags = np.arange(len(acf)) * df

    mask_plot        = (lags > 0) & (lags <= args.fit_max_lag)
    lags_plot        = lags[mask_plot]
    acf_plot         = acf[mask_plot]
    lags_plot_sym    = np.concatenate((-lags_plot[::-1], lags_plot))
    acf_plot_sym     = np.concatenate((acf_plot[::-1],  acf_plot))

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
        amp_guess = max(0.05, float(np.nanmax(acf_lorentz_fit) - np.nanmin(acf_lorentz_fit)))
        off_guess = float(np.nanmedian(acf_lorentz_fit[-max(3, int(0.2 * acf_lorentz_fit.size)):]))

        best_1c = fit_with_restarts(
            lorentzian, lags_lorentz_fit, acf_lorentz_fit,
            p0_list=[
                [d_base,       amp_guess, off_guess],
                [d_base * 0.5, amp_guess, off_guess],
                [d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([1e-6, 0.0, -1.5], [np.inf, 2.5, 1.5]),
        )
        best_2c = fit_with_restarts(
            lorentzian_2c, lags_lorentz_fit, acf_lorentz_fit,
            p0_list=[
                [0.5, d_base * 0.3, d_base * 1.5, amp_guess, off_guess],
                [0.7, d_base * 0.2, d_base * 3.0, amp_guess, off_guess],
                [0.3, d_base * 0.6, d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([0.0, 1e-6, 1e-6, 0.0, -1.5], [1.0, np.inf, np.inf, 2.5, 1.5]),
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
                [0.0, 0.0, 1e-6, 1e-6, 1e-6, 0.0, -1.5],
                [1.0, 1.0, np.inf, np.inf, np.inf, 2.5, 1.5],
            ),
            maxfev=80000,
        )

        def _make_result(best, k):
            if best is None:
                return dict(error="all initialisations failed")
            popt, pcov, ymod = best
            return dict(popt=popt, pcov=pcov, ymod=ymod,
                        **build_fit_diagnostics(acf_lorentz_fit, ymod, k=k))

        r1 = _make_result(best_1c, k=3)
        r2 = _make_result(best_2c, k=5)
        r3 = _make_result(best_3c, k=7)

        fit_models = [
            ("1-component", r1, lorentzian),
            ("2-component", r2, lorentzian_2c),
            ("3-component", r3, lorentzian_3c),
        ]

        # Determine best model by AIC
        valid = [(name, r) for name, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
        best_name, best_result = min(valid, key=lambda x: x[1]["aic"]) if valid else ("1-component", r1)
        best_n_comp = int(best_name[0])
        best_fit    = best_result

        # Extract Δν_d from best model (primary component = highest-weight one)
        if "popt" in best_result:
            components, _, _ = _decode_lorentzian_components(best_n_comp, best_result["popt"])
            primary_comp     = max(components, key=lambda x: x[0])
            delta_nu_d       = primary_comp[1]
        else:
            delta_nu_d = d_base

    # ------------------------------------------------------------------
    # Macquart modulation index
    # ------------------------------------------------------------------
    mac_raw  = estimate_macquart_modulation_index(freq, raw_spectrum, corrected=False,
                                                   fit_max_lag_mhz=args.fit_max_lag)
    mac_corr = estimate_macquart_modulation_index(freq, raw_spectrum, corrected=True,
                                                   fit_max_lag_mhz=args.fit_max_lag)
    m2_raw,  dnu_raw,  lags_raw,  acov_raw  = mac_raw
    m2_corr, dnu_corr, lags_corr, acov_corr = mac_corr

    # Per-component Macquart Δν_d: window the ACF around each Lorentzian component's scale
    mac_dnu_per_component: list[tuple[float, float | None, float | None]] = []
    # list of (lorentz_dnu, mac_dnu_raw, mac_dnu_corr)
    if fit_models and "popt" in best_fit:
        components, _, _ = _decode_lorentzian_components(best_n_comp, best_fit["popt"])
        active_components = [(w, d) for (w, d) in components if w > 0.05]
        if len(active_components) > 1:
            # sort by scale
            active_components = sorted(active_components, key=lambda x: x[1])
            # define lag windows: each component owns the range from half its scale
            # to midpoint between it and the next (or fit_max_lag for the last)
            for i, (w, d) in enumerate(active_components):
                lo = d * 0.1
                if i + 1 < len(active_components):
                    hi = (d + active_components[i + 1][1]) / 2.0
                else:
                    hi = args.fit_max_lag
                hi = min(hi, args.fit_max_lag)
                mdnu_raw  = macquart_dnu_from_window(freq, raw_spectrum,  lo, hi, corrected=False)
                mdnu_corr = macquart_dnu_from_window(freq, raw_spectrum,  lo, hi, corrected=True)
                mac_dnu_per_component.append((d, mdnu_raw, mdnu_corr))
        else:
            # single active component: use the overall Macquart Δν_d
            mac_dnu_per_component.append((delta_nu_d, dnu_raw, dnu_corr))
    else:
        mac_dnu_per_component.append((delta_nu_d, dnu_raw, dnu_corr))

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

        valid = [(n, r) for n, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
        if valid:
            best_aic_n, best_aic_r = min(valid, key=lambda x: x[1]["aic"])
            best_bic_n, best_bic_r = min(valid, key=lambda x: x[1]["bic"])
            sorted_aic = sorted(valid, key=lambda x: x[1]["aic"])
            daic_runner = sorted_aic[1][1]["aic"] - sorted_aic[0][1]["aic"] if len(sorted_aic) > 1 else None
            print(f"\n  Best by AIC : {best_aic_n}  (AIC={best_aic_r['aic']:.3f}"
                  + (f", ΔAIC to runner-up={daic_runner:.2f}" if daic_runner else "") + ")")
            print(f"  Best by BIC : {best_bic_n}  (BIC={best_bic_r['bic']:.3f})")

        # Component parameters for best model
        if best_fit and "popt" in best_fit:
            components, A_fit, C_fit = _decode_lorentzian_components(best_n_comp, best_fit["popt"])
            print(f"\n  Best model ({best_n_comp}-component) parameters:")
            print(f"    A (amplitude) = {A_fit:.4f},  C (offset) = {C_fit:.4f}")
            for i, (w, d) in enumerate(components):
                try:
                    d_err = float(np.sqrt(np.diag(best_fit["pcov"]))[
                        [1, 2, 3][min(i, 2)]  # rough index; exact for 1c
                    ])
                except Exception:
                    d_err = None
                err_str = f" ± {d_err:.4f}" if d_err is not None and np.isfinite(d_err) else ""
                print(f"    Component {i+1}: Δν_d = {d:.4f}{err_str} MHz,  weight = {w:.4f}")

    # Macquart modulation index
    print(f"\n  Macquart (2019) modulation index:")
    if m2_raw is not None:
        m_raw = np.sqrt(max(m2_raw, 0.0))
        dnu_raw_str = f"{dnu_raw:.4f} MHz" if dnu_raw is not None else "N/A"
        print(f"    m  (raw mean)         = {m_raw:.6f}   Δν_d = {dnu_raw_str}")
    if m2_corr is not None:
        m_corr = np.sqrt(max(m2_corr, 0.0))
        dnu_corr_str = f"{dnu_corr:.4f} MHz" if dnu_corr is not None else "N/A"
        print(f"    m  (ν^-1.5 corrected) = {m_corr:.6f}   Δν_d = {dnu_corr_str}")

    if len(mac_dnu_per_component) > 1:
        print(f"\n  Macquart Δν_d per Lorentzian component (windowed ACF):")
        for i, (lor_d, mdnu_r, mdnu_c) in enumerate(mac_dnu_per_component):
            r_str = f"{mdnu_r:.4f} MHz" if mdnu_r is not None else "N/A"
            c_str = f"{mdnu_c:.4f} MHz" if mdnu_c is not None else "N/A"
            print(f"    Component {i+1} (Lorentzian Δν_d={lor_d:.4f} MHz): "
                  f"Macquart raw={r_str},  corrected={c_str}")

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
        print(f"    {'Band':<6} {'ν_c(MHz)':>10} {'Δν_d(MHz)':>12} {'err':>10} "
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
            if best_fit and "pcov" in best_fit:
                try:
                    d_err = float(np.sqrt(np.diag(best_fit["pcov"]))[0])
                    delta_nu_d_scaled_err = d_err * abs(scale_factor)
                except Exception:
                    pass
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
    lg_source       = "--lg-kpc"
    _ne2025_s_kpc   = None   # set below if NE2025 is run
    _ne2025_cn2     = None
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
                lg_kpc_for_calc, cn2_peak = estimate_lg_kpc_from_ne2025(
                    gl_for_lg, gb_for_lg, ds_kpc_for_calc,
                    max_dist_kpc=args.lg_max_dist_kpc)
                lg_source = "NE2025"
                # Keep the full profile for scattering predictions
                _ne2025_s_kpc, _ne2025_cn2 = get_cn2_profile(
                    gl_for_lg, gb_for_lg, da_kpc=args.lg_max_dist_kpc)
            except Exception as e:
                print(f"\nNE2025 L_g estimate failed: {e}")
        else:
            print("\nNE2025 L_g estimate skipped: provide --gl-deg/--gb-deg or --ra-hms/--dec-dms")

    # Modulation index for two-screen
    modulation_index = None
    if args.mg is None:
        if m2_corr is not None and m2_corr > 0:
            modulation_index = float(np.sqrt(m2_corr))
        elif m2_raw is not None and m2_raw > 0:
            modulation_index = float(np.sqrt(m2_raw))
    mg_for_calc = args.mg if args.mg is not None else modulation_index

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
        if lg_source == "NE2025" and _ne2025_s_kpc is not None and ds_kpc_for_calc is not None:
            scatt_ref_mhz = float(args.scatt_ref_freq_mhz) if args.scatt_ref_freq_mhz else nu_for_two_screen
            try:
                ne_pred = ne2025_scattering_prediction(
                    _ne2025_s_kpc, _ne2025_cn2,
                    lg_kpc=lg_kpc_for_calc,
                    ds_kpc=ds_kpc_for_calc,
                    nu_ref_mhz=scatt_ref_mhz,
                    v_iss_km_s=args.iss_velocity_km_s,
                )
                print_ne2025_scattering_prediction(ne_pred, lg_kpc_for_calc, ds_kpc_for_calc)
            except Exception as e:
                print(f"\n  NE2025 scattering prediction failed: {e}")

        print_two_screen_results(
            ts_results, t_scatt_for_calc, nu_for_two_screen,
            float(args.redshift), mg_for_calc, lg_kpc_for_calc,
            source_label, lg_source,
        )

        # Supplemental: NE2025 summary (if estimated)
        if lg_kpc_for_calc is not None and lg_source == "NE2025":
            print(f"\n  NE2025 L_g details:")
            print(f"    l={gl_for_lg:.4f} deg, b={gb_for_lg:.4f} deg")
            print(f"    L_g = {lg_kpc_for_calc:.4f} kpc,  Cn²_peak = {cn2_peak:.4e} m^{{-20/3}}")
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

    # Spectrum + normalised ACF
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(freq, spectrum)
    axs[0].set_xlabel("Frequency (MHz)")
    axs[0].set_ylabel("Normalised intensity")
    axs[0].set_title("Burst spectrum")

    axs[1].plot(lags_plot_sym, acf_plot_sym, label="ACF")
    if delta_nu_d is not None and fit_models and best_fit and "popt" in best_fit:
        model_fn = [lorentzian, lorentzian_2c, lorentzian_3c][best_n_comp - 1]
        axs[1].plot(lags_plot_sym, model_fn(np.abs(lags_plot_sym), *best_fit["popt"]),
                    "--", label=f"Best fit ({best_n_comp}c)")
    elif delta_nu_d is not None:
        axs[1].set_title(f"Δν_d = {delta_nu_d:.2f} MHz (provided)")
    axs[1].set_xlabel("Δν (MHz)")
    axs[1].set_ylabel("ACF")
    axs[1].legend()
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
        print(f"\nSaved spectrum+ACF plot to {args.output}")
    else:
        plt.show()
    plt.close(fig)

    # Macquart diagnostics
    plot_macquart_diagnostics(
        freq, raw_spectrum, mac_raw, mac_corr,
        output=args.output, fit_max_lag_mhz=args.fit_max_lag,
    )

    # Lorentzian component diagnostics
    if fit_models:
        plot_lorentzian_diagnostics(
            lags_plot_sym, acf_plot_sym,
            lags_lorentz_fit, acf_lorentz_fit,
            fit_models, output=args.output,
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
