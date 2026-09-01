import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from frbop.scop.gating import find_burst_window, select_peaks_manual
from frbop.scop.models import scattered_gaussian
from frbop.scop.plotting import (plot_pa_summary, plot_subband_diagnostic,
                                 plot_subband_diagnostic_pa, plot_subband_pa)
from frbop.scop.scattering_index import fit_scattering_index_from_frequencies
from frbop.utils.peaks import (parse_peak_index_pairs,
                               split_frequency_bands_equal,
                               split_frequency_bands_equal_snr)
from frbop.utils.plotting import (pub_figsize, savefig, set_pub_col,
                                  set_pub_style)
from frbop.utils.scrunch import rescale_peak_indices, tscrunch_array


def main():
    parser = argparse.ArgumentParser(description="Fit scattering timescale from dynamic spectrum files")
    parser.add_argument("ds", nargs="?", default="FRB_250607_htr_dsI.npy", help="Dynamic spectrum .npy file (nfreq x ntime) or Stokes cube (4 x nfreq x ntime); auto-detected")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy", help="Frequency axis .npy file [MHz]")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy", help="Time axis .npy file [ms]")
    parser.add_argument(
        "--ref-freq",
        type=float,
        default=None,
        help="Reference frequency in MHz. Defaults to the lowest frequency in the file.",
    )
    parser.add_argument(
        "--scattering-index", "--t-idx",
        type=float,
        default=-4.0,
        help="Scattering index alpha where tau(nu) = tau_0 * (nu/nu_ref)^alpha. Default -4.0 for Kolmogorov. Used when --fit-index is not set.",
    )
    parser.add_argument(
        "--fit-index", "--fit-idx",
        action="store_true",
        help="Fit the scattering index by fitting tau at each frequency (or frequency band) and fitting a power law.",
    )
    parser.add_argument(
        "--freq-bands",
        type=int,
        default=None,
        help="Divide the spectrum into N equal contiguous frequency bands for scattering index fitting (requires --fit-index).",
    )
    parser.add_argument(
        "--freq-bands-snr", "--freq-snr",
        type=int,
        default=None,
        help="Divide the spectrum into N contiguous bands with equal total S/N for scattering index fitting (requires --fit-index).",
    )
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for time series (bins)")
    parser.add_argument(
        "-tscr", "--tscrunch", "-ts",
        type=int,
        default=1,
        help="Time scrunch factor applied before burst finding and scattering fits (average every N time bins)",
    )
    parser.add_argument("--manual-peaks", action="store_true", help="Manually select one or more on-pulse regions by clicking start/end bounds")
    parser.add_argument("--peak-indices", nargs="*", type=int, default=None, help="Manually specify peak indices as pairs: start1 end1 start2 end2 ...")
    parser.add_argument("--threshold-sigma", "--threshold", type=float, default=3.0, help="Threshold in robust sigmas for pulse gating")
    parser.add_argument("--pad", type=int, default=50, help="Padding added to detected pulse window (bins)")
    parser.add_argument("--fallback-window", "--fallback", type=int, default=200, help="Fallback half-window size if detection fails")
    parser.add_argument("--output", default=None, help="Optional output filename for plot (PNG)")
    parser.add_argument('--pub-col', type=float, default=2, help='Publication figure column count (1, 2, 3, ...). Default: 2')

    args = parser.parse_args()

    # Load data — auto-detect Stokes cube (4 x nfreq x ntime) vs plain DS (nfreq x ntime)
    ds_raw = np.load(args.ds)
    freq = np.load(args.freq)
    time = np.load(args.time)
    time = time.astype(float)

    has_stokes = ds_raw.ndim == 3 and ds_raw.shape[0] == 4
    if has_stokes:
        print("Detected Stokes cube (4 layers); using I for analysis, Q/U for PA")
        ds = ds_raw[0].copy()      # I
        ds_q_full = ds_raw[1].copy()  # Q
        ds_u_full = ds_raw[2].copy()  # U
    else:
        ds = ds_raw.copy()
        ds_q_full = None
        ds_u_full = None

    if args.tscrunch < 1:
        raise ValueError(f"--tscrunch must be >= 1, got {args.tscrunch}")

    ref_freq = float(args.ref_freq) if args.ref_freq is not None else float(np.nanmedian(freq))
    band_center_freq = float(np.nanmean(freq))
    scattering_index = float(args.scattering_index)
    fit_index_mode = bool(args.fit_index)
    scattering_scale = (band_center_freq / ref_freq) ** scattering_index

    # Ensure dynspec is (nfreq, ntime)
    if ds.shape[0] != len(freq):
        ds = ds.T
        if has_stokes and ds_q_full is not None:
            ds_q_full = ds_q_full.T
            ds_u_full = ds_u_full.T

    # Enforce descending frequency (high → low)
    if len(freq) > 1 and freq[0] < freq[-1]:
        freq = freq[::-1].copy()
        ds = ds[::-1, :]
        if has_stokes and ds_q_full is not None:
            ds_q_full = ds_q_full[::-1, :]
            ds_u_full = ds_u_full[::-1, :]

    if args.tscrunch > 1:
        print(f"Applying time scrunch factor {args.tscrunch} before gating/fitting")
        if args.peak_indices is not None:
            original_peak_indices = list(args.peak_indices)
            args.peak_indices = rescale_peak_indices(original_peak_indices, args.tscrunch)
            print(f"Peak indices scaled {original_peak_indices} -> {args.peak_indices}")
        n_time_original = time.size
        ds = tscrunch_array(ds, args.tscrunch, axis=1)
        time = tscrunch_array(time, args.tscrunch, axis=0)
        if has_stokes and ds_q_full is not None:
            ds_q_full = tscrunch_array(ds_q_full, args.tscrunch, axis=1)
            ds_u_full = tscrunch_array(ds_u_full, args.tscrunch, axis=1)
        print(f"Time samples: {n_time_original} -> {time.size}")

    nfreq, ntime = ds.shape

    print(f"nchan={nfreq}, ntime={ntime}")
    print(f"Time resolution = {time[1] - time[0]:.6e} ms")
    print(f"Reference frequency = {ref_freq:.6f} MHz")
    print(f"Band-center frequency = {band_center_freq:.6f} MHz")
    if fit_index_mode:
        print("Scattering index mode: will fit index from tau measurements")
    else:
        print(f"Scattering index: {scattering_index:.1f} (fixed)")

    # Collapse over frequency to find burst in time
    ts = np.nanmean(ds, axis=0)
    if args.smooth > 1:
        ts_smooth = np.convolve(ts, np.ones(args.smooth) / args.smooth, mode="same")
    else:
        ts_smooth = ts

    peak_idx = int(np.argmax(ts_smooth))

    onpulse_mask = np.zeros(ntime, dtype=bool)

    if args.peak_indices is not None and len(args.peak_indices) > 0:
        regions = parse_peak_index_pairs(args.peak_indices, ntime)
        for start_idx, end_idx in regions:
            s = start_idx
            e = end_idx
            onpulse_mask[s:e] = True
        print(f"Manual peak indices regions (start:end) = {regions}")

    elif args.manual_peaks:
        manual_regions = select_peaks_manual(
            time,
            ts,
            title='Click start/end bounds for each peak (close window when done)',
            x_label='Time [ms]',
            y_label=r'S [arb.]',
            exclusive_end=True,
        )
        clipped_regions = []
        for start_idx, end_idx in manual_regions:
            s = start_idx
            e = end_idx
            onpulse_mask[s:e] = True
            clipped_regions.append((s, e))

        if np.any(onpulse_mask):
            print(f"Manual gating regions (start:end) = {clipped_regions}")
        else:
            print("Manual gating produced no valid samples; falling back to automatic window")

    if not np.any(onpulse_mask):
        tmin, tmax = find_burst_window(ts, peak_idx, smooth_win=args.smooth, threshold_sigma=args.threshold_sigma, pad=args.pad, fallback_window=args.fallback_window)
        tmin = max(0, tmin)
        tmax = min(ntime, tmax)
        onpulse_mask[tmin:tmax] = True
        print(f"Gating from {tmin} to {tmax}")

    burst_ds = ds[:, onpulse_mask]

    # Estimate off-pulse mean robustly from complement of on-pulse mask
    off_pulse = ds[:, :len(time)//10] if np.any(~onpulse_mask) else np.empty((nfreq, 0))

    if off_pulse.size > 0:
        bandpass = np.nanmean(off_pulse, axis=1)
    else:
        # fallback: use low percentile across time to estimate baseline
        bandpass = np.percentile(ds, 10, axis=1)

    burst_ds = burst_ds - bandpass[:, None]

    # Baseline-subtract Stokes Q/U if available
    burst_ds_q = None
    burst_ds_u = None
    if has_stokes and ds_q_full is not None:
        off_pulse_q = ds_q_full[:, :len(time)//10] if np.any(~onpulse_mask) else np.empty((nfreq, 0))
        off_pulse_u = ds_u_full[:, :len(time)//10] if np.any(~onpulse_mask) else np.empty((nfreq, 0))
        bandpass_q = np.nanmean(off_pulse_q, axis=1) if off_pulse_q.size > 0 else np.percentile(ds_q_full, 10, axis=1)
        bandpass_u = np.nanmean(off_pulse_u, axis=1) if off_pulse_u.size > 0 else np.percentile(ds_u_full, 10, axis=1)
        burst_ds_q = ds_q_full[:, onpulse_mask] - bandpass_q[:, None]
        burst_ds_u = ds_u_full[:, onpulse_mask] - bandpass_u[:, None]

    # Per-channel off-pulse RMS for SNR weighting
    off_pulse_rms = np.nanstd(off_pulse, axis=1) if off_pulse.size > 0 else None

    # Optional frequency banding for scattering-index fitting and/or PA plotting
    band_regions = None
    if args.freq_bands_snr is not None:
        band_snr_weights = None
        if off_pulse_rms is not None:
            raw_spectrum = np.nanmean(burst_ds, axis=1)
            band_snr_weights = np.zeros_like(freq, dtype=float)
            valid = np.isfinite(raw_spectrum) & np.isfinite(off_pulse_rms) & (off_pulse_rms > 0)
            band_snr_weights[valid] = np.maximum(raw_spectrum[valid] / off_pulse_rms[valid], 0.0)
        else:
            print("Equal-SNR banding requested but off-pulse RMS is unavailable; falling back to equal-width bands.")
        band_regions = split_frequency_bands_equal_snr(freq, band_snr_weights, args.freq_bands_snr)
        print(f"\nSubband PA / scattering-index frequency bands: {len(band_regions)} equal-SNR bands")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: {freq[start]:.3f}–{freq[stop - 1]:.3f} MHz ({stop - start} channels)")
    elif args.freq_bands is not None:
        band_regions = split_frequency_bands_equal(freq, args.freq_bands)
        print(f"\nSubband PA / scattering-index frequency bands: {len(band_regions)} equal frequency bands")
        for i, (start, stop) in enumerate(band_regions, start=1):
            print(f"  Band {i}: {freq[start]:.3f}–{freq[stop - 1]:.3f} MHz ({stop - start} channels)")

    # Optionally fit scattering index from per-frequency (or per-band) tau measurements
    fitted_index = None
    fitted_index_err = None
    tau_at_ref = None
    if fit_index_mode:
        fit_details = None
        if band_regions is not None:
            fitted_index, tau_at_ref, fitted_index_err, n_freq_fitted, fit_details = \
                fit_scattering_index_from_frequencies(
                    burst_ds, freq, time, onpulse_mask, ref_freq=ref_freq,
                    band_regions=band_regions, return_details=True,
                )
        else:
            fitted_index, tau_at_ref, fitted_index_err, n_freq_fitted = \
                fit_scattering_index_from_frequencies(
                    burst_ds, freq, time, onpulse_mask, ref_freq=ref_freq,
                    band_regions=None,
                )
        if fitted_index is not None:
            scattering_index = float(fitted_index)
            scattering_scale = (band_center_freq / ref_freq) ** scattering_index
            label = "frequency bands" if band_regions is not None else "frequency channels"
            print(f"\nFitted scattering index from {n_freq_fitted} {label}:")
            if fitted_index_err is not None and np.isfinite(fitted_index_err):
                print(f"  alpha = {scattering_index:.3f} ± {fitted_index_err:.3f}")
            else:
                print(f"  alpha = {scattering_index:.3f}")
            if tau_at_ref > 0:
                print(f"  tau({ref_freq:.3f} MHz) = {tau_at_ref:.6f} ms")
            else:
                print(f"  tau({ref_freq:.3f} MHz) = {tau_at_ref:.3e} ms (underflowed; try different freq range)")

        # Shared plotting setup
        set_pub_col(args.pub_col)
        set_pub_style(use_latex=False)
        fig_width, _ = pub_figsize()
        t_burst_plot = time[onpulse_mask]

        # Combined subband diagnostic + PA plot (when Stokes data is available)
        pa_band_info = None
        if band_regions is not None and fit_details is not None and len(fit_details['freq']) > 0 and has_stokes and burst_ds_q is not None and burst_ds_u is not None:
            plot_subband_diagnostic(
                fit_details, t_burst_plot, args.output,
                scattering_index, fitted_index_err, tau_at_ref, ref_freq,
            )
            sorted_bands = sorted(
                [(float(np.nanmean(freq[lo:hi])), lo, hi) for lo, hi in band_regions if hi > lo],
                key=lambda x: -x[0],
            )
            pa_band_info = plot_subband_diagnostic_pa(
                fit_details, t_burst_plot, args.output,
                scattering_index, fitted_index_err, tau_at_ref, ref_freq,
                sorted_bands, burst_ds, burst_ds_q, burst_ds_u,
                freq, ds, ntime,
                ds_q_full=ds_q_full,
                ds_u_full=ds_u_full,
            )
            plot_pa_summary(pa_band_info, t_burst_plot, fig_width, args.output)
        elif band_regions is not None and fit_details is not None and len(fit_details['freq']) > 0:
            plot_subband_diagnostic(
                fit_details, t_burst_plot, args.output,
                scattering_index, fitted_index_err, tau_at_ref, ref_freq,
            )
        elif band_regions is not None and has_stokes and burst_ds_q is not None and burst_ds_u is not None:
            sorted_bands = sorted(
                [(float(np.nanmean(freq[lo:hi])), lo, hi) for lo, hi in band_regions if hi > lo],
                key=lambda x: -x[0],
            )
            pa_band_info = plot_subband_pa(
                sorted_bands, burst_ds, burst_ds_q, burst_ds_u,
                t_burst_plot, None, freq, fig_width, ds, ntime, args.output,
                ds_q_full=ds_q_full,
                ds_u_full=ds_u_full,
            )
            plot_pa_summary(pa_band_info, t_burst_plot, fig_width, args.output)

    # Frequency-integrated pulse profile and scattered-Gaussian fit for t_scatt
    pulse_profile = np.nanmean(burst_ds, axis=0)
    t_burst = time[onpulse_mask]
    t_scatt_fit_ms = None
    t_scatt_fit_err_ms = None

    if pulse_profile.size >= 5:
        prof_max_idx = int(np.argmax(pulse_profile))
        mu0 = float(t_burst[prof_max_idx])
        p_low = np.percentile(pulse_profile, 5)
        p_high = np.percentile(pulse_profile, 95)
        amp0 = max(1e-6, float(p_high - p_low))
        offset0 = float(p_low)
        width_guess = max(float(np.abs(t_burst[-1] - t_burst[0])) / 20.0, float(np.abs(time[1] - time[0])))
        sigma0 = width_guess
        tau0 = width_guess

        dt_ms = float(np.abs(time[1] - time[0])) if time.size > 1 else 1e-3
        burst_duration = float(t_burst[-1] - t_burst[0])
        lower = [0.0, float(t_burst[0]), dt_ms * 0.5, dt_ms * 0.5, -np.inf]
        upper = [np.inf, float(t_burst[-1]), burst_duration * 0.5, burst_duration * 0.5, np.inf]

        try:
            popt_t, pcov_t = curve_fit(
                scattered_gaussian,
                t_burst,
                pulse_profile,
                p0=[amp0, mu0, sigma0, tau0, offset0],
                bounds=(lower, upper),
                maxfev=50000,
            )
            t_scatt_fit_ms = float(popt_t[3])
            t_scatt_ref_ms = t_scatt_fit_ms * scattering_scale
            try:
                t_scatt_fit_err_ms = float(np.sqrt(np.diag(pcov_t))[3])
            except Exception:
                t_scatt_fit_err_ms = None
            t_scatt_ref_err_ms = None
            if t_scatt_fit_err_ms is not None and np.isfinite(t_scatt_fit_err_ms):
                t_scatt_ref_err_ms = t_scatt_fit_err_ms * scattering_scale

            # Calculate FWHM
            fwhm_intrinsic = 2 * np.sqrt(2 * np.log(2)) * popt_t[2]
            fwhm_scatter = 2 * np.log(2) * t_scatt_ref_ms

            print("\n===== Pulse-Profile Scattering Fit =====")
            if t_scatt_fit_err_ms is not None and np.isfinite(t_scatt_fit_err_ms):
                print(f"t_scatt = {t_scatt_fit_ms:.6f} ± {t_scatt_fit_err_ms:.6f} ms (scattered Gaussian)")
            else:
                print(f"t_scatt = {t_scatt_fit_ms:.6f} ms (scattered Gaussian)")
            if t_scatt_ref_err_ms is not None and np.isfinite(t_scatt_ref_err_ms):
                print(
                    f"t_scatt(ref={ref_freq:.6f} MHz, alpha={scattering_index:.1f}) = "
                    f"{t_scatt_ref_ms:.6f} ± {t_scatt_ref_err_ms:.6f} ms"
                )
            else:
                print(
                    f"t_scatt(ref={ref_freq:.6f} MHz, alpha={scattering_index:.1f}) = "
                    f"{t_scatt_ref_ms:.6f} ms"
                )
            index_source = "fitted" if fit_index_mode else "fixed"
            print(f"Scaling factor (band center to ref, alpha={index_source}) = {scattering_scale:.6f}")
            print(f"FWHM_intrinsic = {fwhm_intrinsic:.6f} ms")
            print(f"FWHM_scatter = {fwhm_scatter:.6f} ms")

            # Optional plot with FWHM markers (publication style)
            if args.output:
                set_pub_col(args.pub_col)
                set_pub_style(use_latex=False)

                fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.7))
                ax.plot(t_burst, pulse_profile, 'k-', label='Data')
                fit_y = scattered_gaussian(t_burst, *popt_t)
                ax.plot(t_burst, fit_y, 'r--', label=rf'Fit: $\tau$ = {t_scatt_fit_ms:.3f} ms')

                # Mark FWHM ranges as vertical lines centered on mu
                mu = float(popt_t[1])
                left_i = mu - 0.5 * fwhm_intrinsic
                right_i = mu + 0.5 * fwhm_intrinsic
                left_s = mu - 0.5 * fwhm_scatter
                right_s = mu + 0.5 * fwhm_scatter

                ax.axvline(left_i, color='C1', linestyle=':', linewidth=1)
                ax.axvline(right_i, color='C1', linestyle=':', linewidth=1, label=rf'FWHM = {fwhm_intrinsic:.3f} ms')
                ax.axvline(left_s, color='C2', linestyle='-.', linewidth=1)
                ax.axvline(right_s, color='C2', linestyle='-.', linewidth=1, label=rf'FWHM$_\tau$ = {fwhm_scatter:.3f} ms')

                # Text annotations above the profile
                #ylim = ax.get_ylim()
                #y_text = ylim[0] + 0.92 * (ylim[1] - ylim[0])
                #y_text2 = ylim[0] + 0.84 * (ylim[1] - ylim[0])
                #ax.text(mu, y_text, f'Intrinsic FWHM = {fwhm_intrinsic:.3f} ms', ha='center', va='top', color='C1', fontsize=styles['annotation'])
                #ax.text(mu, y_text2, f'Scatter FWHM = {fwhm_scatter:.3f} ms', ha='center', va='top', color='C2', fontsize=styles['annotation'])

                ax.set_xlabel('Time [ms]')
                ax.set_ylabel(r'S [arb.]')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                savefig(args.output, dpi=600, fig=fig)
                print(f"Plot saved to {args.output}")

        except Exception as e:
            print("Pulse-profile t_scatt fit failed:", e)
    else:
        print("Insufficient data points for scattering fit.")


if __name__ == "__main__":
    main()
