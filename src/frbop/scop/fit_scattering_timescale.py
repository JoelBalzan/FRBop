import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from frbop.scop.gating import find_burst_window, select_peaks_manual
from frbop.scop.models import scattered_gaussian
from frbop.scop.scattering_index import fit_scattering_index_from_frequencies
from frbop.utils.peaks import (
    parse_peak_index_pairs,
    split_frequency_bands_equal,
    split_frequency_bands_equal_snr,
)
from frbop.utils.plotting import pub_figsize, savefig_rasterized, set_pub_col, set_pub_style, IBM_PALETTE


def main():
    parser = argparse.ArgumentParser(description="Fit scattering timescale from dynamic spectrum files")
    parser.add_argument("ds", nargs="?", default="FRB_250607_htr_dsI.npy", help="Dynamic spectrum .npy file (nfreq x ntime)")
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
    parser.add_argument("--manual-peaks", action="store_true", help="Manually select one or more on-pulse regions by clicking start/end bounds")
    parser.add_argument("--peak-indices", nargs="*", type=int, default=None, help="Manually specify peak indices as pairs: start1 end1 start2 end2 ...")
    parser.add_argument("--threshold-sigma", "--threshold", type=float, default=3.0, help="Threshold in robust sigmas for pulse gating")
    parser.add_argument("--pad", type=int, default=50, help="Padding added to detected pulse window (bins)")
    parser.add_argument("--fallback-window", "--fallback", type=int, default=200, help="Fallback half-window size if detection fails")
    parser.add_argument("--output", default=None, help="Optional output filename for plot (PNG)")
    parser.add_argument('--pub-col', type=float, default=2, help='Publication figure column count (1, 2, 3, ...). Default: 2')

    args = parser.parse_args()

    # Load data
    ds = np.load(args.ds)
    freq = np.load(args.freq)
    time = np.load(args.time)
    time = time.astype(float)
    ref_freq = float(args.ref_freq) if args.ref_freq is not None else float(np.nanmedian(freq))
    band_center_freq = float(np.nanmean(freq))
    scattering_index = float(args.scattering_index)
    fit_index_mode = bool(args.fit_index)
    scattering_scale = (band_center_freq / ref_freq) ** scattering_index

    # Ensure dynspec is (nfreq, ntime)
    if ds.shape[0] != len(freq):
        ds = ds.T

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

    # Per-channel off-pulse RMS for SNR weighting
    off_pulse_rms = np.nanstd(off_pulse, axis=1) if off_pulse.size > 0 else None

    # Optionally fit scattering index from per-frequency (or per-band) tau measurements
    fitted_index = None
    fitted_index_err = None
    tau_at_ref = None
    if fit_index_mode:
        # Build frequency-band regions if requested
        band_regions = None
        if args.freq_bands_snr is not None:
            band_snr_weights = None
            if off_pulse_rms is not None:
                raw_spectrum = np.nanmean(burst_ds, axis=1)
                band_snr_weights = np.zeros_like(freq, dtype=float)
                valid = np.isfinite(raw_spectrum) & np.isfinite(off_pulse_rms) & (off_pulse_rms > 0)
                band_snr_weights[valid] = np.maximum(raw_spectrum[valid] / off_pulse_rms[valid], 0.0)
            elif off_pulse_rms is None:
                print("Equal-SNR banding requested but off-pulse RMS is unavailable; falling back to equal-width bands.")
            band_regions = split_frequency_bands_equal_snr(freq, band_snr_weights, args.freq_bands_snr)
            print(f"\nScattering index: {len(band_regions)} equal-SNR frequency bands")
            for i, (start, stop) in enumerate(band_regions, start=1):
                print(f"  Band {i}: {freq[start]:.3f}–{freq[stop - 1]:.3f} MHz ({stop - start} channels)")
        elif args.freq_bands is not None:
            band_regions = split_frequency_bands_equal(freq, args.freq_bands)
            print(f"\nScattering index: {len(band_regions)} equal frequency bands")
            for i, (start, stop) in enumerate(band_regions, start=1):
                print(f"  Band {i}: {freq[start]:.3f}–{freq[stop - 1]:.3f} MHz ({stop - start} channels)")

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

        # Subband diagnostic plot
        if band_regions is not None and fit_details is not None and len(fit_details['freq']) > 0:
            set_pub_col(args.pub_col)
            set_pub_style(use_latex=False)
            n_bands = len(fit_details['freq'])
            t_burst_plot = time[onpulse_mask]
            fig_width, _ = pub_figsize()
            fig = plt.figure(figsize=(fig_width * 2, max(4, n_bands * 1.5)))
            gs = plt.GridSpec(n_bands, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.35)

            # Left column: subband profiles with fits
            for i in range(n_bands):
                ax = fig.add_subplot(gs[i, 0])
                profile = fit_details['profile'][i]
                popt = fit_details['popt'][i]
                tau_val = fit_details['tau'][i]
                freq_val = fit_details['freq'][i]
                fit_curve = scattered_gaussian(t_burst_plot, *popt)

                ax.plot(t_burst_plot, profile, 'k-', linewidth=1.0)
                ax.plot(t_burst_plot, fit_curve, color=IBM_PALETTE[2], linewidth=1.5, label=f'$\\tau={tau_val:.3f}$ ms')
                ax.set_ylabel(r'S [arb.]')
                ax.set_yticklabels([])
                ax.set_ylim(bottom=np.nanmin(profile) * 1.1, top=np.nanmax(profile) * 1.3)
                ax.text(0.02, 0.95, f'{freq_val:.1f} MHz', transform=ax.transAxes,
                        va='top', fontsize=8)
                if i == n_bands - 1:
                    ax.set_xlabel('Time [ms]')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            # Right column: tau vs frequency with power-law fit
            ax_tau = fig.add_subplot(gs[:, 1])
            band_freqs = np.array(fit_details['freq'])
            band_taus = np.array(fit_details['tau'])
            ax_tau.plot(band_freqs, band_taus, 'ko', markersize=4, label='Measured τ')

            # Power-law fit line
            freq_grid = np.linspace(band_freqs.min(), band_freqs.max(), 200)
            tau_grid = tau_at_ref * (freq_grid / ref_freq) ** scattering_index
            ax_tau.plot(freq_grid, tau_grid, 'b-', linewidth=1.5,
                        label=f'α={scattering_index:.2f}±{fitted_index_err:.2f}')
            ax_tau.set_xlabel('Frequency [MHz]')
            ax_tau.set_ylabel('τ [ms]')
            ax_tau.legend()
            ax_tau.grid(True, alpha=0.3)

            # Save or show
            if args.output:
                base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'png')
                bands_output = f"{base}_subbands.{ext}"
                savefig_rasterized(bands_output, dpi=300, fig=fig)
                print(f"Subband diagnostic plot saved to {bands_output}")
            plt.close(fig)

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
                savefig_rasterized(args.output, dpi=300, fig=fig)
                print(f"Plot saved to {args.output}")

        except Exception as e:
            print("Pulse-profile t_scatt fit failed:", e)
    else:
        print("Insufficient data points for scattering fit.")


if __name__ == "__main__":
    main()
