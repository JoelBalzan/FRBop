import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

try:
    from lmfit import Model, Parameters
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False

from frbop.utils.peaks import parse_peak_index_pairs, select_peaks_manual

# Publication-friendly sizing (in inches) adapted from RM plotting helpers
TWO_COLUMN_WIDTH_IN = 7.1
SINGLE_COLUMN_WIDTH_IN = 4.8


def _pub_figsize(single_column: bool = True, height_ratio: float = 0.62, min_height: float = 3.0):
    """Return a figure size (width, height) in inches suitable for LaTeX figures.

    By default returns a single-column width for a two-column layout. Set
    `single_column=False` to get a full two-column width.
    """
    width = SINGLE_COLUMN_WIDTH_IN if single_column else TWO_COLUMN_WIDTH_IN
    height = max(min_height, width * height_ratio)
    return (width, height)


def _plot_style():
    """Return a small dict of plotting sizes used for publication-style figures."""
    return {
        'title': 11,
        'label': 10,
        'tick': 8,
        'legend': 8,
        'annotation': 7,
        'line': 1.2,
    }


def scattered_gaussian(t, amp, mu, sigma, tau, offset):
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)
    arg = (sigma / tau - (t - mu) / sigma) / np.sqrt(2.0)
    # Clip exponent to prevent overflow (max ~700 for float64)
    exponent = (sigma**2) / (2.0 * tau**2) - (t - mu) / tau
    exponent = np.clip(exponent, -700, 100)
    expo = np.exp(exponent)
    return offset + 0.5 * amp * expo * erfc(arg)


def linear(x, slope, intercept):
    """Simple linear model for log-space power-law fitting."""
    return slope * x + intercept



def find_burst_window(ts, peak_idx, smooth_win=5, threshold_sigma=3.0, pad=50, fallback_window=200):
    """Find contiguous burst window around peak using robust thresholding.

    Returns (tmin, tmax) inclusive-exclusive indices.
    """
    # smooth
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        ts_smooth = np.convolve(ts, kernel, mode="same")
    else:
        ts_smooth = ts

    med = np.median(ts_smooth)
    mad = np.median(np.abs(ts_smooth - med))
    # robust sigma estimate
    sigma_est = 1.4826 * mad if mad > 0 else np.std(ts_smooth)
    thresh = med + threshold_sigma * sigma_est

    above = np.where(ts_smooth > thresh)[0]
    if above.size > 0:
        # find contiguous segments
        breaks = np.where(np.diff(above) > 1)[0]
        segments = []
        start = 0
        for b in breaks:
            segments.append(above[start : b + 1])
            start = b + 1
        segments.append(above[start:])

        # pick segment containing peak_idx, else the largest
        chosen = None
        for seg in segments:
            if peak_idx in seg:
                chosen = seg
                break
        if chosen is None:
            chosen = max(segments, key=lambda s: s.size)

        tmin = max(0, chosen[0] - pad)
        tmax = chosen[-1] + 1 + pad
        return tmin, tmax

    # fallback to fixed window around peak
    tmin = max(0, peak_idx - fallback_window)
    tmax = peak_idx + fallback_window
    return tmin, tmax


def fit_scattering_index_from_frequencies(ds, freq, time, onpulse_mask, ref_freq=None, time_axis_label="Time (ms)"):
    """Fit the scattering index by independently fitting tau at each frequency.
    
    Parameters:
    -----------
    ds : ndarray
        Dynamic spectrum (nfreq x ntime_burst), already sliced to on-pulse region
    freq : ndarray
        Frequency array (MHz)
    time : ndarray
        Time array (ms) - ORIGINAL unsliced array
    onpulse_mask : ndarray
        Boolean mask marking on-pulse region in original time array
    ref_freq : float, optional
        Reference frequency (MHz) for tau computation. Defaults to min(freq).
        
    Returns (fitted_index, tau_at_ref, index_err, n_freq_fitted) or (None, None, None, 0) if fit fails.
    """
    nfreq, ntime_burst = ds.shape
    t_burst = time[onpulse_mask]
    tau_values = []
    freq_values = []
    
    for i_freq in range(nfreq):
        profile = ds[i_freq, :]
        if profile.size < 5:
            continue
        
        prof_max_idx = int(np.argmax(profile))
        mu0 = float(t_burst[prof_max_idx])
        p_low = np.percentile(profile, 5)
        p_high = np.percentile(profile, 95)
        amp0 = max(1e-6, float(p_high - p_low))
        offset0 = float(p_low)
        width_guess = max(float(np.abs(t_burst[-1] - t_burst[0])) / 20.0, float(np.abs(time[1] - time[0])))
        sigma0 = width_guess
        tau0 = width_guess
        
        dt_ms = float(np.abs(time[1] - time[0])) if time.size > 1 else 1e-3
        burst_duration = float(t_burst[-1] - t_burst[0])
        
        # Bounds: [amp, mu, sigma, tau, offset]
        # Tighten sigma and tau to prevent overflow: max ~half the burst duration
        lower = [0.0, float(t_burst[0]), dt_ms * 0.5, dt_ms * 0.5, -np.inf]
        upper = [np.inf, float(t_burst[-1]), burst_duration * 0.5, burst_duration * 0.5, np.inf]
        
        try:
            popt, _ = curve_fit(
                scattered_gaussian,
                t_burst,
                profile,
                p0=[amp0, mu0, sigma0, tau0, offset0],
                bounds=(lower, upper),
                maxfev=5000,
            )
            tau_fit = float(popt[3])
            if tau_fit > 0:
                tau_values.append(tau_fit)
                freq_values.append(float(freq[i_freq]))
        except Exception:
            continue
    
    if len(tau_values) < 3:
        print("Warning: could not fit tau at enough frequencies to estimate index")
        return None, None, None, 0
    
    if ref_freq is None:
        ref_freq = float(np.nanmin(freq_values))
    
    tau_values = np.array(tau_values)
    freq_values = np.array(freq_values)
    
    # Compute geometric mean frequency for stable evaluation
    freq_geom_mean = float(np.exp(np.mean(np.log(freq_values))))
    
    # Fit power law using lmfit with MCMC if available, otherwise use polyfit
    log_freq = np.log(freq_values)
    log_tau = np.log(tau_values)
    
    # Estimate uncertainties in log space
    log_err = np.ones_like(log_freq) * 0.2  # conservative estimate
    
    try:
        # from https://github.com/fjankowsk/scatfit/tree/master
        if not HAS_LMFIT:
            raise ImportError("lmfit not available, using polyfit fallback")
        
        # Use lmfit for robust fitting with MCMC
        model = Model(linear)
        params = Parameters()
        params.add('slope', value=-4.0, vary=True)
        params.add('intercept', value=np.mean(log_tau), vary=True)
        
        # Least-squares fit first (initialization)
        fitresult_ls = model.fit(
            data=log_tau, 
            x=log_freq, 
            weights=1.0 / log_err,
            params=params, 
            method="leastsq"
        )
        
        if not fitresult_ls.success:
            print("Warning: least-squares fit did not converge, using simpler approach")
            raise RuntimeError("LS fit failed")
        
        # MCMC fit with proper uncertainty estimation
        emcee_kws = dict(steps=2000, burn=700, thin=10, is_weighted=True, progress=False)
        emcee_params = fitresult_ls.params.copy()
        
        fitresult_mcmc = model.fit(
            data=log_tau,
            x=log_freq,
            weights=1.0 / log_err,
            params=emcee_params,
            method="emcee",
            fit_kws=emcee_kws,
        )
        
        # Extract results from MCMC
        alpha_fitted = fitresult_mcmc.best_values['slope']
        intercept_fitted = fitresult_mcmc.best_values['intercept']
        
        # Compute uncertainties from flatchain
        samples = fitresult_mcmc.flatchain
        slope_samples = samples['slope']
        intercept_samples = samples['intercept']
        
        # 16th and 84th percentiles for 1-sigma errors
        slope_err = np.std(slope_samples)
        
        # Warn if alpha is positive (unphysical for scattering)
        if alpha_fitted > 0:
            print(f"Warning: fitted alpha = {alpha_fitted:.3f} is positive (unphysical). Scattering index should be negative.")
        
        # Evaluate at geometric mean (most stable), then scale to ref_freq
        log_tau_at_geom_mean = intercept_fitted - alpha_fitted * np.log(freq_geom_mean)
        tau_at_geom_mean = np.exp(np.clip(log_tau_at_geom_mean, -700, 700))
        tau_at_ref = tau_at_geom_mean * (ref_freq / freq_geom_mean) ** alpha_fitted
        
        return alpha_fitted, tau_at_ref, slope_err, len(tau_values)
        
    except Exception as e:
        # Fallback to simple polyfit if lmfit not available or MCMC fails
        print(f"Warning: MCMC fit issue ({e}), falling back to polyfit")
        try:
            coeffs = np.polyfit(log_freq, log_tau, 1)
            alpha_fitted = coeffs[0]
            c_fitted = coeffs[1]
            
            # Warn if alpha is positive (unphysical for scattering)
            if alpha_fitted > 0:
                print(f"Warning: fitted alpha = {alpha_fitted:.3f} is positive (unphysical). Scattering index should be negative.")
            
            # Evaluate at geometric mean (most stable), then scale to ref_freq
            log_tau_at_geom_mean = c_fitted - alpha_fitted * np.log(freq_geom_mean)
            tau_at_geom_mean = np.exp(np.clip(log_tau_at_geom_mean, -700, 700))
            tau_at_ref = tau_at_geom_mean * (ref_freq / freq_geom_mean) ** alpha_fitted
            
            # Simple error estimate
            log_tau_fit = np.polyval(coeffs, log_freq)
            residuals = log_tau - log_tau_fit
            rms_residual = np.sqrt(np.mean(residuals**2))
            alpha_err = rms_residual * np.sqrt(len(log_freq)) / np.sqrt(np.sum((log_freq - np.mean(log_freq))**2))
            
            return alpha_fitted, tau_at_ref, alpha_err, len(tau_values)
        except Exception as e2:
            print(f"Warning: polyfit fallback also failed: {e2}")
            return None, None, None, 0


def main():
    parser = argparse.ArgumentParser(description="Fit scattering timescale from dynamic spectrum files")
    parser.add_argument("ds", nargs="?", default="FRB_250607_htr_dsI.npy", help="Dynamic spectrum .npy file (nfreq x ntime)")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy", help="Frequency axis .npy file (MHz)")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy", help="Time axis .npy file (ms)")
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
        help="Fit the scattering index by independently fitting tau at each frequency and fitting a power law.",
    )
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for time series (bins)")
    parser.add_argument("--manual-peaks", action="store_true", help="Manually select one or more on-pulse regions by clicking start/end bounds")
    parser.add_argument("--peak-indices", nargs="*", type=int, default=None, help="Manually specify peak indices as pairs: start1 end1 start2 end2 ...")
    parser.add_argument("--threshold-sigma", type=float, default=3.0, help="Threshold in robust sigmas for pulse gating")
    parser.add_argument("--pad", type=int, default=50, help="Padding added to detected pulse window (bins)")
    parser.add_argument("--fallback-window", type=int, default=200, help="Fallback half-window size if detection fails")
    parser.add_argument("--output", default=None, help="Optional output filename for plot (PNG)")

    args = parser.parse_args()

    # Load data
    ds = np.load(args.ds)
    freq = np.load(args.freq)
    time = np.load(args.time)
    time = time.astype(float)
    ref_freq = float(args.ref_freq) if args.ref_freq is not None else float(np.nanmin(freq))
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
        print(f"Scattering index mode: will fit index from per-frequency tau measurements")
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
            x_label='Time (ms)',
            y_label='Flux',
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

    # Optionally fit scattering index from per-frequency tau measurements
    fitted_index = None
    fitted_index_err = None
    tau_at_ref = None
    if fit_index_mode:
        fitted_index, tau_at_ref, fitted_index_err, n_freq_fitted = fit_scattering_index_from_frequencies(
            burst_ds, freq, time, onpulse_mask, ref_freq=ref_freq
        )
        if fitted_index is not None:
            scattering_index = float(fitted_index)
            scattering_scale = (band_center_freq / ref_freq) ** scattering_index
            print(f"\nFitted scattering index from {n_freq_fitted} frequency channels:")
            if fitted_index_err is not None and np.isfinite(fitted_index_err):
                print(f"  alpha = {scattering_index:.3f} ± {fitted_index_err:.3f}")
            else:
                print(f"  alpha = {scattering_index:.3f}")
            if tau_at_ref > 0:
                print(f"  tau({ref_freq:.3f} MHz) = {tau_at_ref:.6f} ms")
            else:
                print(f"  tau({ref_freq:.3f} MHz) = {tau_at_ref:.3e} ms (underflowed; try different freq range)")

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
                styles = _plot_style()
                plt.rcParams.update({
                    'axes.titlesize': styles['title'],
                    'axes.labelsize': styles['label'],
                    'xtick.labelsize': styles['tick'],
                    'ytick.labelsize': styles['tick'],
                    'legend.fontsize': styles['legend'],
                    'lines.linewidth': styles['line'],
                })

                # Use single-column width by default so figure fits in a 2-col LaTeX layout
                fig, ax = plt.subplots(figsize=_pub_figsize(single_column=True))
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

                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Flux')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                # Save with higher DPI suitable for publication
                plt.savefig(args.output, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {args.output}")

        except Exception as e:
            print("Pulse-profile t_scatt fit failed:", e)
    else:
        print("Insufficient data points for scattering fit.")


if __name__ == "__main__":
    main()