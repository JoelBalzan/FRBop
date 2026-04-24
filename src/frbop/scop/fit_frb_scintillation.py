import argparse
import os
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc


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


def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2:]


def select_peaks_manual(stokes_i: np.ndarray, time_ms: np.ndarray) -> List[Tuple[int, int]]:
    """
    Manually select peak bounds by clicking on the pulse profile.

    Click pairs of points (start, end) for each peak. Close the window when done.
    Returns a list of (start_idx, end_idx) tuples (inclusive start, exclusive end-like indices).
    """
    time_series = np.mean(stokes_i, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_ms, time_series, color='k', linewidth=1)
    ax.set_title('Click start/end bounds for each peak (close window to finish)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Flux')
    ax.grid(True, alpha=0.3)
    cursor_line = ax.axvline(time_ms[0] if time_ms.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)

    times: List[float] = []

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_line.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x = float(event.xdata)
        times.append(x)
        ax.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
        if len(times) % 2 == 0:
            start_t, end_t = sorted((times[-2], times[-1]))
            ax.axvspan(start_t, end_t, color='tab:orange', alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    n_time = time_ms.size
    if len(times) < 2:
        return [(0, n_time)]

    if len(times) % 2 != 0:
        times = times[:-1]

    peak_regions: List[Tuple[int, int]] = []
    for i in range(0, len(times), 2):
        start_t, end_t = sorted((times[i], times[i + 1]))
        start_idx = int(np.argmin(np.abs(time_ms - start_t)))
        end_idx = int(np.argmin(np.abs(time_ms - end_t)))
        start = min(start_idx, end_idx)
        end_exclusive = min(time_ms.size, max(start_idx, end_idx) + 1)
        peak_regions.append((start, end_exclusive))

    return peak_regions


def lorentzian(delta_nu, delta_nu_d):
    return 1.0 / (1.0 + (delta_nu / delta_nu_d) ** 2)


def scattered_gaussian(t, amp, mu, sigma, tau, offset):
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)
    arg = (sigma / tau - (t - mu) / sigma) / np.sqrt(2.0)
    expo = np.exp((sigma**2) / (2.0 * tau**2) - (t - mu) / tau)
    return offset + 0.5 * amp * expo * erfc(arg)


def estimate_ds_kpc_from_redshift(z, h0_km_s_mpc=67.4, omega_m=0.315, omega_lambda=0.685, n_steps=4096):
    """Estimate angular-diameter distance D_s (kpc) from redshift in flat ΛCDM."""
    if z <= 0:
        return 0.0
    if h0_km_s_mpc <= 0:
        raise ValueError("h0_km_s_mpc must be > 0")
    if omega_m < 0 or omega_lambda < 0:
        raise ValueError("omega_m and omega_lambda must be >= 0")

    c_km_s = 299792.458
    z_grid = np.linspace(0.0, z, int(max(256, n_steps)))
    ez = np.sqrt(omega_m * (1.0 + z_grid) ** 3 + omega_lambda)
    dc_mpc = (c_km_s / h0_km_s_mpc) * np.trapezoid(1.0 / ez, z_grid)
    da_mpc = dc_mpc / (1.0 + z)
    return float(da_mpc * 1e3)


def estimate_lg_kpc_from_ne2025_peak_cn2(ldeg, bdeg, max_dist_kpc=50.0):
    """Estimate L_g as the distance of peak Cn2 along LoS using mwprop NE2025."""
    try:
        from mwprop.nemod.NE2025 import ne2025
    except Exception as exc:
        raise RuntimeError("mwprop NE2025 import failed; install mwprop + dependencies") from exc

    if max_dist_kpc <= 0:
        raise ValueError("max_dist_kpc must be > 0")

    with tempfile.TemporaryDirectory(prefix="mwprop_ne2025_") as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            ne2025(
                ldeg=float(ldeg),
                bdeg=float(bdeg),
                dmd=float(max_dist_kpc),
                ndir=-1,
                classic=False,
                dmd_only=False,
                do_analysis=True,
                plotting=False,
                verbose=False,
            )
            out_dir = os.path.join(tmpdir, "output_ne2025p")
            candidate_paths = [
                os.path.join(out_dir, "f25_d2dm_ne_dsm_vs_s.txt"),
                os.path.join(out_dir, "f25_dm2d_ne_dsm_vs_s.txt"),
            ]
            table_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if table_path is None:
                raise RuntimeError("NE2025 analysis file not found (expected f25_*_ne_dsm_vs_s.txt)")

            d_vals = []
            cn2_vals = []
            with open(table_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) < 6:
                        continue
                    try:
                        d = float(parts[0])
                        cn2 = float(parts[5])
                    except ValueError:
                        continue
                    if np.isfinite(d) and np.isfinite(cn2):
                        d_vals.append(d)
                        cn2_vals.append(cn2)

            if len(d_vals) == 0:
                raise RuntimeError("No numeric LoS rows parsed from NE2025 output table")

            d_arr = np.asarray(d_vals)
            cn2_arr = np.asarray(cn2_vals)
            idx = int(np.argmax(cn2_arr))
            lg_kpc = float(d_arr[idx])
            cn2_peak = float(cn2_arr[idx])
            return lg_kpc, cn2_peak
        finally:
            os.chdir(old_cwd)


def radec_to_galactic_deg(ra_hms, dec_dms):
    """Convert ICRS RA/Dec strings to Galactic (l, b) in degrees."""
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except Exception as exc:
        raise RuntimeError("Astropy is required for RA/Dec to Galactic conversion") from exc

    c_icrs = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    c_gal = c_icrs.galactic
    return float(c_gal.l.deg), float(c_gal.b.deg)


def main():
    parser = argparse.ArgumentParser(description="Fit FRB scintillation from dynamic spectrum files")
    parser.add_argument("ds", nargs="?", default="FRB_250607_htr_dsI.npy", help="Dynamic spectrum .npy file (nfreq x ntime)")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy", help="Frequency axis .npy file (MHz)")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy", help="Time axis .npy file (ms)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for time series (bins)")
    parser.add_argument("--manual-peaks", action="store_true", help="Manually select one or more on-pulse regions by clicking start/end bounds")
    parser.add_argument("--threshold-sigma", type=float, default=3.0, help="Threshold in robust sigmas for pulse gating")
    parser.add_argument("--pad", type=int, default=50, help="Padding added to detected pulse window (bins)")
    parser.add_argument("--fallback-window", type=int, default=200, help="Fallback half-window size if detection fails")
    parser.add_argument("--fit-max-lag", type=float, default=50.0, help="Max lag (MHz) to use in ACF fit")
    parser.add_argument("--output", default=None, help="Optional output filename for plot (PNG)")
    parser.add_argument("--time-acf-model", choices=["exp", "gauss"], default="exp", help="Model for temporal ACF: exponential or gaussian")
    parser.add_argument("--fit-max-tau", type=float, default=100.0, help="Max time lag (ms) to use in temporal ACF fit")
    parser.add_argument("--test-multiple", action="store_true", help="Test single vs. double-component ACF models and report AIC/BIC")
    parser.add_argument("--t-scatt-ms", type=float, default=None, help="Pulse broadening time t_scatt in ms (from pulse-shape fit; used for two-screen distance estimate)")
    parser.add_argument("--redshift", type=float, default=None, help="Source redshift z for two-screen distance estimate")
    parser.add_argument("--ds-kpc", type=float, default=None, help="Source distance D_s in kpc for two-screen distance estimate")
    parser.add_argument("--center-freq-mhz", type=float, default=None, help="Observing frequency ν in MHz for two-screen equations (default: median of freq axis)")
    parser.add_argument("--mg", type=float, default=None, help="Galactic scintillation modulation index m_g (<=1). If omitted, uses measured spectral modulation index m when available")
    parser.add_argument("--lg-kpc", type=float, default=None, help="Assumed Galactic-screen distance L_g in kpc to infer L_x from L_x L_g")
    parser.add_argument("--estimate-lg-ne2025", action="store_true", help="Estimate L_g from NE2025 as the distance of peak Cn2 along the line of sight")
    parser.add_argument("--gl-deg", type=float, default=None, help="Galactic longitude l (deg) for NE2025 L_g estimate")
    parser.add_argument("--gb-deg", type=float, default=None, help="Galactic latitude b (deg) for NE2025 L_g estimate")
    parser.add_argument("--ra-hms", type=str, default=None, help="ICRS right ascension in sexagesimal hour format (e.g., '12:34:56.7')")
    parser.add_argument("--dec-dms", type=str, default=None, help="ICRS declination in sexagesimal degree format (e.g., '-45:12:34.5')")
    parser.add_argument("--lg-max-dist-kpc", type=float, default=50.0, help="Max distance (kpc) for NE2025 LoS sampling when estimating L_g")
    parser.add_argument("--h0-km-s-mpc", type=float, default=67.8, help="Hubble constant for D_s(z) estimator when --ds-kpc is omitted")
    parser.add_argument("--omega-m", type=float, default=0.31, help="Matter density Ω_m for D_s(z) estimator")
    parser.add_argument("--omega-lambda", type=float, default=0.69, help="Dark-energy density Ω_Λ for D_s(z) estimator")

    args = parser.parse_args()

    ds = np.load(args.ds)
    freq = np.load(args.freq)
    time = np.load(args.time)
    # keep loaded time in milliseconds
    time = time.astype(float)

    # Ensure dynspec is (nfreq, ntime)
    if ds.shape[0] != len(freq):
        ds = ds.T

    nfreq, ntime = ds.shape
    df = np.abs(freq[1] - freq[0])

    print(f"nchan={nfreq}, ntime={ntime}")
    print(f"Channel width = {df:.4f} MHz")
    print(f"Time resolution = {time[1] - time[0]:.6e} ms")

    # Collapse over frequency to find burst in time
    ts = np.mean(ds, axis=0)
    if args.smooth > 1:
        ts_smooth = np.convolve(ts, np.ones(args.smooth) / args.smooth, mode="same")
    else:
        ts_smooth = ts

    peak_idx = int(np.argmax(ts_smooth))

    onpulse_mask = np.zeros(ntime, dtype=bool)
    if args.manual_peaks:
        manual_regions = select_peaks_manual(ds, time)
        clipped_regions = []
        for start_idx, end_idx in manual_regions:
            s = int(np.clip(start_idx, 0, ntime - 1))
            e = int(np.clip(end_idx, 0, ntime))
            if e <= s:
                e = min(ntime, s + 1)
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
    else:
        on_idx = np.where(onpulse_mask)[0]
        tmin = int(on_idx[0])
        tmax = int(on_idx[-1] + 1)

    burst_ds = ds[:, onpulse_mask]

    # Estimate off-pulse mean robustly from complement of on-pulse mask
    off_pulse = ds[:, ~onpulse_mask] if np.any(~onpulse_mask) else np.empty((nfreq, 0))

    if off_pulse.size > 0:
        bandpass = np.mean(off_pulse, axis=1)
    else:
        # fallback: use low percentile across time to estimate baseline
        bandpass = np.percentile(ds, 10, axis=1)

    burst_ds = burst_ds - bandpass[:, None]

    # Frequency-integrated pulse profile and scattered-Gaussian fit for t_scatt
    pulse_profile = np.mean(burst_ds, axis=0)
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
        lower = [0.0, float(t_burst[0]), dt_ms * 0.1, dt_ms * 0.1, -np.inf]
        upper = [np.inf, float(t_burst[-1]), np.inf, np.inf, np.inf]

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
            try:
                t_scatt_fit_err_ms = float(np.sqrt(np.diag(pcov_t))[3])
            except Exception:
                t_scatt_fit_err_ms = None

            print("\n===== Pulse-Profile Scattering Fit =====")
            if t_scatt_fit_err_ms is not None and np.isfinite(t_scatt_fit_err_ms):
                print(f"t_scatt = {t_scatt_fit_ms:.6f} ± {t_scatt_fit_err_ms:.6f} ms (scattered Gaussian)")
            else:
                print(f"t_scatt = {t_scatt_fit_ms:.6f} ms (scattered Gaussian)")
        except Exception as e:
            print("Pulse-profile t_scatt fit failed:", e)

    # Collapse in time → 1D spectrum
    raw_spectrum = np.mean(burst_ds, axis=1)
    spectrum = raw_spectrum.copy()

    # Normalize robustly
    spec_med = np.median(spectrum)
    spec_mad = np.median(np.abs(spectrum - spec_med))
    spec_sigma = 1.4826 * spec_mad if spec_mad > 0 else np.std(spectrum)
    if spec_sigma == 0:
        spectrum = spectrum - spec_med
    else:
        spectrum = (spectrum - spec_med) / spec_sigma

    # Frequency ACF
    acf = autocorr(spectrum)
    if acf[0] != 0:
        acf = acf / acf[0]

    lags = np.arange(len(acf)) * df  # MHz

    # Limit fit range (exclude zero lag to avoid forcing normalization)
    mask = (lags > 0) & (lags < args.fit_max_lag)
    lags_fit = lags[mask]
    acf_fit = acf[mask]

    delta_nu_d = None
    delta_nu_d_err = None
    try:
        popt, pcov = curve_fit(lorentzian, lags_fit, acf_fit, p0=[1.0], maxfev=5000)
        delta_nu_d = popt[0]
        delta_nu_d_err = np.sqrt(np.diag(pcov))[0]

        print("\n===== Scintillation Result =====")
        print(f"Δν_d = {delta_nu_d:.3f} ± {delta_nu_d_err:.3f} MHz")
    except Exception as e:
        print("Fit failed:", e)
            # Time axis is assumed to be in milliseconds

    t_scatt_for_calc_ms = args.t_scatt_ms if args.t_scatt_ms is not None else t_scatt_fit_ms

    ds_kpc_for_calc = args.ds_kpc
    if ds_kpc_for_calc is None and args.redshift is not None:
        try:
            ds_kpc_for_calc = estimate_ds_kpc_from_redshift(
                float(args.redshift),
                h0_km_s_mpc=float(args.h0_km_s_mpc),
                omega_m=float(args.omega_m),
                omega_lambda=float(args.omega_lambda),
            )
            print(
                f"Estimated D_s from redshift z={args.redshift:.6f}: "
                f"D_s={ds_kpc_for_calc:.6e} kpc "
                f"(H0={args.h0_km_s_mpc:.3f}, Ω_m={args.omega_m:.3f}, Ω_Λ={args.omega_lambda:.3f})"
            )
        except Exception as e:
            print(f"Could not estimate D_s from redshift: {e}")

    lg_kpc_for_calc = args.lg_kpc
    if lg_kpc_for_calc is None and args.estimate_lg_ne2025:
        gl_for_lg = args.gl_deg
        gb_for_lg = args.gb_deg

        if gl_for_lg is None or gb_for_lg is None:
            if args.ra_hms is not None and args.dec_dms is not None:
                try:
                    gl_for_lg, gb_for_lg = radec_to_galactic_deg(args.ra_hms, args.dec_dms)
                    print(
                        f"Converted ICRS -> Galactic: RA={args.ra_hms}, Dec={args.dec_dms} "
                        f"-> l={gl_for_lg:.6f} deg, b={gb_for_lg:.6f} deg"
                    )
                except Exception as e:
                    print(f"RA/Dec to Galactic conversion failed: {e}")

        if gl_for_lg is None or gb_for_lg is None:
            print("NE2025 L_g estimate skipped: provide --gl-deg/--gb-deg or --ra-hms/--dec-dms")
        else:
            try:
                lg_kpc_for_calc, cn2_peak = estimate_lg_kpc_from_ne2025_peak_cn2(
                    gl_for_lg,
                    gb_for_lg,
                    max_dist_kpc=args.lg_max_dist_kpc,
                )
                print("\n===== NE2025 Galactic Screen Estimate =====")
                print(
                    f"Peak Cn2 at L_g={lg_kpc_for_calc:.6f} kpc "
                    f"(l={gl_for_lg:.3f} deg, b={gb_for_lg:.3f} deg, "
                    f"Cn2_peak={cn2_peak:.6e}, max_dist={args.lg_max_dist_kpc:.2f} kpc)"
                )
            except Exception as e:
                print(f"NE2025 L_g estimate failed: {e}")

    can_do_two_screen = (
        delta_nu_d is not None
        and t_scatt_for_calc_ms is not None
        and args.redshift is not None
        and ds_kpc_for_calc is not None
    )

    modulation_index = None
    if args.mg is None and can_do_two_screen:
        finite_spec = raw_spectrum[np.isfinite(raw_spectrum)]
        if finite_spec.size > 1:
            mean_spec = float(np.mean(finite_spec))
            std_spec = float(np.std(finite_spec))
            if mean_spec > 0:
                modulation_index = std_spec / mean_spec
                print("\n===== Spectral Modulation =====")
                print(f"m = {modulation_index:.6f}")
            else:
                print("\n===== Spectral Modulation =====")
                print("m could not be computed (mean spectrum <= 0 after baseline subtraction)")

    mg_for_calc = args.mg if args.mg is not None else modulation_index

    if delta_nu_d is not None and t_scatt_for_calc_ms is not None and args.redshift is not None and ds_kpc_for_calc is not None:
        if t_scatt_for_calc_ms <= 0:
            print("Two-screen estimate skipped: --t-scatt-ms must be > 0")
        elif ds_kpc_for_calc <= 0:
            print("Two-screen estimate skipped: D_s must be > 0 (from --ds-kpc or redshift estimate)")
        elif args.redshift < 0:
            print("Two-screen estimate skipped: --redshift must be >= 0")
        else:
            nu_obs_mhz = float(args.center_freq_mhz) if args.center_freq_mhz is not None else float(np.median(freq))
            nu_obs_hz = nu_obs_mhz * 1e6
            nu_dc_hz = float(delta_nu_d) * 1e6
            t_scatt_s = float(t_scatt_for_calc_ms) * 1e-3
            ds_kpc = float(ds_kpc_for_calc)
            z = float(args.redshift)
            mg = float(mg_for_calc) if mg_for_calc is not None else None

            c_val = 2.0 * np.pi * nu_dc_hz * t_scatt_s
            geom_prefactor = (ds_kpc ** 2) / (2.0 * np.pi * (nu_obs_hz ** 2) * (1.0 + z))
            lxlg_upper_kpc2 = geom_prefactor * (nu_dc_hz / t_scatt_s)
            lxlg_partial_kpc2 = None
            if mg is not None and mg > 0:
                lxlg_partial_kpc2 = lxlg_upper_kpc2 / (mg ** 2)

            print("\n===== Two-Screen Distance Estimate =====")
            t_source = "--t-scatt-ms" if args.t_scatt_ms is not None else "pulse-profile fit"
            if mg is not None:
                mg_source = "--mg" if args.mg is not None else "measured m"
                print(f"Using ν_DC={delta_nu_d:.6f} MHz, t_scatt={t_scatt_for_calc_ms:.6f} ms ({t_source}), ν={nu_obs_mhz:.3f} MHz, z={z:.6f}, m_g={mg:.6f} ({mg_source})")
            else:
                print(f"Using ν_DC={delta_nu_d:.6f} MHz, t_scatt={t_scatt_for_calc_ms:.6f} ms ({t_source}), ν={nu_obs_mhz:.3f} MHz, z={z:.6f}")
            print(f"C = 2π ν_DC t_scatt = {c_val:.3e}")
            print(f"Eq.(2)-style upper limit (m_g=1): L_x L_g <= {lxlg_upper_kpc2:.6e} kpc^2")

            if mg is None:
                print("Eq.(4) not evaluated: m_g unavailable (provide --mg or enable measured m fallback)")
            elif mg <= 0:
                print("Eq.(4) not evaluated: m_g must be > 0")
            elif mg < 1.0 and lxlg_partial_kpc2 is not None:
                print(f"Eq.(4)-style partial-modulation estimate (m_g={mg:.3f}): L_x L_g ≈ {lxlg_partial_kpc2:.6e} kpc^2")
            elif mg > 1.0:
                print(f"Warning: m_g={mg:.3f} > 1 is unphysical for modulation index; partial-modulation estimate may be invalid")

            if lg_kpc_for_calc is not None:
                if lg_kpc_for_calc <= 0:
                    print("L_x inference skipped: L_g must be > 0")
                else:
                    lg_kpc = float(lg_kpc_for_calc)
                    lx_upper_kpc = lxlg_upper_kpc2 / lg_kpc
                    lg_source = "--lg-kpc" if args.lg_kpc is not None else "NE2025 Cn2 peak"
                    print(f"Using L_g={lg_kpc:.6f} kpc ({lg_source}) -> L_x <= {lx_upper_kpc:.6e} kpc (from Eq. 2 upper limit)")
                    if mg is not None and mg < 1.0 and lxlg_partial_kpc2 is not None:
                        lx_partial_kpc = lxlg_partial_kpc2 / lg_kpc
                        print(f"Using L_g={lg_kpc:.6f} kpc ({lg_source}) -> L_x ≈ {lx_partial_kpc:.6e} kpc (from Eq. 4 partial-modulation)")

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(freq, spectrum)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Intensity")
    plt.title("Burst Spectrum")

    plt.subplot(1, 2, 2)
    plt.plot(lags_fit, acf_fit, label="ACF")
    if delta_nu_d is not None:
        plt.plot(lags_fit, lorentzian(lags_fit, delta_nu_d), "--", label="Lorentzian fit")
        plt.title(f"Δν_d = {delta_nu_d:.2f} MHz")
    else:
        plt.title("ACF (fit failed)")
    plt.xlabel("Δν (MHz)")
    plt.ylabel("ACF")
    plt.legend()

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

    # ---------------------------
    # Temporal ACF: scintillation timescale
    # ---------------------------
    dt = np.abs(time[1] - time[0])
    # compute ACF per frequency channel within burst window
    acf_channels = []
    for i in range(nfreq):
        x = burst_ds[i, :]
        a = autocorr(x)
        if a.size == 0 or a[0] == 0:
            continue
        a = a / a[0]
        acf_channels.append(a)

    if len(acf_channels) == 0:
        print("Temporal ACF: insufficient data to compute per-channel ACFs")
        return

    # pad/truncate to same length
    min_len = min(a.size for a in acf_channels)
    acf_mat = np.vstack([a[:min_len] for a in acf_channels])
    acf_time = np.mean(acf_mat, axis=0)
    taus = np.arange(len(acf_time)) * dt

    # Exclude zero lag from temporal fit as well (t=0 is trivially 1)
    mask_t = (taus > 0) & (taus <= args.fit_max_tau)
    taus_fit = taus[mask_t]
    acf_time_fit = acf_time[mask_t]

    if taus_fit.size < 3:
        print("Temporal ACF: not enough points to fit timescale")
        return

    # Define single and double component temporal models
    if args.time_acf_model == "exp":
        def single_temporal(t, tau):
            return np.exp(-t / tau)

        def double_temporal(t, f, tau1, tau2):
            return f * np.exp(-t / tau1) + (1 - f) * np.exp(-t / tau2)
    else:
        def single_temporal(t, tau):
            return np.exp(-(t / tau) ** 2)

        def double_temporal(t, f, tau1, tau2):
            return f * np.exp(-(t / tau1) ** 2) + (1 - f) * np.exp(-(t / tau2) ** 2)

    def compute_aic_bic(y, ymod, k):
        resid = y - ymod
        rss = np.sum(resid ** 2)
        n = y.size
        if rss <= 0:
            rss = 1e-12
        aic = 2 * k + n * np.log(rss / n)
        bic = k * np.log(n) + n * np.log(rss / n)
        return aic, bic, rss

    results = {}

    # Fit single-component
    try:
        # initial guess for tau: use point where acf drops to ~1/e or half
        if np.any(acf_time_fit < np.exp(-1)):
            idx = np.argmax(acf_time_fit < np.exp(-1))
            p0 = [max(1e-6, taus_fit[idx])]
        else:
            p0 = [max(1e-6, taus_fit[len(taus_fit) // 10])]
        popt_s, pcov_s = curve_fit(single_temporal, taus_fit, acf_time_fit, p0=p0, bounds=(1e-9, np.inf), maxfev=10000)
        ymod_s = single_temporal(taus_fit, *popt_s)
        aic_s, bic_s, rss_s = compute_aic_bic(acf_time_fit, ymod_s, k=1)
        results['single'] = dict(popt=popt_s, pcov=pcov_s, aic=aic_s, bic=bic_s, rss=rss_s, ymod=ymod_s)
    except Exception as e:
        results['single'] = dict(error=str(e))

    # Fit double-component
    try:
        # initial guess: f~0.5, taus from single guess
        if 'popt_s' in locals():
            tau_guess = float(popt_s[0])
        else:
            tau_guess = max(1e-6, taus_fit[len(taus_fit) // 10])
        p0 = [0.5, tau_guess / 3.0, tau_guess * 3.0]
        bounds = ([0.0, 1e-9, 1e-9], [1.0, np.inf, np.inf])
        popt_d, pcov_d = curve_fit(double_temporal, taus_fit, acf_time_fit, p0=p0, bounds=bounds, maxfev=20000)
        ymod_d = double_temporal(taus_fit, *popt_d)
        aic_d, bic_d, rss_d = compute_aic_bic(acf_time_fit, ymod_d, k=3)
        results['double'] = dict(popt=popt_d, pcov=pcov_d, aic=aic_d, bic=bic_d, rss=rss_d, ymod=ymod_d)
    except Exception as e:
        results['double'] = dict(error=str(e))

    # Report results and prefer lower AIC/BIC
    print("\n===== Temporal ACF Model Comparison =====")
    for key in ('single', 'double'):
        r = results.get(key, {})
        if 'error' in r:
            print(f"{key}: fit failed: {r['error']}")
        else:
            print(f"{key}: aic={r['aic']:.2f}, bic={r['bic']:.2f}, rss={r['rss']:.4e}")

    preferred = None
    if 'aic' in results.get('single', {}) and 'aic' in results.get('double', {}):
        preferred = 'double' if results['double']['aic'] < results['single']['aic'] else 'single'
        print(f"Preferred (AIC): {preferred}")

    # Choose final model to report
    final = results.get(preferred or 'single')
    if 'popt' in final:
        if preferred == 'double':
            f, tau1, tau2 = final['popt']
            print(f"\nBest-fit double temporal: f={f:.3f}, tau1={tau1:.6f} ms, tau2={tau2:.6f} ms")
            tau_report = (tau1, tau2)
        else:
            tau = float(final['popt'][0])
            print(f"\nBest-fit single temporal: tau={tau:.6f} ms")
            tau_report = (tau,)
    else:
        print("Temporal fits failed; no timescale to report")

    # plot temporal ACF with both fits
    plt.figure(figsize=(6, 4))
    plt.plot(taus_fit, acf_time_fit, label="Temporal ACF")
    if 'single' in results and 'popt' in results['single']:
        plt.plot(taus_fit, results['single']['ymod'], '--', label='single fit')
    if 'double' in results and 'popt' in results['double']:
        plt.plot(taus_fit, results['double']['ymod'], ':', label='double fit')
    plt.xlabel("τ (ms)")
    plt.ylabel("ACF")
    plt.title("Temporal ACF (averaged over frequency)")
    plt.legend()
    if args.output:
        base, ext = os.path.splitext(args.output)
        out_time = base + "_time" + (ext if ext else ".png")
        plt.savefig(out_time, dpi=200)
        print(f"Saved temporal ACF plot to {out_time}")
    else:
        plt.show()

    # ---------------------------
    # Frequency ACF: compare single vs double Lorentzian
    # ---------------------------
    try:
        # Fit single Lorentzian (already attempted earlier; reuse if available)
        popt_f_single = None
        try:
            popt_f_single, pcov_f_single = curve_fit(lorentzian, lags_fit, acf_fit, p0=[1.0], maxfev=5000)
            ymod_fs = lorentzian(lags_fit, *popt_f_single)
            aic_fs, bic_fs, rss_fs = compute_aic_bic(acf_fit, ymod_fs, k=1)
            f_single = dict(popt=popt_f_single, pcov=pcov_f_single, aic=aic_fs, bic=bic_fs, rss=rss_fs, ymod=ymod_fs)
        except Exception as e:
            f_single = dict(error=str(e))

        # double Lorentzian: f*(L1) + (1-f)*(L2)
        def double_lorentz(delta_nu, f, d1, d2):
            return f / (1.0 + (delta_nu / d1) ** 2) + (1 - f) / (1.0 + (delta_nu / d2) ** 2)

        try:
            # initial guesses
            p0 = [0.5, popt_f_single[0] / 3.0 if popt_f_single is not None else 1.0, popt_f_single[0] * 3.0 if popt_f_single is not None else 10.0]
            bounds = ([0.0, 1e-9, 1e-9], [1.0, np.inf, np.inf])
            popt_fd, pcov_fd = curve_fit(double_lorentz, lags_fit, acf_fit, p0=p0, bounds=bounds, maxfev=20000)
            ymod_fd = double_lorentz(lags_fit, *popt_fd)
            aic_fd, bic_fd, rss_fd = compute_aic_bic(acf_fit, ymod_fd, k=3)
            f_double = dict(popt=popt_fd, pcov=pcov_fd, aic=aic_fd, bic=bic_fd, rss=rss_fd, ymod=ymod_fd)
        except Exception as e:
            f_double = dict(error=str(e))

        print("\n===== Frequency ACF Model Comparison =====")
        for key, r in (('single', f_single), ('double', f_double)):
            if 'error' in r:
                print(f"{key}: fit failed: {r['error']}")
            else:
                print(f"{key}: aic={r['aic']:.2f}, bic={r['bic']:.2f}, rss={r['rss']:.4e}")

        if 'aic' in f_single and 'aic' in f_double:
            pref = 'double' if f_double['aic'] < f_single['aic'] else 'single'
            print(f"Preferred (AIC) for frequency ACF: {pref}")

        # If double fit succeeded, report the two component bandwidths (d1, d2)
        if 'popt' in f_double:
            fval, d1, d2 = f_double['popt']
            try:
                perr = np.sqrt(np.diag(f_double['pcov']))
                f_err, d1_err, d2_err = perr
            except Exception:
                f_err = d1_err = d2_err = float('nan')
            print(f"\nDouble-fit bandwidths: d1 = {d1:.6f} ± {d1_err:.6f} MHz, d2 = {d2:.6f} ± {d2_err:.6f} MHz (mix f={fval:.3f} ± {f_err:.3f})")

        # plot frequency ACF fits
        plt.figure(figsize=(6, 4))
        plt.plot(lags_fit, acf_fit, label='Freq ACF')
        if 'popt' in f_single:
            plt.plot(lags_fit, f_single['ymod'], '--', label='single lorentz')
        if 'popt' in f_double:
            plt.plot(lags_fit, f_double['ymod'], ':', label='double lorentz')
        plt.xlabel('Δν (MHz)')
        plt.ylabel('ACF')
        plt.legend()
        if args.output:
            base, ext = os.path.splitext(args.output)
            out_freq = base + '_freq' + (ext if ext else '.png')
            plt.savefig(out_freq, dpi=200)
            print(f"Saved frequency ACF plot to {out_freq}")
        else:
            plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

