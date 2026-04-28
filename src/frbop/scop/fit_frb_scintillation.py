import argparse
import os
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.cosmology import WMAP5
from scipy.optimize import curve_fit
from scipy.special import erfc

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual


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
    x = np.asarray(x)
    mean = np.nanmean(x)

    if mean == 0:
        return np.zeros_like(x)

    delta = (x - mean) / mean  # fractional fluctuations

    result = np.correlate(delta, delta, mode="full")
    acf = result[result.size // 2:]

    # normalise
    if acf[0] != 0:
        acf /= acf[0]

    return acf


def compute_aic_bic(y, ymod, k):
    resid = y - ymod
    rss = np.nansum(resid ** 2)
    n = y.size
    if rss <= 0:
        rss = 1e-12
    aic = 2 * k + n * np.log(rss / n)
    bic = k * np.log(n) + n * np.log(rss / n)
    return aic, bic, rss


def select_peaks_manual(stokes_i: np.ndarray, time_ms: np.ndarray) -> List[Tuple[int, int]]:
    """Manually select peak bounds by clicking on the pulse profile."""
    time_series = np.nanmean(stokes_i, axis=0)
    return shared_select_peaks_manual(
        time_ms,
        time_series,
        title='Click start/end bounds for each peak (close window when done)',
        x_label='Time (ms)',
        y_label='Flux',
        exclusive_end=True,
    )


def lorentzian(delta_nu, delta_nu_d, A, C):
    return C + A / (1.0 + (delta_nu / delta_nu_d) ** 2)


def lorentzian_1c(delta_nu, d1, A, C):
    return C + A / (1.0 + (delta_nu / d1) ** 2)


def lorentzian_2c(delta_nu, w1, d1, dd12, A, C):
    d2 = d1 + dd12
    return C + A * (
        w1 / (1.0 + (delta_nu / d1) ** 2)
        + (1.0 - w1) / (1.0 + (delta_nu / d2) ** 2)
    )


def lorentzian_3c(delta_nu, a, b, d1, dd12, dd23, A, C):
    d2 = d1 + dd12
    d3 = d2 + dd23
    w1 = a
    w2 = (1.0 - a) * b
    w3 = (1.0 - a) * (1.0 - b)
    return C + A * (
        w1 / (1.0 + (delta_nu / d1) ** 2)
        + w2 / (1.0 + (delta_nu / d2) ** 2)
        + w3 / (1.0 + (delta_nu / d3) ** 2)
    )


def fit_with_restarts(model_fn, x, y, p0_list, bounds, maxfev=30000):
    best = None
    best_rss = np.inf
    for p0 in p0_list:
        try:
            popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            ymod = model_fn(x, *popt)
            rss = np.nansum((y - ymod) ** 2)
            if np.isfinite(rss) and rss < best_rss:
                best_rss = rss
                best = (popt, pcov, ymod)
        except Exception:
            continue
    return best


def build_fit_diagnostics(y, ymod, k):
    aic, bic, rss = compute_aic_bic(y, ymod, k)
    n = y.size
    rmse = np.sqrt(rss / max(n, 1))
    tss = np.nansum((y - np.nanmean(y)) ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    if n > (k + 1):
        aicc = aic + (2.0 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.nan
    return dict(aic=aic, bic=bic, aicc=aicc, rss=rss, rmse=rmse, r2=r2)


def scattered_gaussian(t, amp, mu, sigma, tau, offset):
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)
    arg = (sigma / tau - (t - mu) / sigma) / np.sqrt(2.0)
    expo = np.exp((sigma**2) / (2.0 * tau**2) - (t - mu) / tau)
    return offset + 0.5 * amp * expo * erfc(arg)


def estimate_ds_kpc_from_redshift(z):
    """Estimate angular-diameter distance D_s (kpc) from redshift in flat ΛCDM."""
    return Distance(z=z, cosmology=WMAP5).to(u.kpc).value


def estimate_lg_kpc_from_ne2025(ldeg, bdeg, da_kpc, max_dist_kpc=50.0):
    """
    Estimate effective Galactic screen distance L_g using NE2025 Cn² profile
    with geometric weighting appropriate for the two-screen scattering model.

    The effective distance is:

        L_g = Σ_s [ s · Cn²(s) · s·(1 - s/Da) ] / Σ_s [ Cn²(s) · s·(1 - s/Da) ]

    The geometric factor s·(1 - s/Da) is the two-screen scattering leverage:
    a screen at distance s from the observer (source at Da) contributes
    angular broadening ∝ s·(1 - s/Da), peaking at s = Da/2.

    Falls back to plain Cn²-weighted mean if da_kpc is None or <= 0.

    Parameters
    ----------
    ldeg, bdeg   : Galactic longitude/latitude (deg)
    da_kpc       : Angular-diameter distance to source (kpc)
    max_dist_kpc : Maximum LoS distance to sample (kpc)

    Returns
    -------
    lg_eff   : Geometrically-weighted effective screen distance (kpc)
    cn2_peak : Peak Cn² value along LoS (m^{-20/3})
    """
    s, cn2 = get_cn2_profile(ldeg, bdeg, da_kpc=max_dist_kpc)

    # Diagnostic plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, cn2, color='tab:blue', lw=1.2, label=r'$C_n^2$')
    ax.set_xlabel("Distance from observer (kpc)")
    ax.set_ylabel(r"$C_n^2$ (m$^{-20/3}$)")
    ax.set_title(f"NE2025  (l={ldeg:.2f}°, b={bdeg:.2f}°)")
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    cn2_total = np.nansum(cn2)
    if cn2_total == 0.0:
        print("Warning: Cn² profile is all zeros — check coordinates and NE2025 model flag.")
        lg_eff = float(s[0])
        ax.legend()
        plt.tight_layout()
        plt.show()
        return lg_eff, 0.0


    lg_peak = float(s[np.argmax(cn2)])

    ax.axvline(lg_peak, color='tab:green', lw=1.0, ls='--', label=f'L_g peak = {lg_peak:.3f} kpc')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

    print("\n===== NE2025 Galactic Screen Estimate =====")
    print(f"  L_g peak      ({'Cn² maximum':>20s}) = {lg_peak:.4f} kpc")
    print(f"  Cn²_peak                          = {np.max(cn2):.4e} m^{{-20/3}}")

    return lg_peak, float(np.max(cn2))


def radec_to_galactic_deg(ra_hms, dec_dms):
    """Convert ICRS RA/Dec strings to Galactic (l, b) in degrees."""
    c_icrs = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    c_gal = c_icrs.galactic
    return float(c_gal.l.deg), float(c_gal.b.deg)


def get_cn2_profile(l_deg, b_deg, da_kpc, ndir=-1):
    import os

    import mwprop.nemod.NE2025 as _ne2025_mod

    ne2025    = _ne2025_mod.ne2025
    sm_factor = _ne2025_mod.sm_factor

    outdir = os.path.join(os.getcwd(), 'output_ne2025p')
    os.makedirs(outdir, exist_ok=True)

    Dn, Dv, Du, Dd = ne2025(
        l_deg, b_deg, da_kpc, ndir,
        classic=False,
        dmd_only=False,
        do_analysis=True,
        plotting=False,
        verbose=False,
    )

    prefix = "d2dm" if ndir < 0 else "dm2d"
    f25 = os.path.join(outdir, f'f25_{prefix}_ne_dsm_vs_s.txt')
    if not os.path.exists(f25):
        raise FileNotFoundError(f"NE2025 LoS profile not found at {f25}")

    # Columns: d, x, y, z, ne, Cn2, w, c, v, t, dm, nea, Fa
    data = np.loadtxt(f25, skiprows=3)
    s   = data[:, 0]   # distance (kpc)
    ne  = data[:, 4]   # ne (cm^-3)
    cn2 = data[:, 5]   # Cn2 (m^-20/3) -- direct, no sm_factor needed

    nonzero = np.where(ne != 0)[0]
    if nonzero.size > 0:
        indkeep = min(int(1.1 * nonzero[-1]), s.size)
        s   = s[:indkeep]
        cn2 = cn2[:indkeep]

    return s, cn2


def main():
    parser = argparse.ArgumentParser(description="Fit FRB scintillation from dynamic spectrum files")
    parser.add_argument("ds", nargs="?", default="FRB_250607_htr_dsI.npy", help="Dynamic spectrum .npy file (nfreq x ntime)")
    parser.add_argument("--freq", default="FRB_250607_htr_freq.npy", help="Frequency axis .npy file (MHz)")
    parser.add_argument("--time", default="FRB_250607_htr_time.npy", help="Time axis .npy file (ms)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for time series (bins)")
    parser.add_argument("--manual-peaks", action="store_true", help="Manually select one or more on-pulse regions by clicking start/end bounds")
    parser.add_argument('--peak-indices', nargs='*', type=int, default=None, help='Manually specify peak indices as pairs: start1 end1 start2 end2 ...')
    parser.add_argument("--threshold-sigma", type=float, default=3.0, help="Threshold in robust sigmas for pulse gating")
    parser.add_argument("--pad", type=int, default=50, help="Padding added to detected pulse window (bins)")
    parser.add_argument("--fallback-window", type=int, default=200, help="Fallback half-window size if detection fails")
    parser.add_argument("--fit-max-lag", type=float, default=8.0, help="Max lag (MHz) to use in ACF fit")
    parser.add_argument("--dnu-mhz", type=float, default=None, help="Directly provide scintillation bandwidth Δν_d in MHz and skip frequency ACF Lorentzian fitting")
    parser.add_argument("--output", default=None, help="Optional output filename for plot (PNG)")
    parser.add_argument("--time-acf-model", choices=["exp", "gauss"], default="exp", help="Model for temporal ACF: exponential or gaussian")
    parser.add_argument("--fit-max-tau", type=float, default=100.0, help="Max time lag (ms) to use in temporal ACF fit")
    parser.add_argument("--tau-ms", type=float, default=None, help="Pulse broadening time t_scatt in ms (from pulse-shape fit; used for two-screen distance estimate)")
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
    ts = np.nanmean(ds, axis=0)
    if args.smooth > 1:
        ts_smooth = np.convolve(ts, np.ones(args.smooth) / args.smooth, mode="same")
    else:
        ts_smooth = ts

    peak_idx = int(np.argmax(ts_smooth))

    onpulse_mask = np.zeros(ntime, dtype=bool)
    if args.peak_indices is not None and len(args.peak_indices) > 0:
        clipped_regions = parse_peak_index_pairs(args.peak_indices, ntime)
        for start_idx, end_idx in clipped_regions:
            onpulse_mask[start_idx:end_idx] = True

        if np.any(onpulse_mask):
            print(f"Peak-index gating regions (start:end) = {clipped_regions}")
        else:
            print("Peak-index gating produced no valid samples; falling back to automatic window")
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
    else:
        on_idx = np.where(onpulse_mask)[0]
        tmin = int(on_idx[0])
        tmax = int(on_idx[-1] + 1)

    burst_ds = ds[:, onpulse_mask]

    # Estimate off-pulse mean robustly from complement of on-pulse mask
    off_pulse = ds[:, ~onpulse_mask] if np.any(~onpulse_mask) else np.empty((nfreq, 0))

    if off_pulse.size > 0:
        bandpass = np.nanmean(off_pulse, axis=1)
    else:
        # fallback: use low percentile across time to estimate baseline
        bandpass = np.percentile(ds, 10, axis=1)

    burst_ds = burst_ds - bandpass[:, None]

    # Frequency-integrated pulse profile and scattered-Gaussian fit for t_scatt
    pulse_profile = np.nanmean(burst_ds, axis=0)
    t_burst = time[onpulse_mask]
    t_scatt_fit_ms = args.tau_ms if args.tau_ms is not None else None
    t_scatt_fit_err_ms = None

    if t_scatt_fit_ms == None and pulse_profile.size >= 5:
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
                print(f"tau_ms = {t_scatt_fit_ms:.6f} ± {t_scatt_fit_err_ms:.6f} ms (scattered Gaussian)")
            else:
                print(f"tau_ms = {t_scatt_fit_ms:.6f} ms (scattered Gaussian)")
        except Exception as e:
            print("Pulse-profile t_scatt fit failed:", e)
    else:
        print(f"tau_ms = {t_scatt_fit_ms} ms")

    # Collapse in time → 1D spectrum
    raw_spectrum = np.nanmean(burst_ds, axis=1)
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

    # Build symmetric lags for plotting (explicitly exclude zero lag)
    mask_plot = (lags > 0) & (lags <= args.fit_max_lag)
    lags_plot = lags[mask_plot]
    acf_plot = acf[mask_plot]
    lags_plot_sym = np.concatenate((-lags_plot[::-1], lags_plot))
    acf_plot_sym = np.concatenate((acf_plot[::-1], acf_plot))

    # Limit fit range (explicitly exclude zero lag from Lorentzian fitting)
    mask_lorentz_fit = (lags > 0) & (lags < args.fit_max_lag) & np.isfinite(acf)
    lags_lorentz_fit = lags[mask_lorentz_fit]
    acf_lorentz_fit = acf[mask_lorentz_fit]

    delta_nu_d = None
    delta_nu_d_err = None
    A_fit = None
    C_fit = None
    if args.dnu_mhz is not None:
        if args.dnu_mhz <= 0:
            print("Provided --dnu-mhz must be > 0; falling back to fitted Δν_d")
        else:
            delta_nu_d = float(args.dnu_mhz)
            print("\n===== Scintillation Result =====")
            print(f"Δν_d = {delta_nu_d:.3f} MHz (provided via --dnu-mhz; fit skipped)")

    if delta_nu_d is None:
        try:
            p0 = [1.0, 1.0, 0.0]  # delta_nu_d, A, C
            bounds = ([1e-6, 0.0, -1.0], [np.inf, 2.0, 1.0])
            popt, pcov = curve_fit(
                lorentzian,
                lags_lorentz_fit,
                acf_lorentz_fit,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )

            delta_nu_d, A_fit, C_fit = popt
            delta_nu_d_err = np.sqrt(np.diag(pcov))[0]

            print("\n===== Scintillation Result =====")
            print(f"Δν_d = {delta_nu_d:.3f} ± {delta_nu_d_err:.3f} MHz")
            if A_fit is not None:
                print("\n===== Spectral Modulation from ACF =====")
                print(f"m = {A_fit**0.5:.6f}")
        except Exception as e:
            print("Fit failed:", e)
                # Time axis is assumed to be in milliseconds

    t_scatt_for_calc_ms = args.tau_ms if args.tau_ms is not None else t_scatt_fit_ms

    ds_kpc_for_calc = args.ds_kpc
    if ds_kpc_for_calc is None and args.redshift is not None:
        try:
            ds_kpc_for_calc = estimate_ds_kpc_from_redshift(float(args.redshift))
            print(
                f"Estimated D_s from redshift z={args.redshift:.6f}: "
                f"D_s={ds_kpc_for_calc:.6e} kpc "
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
                lg_kpc_for_calc, cn2_peak = estimate_lg_kpc_from_ne2025(
                    gl_for_lg,
                    gb_for_lg,
                    ds_kpc_for_calc,
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
            mean_spec = float(np.nanmean(finite_spec))
            std_spec = float(np.std(finite_spec))
            if mean_spec > 0:
                modulation_index = std_spec / mean_spec
                print("\n===== Spectral Modulation from Raw Spectrum =====")
                print(f"m = {modulation_index:.6f}")
            else:
                print("\n===== Spectral Modulation from Raw Spectrum =====")
                print("m could not be computed (mean spectrum <= 0 after baseline subtraction)")

    mg_for_calc = args.mg if args.mg is not None else modulation_index

    if delta_nu_d is not None and t_scatt_for_calc_ms is not None and args.redshift is not None and ds_kpc_for_calc is not None:
        if t_scatt_for_calc_ms <= 0:
            print("Two-screen estimate skipped: --tau-ms must be > 0")
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
            t_source = "--tau-ms" if args.tau_ms is not None else "pulse-profile fit"
            if mg is not None:
                mg_source = "--mg" if args.mg is not None else "measured m"
                print(f"Using ν_DC={delta_nu_d:.6f} MHz, tau_ms={t_scatt_for_calc_ms:.6f} ms ({t_source}), ν={nu_obs_mhz:.3f} MHz, z={z:.6f}, m_g={mg:.6f} ({mg_source})")
            else:
                print(f"Using ν_DC={delta_nu_d:.6f} MHz, tau_ms={t_scatt_for_calc_ms:.6f} ms ({t_source}), ν={nu_obs_mhz:.3f} MHz, z={z:.6f}")
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
                    lg_source = "--lg-kpc" if args.lg_kpc is not None else "NE2025"
                    print(f"Eq. 2: Using L_g={lg_kpc:.3f} kpc ({lg_source}) -> L_x <= {lx_upper_kpc:.3e} kpc")
                    if mg is not None and mg < 1.0 and lxlg_partial_kpc2 is not None:
                        lx_partial_kpc = lxlg_partial_kpc2 / lg_kpc
                        print(f"Eq. 4: Using L_g={lg_kpc:.3f} kpc ({lg_source}) -> L_x ≈ {lx_partial_kpc:.3e} kpc")

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(freq, spectrum)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Intensity")
    plt.title("Burst Spectrum")

    plt.subplot(1, 2, 2)
    plt.plot(lags_plot_sym, acf_plot_sym, label="ACF")
    if delta_nu_d is not None:
        if A_fit is not None and C_fit is not None:
            plt.plot(lags_plot_sym, lorentzian(lags_plot_sym, delta_nu_d, A_fit, C_fit), "--", label="Lorentzian fit")
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
    # Frequency ACF: compare 1 vs 2 vs 3 Lorentzian components
    # ---------------------------
    try:
        if lags_lorentz_fit.size < 8:
            raise RuntimeError("Insufficient positive-lag ACF points for 1/2/3-component comparison")

        if delta_nu_d is not None and np.isfinite(delta_nu_d) and delta_nu_d > 0:
            d_base = float(delta_nu_d)
        else:
            d_base = max(1e-3, args.fit_max_lag / 4.0)

        amp_guess = max(0.05, float(np.nanmax(acf_lorentz_fit) - np.nanmin(acf_lorentz_fit)))
        off_guess = float(np.nanmedian(acf_lorentz_fit[-max(3, int(0.2 * acf_lorentz_fit.size)):]))

        # 1-component model
        fit_1c = {}
        best_1c = fit_with_restarts(
            lorentzian_1c,
            lags_lorentz_fit,
            acf_lorentz_fit,
            p0_list=[
                [d_base, amp_guess, off_guess],
                [d_base * 0.5, amp_guess, off_guess],
                [d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([1e-6, 0.0, -1.5], [np.inf, 2.5, 1.5]),
            maxfev=30000,
        )
        if best_1c is None:
            fit_1c = dict(error="all initializations failed")
        else:
            popt, pcov, ymod = best_1c
            fit_1c = dict(popt=popt, pcov=pcov, ymod=ymod, **build_fit_diagnostics(acf_lorentz_fit, ymod, k=3))

        # 2-component model
        fit_2c = {}
        best_2c = fit_with_restarts(
            lorentzian_2c,
            lags_lorentz_fit,
            acf_lorentz_fit,
            p0_list=[
                [0.5, d_base * 0.3, d_base * 1.5, amp_guess, off_guess],
                [0.7, d_base * 0.2, d_base * 3.0, amp_guess, off_guess],
                [0.3, d_base * 0.6, d_base * 2.0, amp_guess, off_guess],
            ],
            bounds=([0.0, 1e-6, 1e-6, 0.0, -1.5], [1.0, np.inf, np.inf, 2.5, 1.5]),
            maxfev=50000,
        )
        if best_2c is None:
            fit_2c = dict(error="all initializations failed")
        else:
            popt, pcov, ymod = best_2c
            fit_2c = dict(popt=popt, pcov=pcov, ymod=ymod, **build_fit_diagnostics(acf_lorentz_fit, ymod, k=5))

        # 3-component model
        fit_3c = {}
        best_3c = fit_with_restarts(
            lorentzian_3c,
            lags_lorentz_fit,
            acf_lorentz_fit,
            p0_list=[
                [0.3, 0.5, d_base * 0.15, d_base * 0.7, d_base * 2.0, amp_guess, off_guess],
                [0.5, 0.5, d_base * 0.2, d_base * 1.0, d_base * 3.0, amp_guess, off_guess],
                [0.7, 0.4, d_base * 0.1, d_base * 0.8, d_base * 2.5, amp_guess, off_guess],
            ],
            bounds=(
                [0.0, 0.0, 1e-6, 1e-6, 1e-6, 0.0, -1.5],
                [1.0, 1.0, np.inf, np.inf, np.inf, 2.5, 1.5],
            ),
            maxfev=80000,
        )
        if best_3c is None:
            fit_3c = dict(error="all initializations failed")
        else:
            popt, pcov, ymod = best_3c
            fit_3c = dict(popt=popt, pcov=pcov, ymod=ymod, **build_fit_diagnostics(acf_lorentz_fit, ymod, k=7))

        fit_models = [("1-component", fit_1c, lorentzian_1c), ("2-component", fit_2c, lorentzian_2c), ("3-component", fit_3c, lorentzian_3c)]

        print("\n===== Frequency ACF Lorentzian Fit Diagnostics =====")
        print(f"{'Model':<14} {'AIC':>10} {'BIC':>10} {'AICc':>10} {'RSS':>12} {'RMSE':>10} {'R^2':>10}")
        for name, result, _ in fit_models:
            if "error" in result:
                print(f"{name:<14} fit failed: {result['error']}")
            else:
                print(
                    f"{name:<14} {result['aic']:>10.3f} {result['bic']:>10.3f} {result['aicc']:>10.3f} "
                    f"{result['rss']:>12.4e} {result['rmse']:>10.4e} {result['r2']:>10.4f}"
                )

        valid = [(name, result) for name, result, _ in fit_models if "aic" in result and np.isfinite(result["aic"])]
        if valid:
            best_aic_name, best_aic_result = min(valid, key=lambda item: item[1]["aic"])
            best_bic_name, best_bic_result = min(valid, key=lambda item: item[1]["bic"])
            print("\nModel preference summary:")
            print(f"Best by AIC : {best_aic_name} (AIC={best_aic_result['aic']:.3f})")
            print(f"Best by BIC : {best_bic_name} (BIC={best_bic_result['bic']:.3f})")
            if len(valid) > 1:
                sorted_aic = sorted(valid, key=lambda item: item[1]["aic"])
                runner_up = sorted_aic[1]
                print(f"ΔAIC to runner-up: {runner_up[1]['aic'] - sorted_aic[0][1]['aic']:.3f}")
        else:
            print("No valid Lorentzian component fits to compare.")

        # Report component scales and weights for interpretability
        if "popt" in fit_2c:
            w1, d1, dd12, amp2, off2 = fit_2c["popt"]
            d2 = d1 + dd12
            print(
                f"2-component parameters: d1={d1:.6f} MHz, d2={d2:.6f} MHz, "
                f"w1={w1:.3f}, w2={1.0 - w1:.3f}, A={amp2:.3f}, C={off2:.3f}"
            )
        if "popt" in fit_3c:
            a, b, d1, dd12, dd23, amp3, off3 = fit_3c["popt"]
            d2 = d1 + dd12
            d3 = d2 + dd23
            w1 = a
            w2 = (1.0 - a) * b
            w3 = (1.0 - a) * (1.0 - b)
            print(
                f"3-component parameters: d1={d1:.6f} MHz, d2={d2:.6f} MHz, d3={d3:.6f} MHz, "
                f"w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}, A={amp3:.3f}, C={off3:.3f}"
            )

        # Diagnostic comparison plot: separate fit panels + residuals + ΔAIC/ΔBIC bars
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))

        xabs = np.abs(lags_plot_sym)

        # Top row: separate model panels
        ax0 = axs[0, 0]
        ax0.plot(lags_plot_sym, acf_plot_sym, color='k', lw=1.3, label='ACF data')
        if "popt" in fit_1c:
            ax0.plot(lags_plot_sym, lorentzian_1c(xabs, *fit_1c["popt"]), lw=1.6, color='tab:blue', label='1c sum')
        ax0.set_title("1-Component Lorentzian")
        ax0.set_xlabel("Δν (MHz)")
        ax0.set_ylabel("ACF")
        ax0.grid(alpha=0.25)
        ax0.legend(fontsize=8)

        ax01 = axs[0, 1]
        ax01.plot(lags_plot_sym, acf_plot_sym, color='k', lw=1.3, label='ACF data')
        if "popt" in fit_2c:
            w1, d1, dd12, amp2, off2 = fit_2c["popt"]
            d2 = d1 + dd12
            sum2 = lorentzian_2c(xabs, *fit_2c["popt"])
            comp2_1 = amp2 * w1 / (1.0 + (xabs / d1) ** 2)
            comp2_2 = amp2 * (1.0 - w1) / (1.0 + (xabs / d2) ** 2)
            ax01.plot(lags_plot_sym, sum2, lw=1.6, color='tab:red', label='2c sum')
            ax01.plot(lags_plot_sym, comp2_1, ls='--', lw=1.1, alpha=0.9, color='tab:cyan', label='2c comp-1')
            ax01.plot(lags_plot_sym, comp2_2, ls='--', lw=1.1, alpha=0.9, color='tab:purple', label='2c comp-2')
            ax01.plot(lags_plot_sym, np.full_like(lags_plot_sym, off2), ls=':', lw=1.0, alpha=0.8, color='tab:gray', label='2c offset')
        ax01.set_title("2-Component Lorentzian")
        ax01.set_xlabel("Δν (MHz)")
        ax01.set_ylabel("ACF")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=8)

        ax02 = axs[0, 2]
        ax02.plot(lags_plot_sym, acf_plot_sym, color='k', lw=1.3, label='ACF data')
        if "popt" in fit_3c:
            a, b, d1, dd12, dd23, amp3, off3 = fit_3c["popt"]
            d2 = d1 + dd12
            d3 = d2 + dd23
            w1 = a
            w2 = (1.0 - a) * b
            w3 = (1.0 - a) * (1.0 - b)
            sum3 = lorentzian_3c(xabs, *fit_3c["popt"])
            comp3_1 = amp3 * w1 / (1.0 + (xabs / d1) ** 2)
            comp3_2 = amp3 * w2 / (1.0 + (xabs / d2) ** 2)
            comp3_3 = amp3 * w3 / (1.0 + (xabs / d3) ** 2)
            ax02.plot(lags_plot_sym, sum3, lw=1.6, color='tab:orange', label='3c sum')
            ax02.plot(lags_plot_sym, comp3_1, ls='-.', lw=1.1, alpha=0.9, color='tab:red', label='3c comp-1')
            ax02.plot(lags_plot_sym, comp3_2, ls='-.', lw=1.1, alpha=0.9, color='tab:pink', label='3c comp-2')
            ax02.plot(lags_plot_sym, comp3_3, ls='-.', lw=1.1, alpha=0.9, color='tab:brown', label='3c comp-3')
            ax02.plot(lags_plot_sym, np.full_like(lags_plot_sym, off3), ls=':', lw=1.0, alpha=0.8, color='tab:gray', label='3c offset')
        ax02.set_title("3-Component Lorentzian")
        ax02.set_xlabel("Δν (MHz)")
        ax02.set_ylabel("ACF")
        ax02.grid(alpha=0.25)
        ax02.legend(fontsize=8)

        # Bottom-left: residuals of all models
        ax1 = axs[1, 0]
        ax1.axhline(0.0, color='0.5', lw=1)
        for name, result, _ in fit_models:
            if "ymod" in result:
                resid = acf_lorentz_fit - result["ymod"]
                ax1.plot(lags_lorentz_fit, resid, lw=1.3, label=name)
        ax1.set_title("Residuals (positive lags)")
        ax1.set_xlabel("Δν (MHz)")
        ax1.set_ylabel("ACF residual")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        valid_names = [name for name, result, _ in fit_models if "aic" in result and np.isfinite(result["aic"])]
        valid_aic = [result["aic"] for _, result, _ in fit_models if "aic" in result and np.isfinite(result["aic"])]
        valid_bic = [result["bic"] for _, result, _ in fit_models if "bic" in result and np.isfinite(result["bic"])]

        ax2 = axs[1, 1]
        if valid_names:
            min_aic = float(np.min(valid_aic))
            delta_aic = [a - min_aic for a in valid_aic]
            x = np.arange(len(valid_names))
            ax2.bar(x, delta_aic, color='tab:blue', alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_names, rotation=15)
            #ax2.set_title("ΔAIC (lower is better)")
            ax2.set_ylabel("ΔAIC")
            ax2.grid(axis='y', alpha=0.25)
        else:
            ax2.text(0.5, 0.5, "No valid AIC values", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_axis_off()

        ax3 = axs[1, 2]
        if valid_names:
            min_bic = float(np.min(valid_bic))
            delta_bic = [b - min_bic for b in valid_bic]
            x = np.arange(len(valid_names))
            ax3.bar(x, delta_bic, color='tab:green', alpha=0.8)
            ax3.set_xticks(x)
            ax3.set_xticklabels(valid_names, rotation=15)
            #ax3.set_title("ΔBIC (lower is better)")
            ax3.set_ylabel("ΔBIC")
            ax3.grid(axis='y', alpha=0.25)
        else:
            ax3.text(0.5, 0.5, "No valid BIC values", ha='center', va='center', transform=ax3.transAxes)
            ax3.set_axis_off()

        plt.tight_layout()
        if args.output:
            base, ext = os.path.splitext(args.output)
            out_diag = base + '_lorentzian_diagnostics' + (ext if ext else '.pdf')
            plt.savefig(out_diag, dpi=220)
            print(f"Saved Lorentzian diagnostics plot to {out_diag}")
        else:
            plt.show()
    except Exception as e:
        print("Frequency multi-fit diagnostics failed:", e)


if __name__ == "__main__":
    main()