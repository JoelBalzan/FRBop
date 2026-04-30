import argparse
import os
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


# ---------------------------------------------------------------------------
# Burst-window detection
# ---------------------------------------------------------------------------

def find_burst_window(ts, peak_idx, smooth_win=5, threshold_sigma=3.0, pad=50, fallback_window=200):
    """Find contiguous burst window around peak using robust thresholding.

    Returns (tmin, tmax) inclusive-exclusive indices.
    """
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        ts_smooth = np.convolve(ts, kernel, mode="same")
    else:
        ts_smooth = ts

    med = np.median(ts_smooth)
    mad = np.median(np.abs(ts_smooth - med))
    sigma_est = 1.4826 * mad if mad > 0 else np.std(ts_smooth)
    thresh = med + threshold_sigma * sigma_est

    above = np.where(ts_smooth > thresh)[0]
    if above.size > 0:
        breaks = np.where(np.diff(above) > 1)[0]
        segments = []
        start = 0
        for b in breaks:
            segments.append(above[start : b + 1])
            start = b + 1
        segments.append(above[start:])

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

    tmin = max(0, peak_idx - fallback_window)
    tmax = peak_idx + fallback_window
    return tmin, tmax


# ---------------------------------------------------------------------------
# Simple normalised ACF (used for the Lorentzian fitting pipeline)
# ---------------------------------------------------------------------------

def autocorr(x):
    x = np.asarray(x)
    mean = np.nanmean(x)
    if mean == 0:
        return np.zeros_like(x)
    delta = (x - mean) / mean
    result = np.correlate(delta, delta, mode="full")
    acf = result[result.size // 2:]
    if acf[0] != 0:
        acf /= acf[0]
    return acf


# ---------------------------------------------------------------------------
# Macquart (2019) autocovariance estimator
# ---------------------------------------------------------------------------

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


def estimate_macquart_modulation_index(
    freq_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    corrected: bool = False,
    fit_max_lag_mhz: float | None = None,
    min_fit_points: int = 4,
) -> Tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Estimate m² and Δν_d from the mean-normalised spectral autocovariance.

    Implements Macquart et al. (2019):
        C(Δν) = ⟨[F(ν'+Δν) − F̄(ν')] · [F(ν') − F̄(ν')]⟩ / F̄²

    Returns (m2, delta_nu_d, lags_mhz, acov).
    """
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum)
    if np.count_nonzero(finite) < 4:
        return None, None, np.array([]), np.array([])

    freq_mhz = freq_mhz[finite]
    spectrum = spectrum[finite]

    if corrected:
        mean_model = _powerlaw_mean_spectrum(freq_mhz, spectrum, spectral_index=-1.5)
    else:
        mean_level = float(np.nanmean(spectrum))
        if not np.isfinite(mean_level) or mean_level <= 0:
            return None, None, np.array([]), np.array([])
        mean_model = np.full_like(spectrum, mean_level)

    positive = np.isfinite(mean_model) & (mean_model > 0)
    if np.count_nonzero(positive) < 4:
        return None, None, np.array([]), np.array([])

    freq_mhz = freq_mhz[positive]
    spectrum  = spectrum[positive]
    mean_model = mean_model[positive]

    # fractional deviation — do NOT subtract residual mean (would bias m²)
    frac = (spectrum - mean_model) / mean_model

    if np.count_nonzero(np.isfinite(frac)) < 4:
        return None, None, np.array([]), np.array([])

    # channel spacing — use abs() so ascending/descending both work
    if freq_mhz.size > 1:
        df = float(np.median(np.abs(np.diff(freq_mhz))))
    else:
        df = np.nan

    if not np.isfinite(df) or df <= 0:
        return None, None, np.array([]), np.array([])

    result = np.correlate(frac, frac, mode="full")
    acov   = result[result.size // 2:]
    counts = np.arange(frac.size, 0, -1, dtype=float)
    acov   = acov / counts
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
                idx      = int(candidate_idxs[0])
                prev_idx = idx - 1
                x0, x1  = float(lags_mhz[prev_idx]), float(lags_mhz[idx])
                y0, y1  = float(acov[prev_idx]),      float(acov[idx])
                if np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
                    t = (half_power - y0) / (y1 - y0)
                    delta_nu_d = float(x0 + t * (x1 - x0))
                elif np.isfinite(x1):
                    delta_nu_d = x1

    if m2 is None:
        fit_acov  = acov[fit_mask]
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
) -> float | None:
    """Extract a Macquart half-power Δν_d restricted to a specific lag window.

    Useful for isolating the decorrelation scale of one Lorentzian component
    when multiple components are present in the ACF.  The half-power crossing
    is searched only within [lag_lo_mhz, lag_hi_mhz].

    Returns Δν_d in MHz, or None if no crossing is found in the window.
    """
    _, _, lags, acov = estimate_macquart_modulation_index(
        freq_mhz, spectrum, corrected=corrected, fit_max_lag_mhz=lag_hi_mhz
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

    idx      = int(idxs[rel[0]])
    prev_idx = idx - 1
    if prev_idx < 0:
        return float(lags[idx])
    x0, x1 = float(lags[prev_idx]), float(lags[idx])
    y0, y1 = float(acov[prev_idx]), float(acov[idx])
    if y1 != y0:
        t = (half_power - y0) / (y1 - y0)
        return float(x0 + t * (x1 - x0))
    return float(x1)


# ---------------------------------------------------------------------------
# Lorentzian models
# ---------------------------------------------------------------------------

def lorentzian(delta_nu, delta_nu_d, A, C):
    return C + A / (1.0 + (delta_nu / delta_nu_d) ** 2)


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


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def compute_aic_bic(y, ymod, k):
    resid = y - ymod
    rss   = np.nansum(resid ** 2)
    n     = y.size
    if rss <= 0:
        rss = 1e-12
    aic = 2 * k + n * np.log(rss / n)
    bic = k * np.log(n) + n * np.log(rss / n)
    return aic, bic, rss


def build_fit_diagnostics(y, ymod, k):
    aic, bic, rss = compute_aic_bic(y, ymod, k)
    n    = y.size
    rmse = np.sqrt(rss / max(n, 1))
    tss  = np.nansum((y - np.nanmean(y)) ** 2)
    r2   = 1.0 - rss / tss if tss > 0 else np.nan
    aicc = aic + (2.0 * k * (k + 1)) / (n - k - 1) if n > (k + 1) else np.nan
    return dict(aic=aic, bic=bic, aicc=aicc, rss=rss, rmse=rmse, r2=r2)


def fit_with_restarts(model_fn, x, y, p0_list, bounds, maxfev=30000):
    best     = None
    best_rss = np.inf
    for p0 in p0_list:
        try:
            popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            ymod = model_fn(x, *popt)
            rss  = np.nansum((y - ymod) ** 2)
            if np.isfinite(rss) and rss < best_rss:
                best_rss = rss
                best     = (popt, pcov, ymod)
        except Exception:
            continue
    return best


def _decode_lorentzian_components(n_components, popt):
    """Return list of (weight, delta_nu_d) pairs for each Lorentzian component.

    Works for n_components in {1, 2, 3}.
    """
    if n_components == 1:
        d1, A, C = popt
        return [(1.0, d1)], A, C
    if n_components == 2:
        w1, d1, dd12, A, C = popt
        d2 = d1 + dd12
        return [(w1, d1), (1.0 - w1, d2)], A, C
    if n_components == 3:
        a, b, d1, dd12, dd23, A, C = popt
        d2 = d1 + dd12
        d3 = d2 + dd23
        w1 = a
        w2 = (1.0 - a) * b
        w3 = (1.0 - a) * (1.0 - b)
        return [(w1, d1), (w2, d2), (w3, d3)], A, C
    raise ValueError(f"Unsupported n_components={n_components}")


# ---------------------------------------------------------------------------
# Physical / astrophysical helpers
# ---------------------------------------------------------------------------

def scattered_gaussian(t, amp, mu, sigma, tau, offset):
    sigma = np.maximum(sigma, 1e-12)
    tau   = np.maximum(tau,   1e-12)
    arg      = (sigma / tau - (t - mu) / sigma) / np.sqrt(2.0)
    exponent = np.clip((sigma ** 2) / (2.0 * tau ** 2) - (t - mu) / tau, -100, 100)
    return offset + 0.5 * amp * np.exp(exponent) * erfc(arg)


def scale_scintillation_bandwidth(delta_nu_d_mhz, nu_from_mhz, nu_to_mhz, alpha=4.0):
    """Scale Δν_d ∝ ν^alpha (default α=4)."""
    if nu_from_mhz <= 0 or nu_to_mhz <= 0:
        raise ValueError("Frequencies must be > 0 for Δν_d scaling")
    return float(delta_nu_d_mhz) * (float(nu_to_mhz) / float(nu_from_mhz)) ** float(alpha)


def estimate_ds_kpc_from_redshift(z):
    return Distance(z=z, cosmology=WMAP5).to(u.kpc).value


def radec_to_galactic_deg(ra_hms, dec_dms):
    c = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    return float(c.galactic.l.deg), float(c.galactic.b.deg)


def get_cn2_profile(l_deg, b_deg, da_kpc, ndir=-1):
    import mwprop.nemod.NE2025 as _ne2025_mod
    ne2025 = _ne2025_mod.ne2025
    outdir = os.path.join(os.getcwd(), 'output_ne2025p')
    os.makedirs(outdir, exist_ok=True)
    ne2025(l_deg, b_deg, da_kpc, ndir,
           classic=False, dmd_only=False, do_analysis=True,
           plotting=False, verbose=False)
    prefix = "d2dm" if ndir < 0 else "dm2d"
    f25 = os.path.join(outdir, f'f25_{prefix}_ne_dsm_vs_s.txt')
    if not os.path.exists(f25):
        raise FileNotFoundError(f"NE2025 LoS profile not found at {f25}")
    data = np.loadtxt(f25, skiprows=3)
    s, ne, cn2 = data[:, 0], data[:, 4], data[:, 5]
    nonzero = np.where(ne != 0)[0]
    if nonzero.size > 0:
        indkeep = min(int(1.1 * nonzero[-1]), s.size)
        s, cn2  = s[:indkeep], cn2[:indkeep]
    return s, cn2


def estimate_lg_kpc_from_ne2025(ldeg, bdeg, da_kpc, max_dist_kpc=50.0):
    """Return (lg_peak_kpc, cn2_peak) using NE2025 Cn² peak distance."""
    s, cn2 = get_cn2_profile(ldeg, bdeg, da_kpc=max_dist_kpc)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, cn2, color='tab:blue', lw=1.2, label=r'$C_n^2$')
    ax.set_xlabel("Distance from observer (kpc)")
    ax.set_ylabel(r"$C_n^2$ (m$^{-20/3}$)")
    ax.set_title(f"NE2025  (l={ldeg:.2f}°, b={bdeg:.2f}°)")
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    if np.nansum(cn2) == 0.0:
        print("Warning: Cn² profile is all zeros — check coordinates and NE2025 model.")
        ax.legend(); plt.tight_layout(); plt.show()
        return float(s[0]), 0.0

    lg_peak  = float(s[np.argmax(cn2)])
    cn2_peak = float(np.max(cn2))
    ax.axvline(lg_peak, color='tab:green', lw=1.0, ls='--',
               label=f'L_g peak = {lg_peak:.3f} kpc')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
    return lg_peak, cn2_peak


# ---------------------------------------------------------------------------
# Two-screen distance calculation
# ---------------------------------------------------------------------------

def two_screen_estimate(
    delta_nu_d_mhz: float,
    tau_ms: float,
    nu_obs_mhz: float,
    redshift: float,
    ds_kpc: float,
    mg: float | None,
    lg_kpc: float | None,
    label: str = "",
) -> dict:
    """Compute LxLg (and optionally Lx) from the two-screen scattering model.

    Based on Macquart et al. (2019) Eqs. (2) and (4):
        LxLg ≤ Ds² · νdc / (2π · ν² · (1+z) · τscatt)    [Eq. 2, mg=1 limit]
        LxLg ≈ LxLg_upper / mg²                             [Eq. 4, mg<1]

    Returns a dict with keys: label, dnu, lxlg_upper, lxlg_partial, lx_upper, lx_partial.
    """
    nu_hz   = nu_obs_mhz * 1e6
    nu_dc_hz = delta_nu_d_mhz * 1e6
    t_s      = tau_ms * 1e-3

    geom           = (ds_kpc ** 2) / (2.0 * np.pi * (nu_hz ** 2) * (1.0 + redshift))
    lxlg_upper     = geom * (nu_dc_hz / t_s)
    lxlg_partial   = lxlg_upper / (mg ** 2) if (mg is not None and mg > 0) else None

    lx_upper  = lxlg_upper  / lg_kpc if (lg_kpc is not None and lg_kpc > 0) else None
    lx_partial = lxlg_partial / lg_kpc if (lxlg_partial is not None and lg_kpc is not None and lg_kpc > 0) else None

    return dict(
        label=label,
        dnu_mhz=delta_nu_d_mhz,
        c_val=2.0 * np.pi * nu_dc_hz * t_s,
        lxlg_upper=lxlg_upper,
        lxlg_partial=lxlg_partial,
        lx_upper=lx_upper,
        lx_partial=lx_partial,
    )


def print_two_screen_results(results: list[dict], tau_ms, nu_obs_mhz, redshift, mg, lg_kpc,
                              delta_nu_d_for_calc_source, lg_source):
    print("\n" + "=" * 60)
    print("TWO-SCREEN DISTANCE ESTIMATES")
    print("=" * 60)
    print(f"  τ_scatt         = {tau_ms:.4f} ms")
    print(f"  ν_obs           = {nu_obs_mhz:.3f} MHz")
    print(f"  z               = {redshift:.6f}")
    print(f"  Δν_d source     = {delta_nu_d_for_calc_source}")
    if mg is not None:
        print(f"  m_g             = {mg:.6f}")
    if lg_kpc is not None:
        print(f"  L_g             = {lg_kpc:.4f} kpc  ({lg_source})")

    for r in results:
        print(f"\n  --- {r['label']} ---")
        print(f"    Δν_d                   = {r['dnu_mhz']:.4f} MHz")
        print(f"    C = 2π ν_dc τ          = {r['c_val']:.3e}")
        print(f"    Eq.(2) L_x L_g ≤       {r['lxlg_upper']:.4e} kpc²  (m_g=1 limit)")
        if r['lxlg_partial'] is not None:
            if mg is not None and mg > 1.0:
                print(f"    Eq.(4) skipped: m_g={mg:.3f} > 1 is unphysical")
            else:
                print(f"    Eq.(4) L_x L_g ≈       {r['lxlg_partial']:.4e} kpc²  (m_g={mg:.4f})")
        if r['lx_upper'] is not None:
            print(f"    Eq.(2) L_x ≤           {r['lx_upper']:.4e} kpc")
        if r['lx_partial'] is not None:
            print(f"    Eq.(4) L_x ≈           {r['lx_partial']:.4e} kpc")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_macquart_diagnostics(
    freq_mhz, raw_spectrum,
    raw_result, corrected_result,
    output=None, fit_max_lag_mhz=None,
):
    m2_raw,  dnu_raw,  lags_raw,  acov_raw  = raw_result
    m2_corr, dnu_corr, lags_corr, acov_corr = corrected_result

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    freq_mhz    = np.asarray(freq_mhz,    dtype=float)
    raw_spectrum = np.asarray(raw_spectrum, dtype=float)
    finite      = np.isfinite(freq_mhz) & np.isfinite(raw_spectrum)

    ax0 = axs[0]
    if np.any(finite):
        ax0.plot(freq_mhz[finite], raw_spectrum[finite], color='0.25', lw=1.4, label='Raw spectrum')
        try:
            raw_mean = float(np.nanmean(raw_spectrum[finite]))
            if np.isfinite(raw_mean) and raw_mean > 0:
                ax0.axhline(raw_mean, color='tab:blue', lw=1.1, ls='--', label='Raw mean')
            cm = _powerlaw_mean_spectrum(freq_mhz[finite], raw_spectrum[finite])
            if np.all(np.isfinite(cm)):
                ax0.plot(freq_mhz[finite], cm, color='tab:orange', lw=1.2, ls='--',
                         label=r'$\nu^{-1.5}$ mean model')
        except Exception:
            pass
    ax0.set_title('Macquart spectrum reference')
    ax0.set_xlabel('Frequency (MHz)')
    ax0.set_ylabel('Flux / intensity')
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8)

    ax1 = axs[1]
    plotted = False
    for lags, acov, label, color, dnu in (
        (lags_raw,  acov_raw,  'Raw mean normalisation',  'tab:blue',   dnu_raw),
        (lags_corr, acov_corr, r'$\nu^{-1.5}$ corrected', 'tab:orange', dnu_corr),
    ):
        if lags.size == 0 or acov.size == 0:
            continue
        plotted = True
        mask = np.isfinite(lags) & np.isfinite(acov)
        if fit_max_lag_mhz is not None:
            mask &= lags <= float(fit_max_lag_mhz)
        if not np.any(mask):
            continue
        half = 0.5 * float(acov[0]) if np.isfinite(acov[0]) else np.nan
        ax1.plot(lags[mask], acov[mask], color=color, lw=1.6, label=f'{label} ACF')
        if np.isfinite(half):
            ax1.axhline(half, color=color, ls=':', lw=1.0, alpha=0.85)
        if dnu is not None and np.isfinite(dnu):
            ax1.axvline(dnu, color=color, ls='--', lw=1.1, alpha=0.85)
            ax1.text(dnu, half if np.isfinite(half) else 0.05,
                     f'  νdc≈{dnu:.3f} MHz', color=color, fontsize=8, va='bottom')
    ax1.axhline(0.0, color='0.5', lw=1.0)
    ax1.set_title('Macquart autocovariance')
    ax1.set_xlabel('Frequency lag Δν (MHz)')
    ax1.set_ylabel('Mean-normalised autocovariance')
    ax1.grid(alpha=0.25)
    if plotted:
        ax1.legend(fontsize=8)

    fig.suptitle('Macquart modulation-index diagnostics')
    plt.tight_layout()

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_macquart_diagnostics' + (ext if ext else '.png')
        plt.savefig(out, dpi=220)
        print(f"Saved Macquart diagnostics plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_lorentzian_diagnostics(
    lags_plot_sym, acf_plot_sym,
    lags_lorentz_fit, acf_lorentz_fit,
    fit_models, output=None,
):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    xabs = np.abs(lags_plot_sym)

    panel_cfg = [
        ("1-Component Lorentzian", fit_models[0], lorentzian,    'tab:blue',   axs[0, 0]),
        ("2-Component Lorentzian", fit_models[1], lorentzian_2c, 'tab:red',    axs[0, 1]),
        ("3-Component Lorentzian", fit_models[2], lorentzian_3c, 'tab:orange', axs[0, 2]),
    ]

    comp_colors = [
        ['tab:cyan',   'tab:purple', 'tab:brown'],
        ['tab:red',    'tab:pink',   'tab:brown'],
        ['tab:orange', 'tab:pink',   'tab:brown'],
    ]

    for i, (title, (name, result, _), model_fn, sum_color, ax) in enumerate(panel_cfg):
        ax.plot(lags_plot_sym, acf_plot_sym, color='k', lw=1.3, label='ACF data')
        if "popt" in result:
            n_comp = i + 1
            components, A, C = _decode_lorentzian_components(n_comp, result["popt"])
            sum_curve = model_fn(xabs, *result["popt"])
            ax.plot(lags_plot_sym, sum_curve, lw=1.6, color=sum_color, label=f'{n_comp}c sum')
            for j, (w, d) in enumerate(components):
                comp = A * w / (1.0 + (xabs / d) ** 2)
                ax.plot(lags_plot_sym, comp, ls='--', lw=1.1, alpha=0.9,
                        color=comp_colors[i][j], label=f'comp {j+1} (d={d:.3f} MHz)')
            ax.plot(lags_plot_sym, np.full_like(lags_plot_sym, C), ls=':', lw=1.0,
                    alpha=0.8, color='tab:gray', label='offset')
        ax.set_title(title)
        ax.set_xlabel("Δν (MHz)")
        ax.set_ylabel("ACF")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # Residuals
    ax1 = axs[1, 0]
    ax1.axhline(0.0, color='0.5', lw=1)
    for name, result, _ in fit_models:
        if "ymod" in result:
            ax1.plot(lags_lorentz_fit, acf_lorentz_fit - result["ymod"], lw=1.3, label=name)
    ax1.set_title("Residuals (positive lags)")
    ax1.set_xlabel("Δν (MHz)")
    ax1.set_ylabel("ACF residual")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    # ΔAIC bar
    valid_names = [n for n, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
    valid_aic   = [r["aic"] for _, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
    valid_bic   = [r["bic"] for _, r, _ in fit_models if "bic" in r and np.isfinite(r["bic"])]

    ax2 = axs[1, 1]
    if valid_names:
        daic = [a - min(valid_aic) for a in valid_aic]
        x    = np.arange(len(valid_names))
        ax2.bar(x, daic, color='tab:blue', alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels(valid_names, rotation=15)
        ax2.set_ylabel("ΔAIC"); ax2.grid(axis='y', alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "No valid AIC values", ha='center', va='center', transform=ax2.transAxes)

    ax3 = axs[1, 2]
    if valid_names:
        dbic = [b - min(valid_bic) for b in valid_bic]
        x    = np.arange(len(valid_names))
        ax3.bar(x, dbic, color='tab:green', alpha=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(valid_names, rotation=15)
        ax3.set_ylabel("ΔBIC"); ax3.grid(axis='y', alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "No valid BIC values", ha='center', va='center', transform=ax3.transAxes)

    plt.tight_layout()
    if output:
        base, ext = os.path.splitext(output)
        out = base + '_lorentzian_diagnostics' + (ext if ext else '.pdf')
        plt.savefig(out, dpi=220)
        print(f"Saved Lorentzian diagnostics plot to {out}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Peak selection
# ---------------------------------------------------------------------------

def select_peaks_manual(
    time_axis, profile_or_stokes, *,
    title='Click start/end bounds for each peak (close window when done)',
    x_label='Time (ms)', y_label='Flux', exclusive_end=True,
):
    ts = np.nanmean(profile_or_stokes, axis=0) if profile_or_stokes.ndim == 2 else profile_or_stokes
    return shared_select_peaks_manual(time_axis, ts, title=title, x_label=x_label,
                                      y_label=y_label, exclusive_end=exclusive_end)


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
    parser.add_argument("--threshold-sigma",   type=float, default=3.0)
    parser.add_argument("--pad",               type=int,   default=50)
    parser.add_argument("--fallback-window",   type=int,   default=200)
    parser.add_argument("--fit-max-lag",       type=float, default=8.0)
    parser.add_argument("--dnu-mhz",           type=float, default=None,
                        help="Provide Δν_d directly (skips Lorentzian fitting)")
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
    spectrum     = raw_spectrum.copy()
    spec_med     = np.median(spectrum)
    spec_mad     = np.median(np.abs(spectrum - spec_med))
    spec_sigma   = 1.4826 * spec_mad if spec_mad > 0 else np.std(spectrum)
    spectrum     = (spectrum - spec_med) / spec_sigma if spec_sigma > 0 else spectrum - spec_med

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
    delta_nu_d = float(args.dnu_mhz) if (args.dnu_mhz is not None and args.dnu_mhz > 0) else None

    if delta_nu_d is not None:
        # user-supplied: skip fitting
        fit_models   = []
        best_n_comp  = 1
        best_fit     = None
        d_base       = delta_nu_d
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

    if delta_nu_d is not None and args.dnu_mhz is not None:
        print(f"\n  Δν_d = {delta_nu_d:.4f} MHz  (provided via --dnu-mhz; fitting skipped)")

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


if __name__ == "__main__":
    main()