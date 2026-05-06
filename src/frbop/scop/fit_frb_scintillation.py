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
from frbop.utils.peaks import select_frequency_bands_manual as shared_select_frequency_bands_manual


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
# NE2025 scattering / scintillation predictions
# ---------------------------------------------------------------------------

def ne2025_scattering_prediction(
    s_kpc: np.ndarray,
    cn2: np.ndarray,
    lg_kpc: float,
    ds_kpc: float,
    nu_ref_mhz: float,
    v_iss_km_s: float = 100.0,
) -> dict:
    """Predict τ_scatt, Δν_d, and t_scint from an NE2025 Cn² profile.

    Uses the standard thin-screen scattering measure formalism:

        SM  = ∫ Cn²(s) ds                              [kpc · m^{-20/3}]

    The scattering measure is converted to SI:
        SM_si = SM * kpc_to_m * 1e-20   [m^{-17/3}]  ← Cn² in units of 10^{-20} m^{-20/3}

    Wait — NE2025 Cn² column is already in m^{-20/3} (not ×10^{-20}), so:
        SM_si = SM_kpc_m * kpc_to_m                   [m^{-17/3}]

    Pulse broadening (Cordes & Lazio 2003, eq. 4):
        τ_scatt = A_tau * SM_si * (ν/GHz)^{-4}        [s]

    where A_tau = λ^4 / (2π)^3 * r_e^2  ≈  can be absorbed into the
    Cordes & Lazio empirical constant.  We use the compact form:

        τ_scatt [ms] = 0.154 * SM_si_kpc * (ν_GHz)^{-4}   [Cordes & Lazio 2003]

    where SM_si_kpc is SM integrated in kpc with Cn² in m^{-20/3}.
    The exact pre-factor from C&L 2003 eq.(4) with the standard constants is:

        τ_scatt [s] = (r_e^2 * c / (2π)) * SM_si * eff_geom * λ^4

    but NE2025 exposes this through sm_factor; here we use the direct
    numeric constant that C&L derive:

        τ_scatt [ms] = 0.154 * SM [kpc m^{-20/3}] * (ν [GHz])^{-4}

    (This matches eq.(4) of Cordes & Lazio 2003 to within the geometric factor
    for a uniform screen.  The geometric factor D_eff = L_g*(D_s-L_g)/D_s
    is applied separately.)

    Decorrelation bandwidth:
        Δν_d [MHz] = 1 / (2π * τ_scatt [s]) * 1e-6

    Scintillation timescale (geometric):
        t_scint [s] = sqrt(D_eff_m * c / (2π * ν_hz)) / V_iss_m_s   (Lorentzian scint)
        equivalently: t_scint = (1/V_iss) * sqrt(Δν_d / ν) * D_eff / c  ... use standard form:
        t_scint [s] = r_diff / V_iss  where r_diff = sqrt(c * D_eff / (2π * ν * (2π*B_dc)))
                    ≈ sqrt(c * D_eff / (2π^2 * ν * Δν_d)) / V_iss  ... but simplest is:

    We use the Macquart & Koay (2013) / Taylor & Cordes standard:
        t_scint = (D_eff / V_iss) * sqrt(Δν_d / ν)  × geometric_factor

    The most physically transparent form (Rickett 1977):
        r_diff  = sqrt( c * D_eff_m / (2π * ν_hz) ) × (1/sqrt(2π * τ_scatt_s * ν_hz))
                but this double-counts; use the clean form:

        t_scint [s] = sqrt(c * D_eff_m / (2π * ν_hz)) / V_iss_m_s   ← diffractive scale / V

    Parameters
    ----------
    s_kpc      : distance axis of Cn² profile (kpc)
    cn2        : Cn² values (m^{-20/3})
    lg_kpc     : effective Galactic screen distance (kpc)
    ds_kpc     : source angular-diameter distance (kpc)
    nu_ref_mhz : reference frequency for predictions (MHz)
    v_iss_km_s : assumed ISS transverse velocity (km/s, default 100)

    Returns
    -------
    dict with keys: SM_kpc, tau_scatt_ms, delta_nu_d_mhz, t_scint_s, D_eff_kpc, nu_ref_mhz
    """
    kpc_to_m = 3.085677581e19   # 1 kpc in metres
    c_m_s    = 2.99792458e8     # speed of light m/s

    # Numerical SM = ∫ Cn²(s) ds  [kpc · m^{-20/3}]
    # Use trapezoidal integration over the provided profile.
    s_arr   = np.asarray(s_kpc,  dtype=float)
    cn2_arr = np.asarray(cn2,    dtype=float)
    finite  = np.isfinite(s_arr) & np.isfinite(cn2_arr) & (cn2_arr >= 0)
    if not np.any(finite):
        raise ValueError("Cn² profile has no finite non-negative values")
    SM_kpc = float(np.trapezoid(cn2_arr[finite], s_arr[finite]))   # kpc · m^{-20/3}

    # Convert to SI: m^{-17/3}
    SM_si = SM_kpc * kpc_to_m   # [m^{-17/3}]

    # Effective geometric distance for a thin screen at L_g, source at D_s
    # D_eff = L_g * (D_s - L_g) / D_s  [kpc]
    if ds_kpc > lg_kpc > 0:
        D_eff_kpc = lg_kpc * (ds_kpc - lg_kpc) / ds_kpc
    else:
        D_eff_kpc = lg_kpc   # fallback: treat as observer-side screen
    D_eff_m = D_eff_kpc * kpc_to_m

    nu_hz  = nu_ref_mhz * 1e6
    nu_ghz = nu_ref_mhz / 1e3
    lam_m  = c_m_s / nu_hz   # wavelength

    # Pulse broadening: Cordes & Lazio (2003) eq.(4) numeric constant
    # τ_scatt [s] = (r_e^2 / (2π)) * λ^4 * SM_si  (per unit D_eff already in SM for uniform screen)
    # For a thin screen the full expression is:
    # τ_scatt = (r_e^2 * λ^4 * SM_si * D_eff_m) / (2π * c_m_s)
    # r_e = 2.8179403e-15 m (classical electron radius)
    r_e = 2.8179403e-15
    tau_scatt_s = (r_e**2 * lam_m**4 * SM_si * D_eff_m) / (2.0 * np.pi * c_m_s)
    tau_scatt_ms = tau_scatt_s * 1e3

    # Decorrelation bandwidth  Δν_d = 1 / (2π τ_scatt)
    if tau_scatt_s > 0:
        delta_nu_d_hz  = 1.0 / (2.0 * np.pi * tau_scatt_s)
        delta_nu_d_mhz = delta_nu_d_hz / 1e6
    else:
        delta_nu_d_mhz = np.nan

    # Diffractive scintillation timescale
    # r_diff = sqrt(λ * D_eff_m / (2π))  ... the Fresnel scale modified by scattering:
    # actually r_diff = sqrt(c * D_eff_m / (2π * ν * (2π * ν * τ_scatt)))
    #                 = sqrt(c / (2π * ν)) * sqrt(D_eff_m) / sqrt(2π * ν * τ_scatt)
    # The clean standard form (e.g. Cordes & Rickett 1998):
    #   r_diff = sqrt(c * D_eff_m / (2π^2 * ν^2 * τ_scatt))  ← but this has τ already
    # Simplest equivalent from bandwidth:
    #   r_diff = sqrt(c * D_eff_m * Δν_d) / ν   ... dimension check: m^{1/2}·Hz^{1/2}/Hz → m ✓
    # We use: t_scint = r_diff / V_iss
    V_iss_m_s = v_iss_km_s * 1e3
    if np.isfinite(delta_nu_d_mhz) and delta_nu_d_mhz > 0:
        delta_nu_d_hz_val = delta_nu_d_mhz * 1e6
        r_diff_m  = np.sqrt(c_m_s * D_eff_m * delta_nu_d_hz_val) / nu_hz
        t_scint_s = r_diff_m / V_iss_m_s
    else:
        r_diff_m  = np.nan
        t_scint_s = np.nan

    return dict(
        SM_kpc=SM_kpc,
        SM_si=SM_si,
        D_eff_kpc=D_eff_kpc,
        tau_scatt_ms=tau_scatt_ms,
        delta_nu_d_mhz=delta_nu_d_mhz,
        t_scint_s=t_scint_s,
        r_diff_m=r_diff_m,
        nu_ref_mhz=nu_ref_mhz,
        v_iss_km_s=v_iss_km_s,
    )


def print_ne2025_scattering_prediction(pred: dict, lg_kpc: float, ds_kpc: float) -> None:
    print("\n  NE2025 predicted scattering (Galactic screen):")
    print(f"    Reference frequency    = {pred['nu_ref_mhz']:.3f} MHz")
    print(f"    SM (numerical ∫Cn²ds)  = {pred['SM_kpc']:.4e} kpc m^{{-20/3}}")
    print(f"    SM (SI)                = {pred['SM_si']:.4e} m^{{-17/3}}")
    print(f"    L_g                    = {lg_kpc:.4f} kpc")
    print(f"    D_s                    = {ds_kpc:.4e} kpc")
    print(f"    D_eff                  = {pred['D_eff_kpc']:.4f} kpc")
    print(f"    τ_scatt (predicted)    = {pred['tau_scatt_ms']:.4e} ms")
    print(f"    Δν_d   (predicted)     = {pred['delta_nu_d_mhz']:.4e} MHz")
    if np.isfinite(pred['t_scint_s']):
        print(f"    t_scint (predicted)    = {pred['t_scint_s']:.4e} s  "
              f"(V_ISS = {pred['v_iss_km_s']:.0f} km/s assumed)")
    else:
        print(f"    t_scint (predicted)    = N/A")


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
# Frequency-band scintillation measurement
# ---------------------------------------------------------------------------

def select_frequency_bands_manual(
    freq_axis, spectrum, *,
    title='Click start/end bounds for each frequency band (close window when done)',
    x_label='Frequency (MHz)', y_label='Flux', exclusive_end=True,
):
    return shared_select_frequency_bands_manual(
        freq_axis, spectrum, title=title, x_label=x_label,
        y_label=y_label, exclusive_end=exclusive_end,
    )


def split_frequency_bands_equal(freq_axis: np.ndarray, n_bands: int) -> list[tuple[int, int]]:
    """Split a 1D frequency axis into n approximately equal contiguous bands."""
    freq_axis = np.asarray(freq_axis, dtype=float)
    if freq_axis.ndim != 1:
        raise ValueError(f"freq_axis must be 1D, got shape={freq_axis.shape}")
    if n_bands <= 0:
        raise ValueError("n_bands must be > 0")
    if freq_axis.size == 0:
        return []

    n_bands = min(int(n_bands), int(freq_axis.size))
    regions: list[tuple[int, int]] = []
    for chunk in np.array_split(np.arange(freq_axis.size), n_bands):
        if chunk.size == 0:
            continue
        regions.append((int(chunk[0]), int(chunk[-1]) + 1))
    return regions


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
) -> list[dict]:
    """Measure Δν_d in each sub-band by fitting a Lorentzian to the normalised ACF.

    Uses the same autocorr + fit_with_restarts(lorentzian) pipeline as the
    full-band analysis so results are directly comparable.  The Macquart
    half-power estimator is deliberately avoided here: it is unreliable on
    narrow sub-bands where the ACF is noisy and the half-power crossing is
    poorly defined.

    Each band's fit-max-lag is capped at half the band width so we never try
    to constrain lags that don't exist in the data.  If off_pulse is provided,
    a noise-informed Δν_d uncertainty is estimated and returned in dnu_err_mhz.
    """
    results: list[dict] = []
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum  = np.asarray(spectrum,  dtype=float)
    max_bound_fraction = 0.9
    max_rel_err = 0.75
    min_r2 = 0.5

    for band_idx, (start_idx, stop_idx) in enumerate(band_regions, start=1):
        sub_freq = freq_mhz[start_idx:stop_idx]
        sub_spec = spectrum[start_idx:stop_idx]
        finite   = np.isfinite(sub_freq) & np.isfinite(sub_spec)
        if np.count_nonzero(finite) < 8:
            print(f"  Band {band_idx}: skipped — fewer than 8 finite channels")
            continue

        sub_freq = sub_freq[finite]
        sub_spec = sub_spec[finite]
        band_width_mhz = float(np.nanmax(sub_freq) - np.nanmin(sub_freq))
        df_band        = float(np.median(np.abs(np.diff(sub_freq))))

        # Cap the lag range to half the band width (can't measure lags > band)
        band_fit_max_lag = min(float(fit_max_lag_mhz), 0.5 * band_width_mhz)
        if band_fit_max_lag < df_band * 2:
            print(f"  Band {band_idx}: skipped — fit-max-lag ({band_fit_max_lag:.3f} MHz) "
                  f"too small for channel spacing ({df_band:.3f} MHz)")
            continue

        # Normalise the sub-band spectrum the same way as the full-band pipeline
        sub_med   = np.median(sub_spec)
        sub_mad   = np.median(np.abs(sub_spec - sub_med))
        sub_sigma = 1.4826 * sub_mad if sub_mad > 0 else np.std(sub_spec)
        sub_norm  = (sub_spec - sub_med) / sub_sigma if sub_sigma > 0 else sub_spec - sub_med

        # Build the normalised ACF
        acf_band = autocorr(sub_norm)
        if acf_band[0] != 0:
            acf_band /= acf_band[0]
        lags_band = np.arange(len(acf_band)) * df_band   # MHz

        # Restrict to positive lags within the allowed range (exclude zero lag)
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
                [d_guess,                   amp_guess, off_guess],
                [d_guess * 0.4,             amp_guess, off_guess],
                [d_guess * 2.0,             amp_guess, off_guess],
                [band_fit_max_lag * 0.1,    amp_guess, off_guess],
            ],
            bounds=([df_band * 0.5, 0.0, -1.5], [band_fit_max_lag, 2.5, 1.5]),
            maxfev=20000,
        )

        if best is None:
            print(f"  Band {band_idx}: Lorentzian fit failed (all initialisations diverged)")
            continue

        popt, pcov, ymod = best
        dnu_fit, A_fit, C_fit = popt
        try:
            dnu_err_fit = float(np.sqrt(np.diag(pcov))[0])
        except Exception:
            dnu_err_fit = np.nan

        # Noise-informed fractional uncertainty using off-pulse data, if available.
        dnu_err_noise = np.nan
        n_eff = np.nan
        noise_ratio = np.nan
        if off_pulse is not None and off_pulse.size > 0:
            off_band = off_pulse[start_idx:stop_idx]
            if off_band.ndim == 2 and off_band.size > 0:
                channel_med = np.nanmedian(off_band, axis=1, keepdims=True)
                off_band = off_band - channel_med
                med2 = np.nanmedian(off_band, axis=1, keepdims=True)
                mad = np.nanmedian(np.abs(off_band - med2), axis=1)
                sigma_chan = 1.4826 * mad
                bad = ~np.isfinite(sigma_chan) | (sigma_chan <= 0)
                if np.any(bad):
                    sigma_chan[bad] = np.nanstd(off_band[bad], axis=1)
                sigma_n = np.nanmedian(sigma_chan[np.isfinite(sigma_chan) & (sigma_chan > 0)])

                mean_signal = np.nanmean(sub_spec)
                if not np.isfinite(mean_signal) or mean_signal <= 0:
                    mean_signal = np.nanmean(np.abs(sub_spec))

                n_chan = max(1, int(sub_spec.size))
                n_eff = band_width_mhz / max(dnu_fit, df_band)
                n_eff = float(np.clip(n_eff, 1.0, float(n_chan)))

                if np.isfinite(sigma_n) and sigma_n > 0 and np.isfinite(mean_signal) and mean_signal > 0:
                    noise_ratio = float(sigma_n / mean_signal)
                    frac_err = (noise_ratio ** 2) / np.sqrt(n_eff)
                    dnu_err_noise = float(frac_err * dnu_fit)

        dnu_err = dnu_err_noise if np.isfinite(dnu_err_noise) and dnu_err_noise > 0 else dnu_err_fit

        if dnu_fit <= 0 or not np.isfinite(dnu_fit):
            print(f"  Band {band_idx}: Lorentzian fit returned non-physical "
                  f"Δν_d = {dnu_fit:.4f} MHz; skipping")
            continue

        diag = build_fit_diagnostics(acf_fit, ymod, k=3)
        rel_err = dnu_err_fit / dnu_fit if np.isfinite(dnu_err_fit) and dnu_fit > 0 else np.inf

        if dnu_fit >= max_bound_fraction * band_fit_max_lag:
            print(f"  Band {band_idx}: skipped — Δν_d = {dnu_fit:.4f} MHz is too close to the fit upper bound "
                  f"({band_fit_max_lag:.4f} MHz)")
            continue
        if not np.isfinite(rel_err) or rel_err > max_rel_err:
            print(f"  Band {band_idx}: skipped — fractional uncertainty {rel_err:.2f} exceeds the limit "
                  f"({max_rel_err:.2f})")
            continue
        if not np.isfinite(diag["r2"]) or diag["r2"] < min_r2:
            print(f"  Band {band_idx}: skipped — poor fit quality R² = {diag['r2']:.4f} < {min_r2:.2f}")
            continue

        results.append(dict(
            band_idx=band_idx,
            start_idx=int(start_idx),
            stop_idx=int(stop_idx),
            center_mhz=float(np.nanmean(sub_freq)),
            band_width_mhz=band_width_mhz,
            dnu_mhz=float(dnu_fit),
            dnu_err_mhz=dnu_err,
            dnu_err_fit_mhz=dnu_err_fit,
            dnu_err_noise_mhz=dnu_err_noise,
            n_eff=n_eff,
            noise_ratio=noise_ratio,
            A_fit=float(A_fit),
            C_fit=float(C_fit),
            r2=diag["r2"],
            rmse=diag["rmse"],
            # store ACF arrays for per-band plot
            _lags=lags_band,
            _acf=acf_band,
            _lags_fit=lags_fit,
            _ymod=ymod,
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
    band_dnu_mhz     = np.asarray(band_dnu_mhz,     dtype=float)
    if band_dnu_err_mhz is not None:
        band_dnu_err_mhz = np.asarray(band_dnu_err_mhz, dtype=float)

    finite = (np.isfinite(band_centers_mhz) & np.isfinite(band_dnu_mhz)
              & (band_centers_mhz > 0) & (band_dnu_mhz > 0))
    if band_dnu_err_mhz is not None:
        finite = finite & np.isfinite(band_dnu_err_mhz) & (band_dnu_err_mhz > 0)
    
    if np.count_nonzero(finite) < 2:
        raise ValueError("Need at least two valid bands to fit a scintillation bandwidth power law")

    x = np.log(band_centers_mhz[finite])
    y = np.log(band_dnu_mhz[finite])
    
    # Use weighted polyfit if errors provided
    if band_dnu_err_mhz is not None:
        # Propagate error in dnu to log-space: d(log dnu) ≈ (d dnu) / dnu
        err_log_dnu = band_dnu_err_mhz[finite] / band_dnu_mhz[finite]
        # Weights are inverse variance in log-space
        weights = 1.0 / (err_log_dnu ** 2)
        popt, pcov = np.polyfit(x, y, 1, w=weights, cov=True)
    else:
        popt, pcov = np.polyfit(x, y, 1, cov=True)
    
    alpha_fit, log_norm_fit = popt
    alpha_err = float(np.sqrt(np.diag(pcov)[0])) if np.isfinite(pcov[0, 0]) else np.nan
    log_norm_44 = float(np.mean(y - comparison_alpha * x))

    ref = float(reference_freq_mhz)
    if ref <= 0:
        raise ValueError("reference_freq_mhz must be > 0")

    dnu_ref_fit  = float(np.exp(log_norm_fit  + alpha_fit            * np.log(ref)))
    dnu_ref_44   = float(np.exp(log_norm_44   + comparison_alpha     * np.log(ref)))
    model_fit    = np.exp(log_norm_fit  + alpha_fit        * x)
    model_44     = np.exp(log_norm_44   + comparison_alpha * x)
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


def plot_scintillation_band_power_law(
    band_results: list[dict],
    power_law_fit: dict | None,
    *,
    output=None,
    fit_max_lag_mhz: float | None = None,
):
    # Top row: power-law plot + residual ratio
    # Bottom row: per-band ACF + Lorentzian fits
    n_bands = len(band_results)
    fig = plt.figure(figsize=(max(13.0, 4.5 * n_bands), 10.0))
    gs_top = fig.add_gridspec(1, 2, top=0.95, bottom=0.55, hspace=0.35, wspace=0.3)
    gs_bot = fig.add_gridspec(1, n_bands, top=0.45, bottom=0.07, hspace=0.35, wspace=0.35)
 
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax1 = fig.add_subplot(gs_top[0, 1])
 
    centers = np.array([r["center_mhz"] for r in band_results], dtype=float)
    dnus    = np.array([r["dnu_mhz"]    for r in band_results], dtype=float)
    errs    = np.array([r.get("dnu_err_mhz", np.nan) for r in band_results], dtype=float)
    finite  = np.isfinite(centers) & np.isfinite(dnus) & (centers > 0) & (dnus > 0)
    centers, dnus, errs = centers[finite], dnus[finite], errs[finite]
 
    if centers.size > 0:
        order   = np.argsort(centers)
        centers, dnus, errs = centers[order], dnus[order], errs[order]
        has_err = np.isfinite(errs)
        if np.any(has_err):
            ax0.errorbar(centers[has_err], dnus[has_err], yerr=errs[has_err],
                         fmt='o', color='tab:blue', capsize=3, label='Measured bands')
            if np.any(~has_err):
                ax0.loglog(centers[~has_err], dnus[~has_err], 'o', color='tab:blue')
        else:
            ax0.loglog(centers, dnus, 'o', color='tab:blue', label='Measured bands')
        ax0.set_xscale('log'); ax0.set_yscale('log')
 
    if power_law_fit is not None and centers.size > 0:
        x_fit  = np.asarray(power_law_fit["fit_freq_mhz"], dtype=float)
        grid   = np.logspace(np.log10(float(np.nanmin(x_fit)) * 0.9),
                             np.log10(float(np.nanmax(x_fit)) * 1.1), 256)
        ax0.loglog(grid,
                   np.exp(power_law_fit["log_norm_fit"] + power_law_fit["alpha_fit"] * np.log(grid)),
                   color='tab:orange', lw=1.8,
                   label=f"Fit: α={power_law_fit['alpha_fit']:.2f}")
        ax0.loglog(grid,
                   np.exp(power_law_fit["log_norm_44"] + power_law_fit["comparison_alpha"] * np.log(grid)),
                   color='tab:green', lw=1.5, ls='--', label='Kolmogorov α=4.4')
        ax0.axvline(power_law_fit["reference_freq_mhz"], color='0.45', lw=1.0, ls=':',
                    label=f"ν_c={power_law_fit['reference_freq_mhz']:.1f} MHz")
 
        fit_at  = np.exp(power_law_fit["log_norm_fit"]  + power_law_fit["alpha_fit"]        * np.log(centers))
        fit_44  = np.exp(power_law_fit["log_norm_44"]   + power_law_fit["comparison_alpha"] * np.log(centers))
        ax1.axhline(1.0, color='0.5', lw=1.0, ls=':')
        ax1.plot(centers, dnus / fit_at, 'o-',  color='tab:orange', lw=1.5, label='Data / fit')
        ax1.plot(centers, dnus / fit_44, 's--', color='tab:green',  lw=1.2, label='Data / Kolmogorov')
        ax1.axvline(power_law_fit["reference_freq_mhz"], color='0.45', lw=1.0, ls=':')
        ax1.set_ylim(0.2, 2.5)
 
    ax0.set_xlabel('Frequency (MHz)'); ax0.set_ylabel(r'$\Delta\nu_d$ (MHz)')
    ax0.set_title('Scintillation bandwidth vs frequency'); ax0.grid(alpha=0.25, which='both')
    ax0.legend(fontsize=8)
    ax1.set_xlabel('Frequency (MHz)'); ax1.set_ylabel('Observed / model')
    ax1.set_title('Residual ratio diagnostic'); ax1.grid(alpha=0.25); ax1.legend(fontsize=8)
 
    # Per-band ACF panels
    for k, row in enumerate(band_results):
        ax = fig.add_subplot(gs_bot[0, k])
        lags = row.get("_lags")
        acf  = row.get("_acf")
        lf   = row.get("_lags_fit")
        ym   = row.get("_ymod")
 
        # x-axis limit: the tighter of fit_max_lag_mhz and half the band width
        half_bw = row.get("band_width_mhz", np.nan) / 2.0
        if fit_max_lag_mhz is not None and np.isfinite(half_bw):
            x_max = min(float(fit_max_lag_mhz), half_bw)
        elif fit_max_lag_mhz is not None:
            x_max = float(fit_max_lag_mhz)
        elif np.isfinite(half_bw):
            x_max = half_bw
        else:
            x_max = None
 
        if lags is not None and acf is not None:
            lags_sym = np.concatenate((-lags[1:][::-1], lags))
            acf_sym  = np.concatenate(( acf[1:][::-1], acf))
            ax.plot(lags_sym, acf_sym, color='0.3', lw=1.2, label='ACF')
        if lf is not None and ym is not None:
            lf_sym = np.concatenate((-lf[::-1], lf))
            ym_sym = np.concatenate(( ym[::-1], ym))
            ax.plot(lf_sym, ym_sym, color='tab:orange', lw=1.6, ls='--',
                    label=f"Lorentzian\nΔν_d={row['dnu_mhz']:.3f} MHz")
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(0, color='0.6', lw=0.8)
        if x_max is not None:
            ax.set_xlim(-x_max, x_max)
        ax.set_title(f"Band {row['band_idx']}  ({row['center_mhz']:.0f} MHz)", fontsize=8)
        ax.set_xlabel('Δν (MHz)', fontsize=7); ax.set_ylabel('ACF', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
 
    if output:
        base, ext = os.path.splitext(output)
        out = base + '_scint_bw_powerlaw' + (ext if ext else '.png')
        plt.savefig(out, dpi=220)
        print(f"Saved scintillation bandwidth power-law plot to {out}")
    else:
        plt.show()
    plt.close(fig)



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
