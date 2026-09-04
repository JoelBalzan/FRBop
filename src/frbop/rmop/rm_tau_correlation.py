"""
sigma_RM / |RM| vs tau correlation plots.

Implements the Feng+ (2022) comparison sample of FRBs with measured
rotation-measure dispersion (sigma_RM) and scattering timescale (tau), adds an
optional burst of your own, and over-plots two linear (power-law) fits:

    * one to the Feng+ (2022) sample only
    * one to the combined Feng + your data set

Fits are performed in log-log space. For the sigma_RM panel the fit follows
the convention of Feng+ (2022), who regress tau on sigma_RM and report
``tau ~ sigma_RM^0.81``: we fit ``ln(tau) = a + b*ln(sigma_RM)`` (the FRB
121102 upper limit is included as a point, as in that analysis) and overplot
the resulting line on the sigma_RM-vs-tau axes. The |RM| panel uses a direct
regression of |RM| on tau. Slopes / intercepts (with uncertainties) are
reported and annotated on the figure.

The plotting follows the FRBop publication conventions: ``pub_figsize``,
``set_pub_style``, ``IBM_PALETTE`` and the ``_savefig`` wrapper used across
the rmop module.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from frbop.utils.plotting import IBM_PALETTE, pub_figsize

from .constants import plot_style
from .plotting import _savefig

# ---------------------------------------------------------------------------
# Reference sample (Feng+ 2022)
# ---------------------------------------------------------------------------
# name           RM      sigma_RM   sigma_RM_err  tau      tau_err
# FRB 20121102A  81542   30.9       0.4          <0.43     -
# FRB 20180301A    546    6.3       0.4          -         -
# FRB 20180916B   -115    0.12      0.01          0.009     -
# FRB 20190303A   -411    3.6       0.1           0.19      0.04
# FRB 20190417A   4681    6.1       0.5           0.21      0.06
# FRB 20190520B   2759  218.9      10.2           9.8       2.0
# FRB 20201124A   -684    2.5       0.1           0.59      -
#
# A ``tau`` stored as a negative number marks an upper limit (magnitude is the
# limit); ``None`` means the source has no tau measurement and is omitted from
# the tau correlation axes.

FENG2022_SAMPLE = [
    {"name": "FRB 20121102A", "nu_centre": 1300, "rm": 81542.0, "sigma_rm": 30.9,
     "sigma_rm_err": 0.4, "tau": -0.43, "tau_err": None},
    {"name": "FRB 20180301A", "nu_centre": 1300, "rm": 546.0, "sigma_rm": 6.3,
     "sigma_rm_err": 0.4, "tau": None, "tau_err": None},
    {"name": "FRB 20180916B", "nu_centre": 1300, "rm": -115.0, "sigma_rm": 0.12,
     "sigma_rm_err": 0.01, "tau": 0.009, "tau_err": None},
    {"name": "FRB 20190303A", "nu_centre": 1300, "rm": -411.0, "sigma_rm": 3.6,
     "sigma_rm_err": 0.1, "tau": 0.19, "tau_err": 0.04},
    {"name": "FRB 20190417A", "nu_centre": 1300, "rm": 4681.0, "sigma_rm": 6.1,
     "sigma_rm_err": 0.5, "tau": 0.21, "tau_err": 0.06},
    {"name": "FRB 20190520B", "nu_centre": 1300, "rm": 2759.0, "sigma_rm": 218.9,
     "sigma_rm_err": 10.2, "tau": 9.8, "tau_err": 2.0},
    {"name": "FRB 20201124A", "nu_centre": 1300, "rm": -684.0, "sigma_rm": 2.5,
     "sigma_rm_err": 0.1, "tau": 0.59, "tau_err": None},
]

UTTARKAR2024_SAMPLE = [

    {"name": "FRB 20180924B", "nu_centre": 1271,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.17, "sigma_rm_prime_upper": 4.01,
     "tau": 0.91, "tau_err_low": 0.07, "tau_err_high": 0.06},

    {"name": "FRB 20190102C", "nu_centre": 1271,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.52, "sigma_rm_prime_upper": 4.39,
     "tau": 0.84, "tau_err_low": 0.03, "tau_err_high": 0.05},

    {"name": "FRB 20190608B", "nu_centre": 1271,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.42, "sigma_rm_prime_upper": 4.81,
     "tau": 0.93, "tau_err_low": 0.06, "tau_err_high": 0.05},

    {"name": "FRB 20190611B", "nu_centre": 1271,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 6.41, "sigma_rm_prime_upper": 5.48,
     "tau": 0.85, "tau_err_low": 0.04, "tau_err_high": 0.06},

    {"name": "FRB 20190711A", "nu_centre": 1271,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 8.64, "sigma_rm_prime_upper": 8.93,
     "tau": 0.85, "tau_err_low": 0.20, "tau_err_high": 0.10},

    {"name": "FRB 20191001A", "nu_centre": 1271,
     "sigma_rm": 4.1, "sigma_rm_err_low": 0.10,
     "sigma_rm_err_high": 0.09,
     "sigma_rm_upper": None, "sigma_rm_prime_upper": 2.8,
     "tau": 0.6, "tau_err_low": 0.03,
     "tau_err_high": 0.06},
]


UTTARKAR2026_SAMPLE = [

    {"name": "FRB 20191228A", "nu_centre": 1272,
     "rm": 11.3, "rm_err": 0.8,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.17, "sigma_rm_prime_upper": 3.93,
     "tau": 5.69, "tau_err_low": 0.14, "tau_err_high": 0.15},

    {"name": "FRB 20200430A", "nu_centre": 864,
     "rm": 195.3, "rm_err": 0.6,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 4.41, "sigma_rm_prime_upper": 3.51,
     "tau": 7.27, "tau_err_low": 0.21, "tau_err_high": 0.20},

    {"name": "FRB 20210117A", "nu_centre": 1272,
     "rm": -45.4, "rm_err": 0.7,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 3.05, "sigma_rm_prime_upper": 2.25,
     "tau": 0.09, "tau_err_low": 0.06, "tau_err_high": 0.07},

    {"name": "FRB 20210320C", "nu_centre": 864,
     "rm": 288.8, "rm_err": 0.2,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": None, "sigma_rm_prime_upper": None,
     "tau": 0.38, "tau_err_low": 0.18, "tau_err_high": 0.20},

    {"name": "FRB 20210407E", "nu_centre": 1272,
     "rm": -8.9, "rm_err": 0.5,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 2.53, "sigma_rm_prime_upper": 1.98,
     "tau": 0.27, "tau_err_low": 0.15, "tau_err_high": 0.04},

    {"name": "FRB 20210912A", "nu_centre": 1272,
     "rm": 5.7, "rm_err": 0.4,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.06, "sigma_rm_prime_upper": 4.30,
     "tau": 0.45, "tau_err_low": 0.02, "tau_err_high": 0.02},

    {"name": "FRB 20220501C", "nu_centre": 864,
     "rm": 35.2, "rm_err": 0.4,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 2.07, "sigma_rm_prime_upper": 1.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},

    {"name": "FRB 20220610A", "nu_centre": 1272,
     "rm": 217.0, "rm_err": 2.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 1.70, "sigma_rm_prime_upper": 1.19,
     "tau": 0.54, "tau_err_low": 0.05, "tau_err_high": 0.03},

    {"name": "FRB 20220725A", "nu_centre": 920,
     "rm": -26.0, "rm_err": 2.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 4.13, "sigma_rm_prime_upper": 3.22,
     "tau": 2.40, "tau_err_low": 0.12, "tau_err_high": 0.12},

    {"name": "FRB 20221106A", "nu_centre": 1632,
     "rm": -445.0, "rm_err": 1.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 10.04, "sigma_rm_prime_upper": 8.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},

    {"name": "FRB 20230526A", "nu_centre": 1272,
     "rm": 613.0, "rm_err": 2.0,
     "sigma_rm": 12.62,
     "sigma_rm_err_low": 0.23, "sigma_rm_err_high": 0.25,
     "sigma_rm_upper": None,
     "sigma_rm_prime": 11.23,
     "sigma_rm_prime_err_low": 1.04,
     "sigma_rm_prime_err_high": 0.83,
     "tau": 1.39, "tau_err_low": 0.04,
     "tau_err_high": 0.04},

    {"name": "FRB 20230718A", "nu_centre": 1272,
     "rm": 243.0, "rm_err": 1.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.91, "sigma_rm_prime_upper": 5.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},
]


def _asym(lo, hi):
    """Return a symmetric error from asymmetric low/high errors (mean)."""
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    if lo is not None:
        return float(lo)
    if hi is not None:
        return float(hi)
    return None


def _unified_sources() -> List[Dict]:
    """
    Flatten the three reference samples into a common schema.

    Each source is normalised to: name, sigma_rm, sigma_rm_err,
    sigma_rm_upper (None = no limit), rm (None = no measurement),
    tau (negative -> upper limit on tau, positive -> detection),
    tau_err (symmetric, None if unknown).
    """
    out: List[Dict] = []
    for r in FENG2022_SAMPLE:
        sig = r.get("sigma_rm")
        sig_err = r.get("sigma_rm_err")
        out.append({
            "name": r["name"],
            "group": "feng",
            "nu_centre": r.get("nu_centre", 1300.0),
            "sigma_rm": sig,
            "sigma_rm_err": sig_err,
            "sigma_rm_upper": None,
            "rm": r.get("rm"),
            "tau": r.get("tau"),
            "tau_err": r.get("tau_err"),
        })
    for grp, rlist in (("uttarkar2024", UTTARKAR2024_SAMPLE),
                       ("uttarkar2026", UTTARKAR2026_SAMPLE)):
        for r in rlist:
            sig = r.get("sigma_rm")
            # sigma_RM asymmetric errors (mean) -> symmetric for plotting
            if sig is not None:
                sig_err = _asym(r.get("sigma_rm_err_low"), r.get("sigma_rm_err_high"))
            else:
                sig_err = None
            tau = r.get("tau")
            out.append({
                "name": r["name"],
                "group": grp,
                "nu_centre": r.get("nu_centre", 1300.0),
                "sigma_rm": sig,
                "sigma_rm_err": sig_err,
                "sigma_rm_upper": r.get("sigma_rm_upper") if sig is None else None,
                "rm": r.get("rm"),
                "tau": tau,
                "tau_err": _asym(r.get("tau_err_low"), r.get("tau_err_high")) if tau is not None else None,
            })
    return out


def feng2022_sample() -> List[Dict]:
    """Return the unified reference sample (Feng 2022 + Uttarkar 2024/2026)."""
    return _unified_sources()


def _extract(sample: List[Dict]) -> Tuple[np.ndarray, ...]:
    """Split a sample into parallel arrays of raw values."""
    names = [r["name"] for r in sample]
    group = np.array([r.get("group", "feng") for r in sample])
    rm = np.array([r["rm"] if r["rm"] is not None else np.nan
                   for r in sample], dtype=float)
    sigma_rm = np.array([r["sigma_rm"] if r["sigma_rm"] is not None else np.nan
                         for r in sample], dtype=float)
    sigma_rm_err = np.array([r["sigma_rm_err"] if r["sigma_rm_err"] is not None
                             else np.nan for r in sample], dtype=float)
    sigma_rm_upper = np.array([r["sigma_rm_upper"] if r["sigma_rm_upper"] is not None
                               else np.nan for r in sample], dtype=float)
    has_rm = np.array([r["rm"] is not None for r in sample], dtype=bool)
    tau = np.array([r["tau"] if r["tau"] is not None else np.nan
                    for r in sample], dtype=float)
    tau_err = np.array([r["tau_err"] if r["tau_err"] is not None else np.nan
                        for r in sample], dtype=float)
    nu_centre = np.array([r.get("nu_centre", 1300.0) for r in sample], dtype=float)
    return (names, group, rm, sigma_rm, sigma_rm_err, sigma_rm_upper, has_rm,
            tau, tau_err, nu_centre)


def _tau_arrays(tau: np.ndarray,
                tau_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve upper limits (negative tau) into positive limits + flag array."""
    upper = tau < 0
    t = np.where(upper, -tau, tau)
    t_err = np.where(upper, np.nan, tau_err)
    return t, t_err, upper


def _linear(logx: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear model in log space: ln(y) = a + b*ln(x)."""
    return a + b * logx


def _loglog_fit(x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray] = None
                ) -> Optional[Dict[str, float]]:
    """
    Fit ln(y) = a + b*ln(x) to ``x``/``y``.

    Weighted by sigma_ln y = yerr / y when yerr is available and positive;
    unweighted least squares otherwise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    if np.nansum(finite) < 2:
        return None

    logx = np.log(x[finite])
    logy = np.log(y[finite])

    sigma = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)[finite]
        frac = yerr / y[finite]
        ok = np.isfinite(frac) & (frac > 0)
        if np.any(ok):
            sigma = np.ones_like(logy)
            sigma[ok] = frac[ok]
            sigma[~ok] = 10.0  # de-weight points without positive errors

    try:
        if sigma is not None:
            popt, pcov = curve_fit(_linear, logx, logy, p0=[0.0, 1.0],
                                   sigma=sigma, absolute_sigma=False, maxfev=20000)
        else:
            popt, pcov = curve_fit(_linear, logx, logy, p0=[0.0, 1.0],
                                   maxfev=20000)
    except Exception:
        return None

    perr = np.sqrt(np.diag(pcov))
    return {"a": float(popt[0]), "b": float(popt[1]),
            "a_err": float(perr[0]), "b_err": float(perr[1])}


def _logrange(vals: np.ndarray, pad: float = 0.3) -> Tuple[float, float]:
    """Decade-padded log limits for an array of positive values."""
    finite = vals[np.isfinite(vals) & (vals > 0)]
    if finite.size == 0:
        return (1e-3, 1e3)
    return (float(np.nanmin(finite)) * 10 ** (-pad),
            float(np.nanmax(finite)) * 10 ** pad)


def _fit_summary(tag: str, fit: Optional[Dict[str, float]], lines: List[str],
                 tau_first: bool = False, absrm_first: bool = False) -> None:
    if fit is None:
        lines.append(f"  {tag}: fit failed / insufficient points")
        return
    if np.isfinite(fit["b_err"]):
        if tau_first:
            lines.append(f"  {tag}: tau ~ sigma_RM^{fit['b']:.3f} ± {fit['b_err']:.3f} "
                         f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})")
        elif absrm_first:
            lines.append(f"  {tag}: |RM| ~ sigma_RM^{fit['b']:.3f} ± {fit['b_err']:.3f} "
                         f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})")
        else:
            lines.append(f"  {tag}: b = {fit['b']:.3f} ± {fit['b_err']:.3f} "
                         f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})")
    else:
        if tau_first:
            lines.append(f"  {tag}: tau ~ sigma_RM^{fit['b']:.3f} "
                         f"(a = {fit['a']:.3f})")
        elif absrm_first:
            lines.append(f"  {tag}: |RM| ~ sigma_RM^{fit['b']:.3f} "
                         f"(a = {fit['a']:.3f})")
        else:
            lines.append(f"  {tag}: b = {fit['b']:.3f} (a = {fit['a']:.3f})")


def _slope_text(fit: Optional[Dict[str, float]], tag: str, tau_first: bool,
                absrm_first: bool = False) -> Optional[str]:
    """Format a slope annotation."""
    if fit is None:
        return None
    prefix = f"{tag}"
    if tau_first:
        prefix += r"$\tau \propto \sigma_{\mathrm{RM}}^{"
    elif absrm_first:
        prefix += r"$|\mathrm{RM}| \propto \sigma_{\mathrm{RM}}^{"
    else:
        prefix += "$b = "
    if np.isfinite(fit["b_err"]):
        if tau_first or absrm_first:
            return prefix + f"{fit['b']:.2f} \\pm {fit['b_err']:.2f}" + "}$"
        return prefix + f"{fit['b']:.2f} \\pm {fit['b_err']:.2f}$"
    if tau_first or absrm_first:
        return prefix + f"{fit['b']:.2f}" + "}$"
    return prefix + f"{fit['b']:.2f}$"


def _annotate_slope(ax, text: Optional[str], color: str, yfrac: float,
                    style: Dict) -> None:
    """Add an annotation box to an axis."""
    if text is None:
        return
    ax.text(0.04, yfrac, text, transform=ax.transAxes, fontsize=9,
            color=color, va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))


def plot_rm_tau_correlation(sigma_rm: float,
                            sigma_rm_err: float,
                            rm: float,
                            tau: float,
                            tau_err: Optional[float] = None,
                            sample: Optional[List[Dict]] = None,
                            output_file: str = 'rm_tau_correlation.png',
                            name: str = 'This work',
                            freq_mhz: Optional[float] = None,
                            ref_freq_mhz: float = 1300.0,
                            scattering_index: Optional[float] = None,
                            scattering_index_err: Optional[float] = None,
                            ) -> Dict:
    """
    Plot tau vs sigma_RM (top) and |RM| vs sigma_RM (bottom) correlations.

    Both panels share the (log) x-axis sigma_RM. The top panel follows Feng
   + (2022), who regress tau on sigma_RM (``tau ~ sigma_RM^0.81``); the
    FRB 121102 upper limit is included as a point, matching that analysis.
    The bottom panel regresses |RM| on sigma_RM.

    Reference-sample tau values are rescaled from each source's central
    frequency (``nu_centre``) to ``ref_freq_mhz`` (default 1300 MHz, the Feng+
    convention) using the scattering law ``tau ~ nu**alpha`` with ``alpha = -4``.
    Your burst's ``tau`` is rescaled from ``freq_mhz`` to ``ref_freq_mhz`` with
    the supplied``scattering_index`` (default -4); omit ``freq_mhz`` to leave
    your point unscaled (assumed already at ``ref_freq_mhz``).

    Parameters
    ----------
    sigma_rm : float
        Your burst's measured sigma_RM (rad/m^2).
    sigma_rm_err : float
        Uncertainty on sigma_RM (rad/m^2).
    rm : float
        Your burst's measured RM (rad/m^2).
    tau : float
        Your burst's scattering timescale (ms) at ``freq_mhz`` (if given) or
        at the reference frequency.
    tau_err : float, optional
        Uncertainty on tau (ms).
    sample : list of dict, optional
        Reference sample; defaults to :data:`FENG2022_SAMPLE`.
    output_file : str
        Path for the saved figure.
    name : str
        Legend label for your data point.
    freq_mhz : float, optional
        Observing frequency (MHz) at which ``tau`` was measured. If omitted,
        no frequency rescaling is applied.
    ref_freq_mhz : float
        Reference frequency (MHz) to rescale tau to (default 1300).
    scattering_index : float, optional
        Scattering index alpha in ``tau ~ nu**alpha`` for your burst (default
        -4). Only used when ``freq_mhz`` is given.
    scattering_index_err : float, optional
        Uncertainty on the scattering index, used for error propagation.

    Returns
    -------
    dict
        Array-level data and the four log-log fits
        (``feng_sigma_fit``, ``all_sigma_fit``, ``feng_absrm_fit``,
        ``all_absrm_fit``), each a dict with ``a``/``b``/``a_err``/``b_err``
        or ``None`` if the fit could not be performed. The sigma-RM (tau)
        fits use the Feng+ convention ``ln(tau) = a + b*ln(sigma_RM)``;
        the |RM| fits use ``ln(|RM|) = a + b*ln(sigma_RM)``.
    """
    if sample is None:
        sample = feng2022_sample()

    if tau <= 0:
        raise ValueError("tau must be > 0 (log-scale axes)")
    if sigma_rm <= 0:
        raise ValueError("sigma_rm must be > 0 (log-scale axes)")
    if rm == 0:
        raise ValueError("rm must be non-zero (|RM| axis is log-scale)")

    (names, group, rm_arr, sigma_rm_arr, sigma_rm_err_arr, sigma_rm_upper,
     has_rm, tau_raw, tau_err_raw, nu_centre) = _extract(sample)

    # Rescale the reference-sample tau values from each source's central
    # frequency to the reference frequency using the scattering law
    # tau ~ nu**alpha with alpha = -4 (standard tau ~ nu^-4 assumed).
    ref_alpha = -4.0
    ref_scale = (ref_freq_mhz / nu_centre) ** ref_alpha
    tau_raw = np.where(np.isfinite(tau_raw), tau_raw * ref_scale, tau_raw)
    tau_err_raw = tau_err_raw * ref_scale

    taus, taus_err, upper = _tau_arrays(tau_raw, tau_err_raw)

    # Rescale the user burst tau from its observing frequency to the reference
    # frequency using the user-supplied scattering index (supplied scaling).
    if freq_mhz is not None and freq_mhz != ref_freq_mhz:
        if freq_mhz <= 0 or ref_freq_mhz <= 0:
            raise ValueError("frequencies must be > 0 (MHz)")
        alpha = -4.0 if scattering_index is None else scattering_index
        scale = (ref_freq_mhz / freq_mhz) ** alpha
        tau = tau * scale
        if tau_err is not None and tau_err > 0:
            frac_tau = tau_err / (tau / scale)
            frac_alpha = np.log(ref_freq_mhz / freq_mhz) * (
                scattering_index_err if scattering_index_err is not None else 0.0)
            tau_err = tau * np.sqrt(frac_tau ** 2 + frac_alpha ** 2)

    # Append the user burst (assumed to be a genuine measurement).
    names = names + [name]
    group = np.append(group, "user")
    rm_arr = np.append(rm_arr, rm)
    sigma_rm_arr = np.append(sigma_rm_arr, sigma_rm)
    sigma_rm_err_arr = np.append(sigma_rm_err_arr, sigma_rm_err)
    sigma_rm_upper = np.append(sigma_rm_upper, np.nan)
    has_rm = np.append(has_rm, True)
    taus = np.append(taus, tau)
    taus_err = np.append(taus_err, tau_err if tau_err is not None else np.nan)
    upper = np.append(upper, False)

    user_mask = np.zeros_like(taus, dtype=bool)
    user_mask[-1] = True

    abs_rm = np.abs(rm_arr)

    # ---- Fits (upper limits on sigma_RM excluded; only detections fit) ----
    has_tau = np.isfinite(taus) & (taus > 0)
    sig_det = np.isfinite(sigma_rm_arr) & (sigma_rm_arr > 0)

    # sigma_RM panel (top, tau on the y-axis): paper-convention fit
    # ln(tau) = a + b*ln(sigma_RM), i.e. tau as the dependent variable against
    # sigma_RM (Feng+ 2022 report tau ~ sigma_RM^0.81). The FRB 121102
    # tau upper limit is plotted as a point but excluded from the fit.
    feng_g = group == "feng"
    sigma_fit_members = has_tau & sig_det & ~upper
    feng_sigma = sigma_fit_members & feng_g
    feng_sigma_fit = (_loglog_fit(sigma_rm_arr[feng_sigma], taus[feng_sigma])
                      if np.any(feng_sigma) else None)
    all_sigma_fit = _loglog_fit(sigma_rm_arr[sigma_fit_members], taus[sigma_fit_members])

    # |RM| panel (bottom, |RM| on the y-axis): direct regression of |RM| on
    # sigma_RM for sources with a detected sigma_RM and a measured RM.
    absrm_fit_members = has_rm & np.isfinite(abs_rm) & (abs_rm > 0) & sig_det
    feng_absrm = absrm_fit_members & feng_g
    feng_absrm_fit = (_loglog_fit(sigma_rm_arr[feng_absrm], abs_rm[feng_absrm])
                      if np.any(feng_absrm) else None)
    all_absrm_fit = _loglog_fit(sigma_rm_arr[absrm_fit_members], abs_rm[absrm_fit_members])

    # ---- Terminal summary ----
    summary: List[str] = ["RM vs tau correlation fit summary:"]
    _fit_summary("Feng only sigma_RM ", feng_sigma_fit, summary, tau_first=True)
    _fit_summary("All data sigma_RM  ", all_sigma_fit, summary, tau_first=True)
    _fit_summary("Feng only |RM|     ", feng_absrm_fit, summary, absrm_first=True)
    _fit_summary("All data |RM|      ", all_absrm_fit, summary, absrm_first=True)
    for line in summary:
        print(line)

    # ---- Figure ----
    style = plot_style()

    # x-position for plotting: detection value where detected, otherwise the
    # sigma_RM upper limit. Flag which sources are (only) upper limits.
    sig_is_upper = np.isfinite(sigma_rm_upper)
    sig_plot = np.where(sig_is_upper, sigma_rm_upper, sigma_rm_arr)
    sigma_lim = _logrange(sig_plot)

    # Two stacked panels sharing sigma_RM on the (log) x-axis: top is tau on
    # the y-axis, bottom is |RM| on the y-axis.
    fig_w, _ = pub_figsize(height_ratio=1.0, ncol=2)
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(fig_w, fig_w * 0.62 * 2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.14,
                        hspace=0.08)

    panels = [
        (0, taus, taus_err, _logrange(taus), r'$\tau$ [ms]',
         feng_sigma_fit, all_sigma_fit, True, False),
        (1, abs_rm, None, _logrange(abs_rm), r'$|\mathrm{RM}|$ [rad m$^{-2}$]',
         feng_absrm_fit, all_absrm_fit, False, True),
    ]

    for idx, y, yerr, y_lim, ylabel, feng_fit, all_fit, tau_first, absrm_first in panels:
        ax = axes[idx]
        feng = ~user_mask

        # Top panel needs a (finite) tau value; bottom panel needs |RM|, so it
        # also drops any source without an RM measurement.
        need_tau = idx == 0
        if need_tau:
            has_xy = np.isfinite(y) & np.isfinite(taus)
        else:
            has_xy = has_rm & np.isfinite(y)

        # Split reference sources into the four combinations of sigma_RM state
        # (detection vs upper limit) and tau state (only relevant for top).
        ref = feng & has_xy
        for grp, grp_color, grp_label in (
            ("feng", IBM_PALETTE[0], "Feng+ (2022)"),
            ("uttarkar2024", IBM_PALETTE[1], "Uttarkar+ (2024)"),
            ("uttarkar2026", IBM_PALETTE[3], "Uttarkar+ (2026)"),
        ):
            sig_det = ref & (group == grp) & ~sig_is_upper \
                & np.isfinite(sigma_rm_arr) & (sigma_rm_arr > 0)
            sig_up = ref & (group == grp) & sig_is_upper & (sigma_rm_upper > 0)

            # tau upper-limit detection points (Feng negative-tau convention, top only)
            tau_up_det = sig_det & upper & need_tau
            tau_up_sigu = sig_up & upper & need_tau
            sig_det = sig_det & (~upper | (not need_tau))
            sig_up = sig_up & (~upper | (not need_tau))

            if np.any(sig_det):
                ax.errorbar(
                    sigma_rm_arr[sig_det], y[sig_det],
                    xerr=(
                        sigma_rm_err_arr[sig_det]
                        if np.all(np.isfinite(sigma_rm_err_arr[sig_det]))
                        else None
                    ),
                    yerr=(
                        yerr[sig_det]
                        if (yerr is not None and np.all(np.isfinite(yerr[sig_det])))
                        else None
                    ),
                    fmt='o', ms=6, color=grp_color,
                    markeredgecolor='white', markeredgewidth=0.8,
                    ecolor='0.4', elinewidth=1, capsize=2.5, zorder=3,
                    label=grp_label if idx == 0 else None,
                )
            # sigma_RM upper limits are drawn as left-pointing triangles.
            if np.any(sig_up):
                ax.errorbar(
                    sigma_rm_upper[sig_up], y[sig_up], fmt='<', ms=6,
                    color=grp_color, markeredgecolor='white',
                    markeredgewidth=0.5, ecolor='0.4', elinewidth=1,
                    capsize=2.5, zorder=3,
                )
            # tau upper limits (top panel only): down-pointing triangles.
            if np.any(tau_up_det):
                ax.errorbar(
                    sigma_rm_arr[tau_up_det], y[tau_up_det], fmt='v', ms=6,
                    color=grp_color, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=3,
                )
            if np.any(tau_up_sigu):
                ax.errorbar(
                    sigma_rm_upper[tau_up_sigu], y[tau_up_sigu], fmt='v', ms=6,
                    color=grp_color, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=3,
                )
        if np.any(user_mask):
            ax.errorbar(
                sigma_rm_arr[user_mask], y[user_mask],
                xerr=sigma_rm_err_arr[user_mask],
                yerr=yerr[user_mask] if (yerr is not None and np.isfinite(yerr[user_mask])) else None,
                marker=(10,1,0), ms=8, color=IBM_PALETTE[-1],
                markeredgecolor='white', markeredgewidth=0.5,
                ecolor='0.4', elinewidth=1, capsize=2.5, zorder=4,
                label=name,
            )

        grid_x = np.logspace(np.log10(sigma_lim[0]), np.log10(sigma_lim[1]), 400)
        if feng_fit is not None:
            line_y = np.exp(_linear(np.log(grid_x), feng_fit['a'],
                                    feng_fit['b']))
            ax.plot(grid_x, line_y, color='0.35', lw=style['line'], ls='--',
                    zorder=2, 
                    #label='Feng fit'
                    )
            _annotate_slope(ax, _slope_text(feng_fit, 'Feng+ (2022): ', tau_first, absrm_first),
                            '0.35', 0.88, style)
        if all_fit is not None:
            line_y = np.exp(_linear(np.log(grid_x), all_fit['a'],
                                    all_fit['b']))
            lo_y = np.exp(_linear(np.log(grid_x),
                                  all_fit['a'] - all_fit['a_err'],
                                  all_fit['b'] - all_fit['b_err']))
            hi_y = np.exp(_linear(np.log(grid_x),
                                  all_fit['a'] + all_fit['a_err'],
                                  all_fit['b'] + all_fit['b_err']))
            ax.fill_between(grid_x, lo_y, hi_y,
                            color=IBM_PALETTE[2], alpha=0.15, zorder=1)
            ax.plot(grid_x, line_y, color=IBM_PALETTE[2], lw=style['line'],
                    ls='-', zorder=2, 
                    #label='All data fit'
                    )
            _annotate_slope(ax, _slope_text(all_fit, '', tau_first, absrm_first),
                            IBM_PALETTE[2], 0.70, style)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*sigma_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, which='major', alpha=0.3)
        ax.tick_params(axis='both', labelsize=style['tick'])
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=style['label'])
            ax.legend(fontsize=style['legend'], loc='best')
        else:
            ax.set_ylabel(ylabel, fontsize=style['label'])
        if idx == 1:
            ax.set_xlabel(r'$\sigma_{\mathrm{RM}}$ [rad m$^{-2}$]',
                          fontsize=style['label'])

    _savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"RM-tau correlation plot saved to {output_file}")
    plt.close(fig)

    return {
        'x': taus,
        'xerr': taus_err,
        'sigma_rm': sigma_rm_arr,
        'sigma_rm_err': sigma_rm_err_arr,
        'abs_rm': abs_rm,
        'upper': upper,
        'user_mask': user_mask,
        'names': names,
        'feng_sigma_fit': feng_sigma_fit,
        'all_sigma_fit': all_sigma_fit,
        'feng_absrm_fit': feng_absrm_fit,
        'all_absrm_fit': all_absrm_fit,
    }


def main() -> None:
    """CLI entry point for the RM-tau correlation plots."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plot sigma_RM and |RM| vs tau with the Feng+ (2022) sample "
            "plus your own burst, with power-law fits to the sample and to all data."
        ),
    )
    parser.add_argument("--sigma-rm", type=float, required=True,
                        help="Your burst's sigma_RM (rad/m^2)")
    parser.add_argument("--sigma-rm-err", type=float, required=True,
                        help="Uncertainty on sigma_RM (rad/m^2)")
    parser.add_argument("--rm", type=float, required=True,
                        help="Your burst's RM (rad/m^2)")
    parser.add_argument("--tau", type=float, required=True,
                        help="Your burst's scattering timescale tau (ms)")
    parser.add_argument("--tau-err", type=float, default=None,
                        help="Uncertainty on tau (ms)")
    parser.add_argument("--freq", type=float, default=None,
                        help="Observing frequency (MHz) of your tau measurement; "
                             "if given, tau is rescaled to --ref-freq")
    parser.add_argument("--ref-freq", type=float, default=1300.0,
                        help="Reference frequency (MHz) for tau rescaling "
                             "(Feng+ 2022 convention, default: 1300)")
    parser.add_argument("--scattering-index", type=float, default=None,
                        help="Scattering index alpha in tau ~ nu**alpha "
                             "for your burst (default: -4)")
    parser.add_argument("--scattering-index-err", type=float, default=None,
                        help="Uncertainty on the scattering index (for error propagation)")
    parser.add_argument("--name", default="This work",
                        help="Legend label for your burst (default: 'This work')")
    parser.add_argument("-o", "--output", default="rm_tau_correlation.png",
                        help="Output PNG filename")
    parser.add_argument("--pub-col", type=float, default=2,
                        help="Publication figure column count (1, 2, 3, ...). Default: 2")

    args = parser.parse_args()

    from frbop.utils.plotting import set_pub_col, set_pub_style
    set_pub_col(args.pub_col)
    set_pub_style(use_latex=False)

    plot_rm_tau_correlation(
        sigma_rm=args.sigma_rm,
        sigma_rm_err=args.sigma_rm_err,
        rm=args.rm,
        tau=args.tau,
        tau_err=args.tau_err,
        output_file=args.output,
        name=args.name,
        freq_mhz=args.freq,
        ref_freq_mhz=args.ref_freq,
        scattering_index=args.scattering_index,
        scattering_index_err=args.scattering_index_err,
    )


if __name__ == "__main__":
    main()