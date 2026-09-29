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

Reference samples combined into the fit/plot: Feng+ (2022), Uttarkar+
(2024/2026), and Pandhi+ (2026) [FRB 20220529A].

The plotting follows the FRBop publication conventions: ``pub_figsize``,
``set_pub_style``, ``IBM_PALETTE`` and the ``_savefig`` wrapper used across
the rmop module.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit

from frbop.utils.plotting import IBM_PALETTE, pub_figsize

from ..constants import plot_style
from ..plotting import _savefig
from .galactic_rm import (DEFAULT_SCATTER_WARN_THRESHOLD, DEFAULT_SKYMAP_DIR,
                          SpiceRacsSkymaps, host_frame_correct_sample,
                          host_frame_correct_source)


def _radec(ra_str: str, dec_str: Optional[str] = None) -> Dict[str, float]:
    """
    Parse coordinates into {'ra': deg, 'dec': deg}.

    Accepts either:
      - a single combined string, e.g. "01h16m24.99s +20d37m56.55s"
      - separate RA/Dec strings in any of astropy's sexagesimal formats,
        e.g. _radec("13:52:19.200", "-63:52:39.00")
             _radec("01h16m24.99s", "+20d37m56.55s")
             _radec("13 52 19.200", "-63 52 39.00")
    """
    if dec_str is None:
        c = SkyCoord(ra_str, frame="icrs", unit=(u.hourangle, u.deg))
    else:
        c = SkyCoord(ra=ra_str, dec=dec_str, frame="icrs",
                    unit=(u.hourangle, u.deg))
    return {"ra": c.ra.deg, "dec": c.dec.deg}

# ---------------------------------------------------------------------------
# Reference samples
# ---------------------------------------------------------------------------

# Pandhi+ (2026) — FRB 20220529A. Uncorrected observer-frame values: rm and
# tau are the median (by |RM| and tau, respectively) of the firm CHIME/FRB
# burst-level detections in Table 1; sigma_rm is the observer-frame
# depolarization-fit parameter from Eq. (2)/Fig. 3.
PANDHI2026_SAMPLE = [
    {"name": "FRB 20220529A", "z": 0.1839, "nu_centre": 600,
     "rm": -4.4, "rm_err": 0.4,
     "sigma_rm": 2.50, "sigma_rm_err": 0.02,
     "tau": 0.43, "tau_err": 0.01, **_radec("01h16m24.99s +20d37m56.55s")},
]

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
    {"name": "FRB 20121102A", "z": 0.19273, "nu_centre": 1300, "rm": 81542.0, "sigma_rm": 30.9,
     "sigma_rm_err": 0.4, "tau": -0.43, "tau_err": None, **_radec("05h32m09s +33d05m13s")},
    # Tendulkar et al. 2017, ApJL 834, L7 — https://arxiv.org/abs/1701.01100

    {"name": "FRB 20180301A", "z": 0.3304, "nu_centre": 1300, "rm": 546.0, "sigma_rm": 6.3,
     "sigma_rm_err": 0.4, "tau": None, "tau_err": None, **_radec("06h12m43.4s +04d33m44.8s")},
    # Bhandari et al. 2022, AJ 163, 69 — https://arxiv.org/abs/2108.01282

    {"name": "FRB 20180916B", "z": 0.0337, "nu_centre": 1300, "rm": -115.0, "sigma_rm": 0.12,
     "sigma_rm_err": 0.01, "tau": 0.009, "tau_err": None, **_radec("01h56m40.8s +65d44m24s")},
    # Marcote et al. 2020, Nature 577, 190 — https://arxiv.org/abs/2001.02222

    {"name": "FRB 20190303A", "z": 0.0640, "nu_centre": 1300, "rm": -411.0, "sigma_rm": 3.6,
     "sigma_rm_err": 0.1, "tau": 0.19, "tau_err": 0.04, **_radec("13:52:19.200 +48:15:00.00")},
    # Michilli et al. 2023, ApJ (sub-arcmin CHIME/FRB localizations) — https://arxiv.org/abs/2212.11941

    {"name": "FRB 20190417A", "z": 0.12817, "nu_centre": 1300, "rm": 4681.0, "sigma_rm": 6.1,
     "sigma_rm_err": 0.5, "tau": 0.21, "tau_err": 0.06, **_radec("19:38:52.800 +59:24:36.00")},
    # EVN localization / host paper — https://arxiv.org/abs/2509.05174

    {"name": "FRB 20190520B", "z": 0.241, "nu_centre": 1300, "rm": 2759.0, "sigma_rm": 218.9,
     "sigma_rm_err": 10.2, "tau": 9.8, "tau_err": 2.0, **_radec("16:02:04.270 -11:17:17.32")},
    # Niu et al. 2022, Nature 606, 873 — https://arxiv.org/abs/2110.07418

    {"name": "FRB 20201124A", "z": 0.098, "nu_centre": 1300, "rm": -684.0, "sigma_rm": 2.5,
     "sigma_rm_err": 0.1, "tau": 0.59, "tau_err": None, **_radec("05:07:57.600	+26:11:24.00")},
    # Ravi et al. 2022 (host galaxy + PRS paper) — https://arxiv.org/abs/2106.09710
]

UTTARKAR2024_SAMPLE = [
# ra dec from Day+2021
    {"name": "FRB 20180924B", "nu_centre": 1271, "rm":22, "rm_err":2, "z": 0.3214,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.17, "sigma_rm_prime_upper": 4.01,
     "tau": 0.91, "tau_err_low": 0.07, "tau_err_high": 0.06, **_radec("21:44:25.255	-40:54:00.10")},

    {"name": "FRB 20190102C", "nu_centre": 1271, "rm":-105, "rm_err":1, "z": 0.29,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.52, "sigma_rm_prime_upper": 4.39,
     "tau": 0.84, "tau_err_low": 0.03, "tau_err_high": 0.05, **_radec("21:29:39.760	-79:28:32.50")},

    {"name": "FRB 20190608B", "nu_centre": 1271, "rm":353, "rm_err":2, "z": 0.1178,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.42, "sigma_rm_prime_upper": 4.81,
     "tau": 0.93, "tau_err_low": 0.06, "tau_err_high": 0.05, **_radec("22:16:04.770	-07:53:53.70")},

    {"name": "FRB 20190611B", "nu_centre": 1271, "rm":20, "rm_err":4, "z": 0.378,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 6.41, "sigma_rm_prime_upper": 5.48,
     "tau": 0.85, "tau_err_low": 0.04, "tau_err_high": 0.06, **_radec("21:22:58.940	-79:23:51.30")},

    {"name": "FRB 20190711A", "nu_centre": 1271, "rm":9, "rm_err":2, "z": 0.522,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 8.64, "sigma_rm_prime_upper": 8.93,
     "tau": 0.85, "tau_err_low": 0.20, "tau_err_high": 0.10, **_radec("21:57:40.62	-80:21:28.80")},

    {"name": "FRB 20191001A", "nu_centre": 1271, "rm":55.5, "rm_err":9, "z": 0.23,
     "sigma_rm": 4.1, "sigma_rm_err_low": 0.10, "sigma_rm_err_high": 0.09,
     "sigma_rm_upper": None, "sigma_rm_prime_upper": 2.8,
     "tau": 0.6, "tau_err_low": 0.03, "tau_err_high": 0.06, **_radec("21:33:24.313	-54:44:51.86")},
]


UTTARKAR2026_SAMPLE = [

    {"name": "FRB 20191228A", **_radec("22:57:43.327", "-29:35:38.82"),
     "nu_centre": 1272, "z": 0.2432,
     "rm": 11.3, "rm_err": 0.8,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.17, "sigma_rm_prime_upper": 3.93,
     "tau": 5.69, "tau_err_low": 0.14, "tau_err_high": 0.15},

    {"name": "FRB 20200430A", **_radec("15:18:49.549", "+12:22:34.784"),
     "nu_centre": 864, "z": 0.161,
     "rm": 195.3, "rm_err": 0.6,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 4.41, "sigma_rm_prime_upper": 3.51,
     "tau": 7.27, "tau_err_low": 0.21, "tau_err_high": 0.20},

    {"name": "FRB 20210117A", **_radec("22:39:55.007", "-16:09:05.203"),
     "nu_centre": 1272, "z": 0.214,
     "rm": -45.4, "rm_err": 0.7,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 3.05, "sigma_rm_prime_upper": 2.25,
     "tau": 0.09, "tau_err_low": 0.06, "tau_err_high": 0.07},

    {"name": "FRB 20210320C", **_radec("13:37:50.104", "-16:07:21.611"),
     "nu_centre": 864, "z": 0.28,
     "rm": 288.8, "rm_err": 0.2,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": None, "sigma_rm_prime_upper": None,
     "tau": 0.38, "tau_err_low": 0.18, "tau_err_high": 0.20},

    {"name": "FRB 20210407E", **_radec("05:14:36.230", "+27:03:29.73"),
     "nu_centre": 1272, "z": None,  # no z found in ground based searches
     "rm": -8.9, "rm_err": 0.5,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 2.53, "sigma_rm_prime_upper": 1.98,
     "tau": 0.27, "tau_err_low": 0.15, "tau_err_high": 0.04},

    {"name": "FRB 20210912A", **_radec("23:23:10.441", "-30:24:20.089"),
     "nu_centre": 1272, "z": None,  # z: "The host is best described by a low mass dwarf galaxy at a redshift of z = 1.42+0.32-0.38" -- Clancy in slack
     "rm": 5.7, "rm_err": 0.4,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.06, "sigma_rm_prime_upper": 4.30,
     "tau": 0.45, "tau_err_low": 0.02, "tau_err_high": 0.02},

    {"name": "FRB 20220501C", **_radec("23:29:30.987", "-32:29:26.63"),
     "nu_centre": 864, "z": 0.381,
     "rm": 35.2, "rm_err": 0.4,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 2.07, "sigma_rm_prime_upper": 1.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},

    {"name": "FRB 20220610A", **_radec("23:24:17.583", "-33:30:49.859"),
     "nu_centre": 1272, "z": 1.015,
     "rm": 217.0, "rm_err": 2.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 1.70, "sigma_rm_prime_upper": 1.19,
     "tau": 0.54, "tau_err_low": 0.05, "tau_err_high": 0.03},

    {"name": "FRB 20220725A", **_radec("23:33:15.649", "-35:59:24.92"),
     "nu_centre": 920, "z": 0.1926,
     "rm": -26.0, "rm_err": 2.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 4.13, "sigma_rm_prime_upper": 3.22,
     "tau": 2.40, "tau_err_low": 0.12, "tau_err_high": 0.12},

    {"name": "FRB 20221106A", **_radec("03:46:49.154", "-25:34:11.316"),
     "nu_centre": 1632, "z": 0.204,
     "rm": -445.0, "rm_err": 1.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 10.04, "sigma_rm_prime_upper": 8.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},

    {"name": "FRB 20230526A", **_radec("01:28:55.828", "-52:43:02.39"),
     "nu_centre": 1272, "z": 0.157,
     "rm": 613.0, "rm_err": 2.0,
     "sigma_rm": 12.62,
     "sigma_rm_err_low": 0.23, "sigma_rm_err_high": 0.25,
     "sigma_rm_upper": None,
     "sigma_rm_prime": 11.23,
     "sigma_rm_prime_err_low": 1.04,
     "sigma_rm_prime_err_high": 0.83,
     "tau": 1.39, "tau_err_low": 0.04,
     "tau_err_high": 0.04},

    {"name": "FRB 20230718A", **_radec("08:32:38.863", "-40:27:06.955"),
     "nu_centre": 1272, "z": 0.035,
     "rm": 243.0, "rm_err": 1.0,
     "sigma_rm": None, "sigma_rm_err": None,
     "sigma_rm_upper": 5.91, "sigma_rm_prime_upper": 5.38,
     "tau": None, "tau_err_low": None, "tau_err_high": None},
]

# Balzan et al. (2026)-- FRB 20250607A
BALZAN2026_SAMPLE = [
    {"name": "FRB 20250607A", **_radec("02:32:43.44488639", "-39:20:27.9387337"),  
     "z": 0.2106, "nu_centre": 919.5,
     "rm": -428.2, "rm_err": 0.14,
     "sigma_rm": 3.556142, "sigma_rm_err": 0.033343,
     "tau": 1.25, "tau_err": 0.02},
]

def _find_in_sample(name: str, *samples: List[Dict]) -> Optional[Dict]:
    """Return the first entry matching ``name`` across one or more sample
    lists (e.g. BALZAN2026_SAMPLE, PANDHI2026_SAMPLE), or None."""
    for sample in samples:
        for src in sample:
            if src["name"] == name:
                return src
    return None


def apply_named_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    If --tau-name matches a known FRB in BALZAN2026_SAMPLE/PANDHI2026_SAMPLE/
    etc, fill in any of ra/dec/z/rm/tau/tau-err/tau-freq the user left at
    its argparse default from that source's stored values. Explicit CLI
    flags always win.
    """
    src = _find_in_sample(args.tau_name, BALZAN2026_SAMPLE, PANDHI2026_SAMPLE)
    if src is None:
        return

    field_map = {"ra": "ra", "dec": "dec", "z": "z", "rm": "rm",
                "tau": "tau", "tau_err": "tau_err"}
    for arg_field, src_field in field_map.items():
        if getattr(args, arg_field, None) == parser.get_default(arg_field):
            val = src.get(src_field)
            if val is not None:
                setattr(args, arg_field, val)

    if (src.get("nu_centre") is not None
            and args.tau_freq == parser.get_default("tau_freq")):
        args.tau_freq = src["nu_centre"]

def _asym(lo, hi):
    """Return a symmetric error from asymmetric low/high errors (mean)."""
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    if lo is not None:
        return float(lo)
    if hi is not None:
        return float(hi)
    return None


def _append_simple_group(out: List[Dict], group: str, rlist: List[Dict]) -> None:
    """Append a sample whose entries already use plain (symmetric) errors,
    i.e. the Feng+ (2022) and Pandhi+ (2026) schemas."""
    for r in rlist:
        out.append({
            "name": r["name"],
            "group": group,
            "nu_centre": r.get("nu_centre", 1300.0),
            "sigma_rm": r.get("sigma_rm"),
            "sigma_rm_err": r.get("sigma_rm_err"),
            "sigma_rm_upper": None,
            "rm": r.get("rm"),
            "tau": r.get("tau"),
            "tau_err": r.get("tau_err"),
            "ra": r.get("ra"),
            "dec": r.get("dec"),
            "z": r.get("z")
        })


def _unified_sources() -> List[Dict]:
    """
    Flatten the reference samples into a common schema.

    Each source is normalised to: name, sigma_rm, sigma_rm_err,
    sigma_rm_upper (None = no limit), rm (None = no measurement),
    tau (negative -> upper limit on tau, positive -> detection),
    tau_err (symmetric, None if unknown).
    """
    out: List[Dict] = []
    _append_simple_group(out, "feng", FENG2022_SAMPLE)
    _append_simple_group(out, "pandhi2026", PANDHI2026_SAMPLE)

    for grp, rlist in (("uttarkar2024", UTTARKAR2024_SAMPLE),
                       ("uttarkar2026", UTTARKAR2026_SAMPLE)):
        for r in rlist:
            sig = r.get("sigma_rm")
            # sigma_RM asymmetric errors (mean) -> symmetric for plotting
            sig_err = _asym(r.get("sigma_rm_err_low"), r.get("sigma_rm_err_high")) if sig is not None else None
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
                "ra": r.get("ra"),
                "dec": r.get("dec"),
                "z": r.get("z")
            })
    return out


def feng2022_sample() -> List[Dict]:
    """Return the unified reference sample (Feng+ 2022, Pandhi+ 2026,
    Uttarkar+ 2024/2026)."""
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


# (group, color, legend label) for reference-sample scatter points. Index 2
# of IBM_PALETTE is reserved for the "all data" fit line/band, so reference
# groups avoid it here.
_GROUP_STYLE = (
    ("feng", IBM_PALETTE[0], "Feng+ (2022)"),
    ("uttarkar2024", IBM_PALETTE[1], "Uttarkar+ (2024)"),
    ("uttarkar2026", "#009E73", "Uttarkar+ (2026)"),
    ("pandhi2026", IBM_PALETTE[3], "Pandhi+ (2026)"),
)


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
                            label_suffix: str = '',
                            exclude_upper_limits: bool = False,
                            ) -> Dict:
    """
    Plot tau vs sigma_RM (top) and |RM| vs sigma_RM (bottom) correlations.

    Both panels share the (log) x-axis sigma_RM. The top panel follows Feng+
    (2022), who regress tau on sigma_RM (``tau ~ sigma_RM^0.81``); the
    FRB 121102 upper limit is included as a point, matching that analysis.
    The bottom panel regresses |RM| on sigma_RM.

    Reference-sample tau values are rescaled from each source's central
    frequency (``nu_centre``) to ``ref_freq_mhz`` (default 1300 MHz, the Feng+
    convention) using the scattering law ``tau ~ nu**alpha`` with ``alpha = -4``.
    Your burst's ``tau`` is rescaled from ``freq_mhz`` to ``ref_freq_mhz`` with
    the supplied ``scattering_index`` (default -4); omit ``freq_mhz`` to leave
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
        Reference sample; defaults to the unified Feng+/Pandhi+/Uttarkar+
        sample returned by :func:`feng2022_sample`.
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
    label_suffix : str
        Appended (as a LaTeX subscript) to each axis label -- e.g. pass
        ``'host'`` to render tau_host, |RM_host|, sigma_RM,host for a
        rest-frame-corrected plot. Empty by default (observer-frame labels).
    exclude_upper_limits : bool
        If True, sources with a sigma_RM upper limit or a tau upper limit
        (Feng+ negative-tau convention) are dropped entirely from the plot
        and both fits -- not just from the fit as normal, but from the
        figure altogether. Use this for a "detections only" companion plot.
        Your own burst (``name``) is never excluded by this flag, even if
        its sigma_rm_err is missing, since it is assumed to be a genuine
        measurement.


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
    # frequency using the user-supplied scattering index.
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
    tau_raw = np.append(tau_raw, tau)
    tau_err_raw = np.append(tau_err_raw,
                            tau_err if tau_err is not None else np.nan)
    nu_centre = np.append(nu_centre, ref_freq_mhz)

    if exclude_upper_limits:
        keep = (
            np.isfinite(sigma_rm_arr) & (sigma_rm_arr > 0)   # has a sigma_RM detection
            & np.isfinite(tau_raw) & (tau_raw > 0)            # has a tau detection (not an upper limit)
        )
        names = [n for n, k in zip(names, keep) if k]
        group = group[keep]
        rm_arr = rm_arr[keep]
        sigma_rm_arr = sigma_rm_arr[keep]
        sigma_rm_err_arr = sigma_rm_err_arr[keep]
        sigma_rm_upper = sigma_rm_upper[keep]
        has_rm = has_rm[keep]
        taus = taus[keep]
        taus_err = taus_err[keep]
        upper = upper[keep]
        tau_raw = tau_raw[keep]
        tau_err_raw = tau_err_raw[keep]
        nu_centre = nu_centre[keep]

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

    _tau_sub = r'\tau' if not label_suffix else rf'\tau_{{\mathrm{{{label_suffix}}}}}'
    _rm_sub = r'|\mathrm{RM}|' if not label_suffix else rf'|\mathrm{{RM}}_{{\mathrm{{{label_suffix}}}}}|'
    _sigma_sub = (r'\sigma_{\mathrm{RM}}' if not label_suffix
                 else rf'\sigma_{{\mathrm{{RM}},\mathrm{{{label_suffix}}}}}')

    panels = [
        (0, taus, taus_err, _logrange(taus), rf'${_tau_sub}$ [ms]',
         feng_sigma_fit, all_sigma_fit, True, False),
        (1, abs_rm, None, _logrange(abs_rm), rf'${_rm_sub}$ [rad m$^{{-2}}$]',
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
        for grp, grp_color, grp_label in _GROUP_STYLE:
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
                marker=(10, 1, 0), ms=8, color=IBM_PALETTE[-1],
                markeredgecolor='white', markeredgewidth=0.5,
                ecolor='0.4', elinewidth=1, capsize=2.5, zorder=4,
                label=name,
            )

        grid_x = np.logspace(np.log10(sigma_lim[0]), np.log10(sigma_lim[1]), 400)
        if feng_fit is not None:
            line_y = np.exp(_linear(np.log(grid_x), feng_fit['a'],
                                    feng_fit['b']))
            ax.plot(grid_x, line_y, color='0.35', lw=style['line'], ls='--',
                    zorder=2)
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
                    ls='-', zorder=2)
            _annotate_slope(ax, _slope_text(all_fit, '', tau_first, absrm_first),
                            IBM_PALETTE[2], 0.70, style)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*sigma_lim)
        ax.set_ylim(*y_lim)
        ax.grid(True, which='major', alpha=0.3)
        ax.tick_params(axis='both', labelsize=style['tick'])
        ax.set_ylabel(ylabel, fontsize=style['label'])
        if idx == 0:
            ax.legend(fontsize=style['legend'], loc='best')
        if idx == 1:
            ax.set_xlabel(rf'${_sigma_sub}$ [rad m$^{{-2}}$]',
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


# ---------------------------------------------------------------------------
# Host-frame (rest-frame) correction: Galactic RM subtraction + (1+z) scaling
# ---------------------------------------------------------------------------
# The correction itself (Galactic RM subtraction via SPICE-RACS DR2 sky
# maps, plus (1+z)**2 RM / (1+z)**3 tau rest-frame scaling, following
# Pandhi+ 2026 Appendix C) lives in galactic_rm.py, since it's really a
# property of the sky-map lookup rather than this plotting pipeline.


def plot_rm_tau_correlation_host_frame(
    sigma_rm: float,
    sigma_rm_err: float,
    rm: float,
    tau: float,
    ra: float,
    dec: float,
    z: float,
    tau_err: Optional[float] = None,
    rm_err: Optional[float] = None,
    sample: Optional[List[Dict]] = None,
    skymap_dir: Path = DEFAULT_SKYMAP_DIR,
    output_file: str = 'rm_tau_correlation_host_frame.png',
    name: str = 'This work',
    scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
    exclude_upper_limits: bool = False,
) -> Dict:
    """
    Same as :func:`plot_rm_tau_correlation`, but first applies the
    Galactic-RM-subtraction + (1+z) rest-frame correction (Pandhi+ 2026,
    Appendix C; implemented in galactic_rm.py) to every source in the
    reference sample AND to your own burst, using its ra/dec/z to look up
    the Galactic RM from SPICE-RACS DR2. Sources without ra/dec/z are
    dropped from the reference sample (printed, not raised) rather than
    plotted with uncorrected values.

    Since this plot works entirely in the rest frame, the tau frequency
    rescaling used by :func:`plot_rm_tau_correlation` (nu_centre -> a
    shared reference *observing* frequency) is not applied here -- it
    addresses a different effect (differing telescope bands) than the
    redshift correction this function performs.

    Parameters mirror :func:`plot_rm_tau_correlation`, with two additions:
    ``ra``/``dec`` (deg, ICRS) and ``z`` for your own burst, used to look
    up its Galactic RM the same way as the reference sample.
    """
    if sample is None:
        sample = feng2022_sample()

    skymaps = SpiceRacsSkymaps(skymap_dir=skymap_dir)

    print("Applying host-frame correction to reference sample:")
    corrected_sample = host_frame_correct_sample(
        sample, skymaps=skymaps, scatter_warn_threshold=scatter_warn_threshold)

    print("Applying host-frame correction to your burst:")
    user_src = host_frame_correct_source(
        {"name": name, "ra": ra, "dec": dec, "z": z, "rm": rm, "rm_err": rm_err,
         "tau": tau, "tau_err": tau_err},
        skymaps, scatter_warn_threshold,
    )
    if user_src is None:
        raise ValueError(
            f"Could not apply host-frame correction to your burst '{name}' "
            f"-- check ra/dec/z/rm are all provided and within the "
            f"SPICE-RACS DR2 footprint (Dec <= +49.4 deg)."
        )

    # Freeze nu_centre to a single shared value so plot_rm_tau_correlation's
    # (unrelated) frequency-to-reference-frequency rescale is a no-op --
    # our tau values are already in the rest frame via (1+z)**3 above.
    for src in corrected_sample:
        src["nu_centre"] = 1300.0

    return plot_rm_tau_correlation(
        sigma_rm=sigma_rm,
        sigma_rm_err=sigma_rm_err,
        rm=user_src["rm"],
        tau=abs(user_src["tau"]),
        tau_err=user_src.get("tau_err"),
        sample=corrected_sample,
        output_file=output_file,
        name=name,
        freq_mhz=None,
        ref_freq_mhz=1300.0,
        label_suffix='host',
        exclude_upper_limits=exclude_upper_limits,
    )

    

def main() -> None:
    """CLI entry point for the RM-tau correlation plots."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plot sigma_RM and |RM| vs tau with the Feng+ (2022)/Pandhi+ "
            "(2026)/Uttarkar+ (2024/2026) sample plus your own burst, with "
            "power-law fits to the sample and to all data."
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