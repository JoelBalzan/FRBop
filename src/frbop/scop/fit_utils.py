"""Fitting utilities for scintillation analysis."""

import numpy as np
from scipy.optimize import curve_fit


def compute_aic_bic(y, ymod, k):
    resid = y - ymod
    rss = np.nansum(resid ** 2)
    n = y.size
    if rss <= 0:
        rss = 1e-12
    aic = 2 * k + n * np.log(rss / n)
    bic = k * np.log(n) + n * np.log(rss / n)
    return aic, bic, rss


def build_fit_diagnostics(y, ymod, k):
    aic, bic, rss = compute_aic_bic(y, ymod, k)
    n = y.size
    rmse = np.sqrt(rss / max(n, 1))
    tss = np.nansum((y - np.nanmean(y)) ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    aicc = aic + (2.0 * k * (k + 1)) / (n - k - 1) if n > (k + 1) else np.nan
    return dict(aic=aic, bic=bic, aicc=aicc, rss=rss, rmse=rmse, r2=r2)


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
