"""
RVM fitting with grid-search initialisation, Nelder-Mead refinement,
and MCMC uncertainty estimation (via emcee).

Supports two modes:
  4-param  — phi provided directly (backward compatible).
  5-param  — time (ms) provided; phase = k × time where k is a free
             parameter (the phase sweep rate in rad/ms).  This avoids
             requiring a known rotation period for FRB data.

Follows the psrchive/psrmodel approach:
  1. Grid over α (0–180°) and ζ (0–180°) to locate the global χ² minimum.
     At each grid point, φ₀ is scanned and ψ₀ is solved analytically.
  2. (k-mode) Re-scan the top N grid points over k to refine the sweep rate.
  3. Refine the best guess with Nelder-Mead (scipy).
  4. Sample the posterior with emcee (4 or 5 parameters; L_i marginalised).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .rvm_model import iau_pa_from_qu, rvm_chi2, rvm_pa

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Grid-search helpers
# ──────────────────────────────────────────────

def _optimal_psi0(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
                  phi0: float, alpha: float, zeta: float) -> float:
    """
    Analytic ψ₀ that minimises χ² for a fixed (φ₀, α, ζ).
    Returns ψ₀ in radians.
    """
    dphi = phi - phi0
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_z = np.sin(zeta)
    cos_z = np.cos(zeta)
    f = np.arctan2(sin_a * np.sin(dphi),
                   sin_z * cos_a - cos_z * sin_a * np.cos(dphi))

    cos2f = np.cos(2 * f)
    sin2f = np.sin(2 * f)
    A = q * cos2f + u * sin2f
    B = u * cos2f - q * sin2f
    AA = np.sum(A**2)
    BB = np.sum(B**2)
    AB = np.sum(A * B)

    psi0_opt = 0.25 * np.arctan2(2.0 * AB, AA - BB)
    candidates = [psi0_opt + k * np.pi / 4.0 for k in range(4)]
    best = min(candidates,
               key=lambda ps: rvm_chi2(phi, q, u, phi0, ps, alpha, zeta))
    return best


def _grid_phi0(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
               alpha: float, zeta: float,
               sigma_q: float = 1.0, sigma_u: float = 1.0,
               n_phi: int = 51) -> Tuple[float, float]:
    """
    Scan φ₀ across the observed phase range and return (best φ₀, best χ²).
    """
    lo, hi = float(phi.min()), float(phi.max())
    pad = 0.5 * (hi - lo)
    phi0_grid = np.linspace(lo - pad, hi + pad, n_phi)
    best_chi2 = np.inf
    best_phi0 = phi0_grid[0]
    for p0 in phi0_grid:
        ps0 = _optimal_psi0(phi, q, u, p0, alpha, zeta)
        c2 = rvm_chi2(phi, q, u, p0, ps0, alpha, zeta,
                      sigma_q=sigma_q, sigma_u=sigma_u)
        if c2 < best_chi2:
            best_chi2 = c2
            best_phi0 = p0
    return best_phi0, best_chi2


# ──────────────────────────────────────────────
#  Grid search
# ──────────────────────────────────────────────

def grid_search(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
                n_alpha: int = 40, n_zeta: int = 40,
                n_phi: int = 101,
                sigma_q: float = 1.0, sigma_u: float = 1.0,
                progress: bool = True) -> dict:
    """
    Grid search over α and ζ (0–180°), analytically optimising ψ₀ and φ₀
    at each point.

    Parameters
    ----------
    phi : (N,) array
        Rotational phase.
    q, u : (N,) array
        Stokes Q, U.
    n_alpha, n_zeta : int
        Number of grid points for α and ζ.
    n_phi : int
        Number of φ₀ grid points in each sub-scan.
    sigma_q, sigma_u : float
        Noise per bin.
    progress : bool
        Print progress.

    Returns
    -------
    result : dict
        Keys: phi0, psi0, alpha, zeta, chi2, grid_alpha, grid_zeta, grid_chi2.
    """
    alpha_deg = np.linspace(2.0, 178.0, n_alpha)
    zeta_deg = np.linspace(2.0, 178.0, n_zeta)
    alpha_rad = np.radians(alpha_deg)
    zeta_rad = np.radians(zeta_deg)

    best_chi2 = np.inf
    best = {"phi0": 0.0, "psi0": 0.0, "alpha": 0.0, "zeta": 0.0}

    grid_chi2 = np.full((n_alpha, n_zeta), np.inf)

    for i, a_rad in enumerate(alpha_rad):
        for j, z_rad in enumerate(zeta_rad):
            if abs(a_rad - z_rad) < np.radians(1.0):
                continue
            p0, c2 = _grid_phi0(phi, q, u, a_rad, z_rad,
                                sigma_q=sigma_q, sigma_u=sigma_u,
                                n_phi=n_phi)
            grid_chi2[i, j] = c2
            if c2 < best_chi2:
                best_chi2 = c2
                best["phi0"] = p0
                best["psi0"] = _optimal_psi0(phi, q, u, p0, a_rad, z_rad)
                best["alpha"] = float(a_rad)
                best["zeta"] = float(z_rad)

    best["chi2"] = best_chi2
    best["grid_alpha"] = alpha_rad
    best["grid_zeta"] = zeta_rad
    best["grid_chi2"] = grid_chi2
    return best


# ──────────────────────────────────────────────
#  K-scan at a single grid point
# ──────────────────────────────────────────────

def _scan_k_grid_point(alpha: float, zeta: float,
                       time: np.ndarray, q: np.ndarray, u: np.ndarray,
                       k_grid: np.ndarray,
                       sigma_q: float, sigma_u: float,
                       n_phi: int = 101) -> dict:
    """
    For a fixed (α, ζ) scan over *k_grid* (rad/ms).  For each k compute
    phi = k × time and scan φ₀ with analytic ψ₀.
    """
    best = {"k": k_grid[0], "phi0": 0.0, "psi0": 0.0, "chi2": np.inf}
    for k in k_grid:
        phi = k * time
        p0, c2 = _grid_phi0(phi, q, u, alpha, zeta,
                            sigma_q=sigma_q, sigma_u=sigma_u, n_phi=n_phi)
        if c2 < best["chi2"]:
            best["k"] = k
            best["phi0"] = p0
            best["psi0"] = _optimal_psi0(phi, q, u, p0, alpha, zeta)
            best["chi2"] = c2
    best["alpha"] = float(alpha)
    best["zeta"] = float(zeta)
    return best


def _scan_k_grid_points(grid_result: dict,
                        time: np.ndarray, q: np.ndarray, u: np.ndarray,
                        k_grid: np.ndarray,
                        sigma_q: float, sigma_u: float,
                        n_phi: int = 101,
                        n_points: int = 20) -> dict:
    """
    Re-scan the best *n_points* (α, ζ) grid cells over *k_grid*.
    Returns the best (α, ζ, φ₀, ψ₀, k, χ²) found.
    """
    gchi2 = grid_result["grid_chi2"]
    flat_idx = np.argsort(gchi2.ravel())
    n_valid = min(n_points, len(flat_idx))
    best = {"chi2": np.inf}
    for idx in flat_idx[:n_valid]:
        i = idx // gchi2.shape[1]
        j = idx % gchi2.shape[1]
        a = grid_result["grid_alpha"][i]
        z = grid_result["grid_zeta"][j]
        kr = _scan_k_grid_point(a, z, time, q, u, k_grid,
                                sigma_q, sigma_u, n_phi)
        if kr["chi2"] < best["chi2"]:
            best = kr
    return best


# ──────────────────────────────────────────────
#  Local refinement  (4 or 5 params, L marginalised)
# ──────────────────────────────────────────────

def _chi2_wrapper(params_vec: np.ndarray, phi: np.ndarray,
                  q: np.ndarray, u: np.ndarray,
                  sigma_q: float, sigma_u: float,
                  time: Optional[np.ndarray] = None) -> float:
    """χ² as scalar function of a plain parameter vector for minimize().
    When *time* is provided, phase = k × time (5-param mode)."""
    if time is not None:
        k = params_vec[-1]
        phi0, psi0, alpha, zeta = params_vec[:4]
        phi_in = k * time
    else:
        phi0, psi0, alpha, zeta = params_vec
        phi_in = phi
    return rvm_chi2(phi_in, q, u, phi0, psi0, alpha, zeta,
                    sigma_q=sigma_q, sigma_u=sigma_u)


def _nelder_mead_min(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
                     x0: np.ndarray,
                     sigma_q: float, sigma_u: float,
                     max_nfev: int = 5000,
                     time: Optional[np.ndarray] = None) -> Optional[dict]:
    """Run a single Nelder-Mead minimisation from *x0*."""
    from scipy.optimize import minimize as sp_minimize
    if time is not None:
        bounds = [(None, None), (None, None),
                  (np.radians(2), np.radians(178)),
                  (np.radians(2), np.radians(178)),
                  (1e-12, None)]
    else:
        bounds = [(None, None), (None, None),
                  (np.radians(2), np.radians(178)),
                  (np.radians(2), np.radians(178))]
    try:
        res = sp_minimize(
            _chi2_wrapper, x0,
            args=(phi, q, u, sigma_q, sigma_u, time),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": max_nfev, "xatol": 1e-7, "fatol": 1e-7},
        )
        if not res.success:
            return None
        out = {
            "phi0": float(res.x[0]),
            "psi0": float(res.x[1]),
            "alpha": float(res.x[2]),
            "zeta": float(res.x[3]),
            "chi2": float(res.fun),
            "nfev": res.nfev,
        }
        if time is not None:
            out["k"] = float(res.x[4])
        return out
    except Exception:
        return None


def lm_refine(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
              phi0: float, psi0: float, alpha: float, zeta: float,
              sigma_q: float = 1.0, sigma_u: float = 1.0,
              max_nfev: int = 5000,
              time: Optional[np.ndarray] = None,
              k: Optional[float] = None) -> Optional[dict]:
    """
    Refine RVM parameters with Nelder‑Mead (scipy).

    When *time* is provided (5-param mode), the starting points include
    the best guess and the complementary geometry, each tried at
    ``k``, ``2k`` and ``k/2``.
    """
    if time is not None and k is not None:
        starts = [
            np.array([phi0, psi0, alpha, zeta, k]),
            np.array([phi0, psi0, np.pi - alpha, np.pi - zeta, k]),
            np.array([phi0, psi0, alpha, zeta, k * 2.0]),
            np.array([phi0, psi0, alpha, zeta, k / 2.0]),
        ]
    else:
        starts = [
            np.array([phi0, psi0, alpha, zeta]),
            np.array([phi0, psi0, np.pi - alpha, np.pi - zeta]),
        ]

    best = None
    best_chi2 = np.inf
    for x in starts:
        r = _nelder_mead_min(phi, q, u, x, sigma_q, sigma_u, max_nfev, time=time)
        if r is not None and r["chi2"] < best_chi2:
            best_chi2 = r["chi2"]
            best = r

    if best is None:
        return None

    phi_eff = best["k"] * time if (time is not None and best.get("k") is not None) else phi
    pa_fit = rvm_pa(phi_eff, best["phi0"], best["psi0"],
                    best["alpha"], best["zeta"])
    cos2p = np.cos(2 * pa_fit)
    sin2p = np.sin(2 * pa_fit)
    L_fit = q * cos2p + u * sin2p
    L_fit = np.maximum(L_fit, 0.0)

    best["L_fit"] = L_fit
    best["pa_fit"] = pa_fit
    return best


# ──────────────────────────────────────────────
#  MCMC sampling  (emcee)
# ──────────────────────────────────────────────

def _log_prob(params: np.ndarray, phi: np.ndarray,
              q: np.ndarray, u: np.ndarray,
              sigma_q: float, sigma_u: float,
              time: Optional[np.ndarray] = None) -> float:
    """
    Log-posterior for 4 or 5 RVM parameters.  When *time* is given,
    the last element of *params* is k (rad/ms) and phase = k × time.
    """
    if time is not None:
        k_val = params[-1]
        phi0, psi0, alpha, zeta = params[:4]
        if not np.isfinite(phi0 + psi0 + alpha + zeta + k_val):
            return -np.inf
        if k_val <= 0:
            return -np.inf
        phi_in = k_val * time
    else:
        phi0, psi0, alpha, zeta = params
        phi_in = phi
        if not np.isfinite(phi0 + psi0 + alpha + zeta):
            return -np.inf

    # Priors
    phi_range = float(phi_in.max() - phi_in.min())
    phi_lo, phi_hi = float(phi_in.min()) - phi_range, float(phi_in.max()) + phi_range
    if not (phi_lo <= phi0 <= phi_hi):
        return -np.inf
    if not (-np.pi / 2.0 <= psi0 <= np.pi / 2.0):
        return -np.inf
    if not (np.radians(1) <= alpha <= np.radians(179)):
        return -np.inf
    if not (np.radians(1) <= zeta <= np.radians(179)):
        return -np.inf
    if abs(alpha - zeta) < np.radians(1):
        return -np.inf

    # Likelihood
    pa = rvm_pa(phi_in, phi0, psi0, alpha, zeta)
    cos2p = np.cos(2 * pa)
    sin2p = np.sin(2 * pa)
    l_hat = q * cos2p + u * sin2p
    res_q = q - l_hat * cos2p
    res_u = u - l_hat * sin2p
    chi2 = np.sum(res_q**2 / sigma_q**2 + res_u**2 / sigma_u**2)
    return -0.5 * chi2


def run_mcmc(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
             phi0: float, psi0: float, alpha: float, zeta: float,
             sigma_q: float = 1.0, sigma_u: float = 1.0,
             n_walkers: int = 32, n_steps: int = 3000,
             n_burn: int = 1000, n_thin: int = 5,
             progress: bool = True,
             time: Optional[np.ndarray] = None,
             k: Optional[float] = None) -> Optional[dict]:
    """
    MCMC sampling with emcee.

    In 5-param mode (time + k provided) the parameters are
    [φ₀, ψ₀, α, ζ, k]; otherwise 4-param mode.
    """
    if not HAS_EMCEE:
        logger.error("emcee not installed — cannot run MCMC")
        return None

    if time is not None and k is not None:
        ndim = 5
        p0 = np.array([phi0, psi0, alpha, zeta, k])
        rng = np.random.RandomState(42)
        start = np.empty((n_walkers, ndim))
        # φ₀, ψ₀, α, ζ — additive Gaussian perturbations
        start[:, 0] = p0[0] + max(abs(p0[0]) * 0.01, np.radians(0.5)) * rng.randn(n_walkers)
        start[:, 1] = p0[1] + max(abs(p0[1]) * 0.01, np.radians(0.5)) * rng.randn(n_walkers)
        start[:, 2] = np.clip(p0[2] + np.radians(3.0) * rng.randn(n_walkers),
                              np.radians(2), np.radians(178))
        start[:, 3] = np.clip(p0[3] + np.radians(3.0) * rng.randn(n_walkers),
                              np.radians(2), np.radians(178))
        # k — log-normal perturbations (multiplicative, always > 0)
        start[:, 4] = p0[4] * np.exp(0.15 * rng.randn(n_walkers))
        log_prob_args = (phi, q, u, sigma_q, sigma_u, time)
    else:
        ndim = 4
        p0 = np.array([phi0, psi0, alpha, zeta])
        scales = np.array([
            max(abs(p0[0]) * 0.01, np.radians(0.5)),
            max(abs(p0[1]) * 0.01, np.radians(0.5)),
            np.radians(3.0),
            np.radians(3.0),
        ])
        start = p0 + scales * np.random.RandomState(42).randn(n_walkers, ndim)
        log_prob_args = (phi, q, u, sigma_q, sigma_u, None)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, _log_prob,
        args=log_prob_args,
    )

    try:
        sampler.run_mcmc(start, n_steps, progress=progress)
    except Exception as exc:
        logger.warning(f"MCMC failed: {exc}")
        return None

    chain = sampler.chain[:, n_burn:, :]
    flat = chain.reshape(-1, ndim)[::n_thin]

    logp_chain = sampler.lnprobability[:, n_burn:].ravel()[::n_thin]
    map_idx = int(np.argmax(logp_chain))
    map_params = flat[map_idx]

    phi_map = map_params[-1] * time if (time is not None) else phi
    pa_map = rvm_pa(phi_map, map_params[0], map_params[1],
                    map_params[2], map_params[3])
    cos2p = np.cos(2 * pa_map)
    sin2p = np.sin(2 * pa_map)
    l_map = q * cos2p + u * sin2p
    chi2_map = np.sum((q - l_map * cos2p)**2 / sigma_q**2
                      + (u - l_map * sin2p)**2 / sigma_u**2)

    return {
        "chain": chain,
        "flatchain": flat,
        "map_params": map_params,
        "map_chi2": chi2_map,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "n_burn": n_burn,
        "n_thin": n_thin,
        "ndim": ndim,
    }


# ──────────────────────────────────────────────
#  Top-level fit driver
# ──────────────────────────────────────────────

def fit_rvm(phi: Optional[np.ndarray] = None,
            q: np.ndarray = None, u: np.ndarray = None,
            sigma_q: Optional[float] = None,
            sigma_u: Optional[float] = None,
            time: Optional[np.ndarray] = None,
            time_phase_cycles: Optional[float] = None,
            n_alpha: int = 40, n_zeta: int = 40, n_phi: int = 101,
            n_k: int = 20, k_min_frac: float = 0.1,
            k_max_frac: float = 10.0, n_grid_refine: int = 20,
            do_lm: bool = True,
            do_mcmc: bool = True,
            mcmc_walkers: int = 32, mcmc_steps: int = 3000,
            mcmc_burn: int = 1000, mcmc_thin: int = 5,
            mcmc_progress: bool = True) -> dict:
    """
    Full RVM fitting pipeline.

    Two modes:

    **4-param (backward compatible)** — *phi* is provided directly as the
    rotational phase array.  *time_phase_cycles* can optionally rescale it.

    **5-param (recommended for FRB data)** — *time* (ms) is provided
    instead of *phi*.  The phase is ``φ = k × t`` where **k** is a free
    parameter (phase sweep rate in rad/ms).  This avoids requiring a
    known rotation period.

    Parameters
    ----------
    phi : (N,) array or None
        Rotational phase in radians.  Not used if *time* is provided.
    q, u : (N,) array
        Stokes Q, U data.
    sigma_q, sigma_u : float or None
        Noise per bin.  Estimated from MAD if None.
    time : (N,) array or None
        Time array in ms.  When provided, *phi* is ignored and *k* (rad/ms)
        becomes a free parameter (5-param mode).
    time_phase_cycles : float or None
        Legacy rescaling for *phi* (ignored in 5-param mode).
    n_alpha, n_zeta, n_phi : int
        Grid sizes for the α×ζ search.
    n_k : int
        Number of k values in the log-spaced scan.
    k_min_frac, k_max_frac : float
        Range of the k scan relative to k₀ = 2π/ptp(time).
    n_grid_refine : int
        Number of top grid cells re-scanned over k.
    do_lm : bool
        Run Nelder-Mead refinement after grid search.
    do_mcmc : bool
        Run MCMC.

    Returns
    -------
    result : dict
        Keys documented in *fit_rvm* output description.
    """
    if sigma_q is None:
        sigma_q = 1.0
    if sigma_u is None:
        sigma_u = 1.0

    # ── Detect mode ──────────────────────────────────────────────────
    use_k = time is not None
    result = {"grid": None, "lm": None, "mcmc": None, "use_k": use_k,
              "k": None}

    if use_k:
        time = np.asarray(time, dtype=float).ravel()
        k0 = 2.0 * np.pi / np.ptp(time)
        phi_at_k0 = k0 * time

        # --- 1a. α×ζ grid at default k₀ ---
        grid_result = grid_search(phi_at_k0, q, u,
                                  n_alpha=n_alpha, n_zeta=n_zeta,
                                  n_phi=n_phi,
                                  sigma_q=sigma_q, sigma_u=sigma_u)
        result["grid"] = grid_result

        # --- 1b. K-scan at top N grid cells ---
        k_grid = np.logspace(np.log10(k0 * k_min_frac),
                             np.log10(k0 * k_max_frac), n_k)
        k_best = _scan_k_grid_points(
            grid_result, time, q, u, k_grid,
            sigma_q, sigma_u, n_phi, n_points=n_grid_refine,
        )
        best_k = k_best["k"]
        best_phi0 = k_best["phi0"]
        best_psi0 = k_best["psi0"]
        best_alpha = k_best["alpha"]
        best_zeta = k_best["zeta"]
        phi_norm = best_k * time

    else:
        # --- 1. Standard grid search (4-param) ---
        if time_phase_cycles is not None:
            phi_norm = phi / np.ptp(phi) * 2.0 * np.pi * time_phase_cycles
        else:
            phi_norm = phi.copy()

        grid_result = grid_search(phi_norm, q, u,
                                  n_alpha=n_alpha, n_zeta=n_zeta,
                                  n_phi=n_phi,
                                  sigma_q=sigma_q, sigma_u=sigma_u)
        result["grid"] = grid_result
        best_k = None
        best_phi0 = grid_result["phi0"]
        best_psi0 = grid_result["psi0"]
        best_alpha = grid_result["alpha"]
        best_zeta = grid_result["zeta"]

    # --- 2. Nelder-Mead refinement ---
    if do_lm:
        if use_k:
            lm_result = lm_refine(
                phi_norm, q, u,
                best_phi0, best_psi0, best_alpha, best_zeta,
                sigma_q=sigma_q, sigma_u=sigma_u,
                time=time, k=best_k,
            )
        else:
            lm_result = lm_refine(
                phi_norm, q, u,
                best_phi0, best_psi0, best_alpha, best_zeta,
                sigma_q=sigma_q, sigma_u=sigma_u,
            )

        if lm_result and lm_result["chi2"] < grid_result.get("chi2", np.inf):
            result["lm"] = lm_result
            best_phi0 = lm_result["phi0"]
            best_psi0 = lm_result["psi0"]
            best_alpha = lm_result["alpha"]
            best_zeta = lm_result["zeta"]
            if use_k and lm_result.get("k") is not None:
                best_k = lm_result["k"]
        else:
            result["lm"] = None

    if use_k:
        phi_norm = best_k * time

    # Wrap ψ₀ to [-π/2, π/2] so it passes the MCMC prior
    best_psi0 = (best_psi0 + np.pi / 2) % np.pi - np.pi / 2

    # --- 3. MCMC ---
    if do_mcmc:
        if use_k:
            mcmc_result = run_mcmc(
                phi_norm, q, u,
                best_phi0, best_psi0, best_alpha, best_zeta,
                sigma_q=sigma_q, sigma_u=sigma_u,
                n_walkers=mcmc_walkers, n_steps=mcmc_steps,
                n_burn=mcmc_burn, n_thin=mcmc_thin,
                progress=mcmc_progress,
                time=time, k=best_k,
            )
        else:
            mcmc_result = run_mcmc(
                phi_norm, q, u,
                best_phi0, best_psi0, best_alpha, best_zeta,
                sigma_q=sigma_q, sigma_u=sigma_u,
                n_walkers=mcmc_walkers, n_steps=mcmc_steps,
                n_burn=mcmc_burn, n_thin=mcmc_thin,
                progress=mcmc_progress,
            )
        result["mcmc"] = mcmc_result

    # --- Derived best-fit ---
    if result["mcmc"] is not None:
        b = result["mcmc"]["map_params"]
        result["best_phi0"] = float(b[0])
        result["best_psi0"] = float(b[1])
        result["best_alpha"] = float(b[2])
        result["best_zeta"] = float(b[3])
        if use_k:
            result["best_k"] = float(b[4])
            best_k = float(b[4])
    elif result["lm"] is not None:
        result["best_phi0"] = result["lm"]["phi0"]
        result["best_psi0"] = result["lm"]["psi0"]
        result["best_alpha"] = result["lm"]["alpha"]
        result["best_zeta"] = result["lm"]["zeta"]
        if use_k:
            result["best_k"] = result["lm"]["k"]
    else:
        result["best_phi0"] = result["grid"]["phi0"]
        result["best_psi0"] = result["grid"]["psi0"]
        result["best_alpha"] = result["grid"]["alpha"]
        result["best_zeta"] = result["grid"]["zeta"]
        if use_k:
            result["best_k"] = float(best_k)

    # Ensure ψ₀ is in [-π/2, π/2] for consistency
    result["best_psi0"] = (result["best_psi0"] + np.pi / 2) % np.pi - np.pi / 2

    result["best_beta"] = result["best_zeta"] - result["best_alpha"]
    result["best_chi2"] = (
        result["mcmc"]["map_chi2"] if result["mcmc"] is not None
        else result["lm"]["chi2"] if result["lm"] is not None
        else result["grid"]["chi2"]
    )
    result["sigma_q"] = sigma_q
    result["sigma_u"] = sigma_u

    if use_k:
        phi_best = result.get("best_k", best_k) * time
    else:
        phi_best = phi_norm

    # Resolve ψ₀ ± π/2 ambiguity
    _pa_data = iau_pa_from_qu(q, u)
    _candidates = [result["best_psi0"],
                   result["best_psi0"] + np.pi / 2,
                   result["best_psi0"] - np.pi / 2]
    _best_d = np.inf
    for _ps in _candidates:
        _pa_model = rvm_pa(phi_best, result["best_phi0"], _ps,
                           result["best_alpha"], result["best_zeta"])
        _pw = (_pa_model + np.pi / 2) % np.pi - np.pi / 2
        _d = np.sum(np.abs(_pw - _pa_data))
        if _d < _best_d:
            _best_d = _d
            result["best_psi0"] = _ps

    result["best_pa"] = rvm_pa(
        phi_best,
        result["best_phi0"], result["best_psi0"],
        result["best_alpha"], result["best_zeta"],
    )
    result["best_L"] = (
        result["lm"]["L_fit"] if (result["lm"] is not None
                                  and "L_fit" in result["lm"])
        else None
    )
    result["best_k"] = best_k
    result["phi_fit"] = phi_best

    return result
