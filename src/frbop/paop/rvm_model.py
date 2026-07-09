"""
Rotating Vector Model (RVM) for polarisation angle fitting.

Implements the IAU-convention RVM from Everett & Weisberg (2001),
complex-valued fitting of Stokes Q+iU (following psrchive/psrmodel),
and RM correction to infinite frequency.
"""

import numpy as np


def rvm_pa(phi: np.ndarray, phi0: float, psi0: float,
           alpha: float, zeta: float) -> np.ndarray:
    """
    RVM position angle in IAU convention (Everett & Weisberg 2001).

    ψ(φ) = ψ₀ + arctan[ sin(α) sin(φ-φ₀) /
                         (sin(ζ) cos(α) - cos(ζ) sin(α) cos(φ-φ₀)) ]

    Parameters
    ----------
    phi : array-like
        Rotational phase (e.g. time within burst in radians).
    phi0 : float
        Phase at steepest PA slope (magnetic meridian) in radians.
    psi0 : float
        PA at φ₀ in radians.
    alpha : float
        Magnetic colatitude (radians) — angle between rotation and magnetic axes.
    zeta : float
        Observer colatitude (radians) — angle between rotation axis and line of
        sight.
    beta : float
        Impact parameter β = ζ - α (radians).

    Returns
    -------
    pa : array-like
        Position angle in radians (IAU convention).
    """
    dphi = phi - phi0
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_z = np.sin(zeta)
    cos_z = np.cos(zeta)
    num = sin_a * np.sin(dphi)
    den = sin_z * cos_a - cos_z * sin_a * np.cos(dphi)
    return psi0 + np.arctan2(num, den)


def rvm_beta(alpha: float, zeta: float) -> float:
    """Impact parameter β = ζ - α (radians)."""
    return zeta - alpha


def rvm_complex(phi: np.ndarray, phi0: float, psi0: float,
                alpha: float, zeta: float,
                l0: np.ndarray) -> np.ndarray:
    """
    Complex linear polarisation under the RVM: Q + iU = L₀ e^{2iψ}.

    Parameters
    ----------
    phi : array-like
        Rotational phase values.
    phi0, psi0, alpha, zeta : float
        RVM parameters in radians.
    l0 : array-like
        Linear polarisation amplitude at each phase bin.

    Returns
    -------
    L : complex array
        Complex polarisation Q + iU.
    """
    pa = rvm_pa(phi, phi0, psi0, alpha, zeta)
    return l0 * np.exp(2j * pa)


def rvm_qu(phi: np.ndarray, phi0: float, psi0: float,
           alpha: float, zeta: float,
           l0: np.ndarray) -> tuple:
    """
    Stokes Q and U predicted by the RVM.

    Returns (Q, U) arrays.
    """
    L = rvm_complex(phi, phi0, psi0, alpha, zeta, l0)
    return L.real, L.imag


def iau_pa_from_qu(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    IAU-convention PA from Stokes Q, U: PA = 0.5 arctan₂(U, Q).

    Returns PA in radians in [-π/2, π/2].
    """
    return 0.5 * np.arctan2(u, q)


def intensity_from_qu(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linear polarisation intensity L = √(Q² + U²)."""
    return np.sqrt(q**2 + u**2)


def correct_rm_to_inf(pa_obs: np.ndarray, freq_hz: np.ndarray,
                      rm: float) -> np.ndarray:
    """
    Correct observed PA for Faraday rotation to infinite frequency.

    PA_∞ = PA_obs - RM λ²,   where λ = c / ν.

    Parameters
    ----------
    pa_obs : array-like
        Observed PA in radians.
    freq_hz : array-like
        Observation frequencies in Hz.
    rm : float
        Rotation measure in rad / m².

    Returns
    -------
    pa_corr : array-like
        PA corrected to infinite frequency.
    """
    c = 299792458.0
    lam_sq = (c / freq_hz) ** 2
    return pa_obs - rm * lam_sq


def correct_qu_to_inf(q: np.ndarray, u: np.ndarray,
                      freq_hz: np.ndarray, rm: float) -> tuple:
    """
    De-rotate Q/U data to infinite frequency using known RM.

    P_∞ = P_obs exp(-2i RM λ²)

    Returns (Q_inf, U_inf).
    """
    c = 299792458.0
    lam_sq = (c / freq_hz) ** 2
    p_obs = q + 1j * u
    p_inf = p_obs * np.exp(-2j * rm * lam_sq)
    return p_inf.real, p_inf.imag


def rvm_chi2(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
             phi0: float, psi0: float, alpha: float, zeta: float,
             sigma_q: float = 1.0, sigma_u: float = 1.0) -> float:
    """
    χ² for RVM parameters marginalising over per-bin L amplitudes.

    At each phase bin the optimal linear-amplitude for a given PA is the
    projection of (Q, U) onto the model direction:

        L̂ = arg min |P - L e^{2iψ}|² = Q cos(2ψ) + U sin(2ψ)

    so χ² = Σ [(Q - L̂ cos(2ψ))² + (U - L̂ sin(2ψ))²] / σ².

    Parameters
    ----------
    phi : np.array
        Rotational phase.
    q, u : np.array
        Stokes Q, U data.
    phi0, psi0, alpha, zeta : float
        RVM parameters in radians.
    sigma_q, sigma_u : float
        Per-bin noise standard deviation for Q and U.

    Returns
    -------
    chi2 : float
        Chi-squared value.
    """
    pa = rvm_pa(phi, phi0, psi0, alpha, zeta)
    cos2p = np.cos(2 * pa)
    sin2p = np.sin(2 * pa)
    l_hat = q * cos2p + u * sin2p
    cos_res = q - l_hat * cos2p
    sin_res = u - l_hat * sin2p
    return float(np.sum(cos_res ** 2 / sigma_q ** 2 +
                        sin_res ** 2 / sigma_u ** 2))
