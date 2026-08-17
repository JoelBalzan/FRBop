"""Shared linear-polarisation helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def pa_from_qu(
    data_q: np.ndarray,
    data_u: np.ndarray,
    unwrap: bool = True,
) -> np.ndarray:
    """Position angle from Stokes Q/U in radians, [-pi/2, pi/2].

    When *unwrap* is True the phase-wrapping inherent in arctan2 is removed
    by unwrapping 2·PA before returning.
    """
    pa = 0.5 * np.arctan2(np.asarray(data_u, dtype=float),
                           np.asarray(data_q, dtype=float))
    if unwrap:
        pa = 0.5 * np.unwrap(2.0 * pa)
    return pa


def pa_degrees_from_qu(
    data_q: np.ndarray,
    data_u: np.ndarray,
    unwrap: bool = True,
) -> np.ndarray:
    """Position angle from Stokes Q/U in degrees, wrapped to [-90, 90]."""
    pa_rad = pa_from_qu(data_q, data_u, unwrap=unwrap)
    pa_deg = np.degrees(pa_rad)
    return ((pa_deg + 90.0) % 180.0) - 90.0


def pa_error_rad(
    q: np.ndarray,
    u: np.ndarray,
    sigma_q: np.ndarray,
    sigma_u: np.ndarray,
    eps: float = 1e-20,
) -> np.ndarray:
    """Propagated PA uncertainty in radians.

    Uses the standard error propagation for PA = 0.5·arctan2(U, Q).
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    lin_sq = q ** 2 + u ** 2 + eps
    return 0.5 * np.sqrt(
        (u ** 2 * np.asarray(sigma_q, dtype=float) ** 2
         + q ** 2 * np.asarray(sigma_u, dtype=float) ** 2)
        / (lin_sq ** 2)
    )


def debiased_linear_from_qu(
    data_q: np.ndarray,
    data_u: np.ndarray,
    noise_q: np.ndarray,
    noise_u: np.ndarray,
    cutoff: float = 1.57,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Ricean-debiased linear polarisation from Stokes Q/U."""
    data_q = np.asarray(data_q, dtype=float)
    data_u = np.asarray(data_u, dtype=float)
    noise_q = np.asarray(noise_q, dtype=float)
    noise_u = np.asarray(noise_u, dtype=float)

    l_meas = np.sqrt(data_q ** 2 + data_u ** 2)
    sigma_l = np.sqrt(data_q ** 2 * noise_q ** 2 + data_u ** 2 * noise_u ** 2) / np.maximum(l_meas, eps)
    det = l_meas / np.maximum(sigma_l, eps) >= cutoff

    l_out = np.zeros_like(l_meas)
    l_out[det] = np.sqrt(np.maximum(l_meas[det] ** 2 - sigma_l[det] ** 2, 0.0))
    return l_out, sigma_l, det