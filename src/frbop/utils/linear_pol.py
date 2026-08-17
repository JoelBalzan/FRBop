"""Shared linear-polarisation helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


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