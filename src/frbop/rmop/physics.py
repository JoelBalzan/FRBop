"""
Physics utility functions for RM analysis:
  - sigma_RM detection thresholds (Burn-law)
  - depolarising-medium delta(n_e, B_parallel)
"""

import numpy as np
from scipy.constants import c


def sigma_rm_detection_threshold(freq_center_hz: float) -> float:
    """
    Return the e-fold sigma_RM sensitivity (rad/m^2) at a center frequency.

    Using the Burn-law model:
        P(lambda) = exp[-2 sigma_RM^2 lambda^4]

    The e-fold sensitivity is defined by P(lambda_c) = exp(-1), giving:
        sigma_RM = 1 / (sqrt(2) * lambda_c^2)
    """
    if not np.isfinite(freq_center_hz):
        raise ValueError("freq_center_hz must be finite")
    if freq_center_hz <= 0:
        raise ValueError("freq_center_hz must be > 0")

    lam_c = c / freq_center_hz
    return float(1.0 / (np.sqrt(2.0) * (lam_c ** 2)))


def sigma_rm_detection_threshold_snr(freq_center_hz: float,
                                     pol_snr: float,
                                     nsigma: float = 3.0) -> float:
    """
    Return an S/N-aware sigma_RM detectability threshold (rad/m^2).

    Uses the Burn-law model and requires depolarisation to exceed an
    ``nsigma`` fractional significance:
        1 - P(lambda_c) >= nsigma / SNR
    where
        P(lambda_c) = exp[-2 sigma_RM^2 lambda_c^4].
    """
    if not np.isfinite(freq_center_hz) or freq_center_hz <= 0:
        raise ValueError("freq_center_hz must be finite and > 0")
    if not np.isfinite(pol_snr) or pol_snr <= 0:
        raise ValueError("pol_snr must be finite and > 0")
    if not np.isfinite(nsigma) or nsigma <= 0:
        raise ValueError("nsigma must be finite and > 0")

    # If the data cannot resolve an nsigma fractional drop, mark as undetectable.
    frac_drop = nsigma / pol_snr
    if frac_drop >= 1.0:
        return float(np.inf)

    lam_c = c / freq_center_hz
    p_detect = 1.0 - frac_drop
    return float(np.sqrt(-np.log(p_detect) / (2.0 * (lam_c ** 4))))


def depolarising_medium_delta_ne_b_parallel(sigma_rm: float,
                                            turbulent_radius_pc: float = 21.0,
                                            screen_scale_cm: float = 1e15) -> float:
    """
    Compute delta(n_e, B_parallel) in microGauss/cm^3 from sigma_RM using:

        delta(n_e, B_parallel)
            = 0.2e3 [uG/cm^3]
              * (sigma_RM / 12)
              * (R / 21 pc)^(-1/2)
              * (l_screen / 1e15 cm)^(-1/2)

    Parameters
    ----------
    sigma_rm : float
        RM dispersion in rad/m^2.
    turbulent_radius_pc : float
        Radius of turbulent environment, R, in pc.
    screen_scale_cm : float
        Plasma-screen length scale, l_screen, in cm.
    """
    if not np.isfinite(sigma_rm):
        raise ValueError("sigma_rm must be finite")
    if not np.isfinite(turbulent_radius_pc) or turbulent_radius_pc <= 0:
        raise ValueError("turbulent_radius_pc must be finite and > 0")
    if not np.isfinite(screen_scale_cm) or screen_scale_cm <= 0:
        raise ValueError("screen_scale_cm must be finite and > 0")

    return float(
        0.2e3
        * (sigma_rm / 12.0)
        * (turbulent_radius_pc / 21.0) ** (-0.5)
        * (screen_scale_cm / 1e15) ** (-0.5)
    )
