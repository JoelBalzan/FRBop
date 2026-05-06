"""Model functions for scattering and scintillation fits."""

import numpy as np
from scipy.special import erfc


def scattered_gaussian(t, amp, mu, sigma, tau, offset):
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)
    arg = (sigma / tau - (t - mu) / sigma) / np.sqrt(2.0)
    exponent = (sigma ** 2) / (2.0 * tau ** 2) - (t - mu) / tau
    exponent = np.clip(exponent, -700, 100)
    return offset + 0.5 * amp * np.exp(exponent) * erfc(arg)


def linear(x, slope, intercept):
    """Simple linear model for log-space power-law fitting."""
    return slope * x + intercept


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
