#!/usr/bin/env python3
"""
RM Fitting Script for Stokes IQUV Data

This script performs Rotation Measure (RM) fitting on Stokes IQUV polarisation data
in both frequency and time domains using the RM synthesis technique.

Author: Generated Script
Date: 2 February 2026
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from scipy.constants import c
from scipy.optimize import curve_fit

from frbop.plotting import publication_plot_style, savefig_rasterized

warnings.filterwarnings('ignore')


TWO_COLUMN_WIDTH_IN = 7.1
SINGLE_COLUMN_WIDTH_IN = 4.8


def _pub_figsize(height_ratio: float = 0.62, min_height: float = 3.0) -> Tuple[float, float]:
    """Return a publication-friendly figure size for a two-column layout."""
    height = max(min_height, TWO_COLUMN_WIDTH_IN * height_ratio)
    return TWO_COLUMN_WIDTH_IN, height


def _summarize_posterior(posterior_values: np.ndarray,
                         low_percentile: float = 16.0,
                         high_percentile: float = 84.0) -> Tuple[float, float, float]:
    """Return median and bounds for posterior samples."""
    values = np.asarray(posterior_values)
    median = np.nanmedian(values)
    low = np.nanpercentile(values, low_percentile)
    high = np.nanpercentile(values, high_percentile)
    return median, low, high


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


def time_series_sigma_rm_diagnostic(rm_time: np.ndarray,
                                    weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute descriptive diagnostics for time-series RM measurements.

    Parameters
    ----------
    rm_time : array-like
        RM values per time bin (rad/m^2). NaNs are ignored.
    weights : array-like, optional
        Per-bin non-negative weights for weighted diagnostics. If provided,
        weighted values are computed over bins where RM and weight are finite
        and weight > 0.

    Returns
    -------
    dict
        Dictionary with unweighted and (when possible) weighted diagnostics.
    """
    rm_arr = np.asarray(rm_time, dtype=float)
    valid = np.isfinite(rm_arr)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            'rm_mean': np.nan,
            'sigma_rm_time': np.nan,
            'rm_min': np.nan,
            'rm_max': np.nan,
            'weighted_rm_mean': np.nan,
            'weighted_sigma_rm_time': np.nan,
            'weighted_n': 0,
            'weight_sum': 0.0,
            'n_valid': 0,
            'n_total': int(rm_arr.size),
        }

    rm_valid = rm_arr[valid]
    weighted_rm_mean = np.nan
    weighted_sigma_rm_time = np.nan
    weighted_n = 0
    weight_sum = 0.0

    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.shape == rm_arr.shape:
            w_mask = valid & np.isfinite(w_arr) & (w_arr > 0)
            if np.any(w_mask):
                rm_w = rm_arr[w_mask]
                w = w_arr[w_mask]
                weight_sum = float(np.sum(w))
                weighted_n = int(np.sum(w_mask))
                if weight_sum > 0:
                    weighted_rm_mean = float(np.sum(w * rm_w) / weight_sum)
                    weighted_sigma_rm_time = float(
                        np.sqrt(np.sum(w * (rm_w - weighted_rm_mean) ** 2) / weight_sum)
                    )

    return {
        'rm_mean': float(np.mean(rm_valid)),
        'sigma_rm_time': float(np.std(rm_valid)),
        'rm_min': float(np.min(rm_valid)),
        'rm_max': float(np.max(rm_valid)),
        'weighted_rm_mean': weighted_rm_mean,
        'weighted_sigma_rm_time': weighted_sigma_rm_time,
        'weighted_n': weighted_n,
        'weight_sum': weight_sum,
        'n_valid': n_valid,
        'n_total': int(rm_arr.size),
    }


def _compute_poincare_point_errors(time_series_data: Dict,
                                   point_times: np.ndarray,
                                   noise_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate q/u/v uncertainties per plotted point using off-pulse full dspec noise."""
    if time_series_data is None:
        raise ValueError("time_series_data is required for Poincare error estimates")

    I_cube = np.asarray(time_series_data['I'], dtype=float)
    Q_cube = np.asarray(time_series_data['Q'], dtype=float)
    U_cube = np.asarray(time_series_data['U'], dtype=float)
    V_cube = np.asarray(time_series_data.get('V', np.zeros_like(I_cube)), dtype=float)
    times = np.asarray(time_series_data['time'], dtype=float)
    if I_cube.ndim != 2:
        raise ValueError("time_series_data['I'] must be 2D")

    n_time = len(times)
    time_axis = 0 if I_cube.shape[0] == n_time else 1

    if time_axis == 0:
        I_t = np.nanmean(I_cube, axis=1)
        Q_t = np.nanmean(Q_cube, axis=1)
        U_t = np.nanmean(U_cube, axis=1)
        V_t = np.nanmean(V_cube, axis=1)
    else:
        I_t = np.nanmean(I_cube, axis=0)
        Q_t = np.nanmean(Q_cube, axis=0)
        U_t = np.nanmean(U_cube, axis=0)
        V_t = np.nanmean(V_cube, axis=0)

    n_off = max(1, int(n_time * noise_fraction))

    def _robust_sigma(arr: np.ndarray) -> float:
        sig = float(np.nanstd(arr[:n_off]))
        if np.isfinite(sig) and sig > 0:
            return sig
        mad = np.nanmedian(np.abs(arr[:n_off] - np.nanmedian(arr[:n_off])))
        if np.isfinite(mad) and mad > 0:
            return float(mad / 0.6745)
        return 1e-10

    sigma_i0 = _robust_sigma(I_t)
    sigma_q0 = _robust_sigma(Q_t)
    sigma_u0 = _robust_sigma(U_t)
    sigma_v0 = _robust_sigma(V_t)

    centers = np.asarray(point_times, dtype=float)
    n_pts = centers.size
    sigma_q = np.full(n_pts, np.nan, dtype=float)
    sigma_u = np.full(n_pts, np.nan, dtype=float)
    sigma_v = np.full(n_pts, np.nan, dtype=float)

    if n_pts == 0:
        return sigma_q, sigma_u, sigma_v

    if n_pts == 1:
        boundaries = np.array([-np.inf, np.inf], dtype=float)
    else:
        mids = 0.5 * (centers[:-1] + centers[1:])
        boundaries = np.concatenate(([-np.inf], mids, [np.inf]))

    for i in range(n_pts):
        left = boundaries[i]
        right = boundaries[i + 1]
        if i < n_pts - 1:
            mask_t = (times >= left) & (times < right)
        else:
            mask_t = (times >= left) & (times <= right)

        if not np.any(mask_t):
            idx = int(np.argmin(np.abs(times - centers[i])))
            mask_t = np.zeros_like(times, dtype=bool)
            mask_t[idx] = True

        n_bin = int(np.sum(mask_t))
        n_bin = max(1, n_bin)

        I_m = float(np.nanmean(I_t[mask_t]))
        Q_m = float(np.nanmean(Q_t[mask_t]))
        U_m = float(np.nanmean(U_t[mask_t]))
        V_m = float(np.nanmean(V_t[mask_t]))

        sI = sigma_i0 / np.sqrt(n_bin)
        sQ = sigma_q0 / np.sqrt(n_bin)
        sU = sigma_u0 / np.sqrt(n_bin)
        sV = sigma_v0 / np.sqrt(n_bin)

        denom = I_m + 1e-10
        sigma_q[i] = np.sqrt((sQ / denom) ** 2 + ((Q_m * sI) / (denom ** 2)) ** 2)
        sigma_u[i] = np.sqrt((sU / denom) ** 2 + ((U_m * sI) / (denom ** 2)) ** 2)
        sigma_v[i] = np.sqrt((sV / denom) ** 2 + ((V_m * sI) / (denom ** 2)) ** 2)

    return sigma_q, sigma_u, sigma_v


def _poincare_angle_errors_deg(q: np.ndarray,
                               u: np.ndarray,
                               v: np.ndarray,
                               sigma_q: np.ndarray,
                               sigma_u: np.ndarray,
                               sigma_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate q/u/v errors to lon/lat errors (degrees)."""
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    sq = np.asarray(sigma_q, dtype=float)
    su = np.asarray(sigma_u, dtype=float)
    sv = np.asarray(sigma_v, dtype=float)

    qu2 = q ** 2 + u ** 2 + 1e-20
    r = np.sqrt(q ** 2 + u ** 2 + v ** 2 + 1e-20)

    sigma_lon_rad = np.sqrt((u ** 2 * sq ** 2 + q ** 2 * su ** 2) / (qu2 ** 2))

    x = np.clip(v / r, -1.0, 1.0)
    dxdq = -v * q / (r ** 3)
    dxdu = -v * u / (r ** 3)
    dxdv = (q ** 2 + u ** 2) / (r ** 3)
    dlatdx = 1.0 / np.sqrt(1.0 - x ** 2 + 1e-20)
    sigma_lat_rad = np.sqrt(
        (dlatdx * dxdq * sq) ** 2 +
        (dlatdx * dxdu * su) ** 2 +
        (dlatdx * dxdv * sv) ** 2
    )

    return np.degrees(sigma_lon_rad), np.degrees(sigma_lat_rad)


def _build_circle_segments(n_points: int,
                           segment_pairs: Optional[List[Tuple[int, int]]],
                           filtered_indices: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Create valid inclusive [start, end] segments over plotted points."""
    if n_points <= 0:
        return []

    # Auto-segment mode: when the flag is passed with no explicit pairs,
    # split into contiguous runs from the masking outcome.
    if segment_pairs == []:
        if filtered_indices is None or len(filtered_indices) == 0:
            return [(0, n_points - 1)]
        idx = np.asarray(filtered_indices, dtype=int)
        runs: List[Tuple[int, int]] = []
        run_start = 0
        for i in range(1, len(idx)):
            if idx[i] != (idx[i - 1] + 1):
                if (i - run_start) >= 3:
                    runs.append((run_start, i - 1))
                run_start = i
        if (len(idx) - run_start) >= 3:
            runs.append((run_start, len(idx) - 1))
        return runs if runs else [(0, n_points - 1)]

    if not segment_pairs:
        return [(0, n_points - 1)]

    out: List[Tuple[int, int]] = []
    for s_raw, e_raw in segment_pairs:
        s = max(0, min(int(s_raw), n_points - 1))
        e = max(0, min(int(e_raw), n_points - 1))
        if e < s:
            s, e = e, s
        if (e - s + 1) >= 3:
            out.append((s, e))
    return out if out else [(0, n_points - 1)]


def _fit_circle_on_sphere(points_xyz: np.ndarray,
                          mode: str = 'auto',
                          sample_points: int = 240) -> Optional[Dict[str, np.ndarray]]:
    """Fit a great/small circle to 3D points and return an arc on the unit sphere."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None

    r = np.linalg.norm(pts, axis=1)
    r[r == 0] = 1.0
    X = pts / r[:, None]

    # Great-circle candidate: plane through origin.
    evals_g, evecs_g = np.linalg.eigh(X.T @ X)
    n_g = evecs_g[:, np.argmin(evals_g)]
    d_g = 0.0
    res_g = np.nanstd(X @ n_g)

    # Small-circle candidate: best-fit offset plane.
    mu = np.mean(X, axis=0)
    C = (X - mu).T @ (X - mu)
    evals_s, evecs_s = np.linalg.eigh(C)
    n_s = evecs_s[:, np.argmin(evals_s)]
    d_s = float(np.clip(np.dot(n_s, mu), -0.999, 0.999))
    res_s = np.nanstd((X @ n_s) - d_s)

    mode_l = str(mode).lower()
    if mode_l == 'great':
        n, d, fit_type = n_g, d_g, 'great'
    elif mode_l == 'small':
        n, d, fit_type = n_s, d_s, 'small'
    else:
        if res_g <= (res_s + 1e-12):
            n, d, fit_type = n_g, d_g, 'great'
        else:
            n, d, fit_type = n_s, d_s, 'small'

    n_norm = np.linalg.norm(n)
    if not np.isfinite(n_norm) or n_norm <= 0:
        return None
    n = n / n_norm
    d = float(np.clip(d / n_norm, -0.999, 0.999))

    center = d * n
    radius = float(np.sqrt(max(1.0 - d ** 2, 1e-10)))

    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1_norm = np.linalg.norm(e1)
    if e1_norm <= 0:
        return None
    e1 = e1 / e1_norm
    e2 = np.cross(n, e1)

    # Use only the arc covered by the segment points.
    proj = X - center[None, :]
    t_data = np.arctan2(proj @ e2, proj @ e1)
    t_data = np.unwrap(t_data)
    t_min = float(np.min(t_data))
    t_max = float(np.max(t_data))
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return None
    if abs(t_max - t_min) < 1e-6:
        t_min -= 0.05
        t_max += 0.05

    tt = np.linspace(t_min, t_max, max(50, int(sample_points)))
    arc = center[None, :] + radius * (np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :])
    arc_r = np.linalg.norm(arc, axis=1)
    arc_r[arc_r == 0] = 1.0
    arc = arc / arc_r[:, None]

    return {
        'arc_xyz': arc,
        'fit_type': np.array([fit_type]),
    }


class RMFitter:
    """Class for performing RM fitting on Stokes IQUV data."""
    
    def __init__(self, freq_hz: np.ndarray, stokes_i: np.ndarray, 
                 stokes_q: np.ndarray, stokes_u: np.ndarray, 
                 stokes_v: Optional[np.ndarray] = None):
        """
        Initialize RM Fitter.
        
        Parameters:
        -----------
        freq_hz : array-like
            Frequency array in Hz
        stokes_i : array-like
            Stokes I data (total intensity)
        stokes_q : array-like
            Stokes Q data (linear polarisation)
        stokes_u : array-like
            Stokes U data (linear polarisation)
        stokes_v : array-like, optional
            Stokes V data (circular polarisation)
        """
        self.freq_hz = np.array(freq_hz)
        self.stokes_i = np.array(stokes_i)
        self.stokes_q = np.array(stokes_q)
        self.stokes_u = np.array(stokes_u)
        self.stokes_v = np.array(stokes_v) if stokes_v is not None else None
        
        # Convert frequency to wavelength squared (λ²)
        c = 299792458.0  # Speed of light in m/s
        wavelength_m = c / self.freq_hz
        self.lambda_sq = wavelength_m ** 2
        
        # Calculate complex linear polarisation
        self.linear_pol = self.stokes_q + 1j * self.stokes_u
        
        # Calculate polarisation angle and fraction
        self.pol_angle = 0.5 * np.arctan2(self.stokes_u, self.stokes_q)
        self.pol_fraction = np.sqrt(self.stokes_q**2 + self.stokes_u**2) / (self.stokes_i + 1e-10)
        
    @staticmethod
    def faraday_rotation_model(lambda_sq: np.ndarray, rm: float, 
                               pol_angle_0: float) -> np.ndarray:
        """
        Faraday rotation model: θ(λ²) = θ₀ + RM × λ²
        
        Parameters:
        -----------
        lambda_sq : array-like
            Wavelength squared in m²
        rm : float
            Rotation measure in rad/m²
        pol_angle_0 : float
            Intrinsic polarisation angle in radians
            
        Returns:
        --------
        pol_angle : array-like
            Polarisation angle as function of λ²
        """
        return pol_angle_0 + rm * lambda_sq
    
    @staticmethod
    def complex_pol_model(lambda_sq: np.ndarray, rm: float, 
                         p0_real: float, p0_imag: float) -> np.ndarray:
        """
        Complex polarisation model: P(λ²) = P₀ × exp(2i × RM × λ²)
        
        Parameters:
        -----------
        lambda_sq : array-like
            Wavelength squared in m²
        rm : float
            Rotation measure in rad/m²
        p0_real : float
            Real part of intrinsic polarisation
        p0_imag : float
            Imaginary part of intrinsic polarisation
            
        Returns:
        --------
        complex_pol : array-like
            Complex polarisation as function of λ²
        """
        p0 = p0_real + 1j * p0_imag
        return p0 * np.exp(2j * rm * lambda_sq)
    
    def fit_rm_simple(self) -> Tuple[float, float, float]:
        """
        Simple RM fitting using linear fit to polarisation angle vs λ².
        
        NOTE: This method is kept for internal use (e.g., generating fit lines in plots).
        For actual RM measurements, fit_rm_complex() using RM synthesis is recommended.
        
        Returns:
        --------
        rm : float
            Fitted rotation measure in rad/m²
        rm_err : float
            Error on RM in rad/m²
        pol_angle_0 : float
            Intrinsic polarisation angle in radians
        """
        # Unwrap phase to handle 2π jumps
        pol_angle_unwrapped = np.unwrap(self.pol_angle)
        
        # Weighted linear fit (weight by polarised intensity)
        weights = np.abs(self.linear_pol)
        
        # Fit using polynomial
        coeffs, cov = np.polyfit(self.lambda_sq, pol_angle_unwrapped, 1, w=weights, cov=True)
        
        rm = coeffs[0]  # Slope is RM
        rm_err = np.sqrt(cov[0, 0])
        pol_angle_0 = coeffs[1]  # Intercept is intrinsic angle
        
        return rm, rm_err, pol_angle_0
    
    # The former public wrapper `fit_rm_complex` has been removed. Call
    # `_fit_rm_with_rmtools(rm_range, n_rm)` directly to run RM-Tools RM synthesis.
    
    def _fit_rm_with_rmtools(self, rm_range: Tuple[float, float], n_rm: int,
                              noise_i: Optional[float] = None,
                              noise_q: Optional[float] = None,
                              noise_u: Optional[float] = None) -> Dict:
        """
        RM synthesis using RM-Tools library.
        """
        # Prepare data for RMtools
        # RMtools expects errors; use rough estimate based on noise
        # Allow caller to provide Q/U noise estimates (e.g., from the same
        # off-pulse time region used to estimate I noise). Fall back to per-
        # spectrum estimates if not provided.
        noise_q_local = noise_q if noise_q is not None else (
            np.std(self.stokes_q) if np.std(self.stokes_q) > 0 else 0.01 * np.abs(self.stokes_q).max()
        )
        noise_u_local = noise_u if noise_u is not None else (
            np.std(self.stokes_u) if np.std(self.stokes_u) > 0 else 0.01 * np.abs(self.stokes_u).max()
        )

        # Use provided noise_i (from off-pulse) when available; otherwise use
        # per-spectrum std fallback.
        noise_i_local = noise_i if noise_i is not None else (np.std(self.stokes_i) if np.std(self.stokes_i) > 0 else 0.01)

        dI = np.ones_like(self.stokes_i) * noise_i_local
        dQ = np.ones_like(self.stokes_q) * noise_q_local
        dU = np.ones_like(self.stokes_u) * noise_u_local
        
        # RMtools expects data as [freq_Hz, I, Q, U, dI, dQ, dU]
        data = [self.freq_hz, self.stokes_i, self.stokes_q, self.stokes_u, dI, dQ, dU]
        
        # Call RMtools RM synthesis
        mDict, aDict = run_rmsynth(
            data=data,
            phiMax_radm2=max(abs(rm_range[0]), abs(rm_range[1])),
            dPhi_radm2=None,  # Auto-calculate
            nSamples=10.0,  # Samples across RMSF
            weightType='variance',
            fitRMSF=True,
            noStokesI=False,
            verbose=False,
            showPlots=False,
            debug=False
        )
        
        # Extract results (correct keys from aDict and mDict)
        rm_values = aDict['phiArr_radm2']
        rm_spectrum = aDict['dirtyFDF']  # The Faraday Dispersion Function
        rm_amplitude = np.abs(rm_spectrum)
        
        # Get peak RM from fit
        rm_peak = mDict['phiPeakPIfit_rm2']  # Fitted peak RM
        
        # Calculate SNR
        noise_estimate = mDict['dFDFth']  # Theoretical noise from RMtools
        rm_peak_snr = mDict['snrPIfit']  # SNR from RMtools fit
        
        results = {
            'rm_values': rm_values,
            'rm_spectrum': rm_spectrum,
            'rm_amplitude': rm_amplitude,
            'rm_peak': rm_peak,
            'rm_peak_snr': rm_peak_snr,
            'noise_estimate': noise_estimate,
            'rmtools_dict': mDict,  # Store full RMtools output
            'rmtools_arrays': aDict  # Store arrays from RMtools
        }
        # Run RM-CLEAN to deconvolve the FDF and obtain cleaned estimates
        try:
            # Use a 3-sigma clean threshold by default (negative => sigma)
            mDict_cl, aDict_cl = run_rmclean(mDict, aDict, cutoff=-3, maxIter=1000, gain=0.1,
                                             showPlots=False, verbose=False, saveFigures=False)
            # Extract cleaned RM and observed uncertainty
            rm_clean_peak = mDict_cl.get('phiPeakPIfit_rm2', None)
            rm_clean_err = mDict_cl.get('dPhiObserved_rm2', None)
            pol_angle = mDict_cl.get('polAngleFit_deg', None)
            pol_angle_err = mDict_cl.get('dPolAngleFitObserved_deg', None)

            results.update({
                'rm_clean_peak': rm_clean_peak,
                'rm_clean_err': rm_clean_err,
                'pol_angle_deg': pol_angle,
                'pol_angle_err_deg': pol_angle_err,
                'rmclean_dict': mDict_cl,
                'rmclean_arrays': aDict_cl
            })
        except Exception:
            # If RM-CLEAN fails, continue without cleaned results
            pass
        
        return results
    
    def fit_rm_qufitting(self) -> Dict:
        """
        RM fitting by directly fitting Q and U vs λ² using non-linear least squares.
        
        Returns:
        --------
        results : dict
            Dictionary containing fitted parameters and errors
        """
        def qu_model(lambda_sq, rm, q0, u0):
            """Model for Q and U separately."""
            pol_angle = rm * lambda_sq
            q_model = q0 * np.cos(2*pol_angle) - u0 * np.sin(2*pol_angle)
            u_model = q0 * np.sin(2*pol_angle) + u0 * np.cos(2*pol_angle)
            return np.concatenate([q_model, u_model])
        
        # Concatenate Q and U data
        qu_data = np.concatenate([self.stokes_q, self.stokes_u])
        lambda_sq_double = np.concatenate([self.lambda_sq, self.lambda_sq])
        
        # Initial guess
        p0 = [0.0, np.mean(self.stokes_q), np.mean(self.stokes_u)]
        
        try:
            # Fit
            popt, pcov = curve_fit(lambda x, rm, q0, u0: qu_model(x, rm, q0, u0), self.lambda_sq, qu_data, p0=p0, maxfev=10000)
            
            rm_fit = popt[0]
            q0_fit = popt[1]
            u0_fit = popt[2]
            
            # Errors from covariance matrix
            perr = np.sqrt(np.diag(pcov))
            
            results = {
                'rm': rm_fit,
                'rm_err': perr[0],
                'q0': q0_fit,
                'u0': u0_fit,
                'q0_err': perr[1],
                'u0_err': perr[2],
                'success': True
            }
        except Exception as e:
            print(f"QU fitting failed: {e}")
            results = {'success': False}
        
        return results

    def fit_rm_rmnest(self, gfr: bool = False, free_alpha: bool = False,
                      outdir: str = './', label: str = 'rmnest',
                      sampler: str = 'dynesty') -> Dict:
        """
        RM fitting using RMNest (Bayesian sampling with bilby).

        Parameters:
        -----------
        gfr : bool
            Fit generalised Faraday rotation (includes Stokes V).
        free_alpha : bool
            Allow alpha to vary for GFR model.
        outdir : str
            Output directory for RMNest run.
        label : str
            Label for RMNest run.
        sampler : str
            bilby sampler name (default: dynesty).
        """
        try:
            from rmnest.fit_RM import RMNest
        except Exception as exc:
            raise ImportError(
                "RMNest is not installed. Install with: pip install rmnest"
            ) from exc

        freq_mhz = self.freq_hz / 1e6
        freq_cen = np.median(freq_mhz)
        stokes_v = self.stokes_v if self.stokes_v is not None else np.zeros_like(self.stokes_q)

        rmnest = RMNest(
            freqs=freq_mhz,
            freq_cen=freq_cen,
            s_q=self.stokes_q,
            s_u=self.stokes_u,
            s_v=stokes_v,
            rms_q=None,
            rms_u=None,
            rms_v=None
        )
        rmnest.fit(
            gfr=gfr,
            free_alpha=free_alpha,
            label=label,
            outdir=outdir,
            sampler=sampler
        )

        result = rmnest.result
        param_name = 'grm' if gfr else 'rm'

        if param_name not in result.posterior:
            raise KeyError(f"RMNest posterior missing '{param_name}' parameter")

        median, low, high = _summarize_posterior(result.posterior[param_name].values)

        return {
            'param_name': param_name,
            'median': median,
            'low': low,
            'high': high,
            'rmnest_result': result,
            'rmnest_outdir': outdir,
            'rmnest_label': label,
            'rmnest_post_json': rmnest.post_json_file
        }


def load_stokes_data(i_file: Optional[str] = None,
                     q_file: Optional[str] = None,
                     u_file: Optional[str] = None,
                     v_file: Optional[str] = None,
                     cube_file: Optional[str] = None,
                     stokes_axis: int = 0,
                     freq_file: Optional[str] = None,
                     time_file: Optional[str] = None,
                     time_axis: int = 1,
                     freq_axis: int = 0,
                     freq_unit: str = 'Hz',
                     time_unit: str = 's') -> Tuple:
    """
    Load Stokes parameter data from separate files or an IQUV cube.
    
    Parameters:
    -----------
    i_file : str, optional
        Path to Stokes I file
    q_file : str, optional
        Path to Stokes Q file
    u_file : str, optional
        Path to Stokes U file
    v_file : str, optional
        Path to Stokes V file
    cube_file : str, optional
        Path to Stokes cube file containing I,Q,U,(V).
    stokes_axis : int
        Axis index of the Stokes dimension in ``cube_file``.
    freq_file : str, optional
        Path to frequency file (Hz). If None, generates placeholder frequencies.
    time_file : str, optional
        Path to time file (seconds or other units).
    time_axis : int
        Axis for time dimension in 2D arrays (default: 1)
    freq_axis : int
        Axis for frequency dimension in 2D arrays (default: 0)
    freq_unit : str
        Unit of frequency file: 'Hz', 'MHz', 'GHz' (default: 'Hz')
    time_unit : str
        Unit of time file: 's', 'ms', 'us' (default: 's')
        
    Returns:
    --------
    freq_hz : array
        Frequency array in Hz
    stokes_i : array
        Stokes I data
    stokes_q : array
        Stokes Q data
    stokes_u : array
        Stokes U data
    stokes_v : array or None
        Stokes V data
    time_array : array or None
        Time array if provided
    """
    
    def load_file(filename):
        """Load data from .npy or text file."""
        if filename.endswith('.npy'):
            return np.load(filename)
        else:
            return np.loadtxt(filename)
    
    # Load data either from a cube or separate I/Q/U(/V) files
    if cube_file:
        cube = np.asarray(load_file(cube_file))
        if cube.ndim < 2:
            raise ValueError("Stokes cube must have at least 2 dimensions")

        # Bring Stokes axis to the front for consistent extraction.
        cube = np.moveaxis(cube, stokes_axis, 0)
        n_stokes = cube.shape[0]
        if n_stokes < 3:
            raise ValueError("Stokes cube must contain at least I, Q, U components")

        stokes_i = cube[0]
        stokes_q = cube[1]
        stokes_u = cube[2]
        stokes_v = cube[3] if n_stokes >= 4 else None

        print(f"  Loaded Stokes cube: {cube_file}")
        print(f"    Cube shape (after moveaxis): {cube.shape}")
    else:
        if i_file is None or q_file is None or u_file is None:
            raise ValueError("Provide --stokes-cube or all of --stokes-i/--stokes-q/--stokes-u")
        stokes_i = load_file(i_file)
        stokes_q = load_file(q_file)
        stokes_u = load_file(u_file)
        stokes_v = load_file(v_file) if v_file else None
    
    print(f"  Loaded data shapes:")
    print(f"    Stokes I: {stokes_i.shape}")
    print(f"    Stokes Q: {stokes_q.shape}")
    print(f"    Stokes U: {stokes_u.shape}")
    if stokes_v is not None:
        print(f"    Stokes V: {stokes_v.shape}")
    
    # Handle frequency
    if freq_file:
        freq_hz = load_file(freq_file)
        print(f"  Loaded frequency array: {len(freq_hz)} channels")
        
        # Convert to Hz based on unit
        if freq_unit.lower() == 'mhz':
            freq_hz = freq_hz * 1e6
            print(f"  Converted from MHz to Hz")
        elif freq_unit.lower() == 'ghz':
            freq_hz = freq_hz * 1e9
            print(f"  Converted from GHz to Hz")
        elif freq_unit.lower() != 'hz':
            print(f"  Warning: Unknown frequency unit '{freq_unit}', assuming Hz")
    else:
        # For 2D arrays [n_time, n_freq], use the frequency dimension size
        if stokes_i.ndim == 2:
            n_freq = stokes_i.shape[freq_axis]
            print(f"  Warning: No frequency file provided. Generating {n_freq} placeholder frequencies.")
            freq_hz = np.linspace(1e9, 2e9, n_freq)  # 1-2 GHz example
        else:
            # 1D array
            print("  Warning: No frequency file provided. Generating placeholder frequencies.")
            freq_hz = np.linspace(1e9, 2e9, len(stokes_i))
    
    # Handle time
    time_array = None
    if time_file:
        time_array = load_file(time_file)
        print(f"  Loaded time array: {len(time_array)} samples")
        
        # Convert to seconds based on unit
        if time_unit.lower() == 'ms':
            time_array = time_array * 1e-3
            print(f"  Converted from ms to seconds")
        elif time_unit.lower() == 'us':
            time_array = time_array * 1e-6
            print(f"  Converted from μs to seconds")
        elif time_unit.lower() != 's':
            print(f"  Warning: Unknown time unit '{time_unit}', assuming seconds")
    elif stokes_i.ndim == 2:
        n_time = stokes_i.shape[time_axis]
        time_array = np.arange(n_time)
    
    return freq_hz, stokes_i, stokes_q, stokes_u, stokes_v, time_array


def find_onpulse_window(time_profile: np.ndarray, flux_fraction: float = 0.95) -> Tuple[int, int]:
    """
    Find the smallest contiguous window that contains a given fraction of the total flux.
    
    Parameters:
    -----------
    time_profile : array
        1D array of flux as function of time
    flux_fraction : float
        Fraction of total flux to contain (default: 0.95 for 95%)
        
    Returns:
    --------
    start_idx : int
        Start index of on-pulse window
    end_idx : int
        End index of on-pulse window (inclusive)
    """
    total_flux = np.sum(time_profile)
    target_flux = total_flux * flux_fraction
    n_bins = len(time_profile)
    
    best_window = None
    best_width = n_bins + 1
    
    # Try all possible window sizes
    for width in range(1, n_bins + 1):
        # Try all starting positions for this width
        for start in range(n_bins - width + 1):
            end = start + width
            window_flux = np.sum(time_profile[start:end])
            
            if window_flux >= target_flux:
                if width < best_width:
                    best_width = width
                    best_window = (start, end - 1)  # end-1 for inclusive end
                break  # Found smallest window for this width
    
    if best_window is None:
        # If no window found (shouldn't happen), use full range
        best_window = (0, n_bins - 1)
    
    return best_window


def find_peak_regions(time_profile: np.ndarray, snr_array: Optional[np.ndarray] = None,
                     min_gap_bins: int = 3, min_peak_bins: int = 3, 
                     max_merge_gap: int = 0, snr_threshold: float = 5.0) -> list:
    """
    Identify separate peak regions in time series data by detecting gaps of low signal.
    
    Parameters:
    -----------
    time_profile : array
        1D array of flux as function of time
    snr_array : array, optional
        1D array of SNR values. If provided, uses SNR threshold to identify gaps.
    min_gap_bins : int
        Minimum number of consecutive low-signal bins to separate peaks (default: 3)
    min_peak_bins : int
        Minimum number of consecutive significant bins required for a valid peak (default: 3)
    max_merge_gap : int
        Maximum gap size for merging nearby peaks. If two peaks are separated by
        fewer than this many bins, they will be merged (default: 0, no merging)
    snr_threshold : float
        SNR threshold below which signal is considered low (default: 5.0)
        
    Returns:
    --------
    peak_regions : list of tuples
        List of (start_idx, end_idx) for each peak region (inclusive)
    """
    n_bins = len(time_profile)
    
    # Determine which bins have significant signal
    if snr_array is not None and len(snr_array) == n_bins:
        # Use SNR if available
        significant = snr_array >= snr_threshold
    else:
        # Use flux threshold (median + some factor)
        threshold = np.median(time_profile) + 2.0 * np.std(time_profile)
        significant = time_profile >= threshold
    
    # Find contiguous regions of significant signal
    peak_regions = []
    in_peak = False
    start_idx = 0
    gap_count = 0
    
    for i in range(n_bins):
        if significant[i]:
            if not in_peak:
                # Starting a new peak
                start_idx = i
                in_peak = True
                gap_count = 0
            else:
                # Continuing peak, reset gap counter
                gap_count = 0
        else:
            if in_peak:
                # In a potential gap
                gap_count += 1
                if gap_count >= min_gap_bins:
                    # Gap is large enough, end this peak
                    end_idx = i - gap_count
                    peak_width = end_idx - start_idx + 1
                    # Only add peak if it has enough consecutive significant bins
                    if peak_width >= min_peak_bins:
                        peak_regions.append((start_idx, end_idx))
                    in_peak = False
                    gap_count = 0
    
    # Handle case where we're still in a peak at the end
    if in_peak:
        end_idx = n_bins - 1 - gap_count
        peak_width = end_idx - start_idx + 1
        if peak_width >= min_peak_bins:
            peak_regions.append((start_idx, end_idx))
    
    # Merge nearby peaks if requested
    if max_merge_gap > 0 and len(peak_regions) > 1:
        merged_regions = []
        current_start, current_end = peak_regions[0]
        
        for i in range(1, len(peak_regions)):
            next_start, next_end = peak_regions[i]
            gap_size = next_start - current_end - 1
            
            if gap_size < max_merge_gap:
                # Merge with current region
                current_end = next_end
            else:
                # Save current region and start new one
                merged_regions.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        # Don't forget the last region
        merged_regions.append((current_start, current_end))
        peak_regions = merged_regions
    
    # If no peaks found, return entire range
    if len(peak_regions) == 0:
        peak_regions = [(0, n_bins - 1)]
    
    return peak_regions


def select_peaks_manual(time_ms: np.ndarray, stokes_i: np.ndarray) -> List[Tuple[int, int]]:
    """
    Manually select peak bounds by clicking on the pulse profile.

    Click pairs of points (start, end) for each peak. Close the window when done.
    Returns list of (start_idx, end_idx) in time-sample indices.
    """
    # Build a 1D time profile by averaging over frequency axis sensibly
    if stokes_i.ndim == 2:
        # try to detect orientation: if length matches time_ms, use axis=0
        if stokes_i.shape[0] == len(time_ms):
            time_series = np.mean(stokes_i, axis=1)
        elif stokes_i.shape[1] == len(time_ms):
            time_series = np.mean(stokes_i, axis=0)
        else:
            time_series = np.mean(stokes_i, axis=1)
    else:
        time_series = stokes_i

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_ms * 1e3, time_series, color='k', linewidth=1)
    ax.set_title('Click start/end bounds for each peak (close window to finish)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Flux')
    ax.grid(True, alpha=0.3)
    cursor_line = ax.axvline(time_ms[0] * 1e3, color='tab:blue', alpha=0.4, linewidth=1)

    times: List[float] = []

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_line.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x = float(event.xdata)
        times.append(x / 1e3)  # convert back to seconds (time_ms input expected in s)
        ax.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
        if len(times) % 2 == 0:
            start_t, end_t = sorted((times[-2], times[-1]))
            ax.axvspan(start_t * 1e3, end_t * 1e3, color='tab:orange', alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    if len(times) < 2:
        return [(0, len(time_ms) - 1)]

    if len(times) % 2 != 0:
        times = times[:-1]

    peak_regions: List[Tuple[int, int]] = []
    for i in range(0, len(times), 2):
        start_t, end_t = sorted((times[i], times[i + 1]))
        start_idx = int(np.argmin(np.abs(time_ms - start_t)))
        end_idx = int(np.argmin(np.abs(time_ms - end_t)))
        peak_regions.append((min(start_idx, end_idx), max(start_idx, end_idx)))

    return peak_regions


def fit_rm_time_series(freq_hz: np.ndarray, time_series_data: Dict,
                      method: str = 'rm_synthesis', rm_range: Tuple[float, float] = (-1000, 1000),
                      n_rm: int = 2000, rmnest_gfr: bool = False,
                      rmnest_free_alpha: bool = False, rmnest_outdir: Optional[str] = None,
                      rmnest_label: Optional[str] = None, rmnest_sampler: str = 'dynesty',
                      n_time_bins: Optional[int] = None,
                      noise_fraction: float = 0.1) -> Dict:
    """
    Fit RM for time-series data (multiple time samples).
    
        n_time = len(times) if 'time' in time_series_data else 0
    -----------
    freq_hz : array
        Frequency array in Hz
    time_series_data : dict
        Dictionary with keys 'time', 'I', 'Q', 'U', 'V' (optional)
        where each is a 2D array with time on one axis
    method : str
        Fitting method: 'simple', 'rm_synthesis', 'qu_fitting', or 'rmnest'
    rm_range : tuple
        Range of RM values to search (rad/m²) for rm_synthesis
    n_rm : int
        Number of RM trial values for rm_synthesis
    n_time_bins : int, optional
        Number of time bins to fit (default: no binning)
        
    Returns:
    --------
    results : dict
        Dictionary containing RM values and errors as function of time
    """
    times = time_series_data['time']
    n_time = len(times)
    
    # Determine which axis is time (the one matching length of times array)
    stokes_i_data = time_series_data['I']
    if stokes_i_data.shape[0] == n_time:
        time_axis = 0
    elif stokes_i_data.shape[1] == n_time:
        time_axis = 1
    else:
        time_axis = 0  # default

    # Compute off-pulse-based Q/U noise estimates using the same fraction of
    # samples used elsewhere for I noise estimation. These are used as
    # optional inputs to the RM-tools fitter so dQ/dU reflect time-domain noise.
    if time_axis == 0:
        I_full_for_noise = np.mean(time_series_data['I'], axis=1)
        Q_time = time_series_data['Q']
        U_time = time_series_data['U']
    else:
        I_full_for_noise = np.mean(time_series_data['I'], axis=0)
        Q_time = time_series_data['Q']
        U_time = time_series_data['U']

    n_frac_noise = max(1, int(len(I_full_for_noise) * noise_fraction))
    # select initial (off-pulse) samples
    if time_axis == 0:
        q_off = Q_time[:n_frac_noise, :]
        u_off = U_time[:n_frac_noise, :]
    else:
        q_off = Q_time[:, :n_frac_noise]
        u_off = U_time[:, :n_frac_noise]

    # per-channel std across the off-pulse time samples, reduce to single
    # representative value using median (robust) and ensure non-zero fallback
    q_std_chan = np.nanstd(q_off, axis=0 if time_axis == 0 else 1)
    u_std_chan = np.nanstd(u_off, axis=0 if time_axis == 0 else 1)
    noise_q = np.nanmedian(q_std_chan) if np.nanmedian(q_std_chan) > 0 else (np.nanmean(q_std_chan) if np.nanmean(q_std_chan) > 0 else 1e-10)
    noise_u = np.nanmedian(u_std_chan) if np.nanmedian(u_std_chan) > 0 else (np.nanmean(u_std_chan) if np.nanmean(u_std_chan) > 0 else 1e-10)
    # Off-pulse noise estimate for Stokes I (time-domain)
    noise_i = np.nanstd(I_full_for_noise[:n_frac_noise])
    if noise_i <= 0:
        mad = np.nanmedian(np.abs(I_full_for_noise - np.nanmedian(I_full_for_noise)))
        if mad > 0:
            noise_i = mad / 0.6745
        else:
            noise_i = max(np.nanmedian(I_full_for_noise) * 0.1, 1e-10)
    
    if n_time_bins is None or n_time_bins <= 0 or n_time_bins >= n_time:
        # no binning or Too many bins requested -> keep full resolution
        bin_size = 1
        n_bins_actual = n_time
    else:
        # aim for exactly n_time_bins (or fewer if n_time smaller)
        n_bins_actual = min(n_time_bins, n_time)
        # compute bin size by ceiling division to cover all samples
        bin_size = int(np.ceil(n_time / n_bins_actual))
        # recompute actual bin count in case ceiling produced extra
        n_bins_actual = (n_time + bin_size - 1) // bin_size
        # ensure we don't exceed requested bins due to rounding
        n_bins_actual = min(n_bins_actual, n_time_bins)

    rm_array = np.zeros(n_bins_actual)
    rm_err_array = np.zeros(n_bins_actual)
    pol_angle_0_array = np.zeros(n_bins_actual)
    snr_array = np.zeros(n_bins_actual)
    pol_angle_ref_array = np.zeros(n_bins_actual)  # Polarisation angle at reference freq
    # arrays for derived quantities per bin
    pa_array = np.zeros(n_bins_actual)
    ea_array = np.zeros(n_bins_actual)
    # track PA/EA uncertainties (degrees) so we can apply the
    # ``>50°`` cut described in the user example
    pa_err_array = np.zeros(n_bins_actual)
    ea_err_array = np.zeros(n_bins_actual)

    P_frac_bins = np.zeros(n_bins_actual)
    L_frac_bins = np.zeros(n_bins_actual)
    V_frac_bins = np.zeros(n_bins_actual)
    # new: store normalised Stokes values per bin
    q_bin = np.zeros(n_bins_actual)
    u_bin = np.zeros(n_bins_actual)
    v_bin = np.zeros(n_bins_actual)
    time_binned = np.zeros(n_bins_actual)
    # track total‑intensity S/N for each bin so we can reproduce the
    # same mask later when plotting or generating a Poincaré sphere
    i_snr_array = np.zeros(n_bins_actual)

    # prepare noise estimate for Stokes V if available (needed for EA errors)
    if 'V' in time_series_data:
        # average along time axis to mimic I/Q/U noise calculation
        if time_axis == 0:
            V_full_for_noise = np.mean(time_series_data['V'], axis=1)
        else:
            V_full_for_noise = np.mean(time_series_data['V'], axis=0)
        noise_v = np.nanstd(V_full_for_noise[:n_frac_noise])
        if noise_v <= 0:
            mad_v = np.nanmedian(np.abs(V_full_for_noise - np.nanmedian(V_full_for_noise)))
            noise_v = mad_v / 0.6745 if mad_v > 0 else 1e-10
    else:
        noise_v = 0.0

    for i in range(n_bins_actual):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, n_time)
        if bin_end <= bin_start:
            continue

        time_binned[i] = np.mean(times[bin_start:bin_end])

        # Extract data for this time bin and average in time
        if time_axis == 0:
            stokes_i = np.mean(time_series_data['I'][bin_start:bin_end, :], axis=0)
            stokes_q = np.mean(time_series_data['Q'][bin_start:bin_end, :], axis=0)
            stokes_u = np.mean(time_series_data['U'][bin_start:bin_end, :], axis=0)
            if 'V' in time_series_data:
                stokes_v = np.mean(time_series_data['V'][bin_start:bin_end, :], axis=0)
            else:
                stokes_v = None
        else:
            stokes_i = np.mean(time_series_data['I'][:, bin_start:bin_end], axis=1)
            stokes_q = np.mean(time_series_data['Q'][:, bin_start:bin_end], axis=1)
            stokes_u = np.mean(time_series_data['U'][:, bin_start:bin_end], axis=1)
            if 'V' in time_series_data:
                stokes_v = np.mean(time_series_data['V'][:, bin_start:bin_end], axis=1)
            else:
                stokes_v = None
        
        # Initialize fitter
        fitter = RMFitter(freq_hz, stokes_i, stokes_q, stokes_u, stokes_v)
        
        # store derived polarisation angles & fractions for this bin
        # collapse any remaining frequency axis by taking simple mean values
        q_val = np.mean(stokes_q)
        u_val = np.mean(stokes_u)
        i_val = np.mean(stokes_i)
        v_val = np.mean(stokes_v) if stokes_v is not None else 0.0
        # record I‑S/N for this bin (used later for masking)
        i_snr_array[i] = i_val / (noise_i + 1e-10)
        # normalised Stokes used by Poincaré sphere
        q_bin[i] = q_val / (i_val + 1e-10)
        u_bin[i] = u_val / (i_val + 1e-10)
        v_bin[i] = v_val / (i_val + 1e-10)
        P_amp = np.sqrt(q_val**2 + u_val**2 + v_val**2) + 1e-10
        pa_array[i] = np.degrees(0.5 * np.arctan2(u_val, q_val))
        ea_array[i] = np.degrees(0.5 * np.arcsin(v_val / P_amp))

        # propagate uncertainties for PA/EA at the bin level (degrees)
        # using the same noise estimates derived earlier
        P_lin_sq = q_val**2 + u_val**2 + 1e-20
        pa_sigma_rad = 0.5 * np.sqrt((u_val**2 * noise_q**2 + q_val**2 * noise_u**2) / (P_lin_sq**2))
        pa_err_array[i] = np.degrees(pa_sigma_rad)
        # EA error
        sigma_P = np.sqrt((q_val**2 * noise_q**2 + u_val**2 * noise_u**2)) / (P_amp + 1e-10)
        sigma_VoverP = np.sqrt((noise_v**2 / (P_amp**2)) + ((v_val**2) * (sigma_P**2) / (P_amp**2 + 1e-20)))
        denom = np.sqrt(1.0 - (v_val / P_amp)**2 + 1e-20)
        ea_sigma_rad = 0.5 * (sigma_VoverP / denom)
        ea_err_array[i] = np.degrees(ea_sigma_rad)

        P_frac_bins[i] = P_amp / (i_val + 1e-10)
        L_frac_bins[i] = np.sqrt(q_val**2 + u_val**2) / (i_val + 1e-10)
        # keep sign of Stokes V rather than taking absolute value
        V_frac_bins[i] = v_val / (i_val + 1e-10)

        # Store polarisation angle at reference frequency (before RM correction)
        # Use the median frequency as reference
        ref_idx = len(freq_hz) // 2
        pol_angle_ref_array[i] = fitter.pol_angle[ref_idx]
        
        # Fit based on method
        if method == 'simple':
            # Use RM synthesis instead of simple linear fit for better accuracy
            result = fitter._fit_rm_with_rmtools(rm_range=rm_range, n_rm=n_rm,
                                                 noise_i=noise_i,
                                                 noise_q=noise_q, noise_u=noise_u)
            rm_val = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
            rm_err_val = result.get('rm_clean_err', result.get('rm_err', result.get('noise_estimate', 0) * 2))
            rm_array[i] = rm_val
            snr_array[i] = result.get('rm_peak_snr', np.nan)
            rm_err_array[i] = rm_err_val
            
        elif method == 'rm_synthesis':
            result = fitter._fit_rm_with_rmtools(rm_range=rm_range, n_rm=n_rm,
                                                 noise_i=noise_i,
                                                 noise_q=noise_q, noise_u=noise_u)
            rm_val = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
            rm_err_val = result.get('rm_clean_err', result.get('rm_err', result.get('noise_estimate', 0) * 2))
            rm_array[i] = rm_val
            snr_array[i] = result.get('rm_peak_snr', np.nan)
            # use cleaned error if available, otherwise fallback
            rm_err_array[i] = rm_err_val
            
        elif method == 'qu_fitting':
            result = fitter.fit_rm_qufitting()
            if result['success']:
                rm_array[i] = result['rm']
                rm_err_array[i] = result['rm_err']

        elif method == 'rmnest':
            base_outdir = rmnest_outdir or 'rmnest_time_series'
            base_label = rmnest_label or 'rmnest'
            step_outdir = os.path.join(base_outdir, f"b{i:04d}")
            step_label = f"{base_label}_b{i:04d}"
            result = fitter.fit_rm_rmnest(
                gfr=rmnest_gfr,
                free_alpha=rmnest_free_alpha,
                outdir=step_outdir,
                label=step_label,
                sampler=rmnest_sampler
            )
            median = result['median']
            low = result['low']
            high = result['high']
            rm_array[i] = median
            rm_err_array[i] = max(median - low, high - median)
    
    # derive a boolean mask based on I‑S/N (mimicking the threshold used
    # later in the plotting routine).  use a fixed value of 2.0 for now, the
    # same `snr_i_full < 2.0` cut applied in the fraction panel.
    valid_bins = i_snr_array >= 2.0

    # Apply the same Stokes I S/N mask to RM results so RM from low I S/N
    # bins is not shown or used downstream.  We only mask RM/its error and
    # the per-bin SNR here; PA/EA masking is handled separately above.
    if valid_bins.size == rm_array.size:
        rm_array[~valid_bins] = np.nan
        rm_err_array[~valid_bins] = np.nan
        snr_array[~valid_bins] = np.nan

    # also identify bins with very large PA/EA uncertainties; this mirrors the
    # `nongoodphi`/`nongoodpsi` logic from the example snippet provided by the
    # user.  we treat either error above 50° as invalid.
    bad_pa = pa_err_array > 50.0
    bad_ea = ea_err_array > 50.0
    bad_bins = (~valid_bins) | bad_pa | bad_ea

    # apply mask by inserting NaNs into the PA/EA arrays returned to the caller
    pa_array[bad_bins] = np.nan
    ea_array[bad_bins] = np.nan
    pa_err_array[bad_bins] = np.nan
    ea_err_array[bad_bins] = np.nan

    results = {
        'time': time_binned,
        'rm': rm_array,
        'rm_err': rm_err_array,
        'pol_angle_0': pol_angle_0_array,
        'snr': snr_array,
        'pol_angle_ref': pol_angle_ref_array,
        'pa_deg': pa_array,
        'ea_deg': ea_array,
        'pa_err_deg': pa_err_array,
        'ea_err_deg': ea_err_array,
        'P_frac_bin': P_frac_bins,
        'L_frac_bin': L_frac_bins,
        'V_frac_bin': V_frac_bins,
        'q_bin': q_bin,
        'u_bin': u_bin,
        'v_bin': v_bin,
        'is_binned': n_bins_actual != n_time,
        'time_bin_size': bin_size,
        'time_bin_count': n_bins_actual,
        'i_snr': i_snr_array,
        'valid_bins': valid_bins,
    }
    
    return results


def plot_poincare_sphere(
                         time_series_data: Dict,
                         output_file: str = 'poincare_sphere.png',
                         snr_threshold: float = 5.0,
                         n_time_bins: Optional[int] = None,
                         noise_fraction: float = 0.1,
                         time_unit: str = 's',
                         interactive: bool = False,
                         force_surface: bool = False,
                         rm_results: Optional[Dict] = None,
                         noise_reference_data: Optional[Dict] = None,
                         circle_fit_mode: Optional[str] = None,
                         circle_fit_segments: Optional[List[Tuple[int, int]]] = None):
    """
    Plot polarisation states on the Poincaré sphere vs. **time only**.

    Frequency-dependent colouring and axes have been removed – this
    routine now _requires_ a time-series data dictionary and will
    average over frequency (or bins) to build the track.  The goal is to
    visualise the temporal evolution of the polarisation state.

    Parameters
    ----------
    time_series_data : dict
        Dictionary with keys ``'time'``, ``'I'``, ``'Q'``, ``'U'`` (and
        optionally ``'V'``) containing 2‑D arrays with shape
        ``(n_time, n_freq)`` or ``(n_freq, n_time)``.  Time is always used
        to colour the points.
    output_file : str
        Output filename for the plot.
    snr_threshold : float
        Minimum linear-polarisation SNR for a point to be plotted.
    n_time_bins : int, optional
        Number of time bins to average over. If ``None`` or ``<=0`` each
        time sample is used individually.
    noise_fraction : float
        Fraction of the time axis used to estimate the off‑pulse noise
        (default ``0.1`` = first 10 % of samples).
    """
    # --------------------------------------------------------------
    # time‑series mode is now mandatory – colour axis = time only
    # --------------------------------------------------------------
    if time_series_data is None:
        raise ValueError("plot_poincare_sphere now requires ``time_series_data``; "
                         "frequency colouring has been removed.")

    I_cube = time_series_data['I']
    Q_cube = time_series_data['Q']
    U_cube = time_series_data['U']
    V_cube = time_series_data.get('V', None)
    times  = time_series_data['time']

    n_time = len(times)
    noise_ref = noise_reference_data if noise_reference_data is not None else time_series_data

    # If rm_results with pre-computed bins are supplied we trust those values
    if rm_results is not None and 'q_bin' in rm_results:
        # use exactly the binned normals that were produced by the time-series fit
        q_norm = np.array(rm_results['q_bin'])
        u_norm = np.array(rm_results['u_bin'])
        v_norm = np.array(rm_results.get('v_bin', np.zeros_like(q_norm)))
        color_axis = np.array(rm_results['time'])
        orig_idx = np.arange(q_norm.size, dtype=int)
        # override local lists used later
        pol_list = np.sqrt(q_norm**2 + u_norm**2)
        # if the fitter supplied a mask, apply it now so that subsequent
        # SNR-based gating only considers the surviving bins
        if 'valid_bins' in rm_results:
            valid = np.asarray(rm_results['valid_bins'], dtype=bool)
            q_norm = q_norm[valid]
            u_norm = u_norm[valid]
            v_norm = v_norm[valid]
            color_axis = color_axis[valid]
            pol_list = pol_list[valid]
            orig_idx = orig_idx[valid]
        # remove any points that were explicitly set to NaN (e.g. from bad
        # PA/EA or other quality cuts).  these would otherwise produce NaNs in
        # pol_list and later kill the SNR mask below.
        notnan = (~np.isnan(q_norm)) & (~np.isnan(u_norm)) & (~np.isnan(v_norm))
        if not np.all(notnan):
            q_norm = q_norm[notnan]
            u_norm = u_norm[notnan]
            v_norm = v_norm[notnan]
            color_axis = color_axis[notnan]
            pol_list = pol_list[notnan]
            orig_idx = orig_idx[notnan]
        # abort early if nothing remains
        if q_norm.size == 0:
            print("Warning: all Poincaré bins were masked; skipping plot.")
            return
    else:
        # Determine time axis
        if I_cube.ndim == 2:
            time_axis = 0 if I_cube.shape[0] == n_time else 1
        else:
            raise ValueError("Time-series data must be 2D (time × frequency).")

        # Optional time binning
        if n_time_bins is None or n_time_bins <= 0:
            bin_size = 1
            n_bins = n_time
        else:
            bin_size = max(1, n_time // n_time_bins)
            n_bins = (n_time + bin_size - 1) // bin_size

        q_list, u_list, v_list = [], [], []
        pol_list, time_list = [], []

        for i in range(n_bins):
            start = i * bin_size
            end   = min((i + 1) * bin_size, n_time)
            if end <= start:
                continue
            time_list.append(np.mean(times[start:end]))
            if time_axis == 0:
                I_sl = I_cube[start:end, :]
                Q_sl = Q_cube[start:end, :]
                U_sl = U_cube[start:end, :]
                V_sl = V_cube[start:end, :] if V_cube is not None else None
            else:
                I_sl = I_cube[:, start:end]
                Q_sl = Q_cube[:, start:end]
                U_sl = U_cube[:, start:end]
                V_sl = V_cube[:, start:end] if V_cube is not None else None
            # Average over frequency
            I_m = np.nanmean(I_sl)
            Q_m = np.nanmean(Q_sl)
            U_m = np.nanmean(U_sl)
            V_m = np.nanmean(V_sl) if V_sl is not None else 0.0
            q_list.append(Q_m / (I_m + 1e-10))
            u_list.append(U_m / (I_m + 1e-10))
            v_list.append(V_m / (I_m + 1e-10))
            pol_list.append(np.sqrt(Q_m**2 + U_m**2))

        q_norm = np.array(q_list)
        u_norm = np.array(u_list)
        v_norm = np.array(v_list)
        color_axis = np.array(time_list)
        orig_idx = np.arange(q_norm.size, dtype=int)
    # scale axis to requested unit and build label
    unit = time_unit.lower()
    if unit == 'ms':
        color_axis = color_axis * 1e3
        color_label = "Time (ms)"
    elif unit == 'us' or unit == 'µs':
        color_axis = color_axis * 1e6
        color_label = "Time (µs)"
    else:
        # default seconds, no scaling
        color_label = "Time (s)"

    # Estimate noise from first noise_fraction of time bins
    n_frac = max(1, int(len(pol_list) * noise_fraction))
    sigma_pol = np.nanstd(pol_list[:n_frac])
    if sigma_pol <= 0:
        sigma_pol = 1e-10
    snr = np.array(pol_list) / (sigma_pol + 1e-10)

    # ------------------------------------------------------------------
    # Apply mask to select points for plotting
    # ------------------------------------------------------------------
    if rm_results is not None and 'valid_bins' in rm_results:
        # we've already applied the valid mask to q_norm/u_norm earlier above,
        # so at this stage all remaining points should be plotted; no further
        # masking is necessary.  preserving the original array length would
        # cause mismatches as seen earlier.
        mask = np.ones(q_norm.shape[0], dtype=bool)
    else:
        mask = snr > snr_threshold
        if np.sum(mask) < 2:
            print(f"Warning: Only {np.sum(mask)} points above SNR threshold "
                  f"{snr_threshold:.1f}. Lowering to 2.0.")
            mask = snr > 2.0
        if np.sum(mask) < 2:
            print("Error: fewer than 2 points survive SNR cut. Cannot plot.")
            return

    filtered_idx = np.asarray(orig_idx[mask], dtype=int)
    q_filt = q_norm[mask]
    u_filt = u_norm[mask]
    v_filt = v_norm[mask]
    color_filt = color_axis[mask]

    # apply force_surface projection
    if force_surface:
        vecs = np.vstack([q_filt, u_filt, v_filt])
        norms = np.linalg.norm(vecs, axis=0)
        norms[norms == 0] = 1.0
        q_filt = q_filt / norms
        u_filt = u_filt / norms
        v_filt = v_filt / norms
        q_filt *= 1.002
        u_filt *= 1.002
        v_filt *= 1.002

    # Per-point uncertainties from off-pulse noise measured on the full dspec.
    sigma_q, sigma_u, sigma_v = _compute_poincare_point_errors(
        noise_ref,
        point_times=np.asarray(color_filt, dtype=float) / (1e3 if unit == 'ms' else (1e6 if unit in ('us', 'µs') else 1.0)),
        noise_fraction=noise_fraction,
    )
    sigma_lon_deg, sigma_lat_deg = _poincare_angle_errors_deg(
        q_filt, u_filt, v_filt, sigma_q, sigma_u, sigma_v
    )

    style = publication_plot_style()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=_pub_figsize(height_ratio=0.92, min_height=6.2))
    ax  = fig.add_subplot(111, projection='3d')

    # ---- Poincaré sphere ----
    u_s = np.linspace(0, 2 * np.pi, 100)
    v_s = np.linspace(0,     np.pi, 100)
    xs  = np.outer(np.cos(u_s), np.sin(v_s))
    ys  = np.outer(np.sin(u_s), np.sin(v_s))
    zs  = np.outer(np.ones_like(u_s), np.cos(v_s))

    # draw a semi-transparent surface rather than a sparse wireframe
    ax.plot_surface(xs, ys, zs, color='lightgray', alpha=0.2, rstride=4, cstride=4,
                    linewidth=0, antialiased=True, zorder=1)
    # add latitude and longitude grid lines for orientation
    n_grid = 12
    # constant latitude (phi) lines
    for lat in np.linspace(0, np.pi, n_grid, endpoint=False)[1:]:
        x_lat = np.cos(u_s) * np.sin(lat)
        y_lat = np.sin(u_s) * np.sin(lat)
        z_lat = np.full_like(u_s, np.cos(lat))
        ax.plot(x_lat, y_lat, z_lat, color='gray', alpha=0.3, linewidth=0.5)
    # constant longitude (theta) lines
    for lon in np.linspace(0, 2*np.pi, n_grid, endpoint=False):
        x_lon = np.cos(lon) * np.sin(v_s)
        y_lon = np.sin(lon) * np.sin(v_s)
        z_lon = np.cos(v_s)
        ax.plot(x_lon, y_lon, z_lon, color='gray', alpha=0.3, linewidth=0.5)

    # ---- Scatter: always the actual filtered bins, not interpolated ----
    sc = ax.scatter(q_filt, u_filt, v_filt,
                    c=color_filt, cmap='viridis',
                    s=60, alpha=1,
                    edgecolors='black', linewidth=0.6, zorder=200,
                    depthshade=True)

    # Add surface-tangent error bars (flattened on the sphere surface).
    # We draw +/- lon and +/- lat segments at constant radius for each point.
    lon_deg = np.degrees(np.arctan2(u_filt, q_filt))
    r_vec = np.sqrt(q_filt**2 + u_filt**2 + v_filt**2)
    lat_deg = np.degrees(np.arcsin(np.clip(v_filt / (r_vec + 1e-20), -1.0, 1.0)))

    def _sph_to_cart(lon_d: float, lat_d: float, radius: float) -> Tuple[float, float, float]:
        lon_r = np.radians(lon_d)
        lat_r = np.radians(lat_d)
        x = radius * np.cos(lat_r) * np.cos(lon_r)
        y = radius * np.cos(lat_r) * np.sin(lon_r)
        z = radius * np.sin(lat_r)
        return float(x), float(y), float(z)

    for i in range(len(q_filt)):
        if not (np.isfinite(lon_deg[i]) and np.isfinite(lat_deg[i]) and np.isfinite(r_vec[i])):
            continue

        dlon = float(sigma_lon_deg[i]) if np.isfinite(sigma_lon_deg[i]) else 0.0
        dlat = float(sigma_lat_deg[i]) if np.isfinite(sigma_lat_deg[i]) else 0.0
        rr = float(r_vec[i])

        if dlon > 0:
            x1, y1, z1 = _sph_to_cart(lon_deg[i] - dlon, lat_deg[i], rr)
            x2, y2, z2 = _sph_to_cart(lon_deg[i] + dlon, lat_deg[i], rr)
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='0.45', linewidth=0.7, alpha=0.6, zorder=150)

        if dlat > 0:
            lat_lo = max(-89.9, lat_deg[i] - dlat)
            lat_hi = min(89.9, lat_deg[i] + dlat)
            x1, y1, z1 = _sph_to_cart(lon_deg[i], lat_lo, rr)
            x2, y2, z2 = _sph_to_cart(lon_deg[i], lat_hi, rr)
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='0.45', linewidth=0.7, alpha=0.6, zorder=150)

    # Optional great/small-circle fits for user-defined segments.
    if circle_fit_mode is not None and len(q_filt) >= 3:
        segments = _build_circle_segments(len(q_filt), circle_fit_segments, filtered_indices=filtered_idx)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(segments))))
        points_xyz = np.column_stack([q_filt, u_filt, v_filt])
        for i_seg, (s_idx, e_idx) in enumerate(segments):
            fit = _fit_circle_on_sphere(points_xyz[s_idx:e_idx + 1], mode=circle_fit_mode)
            if fit is None:
                continue
            arc = fit['arc_xyz']
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                    linestyle='--', linewidth=1.2, alpha=0.9,
                    color=color_cycle[i_seg], zorder=140)

    # adjust view so the bulk of filtered points face the camera
    if len(q_filt) >= 1:
        mean_vec = np.array([np.mean(q_filt), np.mean(u_filt), np.mean(v_filt)])
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            azim = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
            elev = np.degrees(np.arcsin(mean_vec[2] / norm))
            ax.view_init(elev=elev, azim=azim)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02,
                        orientation='horizontal', fraction=0.04)
    cbar.set_label(color_label, fontsize=style['label'], labelpad=5)
    cbar.ax.tick_params(labelsize=style['tick'])


    # ------------------------------------------------------------------
    # Axes, labels, formatting
    # ------------------------------------------------------------------
    # move labels inward and reduce padding
    ax.set_xlabel('Q', fontsize=style['label'], labelpad=-6)
    ax.set_ylabel('U', fontsize=style['label'], labelpad=-6)
    ax.set_zlabel('V', fontsize=style['label'], labelpad=-6)
    # ensure labels are placed on inside of plot volume if supported
    try:
        ax.xaxis.set_label_position('left')
        ax.yaxis.set_label_position('right')
    except Exception:
        pass
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    # drop tick labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # tighten subplot margins for the 3D sphere to reduce surrounding whitespace
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
    if interactive:
        # present an interactive window before saving
        plt.show()
    savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Poincaré sphere plot saved to {output_file}")
    plt.close()



def plot_poincare_projections(
        time_series_data: Dict,
        output_file: str = 'poincare_projections.png',
        projection_type: str = 'all',
        n_time_bins: Optional[int] = None,
        noise_fraction: float = 0.1,
        snr_threshold: float = 5.0,
        time_unit: str = 's',
        force_surface: bool = False,
        rm_results: Optional[Dict] = None,
        center: Optional[Tuple[float, float, float]] = None,
        noise_reference_data: Optional[Dict] = None,
        circle_fit_mode: Optional[str] = None,
        circle_fit_segments: Optional[List[Tuple[int, int]]] = None):
    """
    Generate a 2×2 panel of 2-D cropped projections of the Poincaré sphere.

    The four projections are:

    * **Gnomonic** (central/tangent-plane) – every great-circle arc (Faraday
      rotation path) maps to a straight line, making it ideal for measuring the
      PA rotation rate at a glance.
    * **Stereographic** – conformal (angle-preserving), so the local shape of
      the polarisation ellipse trajectory is faithfully rendered.  Points near
      the projection centre look exactly as they do on the sphere.
    * **Azimuthal equidistant** – preserves arc-length from the projection
      centre, useful for comparing radial excursions in different directions.
    * **Orthographic** – the "view from outside" hemisphere projection.
      Intuitive because it mimics a photograph of the sphere from far away; the
      equatorial Q–U plane shows the linear-polarisation disc.

    All four projections are centred on the mean Stokes vector of the data so
    the track is always near the centre where distortion is smallest.  The
    gnomonic projection is additionally cropped to ±45° from the tangent point
    (a reasonable half-sky for most FRB/pulsar PA swings); the others show the
    full visible hemisphere.

    Colour encodes time (same viridis palette as ``plot_poincare_sphere``).

    Parameters
    ----------
    time_series_data : dict
        Dictionary with keys ``'time'``, ``'I'``, ``'Q'``, ``'U'`` (and
        optionally ``'V'``) containing 2-D arrays (time × frequency or
        frequency × time).
    output_file : str
        Path for the saved PNG.
    n_time_bins : int, optional
        Number of time bins to average into before projecting.  ``None`` keeps
        all samples.
    noise_fraction : float
        Fraction of the time axis used to estimate off-pulse noise.
    snr_threshold : float
        Minimum linear-polarisation SNR threshold.
    time_unit : str
        Time unit label (``'s'``, ``'ms'``, or ``'us'``).
    force_surface : bool
        If ``True`` normalise all vectors to the unit sphere surface before
        projecting.
    rm_results : dict, optional
        Pre-computed binned Stokes parameters from ``fit_rm_time_series``.
        When supplied the function uses the pre-computed ``q_bin``/``u_bin``
        values instead of re-binning from the raw cubes.
    center : tuple(float, float, float), optional
        (x, y, z) unit vector used as the projection centre for all four
        projections.  Defaults to the mean Stokes vector of the data.
    """
    style = publication_plot_style()

    # ------------------------------------------------------------------
    # 1.  Collect & filter Stokes points  (same logic as plot_poincare_sphere)
    # ------------------------------------------------------------------
    if time_series_data is None:
        raise ValueError("plot_poincare_projections requires time_series_data.")

    I_cube = time_series_data['I']
    Q_cube = time_series_data['Q']
    U_cube = time_series_data['U']
    V_cube = time_series_data.get('V', None)
    times  = time_series_data['time']
    n_time = len(times)
    noise_ref = noise_reference_data if noise_reference_data is not None else time_series_data

    if rm_results is not None and 'q_bin' in rm_results:
        q_norm     = np.array(rm_results['q_bin'])
        u_norm     = np.array(rm_results['u_bin'])
        v_norm     = np.array(rm_results.get('v_bin', np.zeros_like(q_norm)))
        color_axis = np.array(rm_results['time'])
        orig_idx = np.arange(q_norm.size, dtype=int)
        pol_list   = np.sqrt(q_norm**2 + u_norm**2)
        if 'valid_bins' in rm_results:
            valid  = np.asarray(rm_results['valid_bins'], dtype=bool)
            q_norm, u_norm, v_norm = q_norm[valid], u_norm[valid], v_norm[valid]
            color_axis = color_axis[valid]
            pol_list   = pol_list[valid]
            orig_idx = orig_idx[valid]
        notnan = (~np.isnan(q_norm)) & (~np.isnan(u_norm)) & (~np.isnan(v_norm))
        q_norm, u_norm, v_norm = q_norm[notnan], u_norm[notnan], v_norm[notnan]
        color_axis = color_axis[notnan]
        pol_list   = pol_list[notnan]
        orig_idx = orig_idx[notnan]
    else:
        if I_cube.ndim != 2:
            raise ValueError("Time-series data must be 2D.")
        time_axis = 0 if I_cube.shape[0] == n_time else 1
        bin_size  = max(1, n_time // n_time_bins) if (n_time_bins and n_time_bins > 0) else 1
        n_bins    = (n_time + bin_size - 1) // bin_size
        q_list, u_list, v_list, pol_list, time_list = [], [], [], [], []
        for i in range(n_bins):
            s, e = i * bin_size, min((i + 1) * bin_size, n_time)
            if e <= s:
                continue
            time_list.append(np.mean(times[s:e]))
            sl = (slice(s, e), slice(None)) if time_axis == 0 else (slice(None), slice(s, e))
            I_m = np.nanmean(I_cube[sl]);  Q_m = np.nanmean(Q_cube[sl])
            U_m = np.nanmean(U_cube[sl])
            V_m = np.nanmean(V_cube[sl]) if V_cube is not None else 0.0
            q_list.append(Q_m / (I_m + 1e-10)); u_list.append(U_m / (I_m + 1e-10))
            v_list.append(V_m / (I_m + 1e-10))
            pol_list.append(np.sqrt(Q_m**2 + U_m**2))
        q_norm     = np.array(q_list)
        u_norm     = np.array(u_list)
        v_norm     = np.array(v_list)
        color_axis = np.array(time_list)
        orig_idx = np.arange(q_norm.size, dtype=int)

    # ---- time-unit scaling ----
    unit = time_unit.lower()
    if unit == 'ms':
        color_axis *= 1e3; color_label = "Time (ms)"
    elif unit in ('us', 'µs'):
        color_axis *= 1e6; color_label = "Time (µs)"
    else:
        color_label = "Time (s)"

    # ---- SNR masking ----
    n_frac   = max(1, int(len(pol_list) * noise_fraction))
    sigma_p  = np.nanstd(pol_list[:n_frac]) or 1e-10
    snr      = np.array(pol_list) / sigma_p
    if rm_results is not None and 'valid_bins' in rm_results:
        mask = np.ones(q_norm.shape[0], dtype=bool)
    else:
        mask = snr > snr_threshold
        if np.sum(mask) < 2:
            mask = snr > 2.0
        if np.sum(mask) < 2:
            print("Error: fewer than 2 points survive SNR cut in projections.")
            return

    filtered_idx = np.asarray(orig_idx[mask], dtype=int)
    q_f = q_norm[mask]
    u_f = u_norm[mask]
    v_f = v_norm[mask]
    c_f = color_axis[mask]

    # ---- optional surface projection ----
    if force_surface:
        r = np.sqrt(q_f**2 + u_f**2 + v_f**2); r[r == 0] = 1.0
        q_f /= r; u_f /= r; v_f /= r

    # ------------------------------------------------------------------
    # 2.  Convert normalised Stokes → sphere longitude / latitude (degrees)
    #     lon = 2ψ = arctan2(U, Q)   (polarisation angle × 2, −180..+180°)
    #     lat = 2χ = arcsin(V / r)   (ellipticity angle × 2, −90..+90°)
    #     r   = sqrt(Q²+U²+V²)       (degree of polarisation, ≤ 1)
    # ------------------------------------------------------------------
    r_f   = np.sqrt(q_f**2 + u_f**2 + v_f**2)
    r_f   = np.where(r_f < 1e-10, 1e-10, r_f)
    lon_f = np.degrees(np.arctan2(u_f, q_f))          # 2ψ  [−180, +180]
    lat_f = np.degrees(np.arcsin(np.clip(v_f / r_f, -1.0, 1.0)))  # 2χ  [−90, +90]

    point_times_s = np.asarray(c_f, dtype=float) / (1e3 if unit == 'ms' else (1e6 if unit in ('us', 'µs') else 1.0))
    sigma_q, sigma_u, sigma_v = _compute_poincare_point_errors(
        noise_ref,
        point_times=point_times_s,
        noise_fraction=noise_fraction,
    )
    sigma_lon_deg, sigma_lat_deg = _poincare_angle_errors_deg(
        q_f, u_f, v_f, sigma_q, sigma_u, sigma_v
    )

    # Optional great/small-circle fits for user-defined segments.
    circle_fits = []
    if circle_fit_mode is not None and len(q_f) >= 3:
        segments = _build_circle_segments(len(q_f), circle_fit_segments, filtered_indices=filtered_idx)
        points_xyz = np.column_stack([q_f, u_f, v_f])
        for i_seg, (s_idx, e_idx) in enumerate(segments):
            fit = _fit_circle_on_sphere(points_xyz[s_idx:e_idx + 1], mode=circle_fit_mode)
            if fit is None:
                continue
            arc = fit['arc_xyz']
            lon_arc = np.degrees(np.arctan2(arc[:, 1], arc[:, 0]))
            lat_arc = np.degrees(np.arcsin(np.clip(arc[:, 2], -1.0, 1.0)))
            circle_fits.append((i_seg, lon_arc, lat_arc))

    # ------------------------------------------------------------------
    # 3.  Projection centre in lon/lat
    # ------------------------------------------------------------------
    if center is not None:
        cx, cy, cz = np.array(center, dtype=float)
        cn = np.sqrt(cx**2 + cy**2 + cz**2)
        cx, cy, cz = (cx/cn, cy/cn, cz/cn) if cn > 1e-10 else (0., 0., 1.)
    else:
        cx = np.mean(q_f); cy = np.mean(u_f); cz = np.mean(v_f)
        cn = np.sqrt(cx**2 + cy**2 + cz**2)
        cx, cy, cz = (cx/cn, cy/cn, cz/cn) if cn > 1e-10 else (0., 0., 1.)

    lon0 = np.degrees(np.arctan2(cy, cx))
    lat0 = np.degrees(np.arcsin(np.clip(cz, -1.0, 1.0)))

    # ------------------------------------------------------------------
    # 4.  Determine crop width/height from data spread (+ 20 % padding)
    #     expressed in metres on the Basemap unit sphere (radius = 1 m).
    # ------------------------------------------------------------------
    try:
        from mpl_toolkits.basemap import Basemap as _Basemap
    except ImportError:
        print("Warning: mpl_toolkits.basemap not available; skipping projection panel.")
        return

    # Use a gnomonic map just to convert lon/lat → metres for extent calc
    _btest = _Basemap(projection='gnom', lat_0=lat0, lon_0=lon0,
                      width=2e7, height=2e7, rsphere=1.0)
    mx_f, my_f = _btest(lon_f, lat_f)
    mx_f = np.array(mx_f, dtype=float); my_f = np.array(my_f, dtype=float)
    fin   = np.isfinite(mx_f) & np.isfinite(my_f)
    if not np.any(fin):
        print("Warning: no finite projected points; skipping projection panel.")
        return
    span  = max(np.ptp(mx_f[fin]), np.ptp(my_f[fin]))
    half  = max(span * 0.5 * 1.20, 0.05)   # 20 % padding, minimum 0.05

    # enforce a minimum angular extent of 30° so that the geometric differences
    # between gnomonic, stereographic, and azimuthal equidistant are visible —
    # at small scales all three collapse to the same flat tangent-plane view.
    # tan(30°) ≈ 0.577 in the unit-sphere gnomonic coordinate system.
    half = max(half, np.tan(np.radians(30)))

    # grid-line spacing: choose 10° or 30° depending on crop size
    ang_half  = np.degrees(np.arctan(half))   # approx angular half-width
    grid_step = 10 if ang_half < 45 else 30

    # ------------------------------------------------------------------
    # 5.  Draw either a 2×2 panel (all) or one selected projection
    # ------------------------------------------------------------------
    projection_map = {
        'gnom': ('gnom', 'Gnomonic\n(great circles → straight lines)'),
        'stere': ('stere', 'Stereographic\n(conformal / angle-preserving)'),
        'aeqd': ('aeqd', 'Azimuthal Equidistant\n(arc-length preserved)'),
        'ortho': ('ortho', 'Orthographic\n(hemisphere view)'),
    }
    proj_key = str(projection_type).lower()
    if proj_key == 'all':
        projections = [
            projection_map['gnom'],
            projection_map['stere'],
            projection_map['aeqd'],
            projection_map['ortho'],
        ]
        fig, axes = plt.subplots(2, 2, figsize=_pub_figsize(height_ratio=1.0, min_height=7.0))
        axes = axes.ravel()
        is_all = True
    else:
        if proj_key not in projection_map:
            raise ValueError(
                "Invalid projection_type. Choose from: all, gnom, stere, aeqd, ortho"
            )
        projections = [projection_map[proj_key]]
        fig, ax_single = plt.subplots(1, 1, figsize=_pub_figsize(height_ratio=0.75, min_height=4.8))
        axes = [ax_single]
        is_all = False

    norm = plt.Normalize(vmin=np.nanmin(c_f), vmax=np.nanmax(c_f))

    for ax, (proj, _title) in zip(axes, projections):
        # ortho shows the full hemisphere; others use the cropped window
        if proj == 'ortho':
            bsmp = _Basemap(projection='ortho', lat_0=lat0, lon_0=lon0,
                            ax=ax, rsphere=1.0)
        else:
            bsmp = _Basemap(projection=proj, lat_0=lat0, lon_0=lon0,
                            width=2*half, height=2*half,
                            ax=ax, rsphere=1.0)

        bsmp.drawmapboundary(fill_color='white', zorder=0)
        if proj == 'ortho':
            # Basemap does not support boundary labels for full-disk ortho views.
            bsmp.drawparallels(
                np.arange(-90, 91, grid_step),
                labels=[False, False, False, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
            bsmp.drawmeridians(
                np.arange(-180, 181, grid_step),
                labels=[False, False, False, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
        else:
            bsmp.drawparallels(
                np.arange(-90, 91, grid_step),
                labels=[True, False, False, True],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
            bsmp.drawmeridians(
                np.arange(-180, 181, grid_step),
                labels=[False, True, True, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)

        # ---- Stokes pole labels ----
        pole_labels = {
            '+Q': ( 0.0,  0.0), '-Q': (180.0,  0.0),
            '+U': (90.0,  0.0), '-U': (-90.0,  0.0),
            '+V': ( 0.0, 90.0), '-V': (  0.0, -90.0),
        }
        for lbl, (plon, plat) in pole_labels.items():
            try:
                px, py = bsmp(plon, plat)
                if np.isfinite(px) and np.isfinite(py):
                    ax.annotate(lbl, (px, py), fontsize=style['annotation'], color='steelblue',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          ec='none', alpha=0.6), zorder=5)
            except Exception:
                pass

        # ---- scatter points ----
        sx, sy = bsmp(lon_f, lat_f)
        sx = np.array(sx, dtype=float); sy = np.array(sy, dtype=float)
        fin_s = np.isfinite(sx) & np.isfinite(sy)
        if np.any(fin_s):
            ax.scatter(sx[fin_s], sy[fin_s],
                       c=c_f[fin_s], cmap='viridis', norm=norm,
                       s=55, edgecolors='black', linewidths=0.6,
                       zorder=4, alpha=1.0)

            # Point error bars in projected plane from lon/lat uncertainties.
            for j in np.where(fin_s)[0]:
                try:
                    x0, y0 = bsmp(lon_f[j], lat_f[j])
                    x_lon, y_lon = bsmp(lon_f[j] + sigma_lon_deg[j], lat_f[j])
                    x_lat, y_lat = bsmp(lon_f[j], lat_f[j] + sigma_lat_deg[j])
                    if not (np.isfinite(x0) and np.isfinite(y0)):
                        continue
                    dx = np.sqrt((x_lon - x0) ** 2 + (x_lat - x0) ** 2)
                    dy = np.sqrt((y_lon - y0) ** 2 + (y_lat - y0) ** 2)
                    if np.isfinite(dx) and np.isfinite(dy):
                        ax.errorbar(x0, y0, xerr=dx, yerr=dy, fmt='none',
                                    ecolor='0.45', elinewidth=0.7, alpha=0.5,
                                    capsize=0, zorder=3)
                except Exception:
                    continue

        # ---- fitted circle arcs ----
        if circle_fits:
            color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(circle_fits))))
            for i_seg, lon_arc, lat_arc in circle_fits:
                tx, ty = bsmp(lon_arc, lat_arc)
                tx = np.asarray(tx, dtype=float)
                ty = np.asarray(ty, dtype=float)
                ok = np.isfinite(tx) & np.isfinite(ty)
                if np.sum(ok) >= 2:
                    ax.plot(tx[ok], ty[ok], linestyle='--', linewidth=1.2,
                            color=color_cycle[i_seg], alpha=0.9, zorder=2)

        # Intentionally keep panels untitled for cleaner publication layout.
        ax.tick_params(axis='both', labelsize=style['tick'])

    # ---- shared colorbar ----
    if is_all:
        fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.08,
                            hspace=0.15, wspace=0.15)
        cax = fig.add_axes([0.25, 0.02, 0.50, 0.016])
    else:
        fig.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.14)
        cax = fig.add_axes([0.22, 0.05, 0.56, 0.025])
    sm  = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(color_label, fontsize=style['label'])
    cb.ax.tick_params(labelsize=style['tick'])

    # ---- suptitle ----
    #fig.suptitle(
    #    f'Poincaré Sphere — 2-D Projections\n'
    #    f'Centre: lon={lon0:.1f}°  lat={lat0:.1f}°  '
    #    f'(Q={cx:.3f}, U={cy:.3f}, V={cz:.3f})',
    #    fontsize=12, y=0.97)

    savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Poincaré projection panel saved to {output_file}")
    plt.close()



def plot_rm_time_series(time_array: np.ndarray, rm_results: Dict,
                       output_file: str = 'rm_time_series.png',
                       time_profile: Optional[np.ndarray] = None,
                       separate_peaks: bool = False,
                       min_gap_bins: int = 3,
                       min_peak_bins: int = 3,
                       max_merge_gap: int = 0,
                       time_series_data: Optional[Dict] = None,
                       freq_hz: Optional[np.ndarray] = None,
                       n_rm_bins: int = 20,
                       noise_fraction: float = 0.1):
    """
    Plot RM as a function of time.
    
    Parameters:
    -----------
    time_array : array
        Time array
    rm_results : dict
        Results from fit_rm_time_series containing 'rm', 'snr', etc.
    output_file : str
        Output filename for plot
    time_profile : array, optional
        Time profile (total intensity) for peak detection
    separate_peaks : bool
        Whether to create separate side-by-side plots for each peak (default: False)
    min_gap_bins : int
        Minimum gap size to separate peaks (default: 3)
    min_peak_bins : int
        Minimum number of consecutive significant bins for a valid peak (default: 3)
    max_merge_gap : int
        Maximum gap size for merging nearby peaks (default: 0, no merging)
    time_series_data : dict, optional
        Original time series data for binned RM analysis
    freq_hz : array, optional
        Frequency array for binned RM analysis
    n_rm_bins : int
        Number of bins for binned RM analysis (default: 20)
    noise_fraction : float
        Fraction of Stokes I samples used when estimating noise for lower-panel
        uncertainty & signal masking (default: 0.1)
    """
    style = publication_plot_style()

    # Identify peak regions if requested
    if separate_peaks and time_profile is not None:
        snr_array = rm_results.get('snr', None)
        peak_regions = find_peak_regions(time_profile, snr_array, min_gap_bins, 
                                        min_peak_bins, max_merge_gap)
        n_peaks = len(peak_regions)
        print(f"\nIdentified {n_peaks} separate peak region(s):")
        for i, (start, end) in enumerate(peak_regions):
            print(f"  Peak {i+1}: bins {start}-{end} ({end-start+1} bins)")
    else:
        peak_regions = [(0, len(time_array) - 1)]
        n_peaks = 1
    
    # Create figure with 3 rows of subplots for each peak
    # Top row: combined Pulse Profile (left y-axis) + RM (right y-axis)
    # Middle row: Polarisation fractions
    # Bottom row: Polarisation Angle (PA) and Ellipticity Angle (EA)
    # sharex ensures lower panels use same time axis as top
    rm_ts_height = max(6.8, 2.25 * 3)
    rm_ts_width = SINGLE_COLUMN_WIDTH_IN if n_peaks == 1 else min(TWO_COLUMN_WIDTH_IN, SINGLE_COLUMN_WIDTH_IN * n_peaks)
    fig, axes = plt.subplots(3, n_peaks, figsize=(rm_ts_width, rm_ts_height), squeeze=False, sharex='col')
    
    # Determine which points have sufficient signal (SNR >= 5 or use other criteria)
    if 'snr' in rm_results and np.any(rm_results['snr'] > 0):
        snr_threshold = 5.0
        good_signal = rm_results['snr'] >= snr_threshold
    else:
        # If no SNR, use all points
        good_signal = np.ones(len(time_array), dtype=bool)
    
    # Plot for each peak region
    for peak_idx, (start_idx, end_idx) in enumerate(peak_regions):
        # Extract data for this peak
        peak_mask = np.zeros(len(time_array), dtype=bool)
        peak_mask[start_idx:end_idx+1] = True
        
        time_peak = time_array[peak_mask]
        good_signal_peak = good_signal[peak_mask]
        # If full time-series data is available, compute full-resolution profiles
        full_time = None
        snr_full = None
        P_frac_full = None
        L_frac_full = None
        V_frac_full = None
        if time_series_data is not None and 'time' in time_series_data:
            full_time = np.asarray(time_series_data['time'])
            # Determine time axis in data arrays
            if time_series_data['I'].ndim == 2:
                if time_series_data['I'].shape[0] == len(full_time):
                    time_axis_dim = 0
                elif time_series_data['I'].shape[1] == len(full_time):
                    time_axis_dim = 1
                else:
                    time_axis_dim = 0
            else:
                time_axis_dim = 0

            # Compute per-sample averages over frequency axis
            if time_axis_dim == 0:
                I_full = np.mean(time_series_data['I'], axis=1)
                Q_full = np.mean(time_series_data['Q'], axis=1)
                U_full = np.mean(time_series_data['U'], axis=1)
                V_full = np.mean(time_series_data['V'], axis=1) if 'V' in time_series_data else np.zeros_like(I_full)

            else:
                I_full = np.mean(time_series_data['I'], axis=0)
                Q_full = np.mean(time_series_data['Q'], axis=0)
                U_full = np.mean(time_series_data['U'], axis=0)
                V_full = np.mean(time_series_data['V'], axis=0) if 'V' in time_series_data else np.zeros_like(I_full)

            pol_int_full = np.sqrt(Q_full**2 + U_full**2)

            P_frac_full = np.sqrt(Q_full**2 + U_full**2 + V_full**2) / (I_full + 1e-10)
            L_full = np.sqrt(Q_full**2 + U_full**2)
            L_frac_full = L_full / (I_full + 1e-10)
            # preserve sign of V fraction (positive or negative circular polarisation)
            V_frac_full = V_full / (I_full + 1e-10)


            # Restrict full-resolution arrays to the same time window as `time_peak`.
            # Pad the window by half the sampling interval to avoid excluding nearby samples
            if len(time_peak) > 0:
                tmin = time_peak.min()
                tmax = time_peak.max()
                if len(full_time) > 1:
                    dt = np.median(np.diff(full_time))
                    pad = dt / 2.0
                else:
                    pad = 0.0
                full_mask = (full_time >= (tmin - pad)) & (full_time <= (tmax + pad))
            else:
                full_mask = np.ones_like(full_time, dtype=bool)

            # If mask is empty (no overlap), fall back to the nearest sample
            if not np.any(full_mask):
                if len(full_time) > 0:
                    centre = 0.5 * (tmin + tmax) if len(time_peak) > 0 else full_time[0]
                    idx = int(np.argmin(np.abs(full_time - centre)))
                    full_mask = np.zeros_like(full_time, dtype=bool)
                    full_mask[idx] = True
                else:
                    full_mask = np.ones_like(full_time, dtype=bool)

            # Estimate noise using initial fraction of Stokes I samples
            n_frac = max(1, int(len(I_full) * noise_fraction))
            noise_est = np.nanstd(I_full[:n_frac])
            if noise_est <= 0:
                # Fallback robust MAD or small fraction of median
                mad = np.nanmedian(np.abs(I_full - np.nanmedian(I_full)))
                if mad > 0:
                    noise_est = mad / 0.6745
                else:
                    noise_est = max(np.nanmedian(I_full) * 0.1, 1e-10)

            # Use noise estimate of Stokes I as the per-sample noise level for polarisation
            snr_full = pol_int_full / (noise_est + 1e-10)
        
        # Top panel: combined pulse profile (left y-axis) and RM (right y-axis)
        ax_top = axes[0, peak_idx]
        ax_top_twin = ax_top.twinx()
        # Ensure RM ticks are on the right and label axis
        ax_top_twin.yaxis.set_label_position('right')
        ax_top_twin.yaxis.tick_right()
        ax_top_twin.set_ylabel('RM (rad/m²)', fontsize=style['label'], color='m')
        ax_top_twin.tick_params(axis='y', colors='m', labelsize=style['tick'])

        rm_peak = rm_results['rm'][peak_mask]

        # Plot all RM points within the time-cropped window.  the only masks
        # applied anywhere in the figure now are the crop itself and the
        # Stokes I S/N > 2 threshold used later for the fraction panel.
        tms = time_peak * 1e3
        rm_err_peak = rm_results.get('rm_err', None)
        if rm_err_peak is not None:
            rm_err_peak = rm_err_peak[peak_mask]
            ax_top_twin.errorbar(tms, rm_peak, yerr=rm_err_peak,
                                 fmt='o-', color='m', markersize=3,
                                 linewidth=1.5, capsize=2, alpha=0.7,
                                 label='RM')
        else:
            ax_top_twin.plot(tms, rm_peak, 'm-o', linewidth=1.5,
                             markersize=3, label='RM')
        # mean line for reference
        if len(rm_peak) > 0:
            rm_mean = np.nanmean(rm_peak)
            rm_std = np.nanstd(rm_peak)
            ax_top_twin.axhline(rm_mean, color='r', linestyle='--',
                                linewidth=2, alpha=0.5,
                                #label=f'Mean RM: {rm_mean:.2f} rad/m²'
                                )

        # Plot pulse profile (intensity) on left axis
        ax_top.plot(full_time[full_mask] * 1e3, I_full[full_mask], 'k-', linewidth=1.5, label='I')
        ax_top.set_ylabel('Intensity (arbitrary units)', fontsize=style['label'])
        ax_top.tick_params(axis='y', labelsize=style['tick'])
        # plot L and V
        ax_top.plot(full_time[full_mask] * 1e3, L_full[full_mask], 'r-', linewidth=1.5, label='L')
        ax_top.plot(full_time[full_mask] * 1e3, V_full[full_mask], 'b-', linewidth=1.5, label='V')
        
        # Compute binned RM if data is available and RM results are not already binned
        if time_series_data is not None and freq_hz is not None and not rm_results.get('is_binned', False):
            # Determine time axis orientation
            if time_series_data['I'].shape[0] == len(time_array):
                time_axis_dim = 0
            else:
                time_axis_dim = 1
            
            # Create bins
            n_time = np.sum(peak_mask)
            bin_size = max(1, n_time // n_rm_bins)
            n_bins_actual = (n_time + bin_size - 1) // bin_size
            
            binned_rm = []
            binned_rm_err = []
            binned_time = []
            
            # Compute off-pulse Q/U noise from same fraction of samples used
            # for I noise estimation so binned fits use consistent noise levels.
            if time_axis_dim == 0:
                I_full_tmp = np.mean(time_series_data['I'], axis=1)
                Q_tmp = time_series_data['Q']
                U_tmp = time_series_data['U']
            else:
                I_full_tmp = np.mean(time_series_data['I'], axis=0)
                Q_tmp = time_series_data['Q']
                U_tmp = time_series_data['U']
            n_frac_tmp = max(1, int(len(I_full_tmp) * noise_fraction))
            if time_axis_dim == 0:
                q_off_tmp = Q_tmp[:n_frac_tmp, :]
                u_off_tmp = U_tmp[:n_frac_tmp, :]
            else:
                q_off_tmp = Q_tmp[:, :n_frac_tmp]
                u_off_tmp = U_tmp[:, :n_frac_tmp]
            q_std_chan_tmp = np.nanstd(q_off_tmp, axis=0 if time_axis_dim == 0 else 1)
            u_std_chan_tmp = np.nanstd(u_off_tmp, axis=0 if time_axis_dim == 0 else 1)
            noise_q_tmp = np.nanmedian(q_std_chan_tmp) if np.nanmedian(q_std_chan_tmp) > 0 else (np.nanmean(q_std_chan_tmp) if np.nanmean(q_std_chan_tmp) > 0 else 1e-10)
            noise_u_tmp = np.nanmedian(u_std_chan_tmp) if np.nanmedian(u_std_chan_tmp) > 0 else (np.nanmean(u_std_chan_tmp) if np.nanmean(u_std_chan_tmp) > 0 else 1e-10)
            # Off-pulse I noise (time-domain) for binned fits
            noise_i_tmp = np.nanstd(I_full_tmp[:n_frac_tmp])
            if noise_i_tmp <= 0:
                mad_tmp = np.nanmedian(np.abs(I_full_tmp - np.nanmedian(I_full_tmp)))
                if mad_tmp > 0:
                    noise_i_tmp = mad_tmp / 0.6745
                else:
                    noise_i_tmp = max(np.nanmedian(I_full_tmp) * 0.1, 1e-10)

            print(f"  Computing binned RM with {n_bins_actual} bins...")
            
            for bin_idx in range(n_bins_actual):
                bin_start = start_idx + bin_idx * bin_size
                bin_end = min(start_idx + (bin_idx + 1) * bin_size, end_idx + 1)
                
                if bin_end <= bin_start:
                    continue
                
                # Average data in this bin
                if time_axis_dim == 0:
                    i_avg = np.mean(time_series_data['I'][bin_start:bin_end, :], axis=0)
                    q_avg = np.mean(time_series_data['Q'][bin_start:bin_end, :], axis=0)
                    u_avg = np.mean(time_series_data['U'][bin_start:bin_end, :], axis=0)
                    v_avg = np.mean(time_series_data['V'][bin_start:bin_end, :], axis=0) if 'V' in time_series_data else None
                else:
                    i_avg = np.mean(time_series_data['I'][:, bin_start:bin_end], axis=1)
                    q_avg = np.mean(time_series_data['Q'][:, bin_start:bin_end], axis=1)
                    u_avg = np.mean(time_series_data['U'][:, bin_start:bin_end], axis=1)
                    v_avg = np.mean(time_series_data['V'][:, bin_start:bin_end], axis=1) if 'V' in time_series_data else None
                
                # Fit RM for this bin using RM synthesis
                fitter = RMFitter(freq_hz, i_avg, q_avg, u_avg, v_avg)
                result = fitter._fit_rm_with_rmtools(rm_range=(-1000, 1000), n_rm=500,
                                                     noise_i=noise_i_tmp,
                                                     noise_q=noise_q_tmp, noise_u=noise_u_tmp)
                rm_fit = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
                # Estimate error from RM-CLEAN if available, otherwise use noise-estimate fallback
                rm_err = result.get('rm_clean_err', result.get('rm_err', result.get('noise_estimate', 0) * 2))
                
                binned_rm.append(rm_fit)
                binned_rm_err.append(rm_err)
                binned_time.append(time_array[(bin_start + bin_end) // 2])
            
            # Plot binned RM on secondary y-axis (right axis)
            ax_top_twin.errorbar(np.array(binned_time) * 1e3, binned_rm, yerr=binned_rm_err,
                                 fmt='o-', color='red', markersize=5, linewidth=2, capsize=3,
                                 label=f'Binned RM ({n_bins_actual} bins)')
            ax_top_twin.set_ylabel('RM (rad/m²)', fontsize=style['label'], color='red')
            ax_top_twin.tick_params(axis='y', labelcolor='red', labelsize=style['tick'])
        
        ax_top.set_xlabel('Time (ms)', fontsize=style['label'])
        if n_peaks > 1:
            ax_top.set_title(f'Peak {peak_idx+1}: Pulse Profile & RM', fontsize=style['title'], fontweight='bold')
        #else:
        #    ax_top.set_title('Pulse Profile & Rotation Measure', fontsize=14, fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        # Combine legends from both axes
        # remove any existing legends on either axis to avoid duplicate boxes
        if ax_top.get_legend() is not None:
            ax_top.get_legend().remove()
        if ax_top_twin.get_legend() is not None:
            ax_top_twin.get_legend().remove()

        handles1, labels1 = ax_top.get_legend_handles_labels()
        handles2, labels2 = ax_top_twin.get_legend_handles_labels()
        if peak_idx == 0:
            # Merge legends from profile (left axis) and RM (right axis),
            # but prefer a single RM entry: keep only the magenta 'RM' handle
            all_handles = list(handles1) + list(handles2)
            all_labels = list(labels1) + list(labels2)
            final_handles = []
            final_labels = []
            seen = set()
            for h, lab in zip(all_handles, all_labels):
                if not lab or lab in seen:
                    continue
                # skip binned RM entries to avoid duplicate RM-type labels
                if 'Binned RM' in lab or 'Binned' in lab:
                    continue
                # if this is the generic 'RM' label, ensure it's the magenta one
                if lab.strip() == 'RM':
                    keep = False
                    try:
                        col = getattr(h, 'get_color', lambda: None)()
                    except Exception:
                        col = None
                    if isinstance(col, str) and col.lower() in ('m', 'magenta'):
                        keep = True
                    else:
                        try:
                            # handle RGBA tuples
                            import numpy as _np
                            colarr = _np.asarray(col)
                            if colarr.size >= 3 and _np.allclose(colarr[:3], _np.array([1.0, 0.0, 1.0]), atol=0.08):
                                keep = True
                        except Exception:
                            keep = False
                    if not keep:
                        continue
                final_handles.append(h)
                final_labels.append(lab)
                seen.add(lab)
            if final_handles:
                ax_top.legend(final_handles, final_labels, fontsize=style['legend'], loc='best')
        ax_top.tick_params(axis='x', labelsize=style['tick'])
        
        # Plot 2 (bottom row): Polarisation fractions vs time
        ax3 = axes[1, peak_idx]
        
        if time_series_data is not None and full_time is not None:
            # Plot full-resolution polarisation fractions with error bands
            # compute noise estimate based on off-pulse I
            # estimate noise using first N samples of Stokes I
            n_frac = max(1, int(len(I_full) * noise_fraction))
            noise_est = np.nanstd(I_full[:n_frac])
            if noise_est <= 0:
                # fallback to robust MAD or small fraction
                mad = np.nanmedian(np.abs(I_full - np.nanmedian(I_full)))
                if mad > 0:
                    noise_est = mad / 0.6745
                else:
                    noise_est = max(np.nanmedian(I_full) * 0.1, 1e-10)
            # fractional uncertainty for all fractions
            err_frac = noise_est / (I_full + 1e-10)

            times_ms = full_time[full_mask] * 1e3
            # determine bins with poor total‑intensity S/N using the full set
            # rather than the already-cropped slices (avoids mismatched shapes)
            snr_i_full = I_full / (noise_est + 1e-10)
            badi_full = snr_i_full < 2.0
            # apply masking globally; NaNs propagate automatically when we later
            # slice by full_mask for plotting
            P_frac_full[badi_full] = np.nan
            L_frac_full[badi_full] = np.nan
            V_frac_full[badi_full] = np.nan

            # boolean mask for plotted subset (crop & good S/N)
            signal_mask = ~badi_full[full_mask]
            if not np.any(signal_mask):
                signal_mask = np.ones_like(signal_mask, dtype=bool)
            # compute rm_mask for informational purposes (not applied)
            rm_mask = np.ones_like(times_ms, dtype=bool)
            if np.any(good_signal_peak):
                good_times = time_peak[good_signal_peak] * 1e3
                dt = np.median(np.diff(time_peak)) * 1e3 if len(time_peak) > 1 else 0
                tol = dt/2 + 1e-9
                rm_mask = np.array([np.any(np.abs(t - good_times) <= tol) for t in times_ms])
            combined = signal_mask  # only mask on I S/N

            def plot_runs(x, y, axis, **kwargs):
                if len(x) == 0:
                    return
                idx = np.arange(len(x))
                splits = np.where(np.diff(idx) != 1)[0] + 1
                for seg in np.split(idx, splits):
                    if len(seg) > 0:
                        axis.plot(x[seg], y[seg], **kwargs)

            # plot P, L, (V) only where both masks true, but split into contiguous
            # segments in the original time-series so gaps are not connected
            full_indices = np.where(full_mask)[0]


            # overlay binned fractions produced during fitting (always in
            # rm_results when time-binning was requested).  this avoids the
            # earlier manual averaging, which sometimes produced nan values.
            if 'P_frac_bin' in rm_results:
                bt = np.asarray(rm_results['time']) * 1e3
                pf_bin = np.asarray(rm_results['P_frac_bin'])
                lf_bin = np.asarray(rm_results['L_frac_bin'])
                vf_bin = np.asarray(rm_results.get('V_frac_bin', []))

                # apply the same mask that was used for the full-resolution
                # fractions panel.  the fitter stored a boolean array indicating
                # which bins passed the I‑S/N cut; fall back to nearest‑sample
                # mapping if it's not available.
                if 'valid_bins' in rm_results:
                    bin_mask = np.asarray(rm_results['valid_bins'], dtype=bool)
                else:
                    # map each binned time to the closest full-time sample and
                    # use the previously computed `combined` mask if present
                    if 'combined' in locals() and full_time is not None:
                        # full_time indices corresponding to the crop
                        idx = np.argmin(np.abs(full_time[:, None] - (bt/1e3)[None, :]), axis=0)
                        bin_mask = combined[idx]
                    else:
                        bin_mask = np.ones_like(bt, dtype=bool)

                if np.any(bin_mask):
                    ax3.plot(bt[bin_mask], pf_bin[bin_mask], 'k--', linewidth=2, zorder=1,
                             label='P/I')
                    #ax3.scatter(bt[bin_mask], pf_bin[bin_mask], 25, 'k', label=None, zorder=20)
                    ax3.plot(bt[bin_mask], lf_bin[bin_mask], 'r--', linewidth=2, zorder=1,
                             label='L/I')
                    ax3.scatter(bt[bin_mask], lf_bin[bin_mask], 25, 'r', label=None, zorder=20)
                    if 'V_frac_bin' in rm_results and vf_bin.size:
                        ax3.plot(bt[bin_mask], vf_bin[bin_mask], 'b--', linewidth=2, zorder=1,
                                 label='V/I')
                        ax3.scatter(bt[bin_mask], vf_bin[bin_mask], 25, 'b', label=None, zorder=20)
                else:
                    combined_idx = full_indices[combined]

                    if combined_idx.size > 0:
                        # find splits where indices are not consecutive
                        seg_splits = np.where(np.diff(combined_idx) != 1)[0] + 1
                        segs = np.split(combined_idx, seg_splits) if len(seg_splits) > 0 else [combined_idx]
                        for seg in segs:
                            if seg.size == 0:
                                continue
                            ax3.plot(full_time[seg] * 1e3, P_frac_full[seg], color='k', linewidth=1.5)
                            ax3.plot(full_time[seg] * 1e3, L_frac_full[seg], color='r', linewidth=1.5)
                            if 'V' in time_series_data:
                                ax3.plot(full_time[seg] * 1e3, V_frac_full[seg], color='b', linewidth=1.5)
                        # add legend entries
                        ax3.plot([], [], color='k', linewidth=1.5)
                        ax3.plot([], [], color='r', linewidth=1.5)
                        if 'V' in time_series_data:
                            ax3.plot([], [], color='b', linewidth=1.5)

            # adjust y limits based on plotted data (use signal_mask within full_mask)
            try:
                sig_idx = full_indices[signal_mask]
                max_frac = np.nanmax([P_frac_full[sig_idx].max(),
                                       L_frac_full[sig_idx].max(),
                                       V_frac_full[sig_idx].max() if 'V' in time_series_data else 0])
                min_frac = np.nanmin([P_frac_full[sig_idx].min(),
                                       L_frac_full[sig_idx].min(),
                                       V_frac_full[sig_idx].min() if 'V' in time_series_data else 0])
            except Exception:
                max_frac = np.nanmax([P_frac_full[full_mask].max(), L_frac_full[full_mask].max(), V_frac_full[full_mask].max() if 'V' in time_series_data else 0])
                min_frac = np.nanmin([P_frac_full[full_mask].min(), L_frac_full[full_mask].min(), V_frac_full[full_mask].min() if 'V' in time_series_data else 0])
            # consider binned data as well (from fit results)
            if 'P_frac_bin' in rm_results:
                max_frac = max(max_frac,
                               np.nanmax(rm_results['P_frac_bin']),
                               np.nanmax(rm_results['L_frac_bin']),
                               np.nanmax(rm_results.get('V_frac_bin', [])) if 'V' in time_series_data else 0)
                min_frac = min(min_frac,
                               np.nanmin(rm_results['P_frac_bin']),
                               np.nanmin(rm_results['L_frac_bin']),
                               np.nanmin(rm_results.get('V_frac_bin', [])) if 'V' in time_series_data else 0)
            # constrain limits within [-1,1]
            lower = max(-1.0, min_frac * 1.1)
            upper = min(1.0, max_frac * 1.1)
            ax3.set_ylim(lower, upper)
            
        ax3.set_xlabel('Time (ms)', fontsize=style['label'])
        ax3.set_ylabel('Polarisation Fraction', fontsize=style['label'])
        if n_peaks > 1:
            ax3.set_title(f'Peak {peak_idx+1}: Polarisation Fractions', fontsize=style['title'], fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=style['legend'])
        ax3.tick_params(axis='both', labelsize=style['tick'])

        # Plot 3 (bottom row): Polarisation Angle (PA) and Ellipticity Angle (EA)
        ax_pa = axes[2, peak_idx]
        if time_series_data is not None and full_time is not None:
            # compute angles from full-resolution Q,U,V
            Q_vals = Q_full[full_mask]
            U_vals = U_full[full_mask]
            V_vals = V_full[full_mask] if 'V' in time_series_data else np.zeros_like(Q_vals)
            times_ms = full_time[full_mask] * 1e3

            # polarisation angle (radians) = 0.5 * atan2(U, Q)
            pa_rad = 0.5 * np.arctan2(U_vals, Q_vals)
            # unwrap then convert to degrees
            pa_deg = np.degrees(np.unwrap(pa_rad))

            # ellipticity angle (radians) = 0.5 * arcsin(V / P)
            P_amp = np.sqrt(Q_vals**2 + U_vals**2 + V_vals**2) + 1e-10
            sin_arg = np.clip(V_vals / P_amp, -1.0, 1.0)
            ea_rad = 0.5 * np.arcsin(sin_arg)
            ea_deg = np.degrees(ea_rad)

            # Estimate Q/U/V noise from off-pulse region (same fraction used for I)
            n_frac_pa = max(1, int(len(I_full) * noise_fraction))
            sigma_Q = np.nanstd(Q_full[:n_frac_pa])
            sigma_U = np.nanstd(U_full[:n_frac_pa])
            sigma_V = np.nanstd(V_full[:n_frac_pa]) if 'V' in time_series_data else 0.0
            # fallback robustly to MAD-derived sigma
            if sigma_Q <= 0:
                mad_q = np.nanmedian(np.abs(Q_full - np.nanmedian(Q_full)))
                sigma_Q = mad_q / 0.6745 if mad_q > 0 else 1e-10
            if sigma_U <= 0:
                mad_u = np.nanmedian(np.abs(U_full - np.nanmedian(U_full)))
                sigma_U = mad_u / 0.6745 if mad_u > 0 else 1e-10
            if sigma_V <= 0 and 'V' in time_series_data:
                mad_v = np.nanmedian(np.abs(V_full - np.nanmedian(V_full)))
                sigma_V = mad_v / 0.6745 if mad_v > 0 else 1e-10

            # PA uncertainty propagation (radians): var(PA) = 1/4 * (U^2*sigma_Q^2 + Q^2*sigma_U^2) / (Q^2+U^2)^2
            P_lin_sq = Q_vals**2 + U_vals**2 + 1e-20
            pa_sigma_rad = 0.5 * np.sqrt((U_vals**2 * sigma_Q**2 + Q_vals**2 * sigma_U**2) / (P_lin_sq**2))
            pa_sigma_deg = np.degrees(pa_sigma_rad)

            # EA uncertainty propagation
            # sigma_P approximated from Q,U uncertainties
            sigma_P = np.sqrt((Q_vals**2 * sigma_Q**2 + U_vals**2 * sigma_U**2)) / (P_amp + 1e-10)
            sigma_VoverP = np.sqrt((sigma_V**2 / (P_amp**2)) + ((V_vals**2) * (sigma_P**2) / (P_amp**2 + 1e-20)))
            denom = np.sqrt(1.0 - (V_vals / P_amp)**2 + 1e-20)
            ea_sigma_rad = 0.5 * (sigma_VoverP / denom)
            ea_sigma_deg = np.degrees(ea_sigma_rad)

            # construct a mask for PA/EA that also removes points with large
            # angle uncertainty, mimicking the ``nongoodphi``/``nongoodpsi``
            # logic from the example.  here we treat either PA or EA error
            # exceeding 50° as bad.
            bad_pa = pa_sigma_deg > 50.0
            bad_ea = ea_sigma_deg > 50.0
            # begin with the I‑S/N mask (``badi`` above, expressed as good=True)
            try:
                mask_pa = combined.copy()
            except Exception:
                mask_pa = np.ones_like(times_ms, dtype=bool)
            mask_pa &= ~bad_pa
            mask_pa &= ~bad_ea

            # plot PA and EA without connecting gaps (small markers)
            def scatter_runs(x, y, axis, **kwargs):
                if len(x) == 0:
                    return
                idx = np.arange(len(x))
                splits = np.where(np.diff(idx) != 1)[0] + 1
                for seg in np.split(idx, splits):
                    if len(seg) > 0:
                        axis.scatter(x[seg], y[seg], **kwargs)

            scatter_runs(times_ms[mask_pa], pa_deg[mask_pa], ax_pa, color='r', s=8, label='PA', zorder=2)
            scatter_runs(times_ms[mask_pa], ea_deg[mask_pa], ax_pa, color='b', s=8, label='EA', zorder=2)

            # add pointwise error bars for PA and EA
            if np.any(mask_pa):
                ax_pa.errorbar(times_ms[mask_pa], pa_deg[mask_pa], yerr=pa_sigma_deg[mask_pa], fmt='none', ecolor='gray', alpha=0.6, capsize=2, zorder=1)
                ax_pa.errorbar(times_ms[mask_pa], ea_deg[mask_pa], yerr=ea_sigma_deg[mask_pa], fmt='none', ecolor='gray', alpha=0.6, capsize=2, zorder=1)

        ax_pa.set_xlabel('Time (ms)', fontsize=style['label'])
        ax_pa.set_ylabel('Angle (deg)', fontsize=style['label'])
        if n_peaks > 1:
            ax_pa.set_title(f'Peak {peak_idx+1}: PA & EA', fontsize=style['title'], fontweight='bold')
        ax_pa.grid(True, alpha=0.3)
        ax_pa.legend(loc='best', fontsize=style['legend'])
        ax_pa.tick_params(axis='both', labelsize=style['tick'])
    
    plt.tight_layout()
    
    savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Time series plot saved to {output_file}")
    plt.close()


def plot_rm_results(fitter: RMFitter, rm_synthesis_result: Dict,
                   output_file: str = 'rm_fitting_results.png',
                   pol_frac_err: Optional[np.ndarray] = None,
                   valid_mask: Optional[np.ndarray] = None,
                   circ_frac_err: Optional[np.ndarray] = None,
                   circ_valid_mask: Optional[np.ndarray] = None,
                   show_frac_panel: bool = True):
    """
    Create comprehensive plots of RM fitting results.
    
    Parameters:
    -----------
    fitter : RMFitter
        RMFitter object with data
    rm_synthesis_result : dict
        Results from RM synthesis
    output_file : str
        Output filename for plot
    pol_frac_err : array, optional
        Per-channel uncertainty for linear fraction (L/I). If provided, this
        uncertainty is used for shaded bands (same data path as Burn-law plots).
    valid_mask : array, optional
        Boolean mask for valid linear-fraction channels.
    circ_frac_err : array, optional
        Per-channel uncertainty for circular fraction (V/I).
    circ_valid_mask : array, optional
        Boolean mask for valid circular-fraction channels.
    show_frac_panel : bool
        If False, omit the third (polarisation-fraction) panel.
    
    Note
    ----
    The Poincaré sphere is no longer produced automatically from here – it
    depends only on time-series information.  Callers who require one should
    invoke :func:`plot_poincare_sphere` explicitly with the appropriate
    ``time_series_data``.
    """
    style = publication_plot_style()
    n_rows = 3 if show_frac_panel else 2
    fig_height = 12 if show_frac_panel else 8
    fig_height = 7.4 if show_frac_panel else 5.2
    fig, axes = plt.subplots(n_rows, 1, figsize=(TWO_COLUMN_WIDTH_IN, fig_height), sharex=False)
    pol_frac_err_arr = None if pol_frac_err is None else np.asarray(pol_frac_err, dtype=float)
    valid_mask_arr = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    circ_frac_err_arr = None if circ_frac_err is None else np.asarray(circ_frac_err, dtype=float)
    circ_valid_mask_arr = None if circ_valid_mask is None else np.asarray(circ_valid_mask, dtype=bool)

    def _robust_channel_noise(series: np.ndarray) -> float:
        """Estimate channel noise robustly from first-difference scatter."""
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return float(np.nanstd(arr)) if arr.size > 0 else np.nan
        diffs = np.diff(arr)
        mad = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
        if np.isfinite(mad) and mad > 0:
            # sigma(diff) ~= sqrt(2) * sigma(noise)
            return float((1.4826 * mad) / np.sqrt(2.0))
        return float(np.nanstd(diffs) / np.sqrt(2.0))

    # Estimate per-channel noise levels for uncertainty propagation.
    # Use robust first-difference noise (less sensitive to real spectral structure).
    sigma_i = _robust_channel_noise(fitter.stokes_i)
    sigma_q = _robust_channel_noise(fitter.stokes_q)
    sigma_u = _robust_channel_noise(fitter.stokes_u)
    sigma_v = _robust_channel_noise(fitter.stokes_v) if fitter.stokes_v is not None else 0.0
    if not np.isfinite(sigma_i) or sigma_i <= 0:
        sigma_i = float(rm_synthesis_result.get('noise_estimate', 1e-10))
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        sigma_q = sigma_i
    if not np.isfinite(sigma_u) or sigma_u <= 0:
        sigma_u = sigma_i
    if fitter.stokes_v is not None and (not np.isfinite(sigma_v) or sigma_v <= 0):
        sigma_v = sigma_i

    used_external_li_err = pol_frac_err_arr is not None and pol_frac_err_arr.shape == np.asarray(fitter.stokes_i).shape
    used_external_vi_err = (
        fitter.stokes_v is not None
        and circ_frac_err_arr is not None
        and circ_frac_err_arr.shape == np.asarray(fitter.stokes_i).shape
    )
    print(
        f"  RM plot uncertainty source: L/I={'Burn-law propagated' if used_external_li_err else 'local fallback'}"
        f", V/I={'Burn-law propagated' if used_external_vi_err else ('local fallback' if fitter.stokes_v is not None else 'N/A')}"
    )
    
    # Plot 1: Polarisation angle vs λ²
    ax1 = axes[0]
    pol_angle_deg = np.degrees(np.unwrap(fitter.pol_angle))
    pol_angle_deg = ((pol_angle_deg + 90.0) % 180.0) - 90.0
    q_vals = np.asarray(fitter.stokes_q, dtype=float)
    u_vals = np.asarray(fitter.stokes_u, dtype=float)
    i_vals = np.asarray(fitter.stokes_i, dtype=float)
    l_meas = np.sqrt(q_vals**2 + u_vals**2)
    lin_sq = q_vals**2 + u_vals**2 + 1e-20
    sigma_l = np.sqrt((q_vals**2 * sigma_q**2 + u_vals**2 * sigma_u**2) / (lin_sq + 1e-20))
    pa_sigma_rad = 0.5 * np.sqrt((u_vals**2 * sigma_q**2 + q_vals**2 * sigma_u**2) / (lin_sq**2))
    pa_sigma_deg = np.degrees(pa_sigma_rad)

    # DM-optimisation style PA masking for frequency spectra:
    # 1) linear-polarisation significance,
    # 2) Stokes-I/SNR validity (reuse passed valid_mask when available),
    # 3) PA uncertainty cut (match time-series style large-error rejection)
    pa_mask = np.isfinite(pol_angle_deg) & np.isfinite(pa_sigma_deg)
    pa_mask &= np.isfinite(l_meas) & np.isfinite(sigma_l) & (sigma_l > 0)
    pa_mask &= l_meas >= (2.0 * sigma_l)
    if valid_mask_arr is not None and valid_mask_arr.shape == i_vals.shape:
        pa_mask &= valid_mask_arr
    else:
        pa_mask &= i_vals > 0
    pa_mask &= pa_sigma_deg <= 50.0

    if np.any(pa_mask):
        ax1.errorbar(fitter.lambda_sq[pa_mask], pol_angle_deg[pa_mask], yerr=pa_sigma_deg[pa_mask],
                     fmt='o', color='k', markersize=4, ecolor='gray',
                     elinewidth=0.9, capsize=2, alpha=0.8)
    
    # Overplot best-fit line using linear fit to the data
    rm_peak = rm_synthesis_result.get('rm_clean_peak', rm_synthesis_result.get('rm_peak'))
    lambda_sq_model = np.linspace(fitter.lambda_sq.min(), fitter.lambda_sq.max(), 100)
    
    # Fit a line to the unwrapped polarisation angles to get both slope and intercept
    # Model: pol_angle = pol_angle_0 + RM * lambda_sq
    if np.sum(pa_mask) >= 2:
        coeffs = np.polyfit(fitter.lambda_sq[pa_mask], np.unwrap(fitter.pol_angle)[pa_mask], 1)
        rm_fit = coeffs[0]  # Slope (RM from linear fit)
        pol_angle_0 = coeffs[1]  # Intercept
        # Use the fitted RM and intercept to plot the model line, then wrap for display
        pol_angle_model = np.degrees(pol_angle_0 + rm_fit * lambda_sq_model)
        pol_angle_model = ((pol_angle_model + 90.0) % 180.0) - 90.0
    else:
        rm_fit = np.nan
        pol_angle_model = None
    rm_err = rm_synthesis_result.get('rm_clean_err', rm_synthesis_result.get('noise_estimate', 0) * 2)
    if pol_angle_model is not None:
        if rm_err > 0:
            label_line = f'RM = {rm_peak:.2f} ± {rm_err:.2f} rad/m² (fit: {rm_fit:.2f})'
        else:
            label_line = f'RM = {rm_peak:.2f} rad/m² (fit: {rm_fit:.2f})'
        ax1.plot(lambda_sq_model, pol_angle_model, 'r-', linewidth=2,
                 label=label_line)
    else:
        if rm_err > 0:
            label_line = f'RM = {rm_peak:.2f} ± {rm_err:.2f} rad/m² (insufficient masked PA points for fit)'
        else:
            label_line = f'RM = {rm_peak:.2f} rad/m² (insufficient masked PA points for fit)'
        ax1.plot([], [], 'r-', linewidth=2, label=label_line)
    
    ax1.set_xlabel('λ² (m²)', fontsize=style['label'])
    ax1.set_ylabel('Polarisation Angle (deg.)', fontsize=style['label'])
    #ax1.set_title('Polarisation Angle vs λ²', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=style['legend'])
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=style['tick'])
    
    # Plot 2: RM Spectrum
    ax2 = axes[1]
    ax2.plot(rm_synthesis_result['rm_values'], 
             rm_synthesis_result['rm_amplitude'], 
             'k-', linewidth=1.5)
    # draw error region if available
    rm_err = rm_synthesis_result.get('rm_clean_err', rm_synthesis_result.get('noise_estimate', 0) * 2)
    if rm_err > 0:
        ax2.axvspan(rm_peak - rm_err, rm_peak + rm_err, color='red', alpha=0.1,
                    label=f'RM error ≈ ±{rm_err:.2f}')
    ax2.axvline(rm_peak, color='r', linestyle='--', linewidth=2,
                label=f'Peak RM = {rm_peak:.2f} rad/m²')
    ax2.axhline(rm_synthesis_result['noise_estimate'], color='gray', 
                linestyle=':', linewidth=1, label='Noise level')
    
    ax2.set_xlabel('RM (rad/m²)', fontsize=style['label'])
    ax2.set_ylabel('|F(RM)|', fontsize=style['label'])
    #ax2.set_title('RM Spectrum (Faraday Dispersion Function)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=style['legend'])
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=style['tick'])
    
    if show_frac_panel:
        # Plot 3: Polarisation fraction vs frequency
        ax3 = axes[2]
        freq_mhz = fitter.freq_hz / 1e6
        l_vals = np.sqrt(q_vals**2 + u_vals**2)
        l_over_i = l_vals / (i_vals + 1e-10)
        if pol_frac_err_arr is not None and pol_frac_err_arr.shape == l_over_i.shape:
            sigma_l_over_i = pol_frac_err_arr
        else:
            sigma_l = np.sqrt((q_vals**2 * sigma_q**2 + u_vals**2 * sigma_u**2) / (l_vals**2 + 1e-20))
            sigma_l_over_i = np.sqrt((sigma_l / (i_vals + 1e-10))**2 +
                                     ((l_vals * sigma_i) / ((i_vals + 1e-10)**2))**2)

        l_plot_mask = np.isfinite(freq_mhz) & np.isfinite(l_over_i) & np.isfinite(sigma_l_over_i)
        if valid_mask_arr is not None and valid_mask_arr.shape == l_over_i.shape:
            l_plot_mask &= valid_mask_arr
        order = np.argsort(freq_mhz)
        freq_sorted = freq_mhz[order]
        l_sorted = l_over_i[order]
        sigma_l_sorted = sigma_l_over_i[order]
        l_mask_sorted = l_plot_mask[order]
        l_plot = np.where(l_mask_sorted, l_sorted, np.nan)
        l_low = np.where(l_mask_sorted, l_sorted - sigma_l_sorted, np.nan)
        l_high = np.where(l_mask_sorted, l_sorted + sigma_l_sorted, np.nan)

        ax3.plot(freq_sorted, l_plot, 'r-', linewidth=2, label='L/I')
        ax3.fill_between(freq_sorted,
                         l_low,
                         l_high,
                         color='r', alpha=0.18, linewidth=0)

        # Add circular polarisation fraction if Stokes V is available
        if fitter.stokes_v is not None:
            v_vals = np.asarray(fitter.stokes_v, dtype=float)
            circ_pol_fraction = v_vals / (i_vals + 1e-10)
            if circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_pol_fraction.shape:
                sigma_v_over_i = circ_frac_err_arr
            else:
                sigma_v_over_i = np.sqrt((sigma_v / (i_vals + 1e-10))**2 +
                                         ((np.abs(v_vals) * sigma_i) / ((i_vals + 1e-10)**2))**2)

            v_plot_mask = np.isfinite(freq_mhz) & np.isfinite(circ_pol_fraction) & np.isfinite(sigma_v_over_i)
            if circ_valid_mask_arr is not None and circ_valid_mask_arr.shape == circ_pol_fraction.shape:
                v_plot_mask &= circ_valid_mask_arr
            elif valid_mask_arr is not None and valid_mask_arr.shape == circ_pol_fraction.shape:
                v_plot_mask &= valid_mask_arr

            v_sorted = circ_pol_fraction[order]
            sigma_v_sorted = sigma_v_over_i[order]
            v_mask_sorted = v_plot_mask[order]
            v_plot = np.where(v_mask_sorted, v_sorted, np.nan)
            v_low = np.where(v_mask_sorted, v_sorted - sigma_v_sorted, np.nan)
            v_high = np.where(v_mask_sorted, v_sorted + sigma_v_sorted, np.nan)

            ax3.plot(freq_sorted, v_plot, 'b-', linewidth=2, label='V/I')
            ax3.fill_between(freq_sorted,
                             v_low,
                             v_high,
                             color='b', alpha=0.14, linewidth=0)

        ax3.set_xlabel('Frequency (MHz)', fontsize=style['label'])
        ax3.set_ylabel('Polarisation Fraction', fontsize=style['label'])
        #ax3.set_title('Polarisation Fraction', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=style['legend'])
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=style['tick'])
    
    plt.tight_layout()
    
    savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_burns_law_fits(fitter: RMFitter,
                        output_file: str = 'burns_law_fit.png',
                        pol_frac_err: Optional[np.ndarray] = None,
                        valid_mask: Optional[np.ndarray] = None,
                        circ_frac_err: Optional[np.ndarray] = None,
                        circ_valid_mask: Optional[np.ndarray] = None,
                        turbulent_radius_pc: float = 21.0,
                        screen_scale_cm: float = 1e15):
    """
    Fit and plot Burn-law depolarisation models for the linear polarisation spectrum.

    Models (as requested):
    - P_Burn(λ)       = exp(-2 * sigma_RM^2 * λ^4)
    - P_mod-Burn(λ)   = P_i * exp(-2 * sigma_RM'^2 * λ^4)
    - P_const(λ)      = P_i
    where P is the linear polarisation fraction (L/I).
    """
    style = publication_plot_style()

    lambda_sq = np.asarray(fitter.lambda_sq, dtype=float)
    freq_hz = np.asarray(fitter.freq_hz, dtype=float)
    pol_frac = np.asarray(fitter.pol_fraction, dtype=float)
    circ_frac = None
    if fitter.stokes_v is not None:
        circ_frac = np.asarray(fitter.stokes_v / (fitter.stokes_i + 1e-10), dtype=float)
    pol_frac_err_arr = None if pol_frac_err is None else np.asarray(pol_frac_err, dtype=float)
    valid_mask_arr = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    circ_frac_err_arr = None if circ_frac_err is None else np.asarray(circ_frac_err, dtype=float)
    circ_valid_mask_arr = None if circ_valid_mask is None else np.asarray(circ_valid_mask, dtype=bool)

    valid = np.isfinite(lambda_sq) & np.isfinite(freq_hz) & (freq_hz > 0) & np.isfinite(pol_frac) & (pol_frac > 0)
    if valid_mask_arr is not None and valid_mask_arr.shape == pol_frac.shape:
        valid &= valid_mask_arr
    if pol_frac_err_arr is not None and pol_frac_err_arr.shape == pol_frac.shape:
        valid &= np.isfinite(pol_frac_err_arr) & (pol_frac_err_arr > 0)
    if np.sum(valid) < 5:
        print("Warning: insufficient valid points for Burn-law fitting; skipping plot.")
        return

    sigma_rm_thresh = None
    sigma_rm_thresh_snr = None
    pol_snr_eff = np.nan
    meas_nsigma = 3.0
    freq_center_mhz = np.nan
    try:
        freq_valid = freq_hz[valid]
        freq_center_hz = float(np.nanmedian(freq_valid))
        freq_center_mhz = freq_center_hz / 1e6
        sigma_rm_thresh = sigma_rm_detection_threshold(freq_center_hz)
    except Exception:
        sigma_rm_thresh = None

    x = lambda_sq[valid]
    freq_mhz = freq_hz[valid] / 1e6
    y = pol_frac[valid]
    yerr = pol_frac_err_arr[valid] if (pol_frac_err_arr is not None and pol_frac_err_arr.shape == pol_frac.shape) else None
    order = np.argsort(freq_mhz)
    x = x[order]
    freq_mhz = freq_mhz[order]
    y = y[order]
    if yerr is not None:
        yerr = yerr[order]

    # Effective linear-polarisation S/N used for detectability reporting.
    if yerr is not None:
        snr_arr = y / (yerr + 1e-20)
        snr_arr = snr_arr[np.isfinite(snr_arr) & (snr_arr > 0)]
        if snr_arr.size > 0:
            pol_snr_eff = float(np.nanmedian(snr_arr))
    else:
        # Fallback SNR estimate when explicit uncertainties are unavailable.
        dy = np.diff(y)
        sigma_y = np.nanstd(dy) / np.sqrt(2.0) if dy.size > 1 else np.nanstd(y)
        if np.isfinite(sigma_y) and sigma_y > 0:
            pol_snr_eff = float(np.nanmedian(y) / sigma_y)

    if sigma_rm_thresh is not None and np.isfinite(pol_snr_eff) and pol_snr_eff > 0:
        try:
            sigma_rm_thresh_snr = sigma_rm_detection_threshold_snr(
                freq_center_hz=freq_center_hz,
                pol_snr=pol_snr_eff,
                nsigma=meas_nsigma,
            )
        except Exception:
            sigma_rm_thresh_snr = None

    x_full = lambda_sq
    freq_mhz_full = freq_hz / 1e6

    def burn_model(l2, sigma_rm):
        return np.exp(-2.0 * (sigma_rm ** 2) * (l2 ** 2))

    def modified_burn_model(l2, p_i, sigma_rm_prime):
        return p_i * np.exp(-2.0 * (sigma_rm_prime ** 2) * (l2 ** 2))

    def constant_model(l2, p_i):
        return np.full_like(l2, p_i, dtype=float)

    # Circular-fraction models:
    # mC(lambda^2) = C0
    # mC(lambda^2) = C0 + C1 * lambda^2
    # mC(lambda^2) = C0 + A * sin(2 * (phi0 + beta * lambda^2))
    def circ_const_model(l2, c0):
        return np.full_like(l2, c0, dtype=float)

    def circ_linear_model(l2, c0, c1):
        return c0 + c1 * l2

    def circ_sine_model(l2, c0, amp, phi0, beta):
        return c0 + amp * np.sin(2.0 * (phi0 + beta * l2))

    def _log10_evidence_bic(y_obs: np.ndarray,
                            y_model: np.ndarray,
                            n_params: int,
                            sigma_obs: Optional[np.ndarray] = None) -> float:
        n = len(y_obs)
        if n <= max(1, n_params):
            return np.nan

        residual = y_obs - y_model
        if sigma_obs is None:
            sigma_level = np.nanstd(y_obs)
            if not np.isfinite(sigma_level) or sigma_level <= 0:
                sigma_level = 1e-10
            sigma = np.full_like(y_obs, sigma_level, dtype=float)
        else:
            sigma = np.array(sigma_obs, dtype=float)
            finite = np.isfinite(sigma) & (sigma > 0)
            if not np.any(finite):
                sigma = np.full_like(y_obs, 1e-10, dtype=float)
            else:
                fallback = np.nanmedian(sigma[finite])
                if not np.isfinite(fallback) or fallback <= 0:
                    fallback = 1e-10
                sigma[~finite] = fallback

        ln_like = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))
        bic = n_params * np.log(n) - 2.0 * ln_like
        ln_z = -0.5 * bic
        return ln_z / np.log(10.0)

    def _trotta_strength(delta_log10_z: float) -> str:
        if delta_log10_z < 0.5:
            return 'inconclusive'
        if delta_log10_z < 1.0:
            return 'substantial'
        if delta_log10_z < 2.0:
            return 'strong'
        return 'decisive'

    p0_guess = float(np.nanmax(y)) if np.nanmax(y) > 0 else 0.1
    p0_guess = min(max(p0_guess, 1e-6), 1.5)
    sigma_guess = 10.0

    burn_popt = None
    burn_perr = None
    mod_popt = None
    mod_perr = None
    const_popt = None
    const_perr = None
    circ_const_popt = None
    circ_const_perr = None
    circ_lin_popt = None
    circ_lin_perr = None
    circ_sin_popt = None
    circ_sin_perr = None

    try:
        burn_popt, burn_pcov = curve_fit(
            burn_model,
            x,
            y,
            p0=[sigma_guess],
            bounds=([0.0], [1e5]),
            maxfev=20000,
        )
        burn_perr = np.sqrt(np.diag(burn_pcov))
    except Exception:
        burn_popt = None

    try:
        mod_popt, mod_pcov = curve_fit(
            modified_burn_model,
            x,
            y,
            p0=[p0_guess, sigma_guess],
            bounds=([0.0, 0.0], [2.0, 1e5]),
            maxfev=30000,
        )
        mod_perr = np.sqrt(np.diag(mod_pcov))
    except Exception:
        mod_popt = None

    try:
        const_popt, const_pcov = curve_fit(
            constant_model,
            x,
            y,
            p0=[p0_guess],
            bounds=([0.0], [2.0]),
            maxfev=10000,
        )
        const_perr = np.sqrt(np.diag(const_pcov))
    except Exception:
        const_popt = None

    burn_y_fit = burn_model(x, *burn_popt) if burn_popt is not None else None
    mod_y_fit = modified_burn_model(x, *mod_popt) if mod_popt is not None else None
    const_y_fit = constant_model(x, *const_popt) if const_popt is not None else None

    # Circular-fraction fits (if Stokes V is available)
    x_c = None
    freq_c = None
    y_c = None
    yerr_c = None
    if circ_frac is not None:
        valid_c = np.isfinite(x_full) & np.isfinite(freq_mhz_full) & np.isfinite(circ_frac)
        if circ_valid_mask_arr is not None and circ_valid_mask_arr.shape == circ_frac.shape:
            valid_c &= circ_valid_mask_arr
        if circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_frac.shape:
            valid_c &= np.isfinite(circ_frac_err_arr) & (circ_frac_err_arr > 0)
        if np.sum(valid_c) >= 5:
            x_c = x_full[valid_c]
            freq_c = freq_mhz_full[valid_c]
            y_c = circ_frac[valid_c]
            yerr_c = circ_frac_err_arr[valid_c] if (circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_frac.shape) else None
            order_c = np.argsort(freq_c)
            x_c = x_c[order_c]
            freq_c = freq_c[order_c]
            y_c = y_c[order_c]
            if yerr_c is not None:
                yerr_c = yerr_c[order_c]

            c0_guess = float(np.nanmedian(y_c))
            amp_guess = max(0.01, 0.5 * float(np.nanmax(y_c) - np.nanmin(y_c)))
            beta_guess = 100.0

            try:
                circ_const_popt, circ_const_pcov = curve_fit(
                    circ_const_model,
                    x_c,
                    y_c,
                    p0=[c0_guess],
                    bounds=([-1.0], [1.0]),
                    maxfev=20000,
                )
                circ_const_perr = np.sqrt(np.diag(circ_const_pcov))
            except Exception:
                circ_const_popt = None

            try:
                circ_lin_popt, circ_lin_pcov = curve_fit(
                    circ_linear_model,
                    x_c,
                    y_c,
                    p0=[c0_guess, 0.0],
                    bounds=([-1.0, -1e5], [1.0, 1e5]),
                    maxfev=30000,
                )
                circ_lin_perr = np.sqrt(np.diag(circ_lin_pcov))
            except Exception:
                circ_lin_popt = None

            if np.sum(np.isfinite(y_c)) >= 8:
                try:
                    circ_sin_popt, circ_sin_pcov = curve_fit(
                        circ_sine_model,
                        x_c,
                        y_c,
                        p0=[c0_guess, amp_guess, 0.0, beta_guess],
                        bounds=([-1.0, 0.0, -np.pi, -1e5], [1.0, 1.0, np.pi, 1e5]),
                        maxfev=60000,
                    )
                    circ_sin_perr = np.sqrt(np.diag(circ_sin_pcov))
                except Exception:
                    circ_sin_popt = None

    circ_const_y_fit = circ_const_model(x_c, *circ_const_popt) if (x_c is not None and circ_const_popt is not None) else None
    circ_lin_y_fit = circ_linear_model(x_c, *circ_lin_popt) if (x_c is not None and circ_lin_popt is not None) else None
    circ_sin_y_fit = circ_sine_model(x_c, *circ_sin_popt) if (x_c is not None and circ_sin_popt is not None) else None

    # Print fit summaries to terminal and save to text file
    summary_lines: List[str] = []
    summary_lines.append("Depolarisation fit summary:")
    summary_lines.append(f"  Output plot: {output_file}")
    print("\nDepolarisation fit summary:")
    print(f"  Output plot: {output_file}")

    if burn_popt is not None:
        if burn_perr is not None and burn_perr.size == 1:
            line = f"  P_Burn: sigma_RM = {burn_popt[0]:.6f} ± {burn_perr[0]:.6f} rad/m^2"
            print(line)
            summary_lines.append(line)
        else:
            line = f"  P_Burn: sigma_RM = {burn_popt[0]:.6f} rad/m^2"
            print(line)
            summary_lines.append(line)
    else:
        line = "  P_Burn: fit failed"
        print(line)
        summary_lines.append(line)

    if burn_popt is not None:
        try:
            delta_burn = depolarising_medium_delta_ne_b_parallel(
                sigma_rm=float(burn_popt[0]),
                turbulent_radius_pc=turbulent_radius_pc,
                screen_scale_cm=screen_scale_cm,
            )
            if burn_perr is not None and burn_perr.size == 1 and np.isfinite(burn_perr[0]) and burn_popt[0] > 0:
                frac_err = float(burn_perr[0] / burn_popt[0])
                delta_burn_err = abs(delta_burn) * frac_err
                line = (f"    delta(n_e, B_parallel) = {delta_burn:.6e} ± {delta_burn_err:.6e} uG/cm^3 "
                        f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
            else:
                line = (f"    delta(n_e, B_parallel) = {delta_burn:.6e} uG/cm^3 "
                        f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
        except Exception as exc:
            line = f"    delta(n_e, B_parallel): not computed ({exc})"
        print(line)
        summary_lines.append(line)

    if sigma_rm_thresh_snr is not None and burn_popt is not None:
        if np.isfinite(sigma_rm_thresh_snr):
            burn_measurable = bool(burn_popt[0] >= sigma_rm_thresh_snr)
            line = (f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                    f"SNR_eff={pol_snr_eff:.2f}, nsigma={meas_nsigma:.1f}, "
                    f"threshold={sigma_rm_thresh_snr:.6f} rad/m^2, "
                    f"fitted={burn_popt[0]:.6f} -> {'measurable' if burn_measurable else 'not measurable'}")
        else:
            line = (f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                    f"SNR_eff={pol_snr_eff:.2f} is too low for a {meas_nsigma:.1f}σ depolarisation detection")
        print(line)
        summary_lines.append(line)
    elif sigma_rm_thresh is not None and burn_popt is not None:
        burn_measurable = bool(burn_popt[0] >= sigma_rm_thresh)
        line = (f"    measurability (fallback e-fold @ {freq_center_mhz:.2f} MHz): "
                f"threshold={sigma_rm_thresh:.6f} rad/m^2, "
                f"fitted={burn_popt[0]:.6f} -> {'measurable' if burn_measurable else 'not measurable'}")
        print(line)
        summary_lines.append(line)

    if mod_popt is not None:
        if mod_perr is not None and mod_perr.size == 2:
            line = (f"  P_mod-Burn: P_i = {mod_popt[0]:.6f} ± {mod_perr[0]:.6f}, "
                    f"sigma_RM' = {mod_popt[1]:.6f} ± {mod_perr[1]:.6f} rad/m^2")
            print(line)
            summary_lines.append(line)
        else:
            line = f"  P_mod-Burn: P_i = {mod_popt[0]:.6f}, sigma_RM' = {mod_popt[1]:.6f} rad/m^2"
            print(line)
            summary_lines.append(line)
    else:
        line = "  P_mod-Burn: fit failed"
        print(line)
        summary_lines.append(line)

    if mod_popt is not None:
        try:
            delta_mod = depolarising_medium_delta_ne_b_parallel(
                sigma_rm=float(mod_popt[1]),
                turbulent_radius_pc=turbulent_radius_pc,
                screen_scale_cm=screen_scale_cm,
            )
            if mod_perr is not None and mod_perr.size == 2 and np.isfinite(mod_perr[1]) and mod_popt[1] > 0:
                frac_err = float(mod_perr[1] / mod_popt[1])
                delta_mod_err = abs(delta_mod) * frac_err
                line = (f"    delta(n_e, B_parallel) = {delta_mod:.6e} ± {delta_mod_err:.6e} uG/cm^3 "
                        f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
            else:
                line = (f"    delta(n_e, B_parallel) = {delta_mod:.6e} uG/cm^3 "
                        f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
        except Exception as exc:
            line = f"    delta(n_e, B_parallel): not computed ({exc})"
        print(line)
        summary_lines.append(line)

    if sigma_rm_thresh_snr is not None and mod_popt is not None:
        if np.isfinite(sigma_rm_thresh_snr):
            mod_measurable = bool(mod_popt[1] >= sigma_rm_thresh_snr)
            line = (f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                    f"SNR_eff={pol_snr_eff:.2f}, nsigma={meas_nsigma:.1f}, "
                    f"threshold={sigma_rm_thresh_snr:.6f} rad/m^2, "
                    f"fitted={mod_popt[1]:.6f} -> {'measurable' if mod_measurable else 'not measurable'}")
        else:
            line = (f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                    f"SNR_eff={pol_snr_eff:.2f} is too low for a {meas_nsigma:.1f}σ depolarisation detection")
        print(line)
        summary_lines.append(line)
    elif sigma_rm_thresh is not None and mod_popt is not None:
        mod_measurable = bool(mod_popt[1] >= sigma_rm_thresh)
        line = (f"    measurability (fallback e-fold @ {freq_center_mhz:.2f} MHz): "
                f"threshold={sigma_rm_thresh:.6f} rad/m^2, "
                f"fitted={mod_popt[1]:.6f} -> {'measurable' if mod_measurable else 'not measurable'}")
        print(line)
        summary_lines.append(line)

    if const_popt is not None:
        if const_perr is not None and const_perr.size == 1:
            line = f"  P_const: P_i = {const_popt[0]:.6f} ± {const_perr[0]:.6f}"
            print(line)
            summary_lines.append(line)
        else:
            line = f"  P_const: P_i = {const_popt[0]:.6f}"
            print(line)
            summary_lines.append(line)
    else:
        line = "  P_const: fit failed"
        print(line)
        summary_lines.append(line)

    if y_c is None:
        line = "  Circular fraction models: skipped (no valid V/I data)"
        print(line)
        summary_lines.append(line)
    else:
        if circ_const_popt is not None:
            if circ_const_perr is not None and circ_const_perr.size == 1:
                line = f"  mC const: C0 = {circ_const_popt[0]:.6f} ± {circ_const_perr[0]:.6f}"
                print(line)
                summary_lines.append(line)
            else:
                line = f"  mC const: C0 = {circ_const_popt[0]:.6f}"
                print(line)
                summary_lines.append(line)
        else:
            line = "  mC const: fit failed"
            print(line)
            summary_lines.append(line)

        if circ_lin_popt is not None:
            if circ_lin_perr is not None and circ_lin_perr.size == 2:
                line = (f"  mC linear: C0 = {circ_lin_popt[0]:.6f} ± {circ_lin_perr[0]:.6f}, "
                        f"C1 = {circ_lin_popt[1]:.6f} ± {circ_lin_perr[1]:.6f}")
                print(line)
                summary_lines.append(line)
            else:
                line = f"  mC linear: C0 = {circ_lin_popt[0]:.6f}, C1 = {circ_lin_popt[1]:.6f}"
                print(line)
                summary_lines.append(line)
        else:
            line = "  mC linear: fit failed"
            print(line)
            summary_lines.append(line)

        if circ_sin_popt is not None:
            if circ_sin_perr is not None and circ_sin_perr.size == 4:
                line = (f"  mC sinusoid: C0 = {circ_sin_popt[0]:.6f} ± {circ_sin_perr[0]:.6f}, "
                        f"A = {circ_sin_popt[1]:.6f} ± {circ_sin_perr[1]:.6f}, "
                        f"phi0 = {circ_sin_popt[2]:.6f} ± {circ_sin_perr[2]:.6f}, "
                        f"beta = {circ_sin_popt[3]:.6f} ± {circ_sin_perr[3]:.6f}")
                print(line)
                summary_lines.append(line)
            else:
                line = (f"  mC sinusoid: C0 = {circ_sin_popt[0]:.6f}, A = {circ_sin_popt[1]:.6f}, "
                        f"phi0 = {circ_sin_popt[2]:.6f}, beta = {circ_sin_popt[3]:.6f}")
                print(line)
                summary_lines.append(line)
        else:
            line = "  mC sinusoid: fit failed"
            print(line)
            summary_lines.append(line)

    # Approximate model comparison via log10 Bayes evidence using BIC proxy
    # (Trotta 2008 style interpretation on Delta log10 evidence).
    linear_log10z = {}
    if burn_y_fit is not None:
        linear_log10z['P_Burn'] = _log10_evidence_bic(y, burn_y_fit, 1, yerr)
    if mod_y_fit is not None:
        linear_log10z['P_mod-Burn'] = _log10_evidence_bic(y, mod_y_fit, 2, yerr)
    if const_y_fit is not None:
        linear_log10z['P_const'] = _log10_evidence_bic(y, const_y_fit, 1, yerr)

    if len(linear_log10z) > 0:
        line = "  Linear models log10 evidence (BIC approximation):"
        print(line)
        summary_lines.append(line)
        for name, val in linear_log10z.items():
            line = f"    {name}: log10(Z) ≈ {val:.6f}"
            print(line)
            summary_lines.append(line)
        if len(linear_log10z) >= 2:
            ranking = sorted(linear_log10z.items(), key=lambda item: item[1], reverse=True)
            best_name, best_val = ranking[0]
            second_name, second_val = ranking[1]
            delta = best_val - second_val
            strength = _trotta_strength(delta)
            line = f"  Preferred linear model: {best_name} over {second_name} (Δlog10Z={delta:.3f}, {strength})"
            print(line)
            summary_lines.append(line)

    best_linear_model = None
    if len(linear_log10z) > 0:
        best_linear_model = max(linear_log10z, key=linear_log10z.get)

    circular_log10z = {}
    if y_c is not None:
        if circ_const_y_fit is not None:
            circular_log10z['mC_const'] = _log10_evidence_bic(y_c, circ_const_y_fit, 1, yerr_c)
        if circ_lin_y_fit is not None:
            circular_log10z['mC_linear'] = _log10_evidence_bic(y_c, circ_lin_y_fit, 2, yerr_c)
        if circ_sin_y_fit is not None:
            circular_log10z['mC_sinusoid'] = _log10_evidence_bic(y_c, circ_sin_y_fit, 4, yerr_c)

    if len(circular_log10z) > 0:
        line = "  Circular models log10 evidence (BIC approximation):"
        print(line)
        summary_lines.append(line)
        for name, val in circular_log10z.items():
            line = f"    {name}: log10(Z) ≈ {val:.6f}"
            print(line)
            summary_lines.append(line)
        if len(circular_log10z) >= 2:
            ranking_c = sorted(circular_log10z.items(), key=lambda item: item[1], reverse=True)
            best_name_c, best_val_c = ranking_c[0]
            second_name_c, second_val_c = ranking_c[1]
            delta_c = best_val_c - second_val_c
            strength_c = _trotta_strength(delta_c)
            line = f"  Preferred circular model: {best_name_c} over {second_name_c} (Δlog10Z={delta_c:.3f}, {strength_c})"
            print(line)
            summary_lines.append(line)

    best_circular_model = None
    if len(circular_log10z) > 0:
        best_circular_model = max(circular_log10z, key=circular_log10z.get)

    summary_txt = os.path.splitext(output_file)[0] + "_fit_summary.txt"
    with open(summary_txt, 'w', encoding='utf-8') as summary_file:
        summary_file.write("\n".join(summary_lines) + "\n")
    print(f"  Fit summary saved to {summary_txt}")

    freq_model_mhz = np.linspace(np.nanmin(freq_mhz), np.nanmax(freq_mhz), 500)
    freq_model_hz = freq_model_mhz * 1e6
    x_model = (c / freq_model_hz) ** 2

    fig, ax = plt.subplots(1, 1, figsize=_pub_figsize(height_ratio=0.62, min_height=4.2))
    if yerr is not None:
        ax.errorbar(freq_mhz, y, yerr=yerr, fmt='o', markersize=4,
                    color='tab:red', marker='s', ecolor='gray', elinewidth=1, capsize=2,
                    alpha=0.8, label=r'$L/I$')
    else:
        ax.scatter(freq_mhz, y, s=28, c='tab:red', marker='s', alpha=0.8, label=r'$L/I$')

    if burn_popt is not None and best_linear_model == 'P_Burn':
        y_burn = burn_model(x_model, *burn_popt)
        burn_label = r"$P_{\mathrm{Burn}}(\lambda)=\exp\left(-2\sigma_{\mathrm{RM}}^2\lambda^4\right)$"
        ax.plot(freq_model_mhz, y_burn, color='tab:purple', linewidth=2, label=burn_label)

    if mod_popt is not None and best_linear_model == 'P_mod-Burn':
        y_mod = modified_burn_model(x_model, *mod_popt)
        mod_label = r"$P_{\mathrm{mod-Burn}}(\lambda)=P_i\exp\left(-2\sigma_{\mathrm{RM}}'^{\,2}\lambda^4\right)$"
        ax.plot(freq_model_mhz, y_mod, color='tab:cyan', linewidth=2, linestyle='--', label=mod_label)

    if const_popt is not None and best_linear_model == 'P_const':
        y_const = constant_model(x_model, *const_popt)
        label_const = r"$P_{\mathrm{const}}(\lambda)=P_i$"
        ax.plot(freq_model_mhz, y_const, color='0.25', linewidth=2, linestyle=':', label=label_const)

    # Plot circular-fraction data and model fits on same figure
    if y_c is not None and freq_c is not None:
        if yerr_c is not None:
            ax.errorbar(freq_c, y_c, yerr=yerr_c, fmt='s', markersize=3,
                        color='tab:blue', ecolor='tab:blue', elinewidth=0.8,
                        capsize=2, alpha=0.8, label=r'$V/I$')
        else:
            ax.scatter(freq_c, y_c, s=16, c='tab:blue', marker='s', alpha=0.8, label=r'$V/I$')

        #if circ_const_popt is not None:
        #    y_cc = circ_const_model(x_model, *circ_const_popt)
        #    c0 = circ_const_popt[0]
        #    if circ_const_perr is not None and circ_const_perr.size == 1:
        #        lbl = rf"$m_C$: $C_0={c0:.3f}\pm{circ_const_perr[0]:.3f}$"
        #    else:
        #        lbl = rf"$m_C$: $C_0={c0:.3f}$"
        #    ax.plot(freq_model_mhz, y_cc, color='tab:blue', linewidth=1.8, linestyle='-', label=lbl)

        if circ_lin_popt is not None and best_circular_model == 'mC_linear':
            y_cl = circ_linear_model(x_model, *circ_lin_popt)
            lbl = r"$m_C(\lambda^2)=C_0 + C_1\lambda^2$"
            ax.plot(freq_model_mhz, y_cl, color='tab:green', linewidth=1.8, linestyle='--', label=lbl)

        if circ_sin_popt is not None and best_circular_model == 'mC_sinusoid':
            y_cs = circ_sine_model(x_model, *circ_sin_popt)
            lbl = r"$m_C(\lambda^2)=C_0 + A\sin\left(2\left(\phi_0 + \beta\lambda^2\right)\right)$"
            ax.plot(freq_model_mhz, y_cs, color='tab:olive', linewidth=1.8, linestyle='-.', label=lbl)

        if circ_const_popt is not None and best_circular_model == 'mC_const':
            y_cc = circ_const_model(x_model, *circ_const_popt)
            lbl = r"$m_C(\lambda^2)=C_0$"
            ax.plot(freq_model_mhz, y_cc, color='tab:green', linewidth=1.8, linestyle='-', label=lbl)

    ax.set_xlabel('Frequency (MHz)', fontsize=style['label'])
    ax.set_ylabel('Polarisation Fraction', fontsize=style['label'])
    #ax.set_title('Depolarisation Fits: Burn, Modified Burn, and Constant', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=style['legend'], loc='best')
    ax.tick_params(axis='both', labelsize=style['tick'])

    plt.tight_layout()
    savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Burn-law fit plot saved to {output_file}")
    plt.close()
    


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='RM Fitting for Stokes IQUV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit single spectrum from text files
  python rm_fitting.py -i stokes_i.txt -q stokes_q.txt -u stokes_u.txt --freq freq.txt
  
  # Fit from .npy files with time averaging
  python rm_fitting.py -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-avg
  
  # Fit with custom RM range
  python rm_fitting.py -i stokes_i.txt -q stokes_q.txt -u stokes_u.txt --rm-range -500 500
  
  # Process 2D data as time series
  python rm_fitting.py -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series

  # Manually click to select on-pulse window
  python rm_fitting.py -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series \
        --onpulse-only --manual-peaks

  # Provide peak start/end indices directly
  python rm_fitting.py -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series \
        --onpulse-only --peak-indices 100 120 200 220
        """
    )
    
    parser.add_argument('-i', '--stokes-i', required=False,
                       help='Stokes I file path')
    parser.add_argument('-q', '--stokes-q', required=False,
                       help='Stokes Q file path')
    parser.add_argument('-u', '--stokes-u', required=False,
                       help='Stokes U file path')
    parser.add_argument('-v', '--stokes-v', default=None,
                       help='Stokes V file path (optional)')
    parser.add_argument('--stokes-cube', default=None,
                       help='Path to Stokes cube with components ordered I,Q,U,(V)')
    parser.add_argument('--stokes-axis', type=int, default=0,
                       help='Axis index of Stokes dimension in --stokes-cube (default: 0)')
    parser.add_argument('--freq', default=None,
                       help='Frequency file (.npy or .txt)')
    parser.add_argument('--freq-unit', default='MHz', choices=['Hz', 'MHz', 'GHz'],
                       help='Frequency unit (default: MHz)')
    parser.add_argument('--time', default=None,
                       help='Time file (.npy or .txt)')
    parser.add_argument('--time-unit', default='ms', choices=['s', 'ms', 'us'],
                       help='Time unit (default: ms)')
    parser.add_argument('--method', choices=['simple', 'rm_synthesis', 'qu_fitting', 'rmnest'], default='rm_synthesis',
                       help='RM fitting method (default: rm_synthesis)')
    parser.add_argument('--rm-range', nargs=2, type=float, default=[-1000, 1000], metavar=('MIN', 'MAX'),
                       help='RM search range in rad/m² (default: -1000 1000)')
    parser.add_argument('--n-rm', type=int, default=2000,
                       help='Number of RM trial values (default: 2000)')
    parser.add_argument('--time-series', action='store_true',
                       help='Process as time series data')
    parser.add_argument('--time-avg', action='store_true',
                       help='Average over time axis for 2D data')
    parser.add_argument('--time-axis', type=int, default=1,
                       help='Time axis for 2D arrays (default: 1)')
    parser.add_argument('--freq-axis', type=int, default=0,
                       help='Frequency axis for 2D arrays (default: 0)')
    parser.add_argument('--onpulse-only', action='store_true',
                       help='Fit RM only within on-pulse window')
    parser.add_argument('--onpulse-fraction', type=float, default=0.95,
                       help='Fraction of flux to include in on-pulse window (default: 0.95)')
    parser.add_argument('--manual-peaks', action='store_true',
                       help='Manually select peak bounds by clicking on the pulse profile')
    parser.add_argument('--peak-indices', nargs='*', type=int, default=None,
                       help='Manually specify peak indices as pairs: start1 end1 start2 end2 ...')
    parser.add_argument('-o', '--output', default='rm_fitting_results',
                       help='Output file prefix (default: rm_fitting_results)')
    parser.add_argument('--ext', default='png',
                       help='Output figure extension (default: png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--hide-rm-frac-panel', action='store_true',
                       help='Hide the 3rd panel (L/I and V/I) in plot_rm_results')
    parser.add_argument('--poincare', action='store_true',
                       help='Generate Poincaré sphere plot (time‑dependent only; requires --time-series)')
    parser.add_argument('--poincare-interactive', action='store_true',
                       help='Display Poincaré plot interactively before saving')
    parser.add_argument('--poincare-surface', action='store_true',
                       help='Force all Poincaré points onto unit sphere surface')
    parser.add_argument('--poincare-projections', nargs='?', const='all', default=None,
                       choices=['all', 'gnom', 'stere', 'aeqd', 'ortho'],
                       help='Generate Poincare projections. Use "all" (default when flag is present) '
                           'for a 2x2 panel, or a single projection type: gnom, stere, aeqd, ortho. '
                           'Requires --time-series and --poincare.')
    parser.add_argument('--poincare-proj-center', type=float, nargs=3,
                       metavar=('CX', 'CY', 'CZ'), default=None,
                       help='Projection centre as a Stokes (Q,U,V) unit vector. '
                            'Defaults to the mean polarisation vector of the data.')
    parser.add_argument('--poincare-circle-fit', nargs='?', const='auto', default=None,
                       choices=['auto', 'great', 'small'],
                       help='Fit circles to Poincare segments: auto (default), great, or small.')
    parser.add_argument('--poincare-circle-segments', nargs='*', type=int, default=None,
                       help='Point-index segment pairs for circle fitting: s1 e1 s2 e2 ... '
                           '(indices refer to plotted Poincare points after masking/binning).')
    parser.add_argument('--separate-peaks', action='store_true',
                       help='Create separate side-by-side plots for each detected peak region')
    parser.add_argument('--min-gap-bins', type=int, default=3,
                       help='Minimum number of low-signal bins to separate peaks (default: 3)')
    parser.add_argument('--min-peak-bins', type=int, default=10,
                       help='Minimum number of consecutive significant bins required for a valid peak (default: 10)')
    parser.add_argument('--max-merge-gap', type=int, default=0,
                       help='Maximum gap size to merge nearby peaks. Peaks separated by fewer bins will be merged (default: 0, no merging)')
    parser.add_argument('--rmnest-gfr', action='store_true',
                       help='Use RMNest generalised Faraday rotation model')
    parser.add_argument('--rmnest-free-alpha', action='store_true',
                       help='Allow alpha to vary for RMNest GFR model')
    parser.add_argument('--rmnest-outdir', default=None,
                       help='Output directory for RMNest results (default: <output>_rmnest)')
    parser.add_argument('--rmnest-label', default=None,
                       help='Label for RMNest run (default: <output>)')
    parser.add_argument('--rmnest-sampler', default='dynesty',
                       help='Sampler for RMNest/Bilby (default: dynesty)')
    parser.add_argument('--time-bins', type=int, default=None,
                       help='Number of time bins to fit in time-series mode (default: no binning)')
    parser.add_argument('--freq-bins', type=int, default=None,
                       help='Number of frequency bins after --time-avg (default: no binning)')
    parser.add_argument('--noise-fraction', type=float, default=0.1,
                       help='Fraction of Stokes I samples used for noise estimation (default: 0.10)')
    parser.add_argument('--turbulent-radius-pc', type=float, default=21.0,
                       help='Radius R of turbulent environment in pc for delta(n_e, B_parallel) (default: 21.0)')
    parser.add_argument('--screen-scale-cm', type=float, default=1e15,
                       help='Plasma-screen scale l_screen in cm for delta(n_e, B_parallel) (default: 1e15)')
    
    args = parser.parse_args()

    # Input validation for Stokes data sources.
    using_cube = args.stokes_cube is not None
    using_separate = any(v is not None for v in (args.stokes_i, args.stokes_q, args.stokes_u, args.stokes_v))
    if using_cube and using_separate:
        parser.error("Use either --stokes-cube or separate --stokes-i/--stokes-q/--stokes-u inputs, not both")
    if (not using_cube) and (args.stokes_i is None or args.stokes_q is None or args.stokes_u is None):
        parser.error("Provide --stokes-cube or all of --stokes-i, --stokes-q, and --stokes-u")

    circle_segments: Optional[List[Tuple[int, int]]] = None
    if args.poincare_circle_segments is not None:
        if len(args.poincare_circle_segments) == 0:
            # Flag present without explicit pairs => auto-segment from masks.
            circle_segments = []
        elif len(args.poincare_circle_segments) % 2 != 0:
            parser.error("--poincare-circle-segments must contain an even number of integers")
        else:
            circle_segments = list(zip(
                args.poincare_circle_segments[0::2],
                args.poincare_circle_segments[1::2]
            ))
    
    print("="*60)
    print("RM FITTING FOR STOKES IQUV DATA")
    print("Using RM-Tools library for RM synthesis")
    print("="*60)
    
    # Load data
    print("\nLoading Stokes data...")
    freq_hz, stokes_i, stokes_q, stokes_u, stokes_v, time_array = load_stokes_data(
        i_file=args.stokes_i,
        q_file=args.stokes_q,
        u_file=args.stokes_u,
        v_file=args.stokes_v,
        cube_file=args.stokes_cube,
        stokes_axis=args.stokes_axis,
        freq_file=args.freq,
        time_file=args.time, time_axis=args.time_axis, freq_axis=args.freq_axis,
        freq_unit=args.freq_unit, time_unit=args.time_unit
    )
    burn_pol_frac_err = None
    burn_valid_mask = None
    burn_circ_frac_err = None
    burn_circ_valid_mask = None
    time_avg_extra_regions: List[Tuple[int, int]] = []
    sigma_i_chan_base = None
    sigma_q_chan_base = None
    sigma_u_chan_base = None
    sigma_v_chan_base = None
    freq_hz_unbinned = None
    stokes_v_full_noise = None
    
    # Handle 2D data
    if stokes_i.ndim == 2:
        print(f"\n  Detected 2D data with shape: {stokes_i.shape}")
        print(f"  Time axis: {args.time_axis}, Frequency axis: {args.freq_axis}")

        # Preserve full-resolution arrays for off-pulse noise estimation used
        # in time-averaged depolarisation error bars.
        stokes_i_full_noise = stokes_i
        stokes_q_full_noise = stokes_q
        stokes_u_full_noise = stokes_u
        stokes_v_full_noise = stokes_v
        if args.time_axis == 0:
            n_time_noise = stokes_i_full_noise.shape[0]
            n_frac_noise = max(1, int(n_time_noise * args.noise_fraction))
            i_off = stokes_i_full_noise[:n_frac_noise, :]
            q_off = stokes_q_full_noise[:n_frac_noise, :]
            u_off = stokes_u_full_noise[:n_frac_noise, :]
            sigma_i_chan = np.nanstd(i_off, axis=0)
            sigma_q_chan = np.nanstd(q_off, axis=0)
            sigma_u_chan = np.nanstd(u_off, axis=0)
            sigma_v_chan = np.nanstd(stokes_v_full_noise[:n_frac_noise, :], axis=0) if stokes_v_full_noise is not None else None
        else:
            n_time_noise = stokes_i_full_noise.shape[1]
            n_frac_noise = max(1, int(n_time_noise * args.noise_fraction))
            i_off = stokes_i_full_noise[:, :n_frac_noise]
            q_off = stokes_q_full_noise[:, :n_frac_noise]
            u_off = stokes_u_full_noise[:, :n_frac_noise]
            sigma_i_chan = np.nanstd(i_off, axis=1)
            sigma_q_chan = np.nanstd(q_off, axis=1)
            sigma_u_chan = np.nanstd(u_off, axis=1)
            sigma_v_chan = np.nanstd(stokes_v_full_noise[:, :n_frac_noise], axis=1) if stokes_v_full_noise is not None else None

        sigma_i_chan = np.where(np.isfinite(sigma_i_chan) & (sigma_i_chan > 0), sigma_i_chan, 1e-10)
        sigma_q_chan = np.where(np.isfinite(sigma_q_chan) & (sigma_q_chan > 0), sigma_q_chan, 1e-10)
        sigma_u_chan = np.where(np.isfinite(sigma_u_chan) & (sigma_u_chan > 0), sigma_u_chan, 1e-10)
        if sigma_v_chan is not None:
            sigma_v_chan = np.where(np.isfinite(sigma_v_chan) & (sigma_v_chan > 0), sigma_v_chan, 1e-10)
        sigma_i_chan_base = sigma_i_chan.copy()
        sigma_q_chan_base = sigma_q_chan.copy()
        sigma_u_chan_base = sigma_u_chan.copy()
        sigma_v_chan_base = sigma_v_chan.copy() if sigma_v_chan is not None else None
        
        # Verify that frequency array length matches the frequency axis dimension
        n_freq_data = stokes_i.shape[args.freq_axis]
        if len(freq_hz) != n_freq_data:
            print(f"\n  WARNING: Frequency array length ({len(freq_hz)}) does not match frequency axis dimension ({n_freq_data})")
            print(f"  Attempting to auto-correct: swapping time and frequency arrays...")
            freq_hz, time_array = time_array, freq_hz
            print(f"  New frequency array length: {len(freq_hz)}")
            if time_array is not None:
                print(f"  New time array length: {len(time_array)}")
        freq_hz_unbinned = np.asarray(freq_hz, dtype=float).copy()
        
        # Detect on-pulse window BEFORE averaging or time series processing
        onpulse_mask = None
        onpulse_regions = None

        if args.manual_peaks or (args.peak_indices is not None and len(args.peak_indices) > 0):
            # interactive selection or explicit indices
            if args.manual_peaks:
                print("\nInteractive peak selection requested...")
                if time_array is None:
                    print("  ERROR: manual peak selection requires a time array (--time)")
                else:
                    peaks = select_peaks_manual(time_array, stokes_i)
                    print(f"  Manual peaks: {peaks}")
                    onpulse_regions = peaks
                    start_idx = min(p[0] for p in peaks)
                    end_idx = max(p[1] for p in peaks)
                    print(f"  Using on-pulse window covering manual peaks: {start_idx} to {end_idx}")
                    onpulse_mask = (start_idx, end_idx)
            elif args.peak_indices is not None:
                # convert flat list to pairs
                pairs = list(zip(args.peak_indices[0::2], args.peak_indices[1::2]))
                if len(pairs) == 0:
                    print("  Warning: --peak-indices provided but no valid pairs found")
                else:
                    print(f"\nUser-specified peak index pairs: {pairs}")
                    onpulse_regions = pairs
                    start_idx = min(p[0] for p in pairs)
                    end_idx = max(p[1] for p in pairs)
                    print(f"  Using on-pulse window covering provided indices: {start_idx} to {end_idx}")
                    onpulse_mask = (start_idx, end_idx)
            else:
                print(f"\nDetecting on-pulse window ({args.onpulse_fraction*100:.1f}% flux)...")

                # Sum over frequency axis to get time profile
                time_profile = np.sum(stokes_i, axis=args.freq_axis)

                # Find on-pulse window
                start_idx, end_idx = find_onpulse_window(time_profile, args.onpulse_fraction)

                if time_array is not None:
                    time_start = time_array[start_idx] * 1e3  # convert to ms
                    time_end = time_array[end_idx] * 1e3
                    print(f"  On-pulse window: time bins {start_idx} to {end_idx}")
                    print(f"  Time range: {time_start:.3f} to {time_end:.3f} ms")
                    print(f"  Window width: {end_idx - start_idx + 1} bins")
                else:
                    print(f"  On-pulse window: bins {start_idx} to {end_idx}")

                onpulse_mask = (start_idx, end_idx)
        
        if args.time_avg:
            print("  Averaging over time axis...")
            n_time_avg_used = n_time_noise
            
            if onpulse_regions is not None and len(onpulse_regions) > 0:
                # In time-avg mode with multiple selected peaks, run each peak
                # separately. The first peak is processed in the main path and
                # remaining peaks are queued for additional runs.
                first_start, first_end = onpulse_regions[0]
                time_avg_extra_regions = onpulse_regions[1:]
                n_time_avg_used = max(1, first_end - first_start + 1)
                print(f"  Using peak 1 on-pulse region: bins {first_start} to {first_end}")
                if len(time_avg_extra_regions) > 0:
                    print(f"  Additional selected peaks to process separately: {len(time_avg_extra_regions)}")

                if args.time_axis == 0:
                    stokes_i = np.mean(stokes_i[first_start:first_end+1, :], axis=0)
                    stokes_q = np.mean(stokes_q[first_start:first_end+1, :], axis=0)
                    stokes_u = np.mean(stokes_u[first_start:first_end+1, :], axis=0)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[first_start:first_end+1, :], axis=0)
                else:
                    stokes_i = np.mean(stokes_i[:, first_start:first_end+1], axis=1)
                    stokes_q = np.mean(stokes_q[:, first_start:first_end+1], axis=1)
                    stokes_u = np.mean(stokes_u[:, first_start:first_end+1], axis=1)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[:, first_start:first_end+1], axis=1)
            elif onpulse_mask is not None:
                start_idx, end_idx = onpulse_mask
                n_time_avg_used = max(1, end_idx - start_idx + 1)
                print(f"  Using only on-pulse region (bins {start_idx} to {end_idx})...")
                
                if args.time_axis == 0:
                    stokes_i = np.mean(stokes_i[start_idx:end_idx+1, :], axis=args.time_axis)
                    stokes_q = np.mean(stokes_q[start_idx:end_idx+1, :], axis=args.time_axis)
                    stokes_u = np.mean(stokes_u[start_idx:end_idx+1, :], axis=args.time_axis)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[start_idx:end_idx+1, :], axis=args.time_axis)
                else:
                    stokes_i = np.mean(stokes_i[:, start_idx:end_idx+1], axis=args.time_axis)
                    stokes_q = np.mean(stokes_q[:, start_idx:end_idx+1], axis=args.time_axis)
                    stokes_u = np.mean(stokes_u[:, start_idx:end_idx+1], axis=args.time_axis)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[:, start_idx:end_idx+1], axis=args.time_axis)
            else:
                stokes_i = np.mean(stokes_i, axis=args.time_axis)
                stokes_q = np.mean(stokes_q, axis=args.time_axis)
                stokes_u = np.mean(stokes_u, axis=args.time_axis)
                if stokes_v is not None:
                    stokes_v = np.mean(stokes_v, axis=args.time_axis)
            
            print(f"  Averaged data shape: {stokes_i.shape}")

            # Noise arrays above are per single-time-sample channel RMS from the
            # off-pulse region; convert to uncertainty on the time-averaged mean.
            noise_scale_timeavg = np.sqrt(max(1, n_time_avg_used))
            sigma_i_chan = sigma_i_chan / noise_scale_timeavg
            sigma_q_chan = sigma_q_chan / noise_scale_timeavg
            sigma_u_chan = sigma_u_chan / noise_scale_timeavg
            if sigma_v_chan is not None:
                sigma_v_chan = sigma_v_chan / noise_scale_timeavg

            # Optional frequency binning after time averaging
            n_freq = len(freq_hz)
            if args.freq_bins is None or args.freq_bins <= 0 or args.freq_bins >= n_freq:
                if args.freq_bins is not None and args.freq_bins >= n_freq:
                    print(f"  Requested --freq-bins={args.freq_bins} >= number of channels ({n_freq}); keeping full resolution.")
            else:
                n_freq_bins_actual = min(args.freq_bins, n_freq)
                freq_bin_size = int(np.ceil(n_freq / n_freq_bins_actual))
                n_freq_bins_actual = (n_freq + freq_bin_size - 1) // freq_bin_size

                freq_hz_binned = np.zeros(n_freq_bins_actual)
                stokes_i_binned = np.zeros(n_freq_bins_actual)
                stokes_q_binned = np.zeros(n_freq_bins_actual)
                stokes_u_binned = np.zeros(n_freq_bins_actual)
                stokes_v_binned = np.zeros(n_freq_bins_actual) if stokes_v is not None else None
                sigma_i_binned = np.zeros(n_freq_bins_actual)
                sigma_q_binned = np.zeros(n_freq_bins_actual)
                sigma_u_binned = np.zeros(n_freq_bins_actual)
                sigma_v_binned = np.zeros(n_freq_bins_actual) if sigma_v_chan is not None else None

                for i_bin in range(n_freq_bins_actual):
                    bin_start = i_bin * freq_bin_size
                    bin_end = min((i_bin + 1) * freq_bin_size, n_freq)
                    if bin_end <= bin_start:
                        continue

                    freq_hz_binned[i_bin] = np.mean(freq_hz[bin_start:bin_end])
                    stokes_i_binned[i_bin] = np.mean(stokes_i[bin_start:bin_end])
                    stokes_q_binned[i_bin] = np.mean(stokes_q[bin_start:bin_end])
                    stokes_u_binned[i_bin] = np.mean(stokes_u[bin_start:bin_end])
                    if stokes_v is not None:
                        stokes_v_binned[i_bin] = np.mean(stokes_v[bin_start:bin_end])
                    n_chan_bin = max(1, bin_end - bin_start)
                    sigma_i_binned[i_bin] = np.sqrt(np.sum(sigma_i_chan[bin_start:bin_end]**2)) / n_chan_bin
                    sigma_q_binned[i_bin] = np.sqrt(np.sum(sigma_q_chan[bin_start:bin_end]**2)) / n_chan_bin
                    sigma_u_binned[i_bin] = np.sqrt(np.sum(sigma_u_chan[bin_start:bin_end]**2)) / n_chan_bin
                    if sigma_v_chan is not None:
                        sigma_v_binned[i_bin] = np.sqrt(np.sum(sigma_v_chan[bin_start:bin_end]**2)) / n_chan_bin

                freq_hz = freq_hz_binned
                stokes_i = stokes_i_binned
                stokes_q = stokes_q_binned
                stokes_u = stokes_u_binned
                if stokes_v is not None:
                    stokes_v = stokes_v_binned
                sigma_i_chan = sigma_i_binned
                sigma_q_chan = sigma_q_binned
                sigma_u_chan = sigma_u_binned
                sigma_v_chan = sigma_v_binned

                print(f"  Frequency-binned data: {n_freq} -> {len(freq_hz)} channels (--freq-bins={args.freq_bins})")

            # Propagate off-pulse Q/U/I noise to linear-fraction uncertainty per channel
            # for Burns-law error bars: f = L/I, L = sqrt(Q^2 + U^2).
            q_val = np.asarray(stokes_q, dtype=float)
            u_val = np.asarray(stokes_u, dtype=float)
            i_val = np.asarray(stokes_i, dtype=float)
            l_val = np.sqrt(q_val**2 + u_val**2)
            sigma_l = np.sqrt((q_val**2 * sigma_q_chan**2 + u_val**2 * sigma_u_chan**2) / (l_val**2 + 1e-20))
            burn_pol_frac_err = np.sqrt((sigma_l / (i_val + 1e-10))**2 +
                                        ((l_val * sigma_i_chan) / ((i_val + 1e-10)**2))**2)

            # Apply the same Stokes-I S/N style sanity cut used elsewhere so
            # very low-I channels do not produce pathological fractional errors.
            i_snr_chan = i_val / (sigma_i_chan + 1e-10)
            burn_valid_mask = i_snr_chan >= 2.0
            burn_pol_frac_err[~burn_valid_mask] = np.nan
            burn_circ_valid_mask = burn_valid_mask.copy()

            if stokes_v is not None and sigma_v_chan is not None:
                v_val = np.asarray(stokes_v, dtype=float)
                burn_circ_frac_err = np.sqrt((sigma_v_chan / (i_val + 1e-10))**2 +
                                             ((np.abs(v_val) * sigma_i_chan) / ((i_val + 1e-10)**2))**2)
                burn_circ_frac_err[~burn_circ_valid_mask] = np.nan
                burn_circ_frac_err = np.where(np.isfinite(burn_circ_frac_err) & (burn_circ_frac_err > 0),
                                              burn_circ_frac_err, np.nan)

            burn_pol_frac_err = np.where(np.isfinite(burn_pol_frac_err) & (burn_pol_frac_err > 0),
                                         burn_pol_frac_err, np.nan)
        elif not args.time_series:
            print("  Note: Data is 2D. Use --time-avg to average over time, or --time-series to process each time bin.")
            print("  Proceeding with first time sample...")
            # Take first time sample
            idx = [slice(None)] * stokes_i.ndim
            idx[args.time_axis] = 0
            stokes_i = stokes_i[tuple(idx)]
            stokes_q = stokes_q[tuple(idx)]
            stokes_u = stokes_u[tuple(idx)]
            if stokes_v is not None:
                stokes_v = stokes_v[tuple(idx)]
            print(f"  Using data shape: {stokes_i.shape}")
    
    print(f"  Frequency range: {freq_hz.min()/1e6:.2f} - {freq_hz.max()/1e6:.2f} MHz")
    print(f"  Number of channels: {len(freq_hz)}")
    
    if not args.time_series:
        # Single spectrum fitting
        print(f"\nPerforming RM fitting using method: {args.method}")
        
        # Initialize fitter
        fitter = RMFitter(freq_hz, stokes_i, stokes_q, stokes_u, stokes_v)
        
        # Perform fitting
        if args.method == 'simple':
            # Use RM synthesis for better accuracy
            result = fitter._fit_rm_with_rmtools(
                rm_range=tuple(args.rm_range),
                n_rm=args.n_rm
            )
            rm_peak_print = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
            rm_err = result.get('rm_clean_err', result.get('noise_estimate', 0) * 2)
            print(f"\nResults (RM Synthesis - Simple Mode):")
            print(f"  Peak RM = {rm_peak_print:.4f} ± {rm_err:.4f} rad/m²")
            print(f"  SNR = {result.get('rm_peak_snr', np.nan):.2f}")
            print(f"  Noise level = {result.get('noise_estimate', 0):.6f}")
            
            # Plotting
            if not args.no_plot:
                plot_rm_results(
                    fitter,
                    result,
                    f"{args.output}_rm_results.{args.ext}",
                    pol_frac_err=burn_pol_frac_err,
                    valid_mask=burn_valid_mask,
                    circ_frac_err=burn_circ_frac_err,
                    circ_valid_mask=burn_circ_valid_mask,
                    show_frac_panel=not args.hide_rm_frac_panel,
                )
            if args.poincare:
                print("Warning: --poincare requested but not running in time-series mode; skipping.")
            
        elif args.method == 'rm_synthesis':
            result = fitter._fit_rm_with_rmtools(
                rm_range=tuple(args.rm_range),
                n_rm=args.n_rm
            )
            rm_peak_print = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
            rm_err = result.get('rm_clean_err', result.get('noise_estimate', 0) * 2)
            print(f"\nResults (RM Synthesis):")
            print(f"  Peak RM = {rm_peak_print:.4f} ± {rm_err:.4f} rad/m²")
            print(f"  SNR = {result.get('rm_peak_snr', np.nan):.2f}")
            print(f"  Noise level = {result.get('noise_estimate', 0):.6f}")
            
            # Save RM spectrum
            output_txt = args.output + '_rm_spectrum.txt'
            np.savetxt(output_txt, 
                      np.column_stack([result['rm_values'], 
                                      result['rm_amplitude']]),
                      header='RM(rad/m2) Amplitude',
                      fmt='%.6f')
            print(f"\nRM spectrum saved to {output_txt}")
            
            # Plotting
            if not args.no_plot:
                plot_rm_results(
                    fitter,
                    result,
                    f"{args.output}_rm_results.{args.ext}",
                    pol_frac_err=burn_pol_frac_err,
                    valid_mask=burn_valid_mask,
                    circ_frac_err=burn_circ_frac_err,
                    circ_valid_mask=burn_circ_valid_mask,
                    show_frac_panel=not args.hide_rm_frac_panel,
                )
            if args.poincare:
                print("Warning: --poincare requested but not running in time-series mode; skipping.")
            
        elif args.method == 'qu_fitting':
            result = fitter.fit_rm_qufitting()
            if result['success']:
                print(f"\nResults (QU Fitting):")
                print(f"  RM = {result['rm']:.4f} ± {result['rm_err']:.4f} rad/m²")
                print(f"  Q₀ = {result['q0']:.6f} ± {result['q0_err']:.6f}")
                print(f"  U₀ = {result['u0']:.6f} ± {result['u0_err']:.6f}")
            else:
                print("QU fitting failed!")

        elif args.method == 'rmnest':
            rmnest_outdir = args.rmnest_outdir or f"{args.output}_rmnest"
            rmnest_label = args.rmnest_label or args.output
            try:
                result = fitter.fit_rm_rmnest(
                    gfr=args.rmnest_gfr,
                    free_alpha=args.rmnest_free_alpha,
                    outdir=rmnest_outdir,
                    label=rmnest_label,
                    sampler=args.rmnest_sampler
                )
            except ImportError as exc:
                print(f"RMNest unavailable: {exc}")
                return

            param_name = result['param_name'].upper()
            median = result['median']
            low = result['low']
            high = result['high']
            print("\nResults (RMNest):")
            print(f"  {param_name} = {median:.4f} +{high - median:.4f}/-{median - low:.4f}")
            print(f"  Output directory: {result['rmnest_outdir']}")
            print(f"  Bilby result: {result['rmnest_post_json']}")

        # Time-averaged branch only: fit and plot Burn-law depolarisation models
        if args.time_avg and not args.no_plot:
            burn_out = f"{args.output}_burns_law.{args.ext}"
            plot_burns_law_fits(fitter, burn_out,
                                pol_frac_err=burn_pol_frac_err,
                                valid_mask=burn_valid_mask,
                                circ_frac_err=burn_circ_frac_err,
                                circ_valid_mask=burn_circ_valid_mask,
                                turbulent_radius_pc=args.turbulent_radius_pc,
                                screen_scale_cm=args.screen_scale_cm)

        # If multiple peaks were selected in time-avg mode, process each
        # remaining peak separately with suffixed output names.
        if args.time_avg and len(time_avg_extra_regions) > 0 and stokes_i_full_noise is not None:
            for i_extra, (pk_start, pk_end) in enumerate(time_avg_extra_regions, start=2):
                print(f"\nProcessing additional selected peak {i_extra}: bins {pk_start} to {pk_end}")
                n_time_pk = max(1, pk_end - pk_start + 1)

                if args.time_axis == 0:
                    stokes_i_pk = np.mean(stokes_i_full_noise[pk_start:pk_end+1, :], axis=0)
                    stokes_q_pk = np.mean(stokes_q_full_noise[pk_start:pk_end+1, :], axis=0)
                    stokes_u_pk = np.mean(stokes_u_full_noise[pk_start:pk_end+1, :], axis=0)
                    stokes_v_pk = (np.mean(stokes_v_full_noise[pk_start:pk_end+1, :], axis=0)
                                   if stokes_v_full_noise is not None else None)
                else:
                    stokes_i_pk = np.mean(stokes_i_full_noise[:, pk_start:pk_end+1], axis=1)
                    stokes_q_pk = np.mean(stokes_q_full_noise[:, pk_start:pk_end+1], axis=1)
                    stokes_u_pk = np.mean(stokes_u_full_noise[:, pk_start:pk_end+1], axis=1)
                    stokes_v_pk = (np.mean(stokes_v_full_noise[:, pk_start:pk_end+1], axis=1)
                                   if stokes_v_full_noise is not None else None)

                freq_pk = freq_hz_unbinned.copy() if freq_hz_unbinned is not None else np.asarray(freq_hz, dtype=float).copy()
                sigma_i_pk = sigma_i_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_q_pk = sigma_q_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_u_pk = sigma_u_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_v_pk = (sigma_v_chan_base.copy() / np.sqrt(n_time_pk)) if sigma_v_chan_base is not None else None

                # Apply optional frequency binning for this peak run
                n_freq_pk = len(freq_pk)
                if args.freq_bins is not None and args.freq_bins > 0 and args.freq_bins < n_freq_pk:
                    n_freq_bins_actual = min(args.freq_bins, n_freq_pk)
                    freq_bin_size = int(np.ceil(n_freq_pk / n_freq_bins_actual))
                    n_freq_bins_actual = (n_freq_pk + freq_bin_size - 1) // freq_bin_size

                    freq_b = np.zeros(n_freq_bins_actual)
                    i_b = np.zeros(n_freq_bins_actual)
                    q_b = np.zeros(n_freq_bins_actual)
                    u_b = np.zeros(n_freq_bins_actual)
                    v_b = np.zeros(n_freq_bins_actual) if stokes_v_pk is not None else None
                    si_b = np.zeros(n_freq_bins_actual)
                    sq_b = np.zeros(n_freq_bins_actual)
                    su_b = np.zeros(n_freq_bins_actual)
                    sv_b = np.zeros(n_freq_bins_actual) if sigma_v_pk is not None else None

                    for i_bin in range(n_freq_bins_actual):
                        bin_start = i_bin * freq_bin_size
                        bin_end = min((i_bin + 1) * freq_bin_size, n_freq_pk)
                        if bin_end <= bin_start:
                            continue
                        n_chan_bin = max(1, bin_end - bin_start)
                        freq_b[i_bin] = np.mean(freq_pk[bin_start:bin_end])
                        i_b[i_bin] = np.mean(stokes_i_pk[bin_start:bin_end])
                        q_b[i_bin] = np.mean(stokes_q_pk[bin_start:bin_end])
                        u_b[i_bin] = np.mean(stokes_u_pk[bin_start:bin_end])
                        if stokes_v_pk is not None:
                            v_b[i_bin] = np.mean(stokes_v_pk[bin_start:bin_end])
                        si_b[i_bin] = np.sqrt(np.sum(sigma_i_pk[bin_start:bin_end]**2)) / n_chan_bin
                        sq_b[i_bin] = np.sqrt(np.sum(sigma_q_pk[bin_start:bin_end]**2)) / n_chan_bin
                        su_b[i_bin] = np.sqrt(np.sum(sigma_u_pk[bin_start:bin_end]**2)) / n_chan_bin
                        if sigma_v_pk is not None:
                            sv_b[i_bin] = np.sqrt(np.sum(sigma_v_pk[bin_start:bin_end]**2)) / n_chan_bin

                    freq_pk = freq_b
                    stokes_i_pk = i_b
                    stokes_q_pk = q_b
                    stokes_u_pk = u_b
                    if stokes_v_pk is not None:
                        stokes_v_pk = v_b
                    sigma_i_pk = si_b
                    sigma_q_pk = sq_b
                    sigma_u_pk = su_b
                    sigma_v_pk = sv_b

                # Build Burn-law point uncertainties/mask for this peak
                l_pk = np.sqrt(stokes_q_pk**2 + stokes_u_pk**2)
                sigma_l_pk = np.sqrt((stokes_q_pk**2 * sigma_q_pk**2 + stokes_u_pk**2 * sigma_u_pk**2) / (l_pk**2 + 1e-20))
                burn_err_pk = np.sqrt((sigma_l_pk / (stokes_i_pk + 1e-10))**2 +
                                      ((l_pk * sigma_i_pk) / ((stokes_i_pk + 1e-10)**2))**2)
                burn_mask_pk = (stokes_i_pk / (sigma_i_pk + 1e-10)) >= 2.0
                burn_err_pk[~burn_mask_pk] = np.nan
                burn_err_pk = np.where(np.isfinite(burn_err_pk) & (burn_err_pk > 0), burn_err_pk, np.nan)
                burn_circ_err_pk = None
                if stokes_v_pk is not None and sigma_v_pk is not None:
                    burn_circ_err_pk = np.sqrt((sigma_v_pk / (stokes_i_pk + 1e-10))**2 +
                                               ((np.abs(stokes_v_pk) * sigma_i_pk) / ((stokes_i_pk + 1e-10)**2))**2)
                    burn_circ_err_pk[~burn_mask_pk] = np.nan
                    burn_circ_err_pk = np.where(np.isfinite(burn_circ_err_pk) & (burn_circ_err_pk > 0),
                                                burn_circ_err_pk, np.nan)

                fitter_pk = RMFitter(freq_pk, stokes_i_pk, stokes_q_pk, stokes_u_pk, stokes_v_pk)
                output_prefix_pk = f"{args.output}_peak{i_extra}"

                if args.method in ['simple', 'rm_synthesis']:
                    result_pk = fitter_pk._fit_rm_with_rmtools(
                        rm_range=tuple(args.rm_range),
                        n_rm=args.n_rm
                    )
                    rm_peak_pk = result_pk.get('rm_clean_peak', result_pk.get('rm_peak', np.nan))
                    rm_err_pk = result_pk.get('rm_clean_err', result_pk.get('noise_estimate', 0) * 2)
                    print(f"  Peak RM = {rm_peak_pk:.4f} ± {rm_err_pk:.4f} rad/m²")
                    print(f"  SNR = {result_pk.get('rm_peak_snr', np.nan):.2f}")

                    output_txt_pk = output_prefix_pk + '_rm_spectrum.txt'
                    np.savetxt(output_txt_pk,
                               np.column_stack([result_pk['rm_values'], result_pk['rm_amplitude']]),
                               header='RM(rad/m2) Amplitude', fmt='%.6f')
                    print(f"  RM spectrum saved to {output_txt_pk}")

                    if not args.no_plot:
                        plot_rm_results(
                            fitter_pk,
                            result_pk,
                            f"{output_prefix_pk}_rm_results.{args.ext}",
                            pol_frac_err=burn_err_pk,
                            valid_mask=burn_mask_pk,
                            circ_frac_err=burn_circ_err_pk,
                            circ_valid_mask=burn_mask_pk,
                            show_frac_panel=not args.hide_rm_frac_panel,
                        )
                        plot_burns_law_fits(
                            fitter_pk,
                            f"{output_prefix_pk}_burns_law.{args.ext}",
                            pol_frac_err=burn_err_pk,
                            valid_mask=burn_mask_pk,
                            circ_frac_err=burn_circ_err_pk,
                            circ_valid_mask=burn_mask_pk,
                            turbulent_radius_pc=args.turbulent_radius_pc,
                            screen_scale_cm=args.screen_scale_cm,
                        )
                elif args.method == 'qu_fitting':
                    result_pk = fitter_pk.fit_rm_qufitting()
                    if result_pk['success']:
                        print(f"  RM = {result_pk['rm']:.4f} ± {result_pk['rm_err']:.4f} rad/m²")
                        if not args.no_plot:
                            plot_burns_law_fits(
                                fitter_pk,
                                f"{output_prefix_pk}_burns_law.{args.ext}",
                                pol_frac_err=burn_err_pk,
                                valid_mask=burn_mask_pk,
                                circ_frac_err=burn_circ_err_pk,
                                circ_valid_mask=burn_mask_pk,
                                turbulent_radius_pc=args.turbulent_radius_pc,
                                screen_scale_cm=args.screen_scale_cm,
                            )
                elif args.method == 'rmnest':
                    outdir_pk = (args.rmnest_outdir or f"{args.output}_rmnest") + f"_peak{i_extra}"
                    label_pk = (args.rmnest_label or args.output) + f"_peak{i_extra}"
                    try:
                        result_pk = fitter_pk.fit_rm_rmnest(
                            gfr=args.rmnest_gfr,
                            free_alpha=args.rmnest_free_alpha,
                            outdir=outdir_pk,
                            label=label_pk,
                            sampler=args.rmnest_sampler
                        )
                        print(f"  RMNest output directory: {result_pk['rmnest_outdir']}")
                    except ImportError as exc:
                        print(f"  RMNest unavailable for peak {i_extra}: {exc}")
    
    else:
        # Time series fitting
        if stokes_i.ndim != 2:
            print("\nError: Time series mode requires 2D data arrays.")
            return

        if args.method == 'rmnest':
            print("\nNote: RMNest time-series fitting can be slow."
                  " Outputs will be written per time bin.")
        
        print(f"\nProcessing time series data: {stokes_i.shape[args.time_axis]} time samples")
        print(f"Using method: {args.method}")
        print("This may take a while...")

        # Keep full dynamic spectrum for off-pulse noise estimates in
        # Poincare error bars, even if on-pulse slicing is requested.
        full_time_series_data = {
            'time': time_array if time_array is not None else np.arange(stokes_i.shape[args.time_axis]),
            'I': stokes_i,
            'Q': stokes_q,
            'U': stokes_u,
        }
        if stokes_v is not None:
            full_time_series_data['V'] = stokes_v
        
        # Prepare time series data dictionary
        if onpulse_mask is not None:
            start_idx, end_idx = onpulse_mask
            print(f"  Processing only on-pulse bins {start_idx} to {end_idx}")
            
            if args.time_axis == 0:
                time_series_data = {
                    'time': (time_array if time_array is not None else np.arange(stokes_i.shape[args.time_axis]))[start_idx:end_idx+1],
                    'I': stokes_i[start_idx:end_idx+1, :],
                    'Q': stokes_q[start_idx:end_idx+1, :],
                    'U': stokes_u[start_idx:end_idx+1, :]
                }
                if stokes_v is not None:
                    time_series_data['V'] = stokes_v[start_idx:end_idx+1, :]
            else:
                time_series_data = {
                    'time': (time_array if time_array is not None else np.arange(stokes_i.shape[args.time_axis]))[start_idx:end_idx+1],
                    'I': stokes_i[:, start_idx:end_idx+1],
                    'Q': stokes_q[:, start_idx:end_idx+1],
                    'U': stokes_u[:, start_idx:end_idx+1]
                }
                if stokes_v is not None:
                    time_series_data['V'] = stokes_v[:, start_idx:end_idx+1]
            
            print(f"  Reduced to {len(time_series_data['time'])} time samples")
        else:
            time_series_data = {
                'time': time_array if time_array is not None else np.arange(stokes_i.shape[args.time_axis]),
                'I': stokes_i,
                'Q': stokes_q,
                'U': stokes_u
            }
            if stokes_v is not None:
                time_series_data['V'] = stokes_v
        
        # Fit RM for each time sample
        rm_results = fit_rm_time_series(
            freq_hz, 
            time_series_data, 
            method=args.method,
            rm_range=tuple(args.rm_range),
            n_rm=args.n_rm,
            rmnest_gfr=args.rmnest_gfr,
            rmnest_free_alpha=args.rmnest_free_alpha,
            rmnest_outdir=args.rmnest_outdir or f"{args.output}_rmnest_ts",
            rmnest_label=args.rmnest_label or args.output,
            rmnest_sampler=args.rmnest_sampler,
            n_time_bins=args.time_bins
        )
        
        l_weights = None
        if 'L_frac_bin' in rm_results:
            l_weights = np.asarray(rm_results['L_frac_bin'], dtype=float) ** 2
        rm_diag = time_series_sigma_rm_diagnostic(rm_results['rm'], weights=l_weights)

        print("\nTime Series Results:")
        print(f"  RM bins used = {rm_diag['n_valid']}/{rm_diag['n_total']}")
        print(f"  Mean RM = {rm_diag['rm_mean']:.4f} rad/m²")
        print(f"  σ_RM(time) = {rm_diag['sigma_rm_time']:.4f} rad/m²")
        if np.isfinite(rm_diag['weighted_sigma_rm_time']):
            print(f"  Weighted Mean RM (L²) = {rm_diag['weighted_rm_mean']:.4f} rad/m²")
            print(f"  Weighted σ_RM(time) (L²) = {rm_diag['weighted_sigma_rm_time']:.4f} rad/m²")
        print(f"  Min RM = {rm_diag['rm_min']:.4f} rad/m²")
        print(f"  Max RM = {rm_diag['rm_max']:.4f} rad/m²")
        
        if 'snr' in rm_results and np.any(rm_results['snr'] > 0):
            print(f"  Mean SNR = {np.nanmean(rm_results['snr']):.2f}")

        # print PA/EA summary if present
        if 'pa_deg' in rm_results:
            print(f"  Mean PA = {np.nanmean(rm_results['pa_deg']):.2f} deg")
            print(f"  Mean EA = {np.nanmean(rm_results['ea_deg']):.2f} deg")
        
        # Save results
        output_txt = args.output + '_time_series.txt'
        header = 'Time(s) RM(rad/m2)'
        data = np.column_stack([rm_results['time'], rm_results['rm']])
        
        if 'rm_err' in rm_results and np.any(rm_results['rm_err'] > 0):
            header += ' RM_err(rad/m2)'
            data = np.column_stack([data, rm_results['rm_err']])
        if 'snr' in rm_results and np.any(rm_results['snr'] > 0):
            header += ' SNR'
            data = np.column_stack([data, rm_results['snr']])
        # add polarisation angle / ellipticity, if available
        if 'pa_deg' in rm_results:
            header += ' PA(deg)'
            data = np.column_stack([data, rm_results['pa_deg']])
        if 'ea_deg' in rm_results:
            header += ' EA(deg)'
            data = np.column_stack([data, rm_results['ea_deg']])
        if 'pa_err_deg' in rm_results:
            header += ' PA_err(deg)'
            data = np.column_stack([data, rm_results['pa_err_deg']])
        if 'ea_err_deg' in rm_results:
            header += ' EA_err(deg)'
            data = np.column_stack([data, rm_results['ea_err_deg']])
        
        np.savetxt(output_txt, data, header=header, fmt='%.6f')
        print(f"\nTime series data saved to {output_txt}")
        
        # Plot time series
        if not args.no_plot:
            # Get time profile for peak detection
            if args.separate_peaks:
                if args.time_axis == 0:
                    # Reconstruct original time profile (before on-pulse filtering)
                    if onpulse_mask is not None:
                        start_idx, end_idx = onpulse_mask
                        full_time_profile = np.zeros(end_idx - start_idx + 1)
                        if time_series_data['I'].ndim == 2:
                            full_time_profile = np.sum(time_series_data['I'], axis=1)
                    else:
                        full_time_profile = np.sum(time_series_data['I'], axis=1) if time_series_data['I'].ndim == 2 else None
                else:
                    if onpulse_mask is not None:
                        start_idx, end_idx = onpulse_mask
                        full_time_profile = np.zeros(end_idx - start_idx + 1)
                        if time_series_data['I'].ndim == 2:
                            full_time_profile = np.sum(time_series_data['I'], axis=0)
                    else:
                        full_time_profile = np.sum(time_series_data['I'], axis=0) if time_series_data['I'].ndim == 2 else None
            else:
                full_time_profile = None
            
            plot_rm_time_series(rm_results['time'], rm_results, 
                              f"{args.output}_time_series.{args.ext}",
                              time_profile=full_time_profile,
                              separate_peaks=args.separate_peaks,
                              min_gap_bins=args.min_gap_bins,
                              min_peak_bins=args.min_peak_bins,
                              max_merge_gap=args.max_merge_gap,
                              time_series_data=time_series_data,
                              freq_hz=freq_hz,
                              noise_fraction=args.noise_fraction)
            # Generate Poincaré sphere if requested – this routine now operates
            # purely on the provided ``time_series_data`` and ignores frequency.
            if args.poincare:
                # For the Poincaré sphere we generally want raw samples, not
                # the same binning used for the RM time-series fit.  override
                # ``n_time_bins`` unless the user explicitly set a positive
                # value on the command line.
                # respect the time-bin setting for the Poincaré plot so the
                # user can limit the number of points.  ``args.time_bins``
                # defaults to 20, matching the original behaviour.
                pt_bins = args.time_bins if args.time_bins and args.time_bins > 0 else None
                plot_poincare_sphere(time_series_data,
                                     f"{args.output}_poincare.{args.ext}",
                                     n_time_bins=pt_bins,
                                     noise_fraction=args.noise_fraction,
                                     time_unit=args.time_unit,
                                     interactive=args.poincare_interactive,
                                     force_surface=args.poincare_surface,
                                     rm_results=rm_results,
                                     noise_reference_data=full_time_series_data,
                                     circle_fit_mode=args.poincare_circle_fit,
                                     circle_fit_segments=circle_segments)
                # Optional 2-D projection panel
                if args.poincare_projections:
                    proj_tag = str(args.poincare_projections).lower()
                    plot_poincare_projections(
                        time_series_data,
                        f"{args.output}_poincare_projections_{proj_tag}.{args.ext}",
                        projection_type=args.poincare_projections,
                        n_time_bins=pt_bins,
                        noise_fraction=args.noise_fraction,
                        time_unit=args.time_unit,
                        force_surface=args.poincare_surface,
                        rm_results=rm_results,
                                noise_reference_data=full_time_series_data,
                           circle_fit_mode=args.poincare_circle_fit,
                           circle_fit_segments=circle_segments,
                        center=tuple(args.poincare_proj_center)
                               if args.poincare_proj_center is not None else None)
    
    print("\n" + "="*60)
    print("RM fitting completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()