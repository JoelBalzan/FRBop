"""
RMFitter class and the fit_rm_time_series driver.
"""

import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from scipy.optimize import curve_fit

from .diagnostics import summarize_posterior

warnings.filterwarnings('ignore')


def debiased_linear_from_qu(
    data_q: np.ndarray,
    data_u: np.ndarray,
    noise_q: np.ndarray,
    noise_u: np.ndarray,
    cutoff: float = 1.57,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Ricean-debiased linear polarisation L = sqrt(Q² + U²).

    Applies Ricean debiasing and a detection cutoff on L/sigma_L.

    Parameters
    ----------
    data_q, data_u : np.ndarray
        Stokes Q and U arrays.
    noise_q, noise_u : np.ndarray
        Per-sample noise estimates for Q and U (same shape as data).
    cutoff : float
        Detection threshold on L/sigma_L (default 1.57 ~ 1-sigma).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    L_out : np.ndarray
        Debiased linear polarisation.
    sigma_L : np.ndarray
        Propagated uncertainty on L.
    det : np.ndarray (bool)
        Detection mask where L/sigma_L >= cutoff.
    """
    L_meas = np.sqrt(data_q ** 2 + data_u ** 2)
    sigma_L = np.sqrt(data_q ** 2 * noise_q ** 2 + data_u ** 2 * noise_u ** 2) / np.maximum(L_meas, eps)
    det = L_meas / np.maximum(sigma_L, eps) >= cutoff

    L_out = np.zeros_like(L_meas)
    L_out[det] = np.sqrt(np.maximum(L_meas[det] ** 2 - sigma_L[det] ** 2, 0.0))

    return L_out, sigma_L, det


def _patch_scipy_bilby_compat() -> None:
    """Patch known bilby/scipy symbol mismatches for older bilby releases."""
    try:
        import scipy.special._ufuncs as _ufuncs
    except Exception:
        return
    alias_map = {
        "btdtr": "bdtr",
        "btdtri": "bdtri",
        "btdtria": "bdtria",
    }
    for missing_name, fallback_name in alias_map.items():
        if hasattr(_ufuncs, missing_name) or not hasattr(_ufuncs, fallback_name):
            continue
        setattr(_ufuncs, missing_name, getattr(_ufuncs, fallback_name))


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
        q_std = float(np.nanstd(self.stokes_q))
        u_std = float(np.nanstd(self.stokes_u))
        q_max = float(np.nanmax(np.abs(self.stokes_q))) if np.any(np.isfinite(self.stokes_q)) else 0.0
        u_max = float(np.nanmax(np.abs(self.stokes_u))) if np.any(np.isfinite(self.stokes_u)) else 0.0
        noise_q_local = noise_q if noise_q is not None else (
            q_std if q_std > 0 else 0.01 * q_max
        )
        noise_u_local = noise_u if noise_u is not None else (
            u_std if u_std > 0 else 0.01 * u_max
        )

        # Use provided noise_i (from off-pulse) when available; otherwise use
        # per-spectrum std fallback.
        i_std = float(np.nanstd(self.stokes_i))
        noise_i_local = noise_i if noise_i is not None else (i_std if i_std > 0 else 0.01)

        dI = np.ones_like(self.stokes_i) * noise_i_local
        dQ = np.ones_like(self.stokes_q) * noise_q_local
        dU = np.ones_like(self.stokes_u) * noise_u_local

        # RMtools expects data as [freq_Hz, I, Q, U, dI, dQ, dU]
        finite = (
            np.isfinite(self.freq_hz)
            & np.isfinite(self.stokes_i)
            & np.isfinite(self.stokes_q)
            & np.isfinite(self.stokes_u)
            & np.isfinite(dI)
            & np.isfinite(dQ)
            & np.isfinite(dU)
        )
        if np.sum(finite) < 4:
            raise ValueError("RM synthesis needs at least 4 finite frequency channels")
        data = [
            self.freq_hz[finite],
            self.stokes_i[finite],
            self.stokes_q[finite],
            self.stokes_u[finite],
            dI[finite],
            dQ[finite],
            dU[finite],
        ]

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

        dphi_peak_pi_fit = mDict.get('dPhiPeakPIfit_rm2', None)
        results = {
            'rm_values': rm_values,
            'rm_spectrum': rm_spectrum,
            'rm_amplitude': rm_amplitude,
            'rm_peak': rm_peak,
            'rm_peak_snr': rm_peak_snr,
            'noise_estimate': noise_estimate,
            'dphi_peak_pi_fit': dphi_peak_pi_fit,
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
        lambda_sq_double = np.concatenate([self.lambda_sq, self.lambda_sq])  # noqa: F841

        # Initial guess
        p0 = [0.0, np.nanmean(self.stokes_q), np.nanmean(self.stokes_u)]

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
        _patch_scipy_bilby_compat()
        for _ in range(4):
            try:
                from rmnest.fit_RM import RMNest
                break
            except ImportError as exc:
                msg = str(exc)
                match = re.search(
                    r"cannot import name '([^']+)' from 'scipy\.special\._ufuncs'",
                    msg,
                )
                if match:
                    import scipy.special._ufuncs as _ufuncs
                    missing = match.group(1)
                    candidates = []
                    if missing.startswith("btd"):
                        candidates.append("bd" + missing[3:])
                    if missing.startswith("std"):
                        candidates.append("sd" + missing[3:])
                    for cand in candidates:
                        if hasattr(_ufuncs, cand):
                            setattr(_ufuncs, missing, getattr(_ufuncs, cand))
                            sys.modules.pop("rmnest", None)
                            sys.modules.pop("bilby", None)
                            break
                    else:
                        raise
                else:
                    raise
        else:
            raise ImportError(
                "RMNest is not installed. Install with: pip install rmnest"
            )

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

        median, low, high = summarize_posterior(result.posterior[param_name].values)

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


def fit_rm_time_series(freq_hz: np.ndarray, time_series_data: Dict,
                       method: str = 'rm_synthesis', rm_range: Tuple[float, float] = (-1000, 1000),
                       n_rm: int = 2000, rmnest_gfr: bool = False,
                       rmnest_free_alpha: bool = False, rmnest_outdir: Optional[str] = None,
                       rmnest_label: Optional[str] = None, rmnest_sampler: str = 'dynesty',
                       n_time_bins: Optional[int] = None,
                       noise_fraction: float = 0.1,
                       offpulse_std: Optional[np.ndarray] = None,
                       exclude_edge_bins: int = 0) -> Dict:
    """
    Fit RM for time-series data (multiple time samples).

    Parameters
    -----------
    freq_hz : array
        Frequency array in Hz
    time_series_data : dict
        Dictionary with keys 'time', 'I', 'Q', 'U', 'V' (optional)
        where each is a 2D array with time on one axis
    method : str
        Fitting method: 'simple', 'rm_synthesis', 'qu_fitting', or 'rmnest'
    rm_range : tuple
        Range of RM values to search (rad/m^2) for rm_synthesis
    n_rm : int
        Number of RM trial values for rm_synthesis
    n_time_bins : int, optional
        Number of time bins to fit (default: no binning)
    noise_fraction : float
        Fraction of time samples used for off-pulse noise estimation (default: 0.1)
    offpulse_std : np.ndarray, optional
        Pre-computed per-channel noise as array of shape (4, n_freq) with rows
        [sigma_i, sigma_q, sigma_u, sigma_v]. When provided, internal noise
        estimation is skipped entirely.
    exclude_edge_bins : int
        Number of frequency bins to exclude from each spectrum edge before
        fitting (default: 0)

    Returns:
    --------
    results : dict
        Dictionary containing RM values and errors as function of time
    """
    times = time_series_data['time']
    n_time = len(times)
    freq_fit = np.asarray(freq_hz, dtype=float)
    n_edge = int(max(0, exclude_edge_bins))
    if n_edge > 0:
        if (2 * n_edge) >= len(freq_fit):
            raise ValueError(
                f"exclude_edge_bins={n_edge} removes all channels for time-series fitting "
                f"(n_freq={len(freq_fit)})."
            )
        freq_fit = freq_fit[n_edge:-n_edge]

    # Determine which axis is time (the one matching length of times array)
    stokes_i_data = time_series_data['I']
    if stokes_i_data.shape[0] == n_time:
        time_axis = 0
    elif stokes_i_data.shape[1] == n_time:
        time_axis = 1
    else:
        time_axis = 0  # default

    # -------------------------------------------------------------------------
    # Noise estimation
    # Use pre-computed per-channel noise if provided, otherwise estimate from
    # the first noise_fraction of the time axis of time_series_data.
    # -------------------------------------------------------------------------
    if offpulse_std is not None:
        sigma_i_chan = np.asarray(offpulse_std[0], dtype=float)
        sigma_q_chan = np.asarray(offpulse_std[1], dtype=float)
        sigma_u_chan = np.asarray(offpulse_std[2], dtype=float)
        sigma_v_chan = np.asarray(offpulse_std[3], dtype=float) if offpulse_std.shape[0] > 3 else None

        # In the offpulse_std branch, scale noise_i for frequency averaging too
        noise_q_perchan = float(np.nanmedian(sigma_q_chan[sigma_q_chan > 0])) if np.any(sigma_q_chan > 0) else 1e-10
        noise_u_perchan = float(np.nanmedian(sigma_u_chan[sigma_u_chan > 0])) if np.any(sigma_u_chan > 0) else 1e-10
        noise_i_perchan = float(np.nanmedian(sigma_i_chan[sigma_i_chan > 0])) if np.any(sigma_i_chan > 0) else 1e-10
        noise_v_perchan = float(np.nanmedian(sigma_v_chan[sigma_v_chan > 0])) if (sigma_v_chan is not None and np.any(sigma_v_chan > 0)) else 0.0

        n_chan = len(sigma_i_chan)
        noise_i = noise_i_perchan / np.sqrt(n_chan)
        noise_q = noise_q_perchan / np.sqrt(n_chan)
        noise_u = noise_u_perchan / np.sqrt(n_chan)
        noise_v = noise_v_perchan / np.sqrt(n_chan) if noise_v_perchan > 0 else 0.0
    else:
        # Estimate noise from the first noise_fraction of time bins
        if time_axis == 0:
            I_full_for_noise = np.nanmean(time_series_data['I'], axis=1)
            Q_time_noise = time_series_data['Q']
            U_time_noise = time_series_data['U']
        else:
            I_full_for_noise = np.nanmean(time_series_data['I'], axis=0)
            Q_time_noise = time_series_data['Q']
            U_time_noise = time_series_data['U']

        n_frac_noise = max(1, int(len(I_full_for_noise) * noise_fraction))

        if time_axis == 0:
            q_off = Q_time_noise[:n_frac_noise, :]
            u_off = U_time_noise[:n_frac_noise, :]
        else:
            q_off = Q_time_noise[:, :n_frac_noise]
            u_off = U_time_noise[:, :n_frac_noise]

        q_std_chan = np.nanstd(q_off, axis=0 if time_axis == 0 else 1)
        u_std_chan = np.nanstd(u_off, axis=0 if time_axis == 0 else 1)
        noise_q = np.nanmedian(q_std_chan) if np.nanmedian(q_std_chan) > 0 else (np.nanmean(q_std_chan) if np.nanmean(q_std_chan) > 0 else 1e-10)
        noise_u = np.nanmedian(u_std_chan) if np.nanmedian(u_std_chan) > 0 else (np.nanmean(u_std_chan) if np.nanmean(u_std_chan) > 0 else 1e-10)

        noise_i = np.nanstd(I_full_for_noise[:n_frac_noise])
        if noise_i <= 0:
            mad = np.nanmedian(np.abs(I_full_for_noise - np.nanmedian(I_full_for_noise)))
            noise_i = mad / 0.6745 if mad > 0 else max(np.nanmedian(I_full_for_noise) * 0.1, 1e-10)

        if 'V' in time_series_data:
            if time_axis == 0:
                V_full_for_noise = np.nanmean(time_series_data['V'], axis=1)
            else:
                V_full_for_noise = np.nanmean(time_series_data['V'], axis=0)
            noise_v = np.nanstd(V_full_for_noise[:n_frac_noise])
            if noise_v <= 0:
                mad_v = np.nanmedian(np.abs(V_full_for_noise - np.nanmedian(V_full_for_noise)))
                noise_v = mad_v / 0.6745 if mad_v > 0 else 1e-10
        else:
            noise_v = 0.0
            noise_i_perchan = noise_i
            noise_q_perchan = noise_q
            noise_u_perchan = noise_u

    # -------------------------------------------------------------------------
    # Binning setup
    # -------------------------------------------------------------------------
    if n_time_bins is None or n_time_bins <= 0 or n_time_bins >= n_time:
        bin_size = 1
        n_bins_actual = n_time
    else:
        n_bins_actual = min(n_time_bins, n_time)
        bin_size = int(np.ceil(n_time / n_bins_actual))
        n_bins_actual = (n_time + bin_size - 1) // bin_size
        n_bins_actual = min(n_bins_actual, n_time_bins)

    # -------------------------------------------------------------------------
    # Output arrays
    # -------------------------------------------------------------------------
    rm_array = np.zeros(n_bins_actual)
    rm_err_array = np.zeros(n_bins_actual)
    pol_angle_0_array = np.zeros(n_bins_actual)
    snr_array = np.zeros(n_bins_actual)
    pol_angle_ref_array = np.zeros(n_bins_actual)
    pa_array = np.zeros(n_bins_actual)
    ea_array = np.zeros(n_bins_actual)
    pa_err_array = np.zeros(n_bins_actual)
    ea_err_array = np.zeros(n_bins_actual)

    P_frac_bins = np.zeros(n_bins_actual)
    L_frac_bins = np.zeros(n_bins_actual)
    V_frac_bins = np.zeros(n_bins_actual)
    q_bin = np.zeros(n_bins_actual)
    u_bin = np.zeros(n_bins_actual)
    v_bin = np.zeros(n_bins_actual)
    time_binned = np.zeros(n_bins_actual)
    i_snr_array = np.zeros(n_bins_actual)
    snr_array_L = np.zeros(n_bins_actual)
    sigma_L_bins = np.zeros(n_bins_actual)
    time_bin_start = np.zeros(n_bins_actual, dtype=int)
    time_bin_end = np.zeros(n_bins_actual, dtype=int)

    pa_corr_deg = np.full(n_bins_actual, np.nan)
    pa_corr_err_deg = np.full(n_bins_actual, np.nan)
    l_corr_frac = np.full(n_bins_actual, np.nan)
    q_corr_bin = np.full(n_bins_actual, np.nan)
    u_corr_bin = np.full(n_bins_actual, np.nan)
    rm_corr = np.full(n_bins_actual, np.nan)
    rm_corr_err = np.full(n_bins_actual, np.nan)

    pa_corr_full = np.full(n_time, np.nan)
    l_corr_full = np.full(n_time, np.nan)

    # -------------------------------------------------------------------------
    # Main loop over time bins
    # -------------------------------------------------------------------------
    n_freq_used = len(freq_fit)

    for i in range(n_bins_actual):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, n_time)
        if bin_end <= bin_start:
            continue

        time_bin_start[i] = bin_start
        time_bin_end[i] = bin_end
        time_binned[i] = np.nanmean(times[bin_start:bin_end])

        # Extract and time-average data for this bin
        if time_axis == 0:
            stokes_i = np.nanmean(time_series_data['I'][bin_start:bin_end, :], axis=0)
            stokes_q = np.nanmean(time_series_data['Q'][bin_start:bin_end, :], axis=0)
            stokes_u = np.nanmean(time_series_data['U'][bin_start:bin_end, :], axis=0)
            stokes_v = np.nanmean(time_series_data['V'][bin_start:bin_end, :], axis=0) if 'V' in time_series_data else None
        else:
            stokes_i = np.nanmean(time_series_data['I'][:, bin_start:bin_end], axis=1)
            stokes_q = np.nanmean(time_series_data['Q'][:, bin_start:bin_end], axis=1)
            stokes_u = np.nanmean(time_series_data['U'][:, bin_start:bin_end], axis=1)
            stokes_v = np.nanmean(time_series_data['V'][:, bin_start:bin_end], axis=1) if 'V' in time_series_data else None

        if n_edge > 0:
            stokes_i = stokes_i[n_edge:-n_edge]
            stokes_q = stokes_q[n_edge:-n_edge]
            stokes_u = stokes_u[n_edge:-n_edge]
            if stokes_v is not None:
                stokes_v = stokes_v[n_edge:-n_edge]

        # Initialize fitter
        fitter = RMFitter(freq_fit, stokes_i, stokes_q, stokes_u, stokes_v)

        # Frequency- and time-averaged Stokes values for this bin
        q_val = np.nanmean(stokes_q)
        u_val = np.nanmean(stokes_u)
        i_val = np.nanmean(stokes_i)
        v_val = np.nanmean(stokes_v) if stokes_v is not None else 0.0

        n_time_in_bin = bin_end - bin_start
        i_snr_array[i] = i_val / (noise_i / np.sqrt(n_time_in_bin) + 1e-10)
        q_bin[i] = q_val / (i_val + 1e-10)
        u_bin[i] = u_val / (i_val + 1e-10)
        v_bin[i] = v_val / (i_val + 1e-10)

        # ----------------------------------------------------------------
        # Noise scaling
        # noise_q/noise_u are per single-channel single-time-bin estimates.
        # After averaging over n_freq_used channels and n_time_in_bin time
        # bins the noise on the mean is reduced by sqrt(n_freq * n_time).
        # ----------------------------------------------------------------
        n_time_in_bin = bin_end - bin_start
        noise_scale = np.sqrt(n_freq_used * n_time_in_bin)
        noise_q_bin = noise_q / noise_scale
        noise_u_bin = noise_u / noise_scale
        noise_v_bin = noise_v / noise_scale if noise_v > 0 else 0.0

        # Linear polarisation with optional Ricean debiasing
        L_meas = np.sqrt(q_val**2 + u_val**2)
        sigma_L = np.sqrt(
            q_val**2 * noise_q_bin**2 + u_val**2 * noise_u_bin**2
        ) / max(L_meas, 1e-12)
        L_snr = L_meas / max(sigma_L, 1e-12)
        snr_array_L[i] = L_snr
        sigma_L_bins[i] = sigma_L
        L_det = L_snr >= 1.57

        if L_det:
            L_val = np.sqrt(max(L_meas**2 - sigma_L**2, 0.0))
        else:
            L_val = L_meas

        P_amp = np.sqrt(L_val**2 + v_val**2) + 1e-10
        pa_array[i] = np.degrees(0.5 * np.arctan2(u_val, q_val))
        ea_array[i] = np.degrees(0.5 * np.arcsin(np.clip(v_val / P_amp, -1.0, 1.0)))

        P_frac_bins[i] = P_amp / (i_val + 1e-10)
        L_frac_bins[i] = L_val / (i_val + 1e-10)
        V_frac_bins[i] = v_val / (i_val + 1e-10)

        P_lin_sq = q_val**2 + u_val**2 + 1e-20
        pa_sigma_rad = 0.5 * np.sqrt(
            (u_val**2 * noise_q_bin**2 + q_val**2 * noise_u_bin**2) / (P_lin_sq**2)
        )

        sigma_P = np.sqrt(q_val**2 * noise_q_bin**2 + u_val**2 * noise_u_bin**2) / (P_amp + 1e-10)
        sigma_VoverP = np.sqrt(
            (noise_v_bin**2 / (P_amp**2))
            + (v_val**2 * sigma_P**2 / (P_amp**4 + 1e-20))
        )
        denom = np.sqrt(max(1.0 - (v_val / P_amp)**2, 1e-10))
        ea_sigma_rad = 0.5 * (sigma_VoverP / denom)

        if not L_det:
            pa_err_array[i] = np.nan
            ea_err_array[i] = np.nan
            pa_array[i] = np.nan
            ea_array[i] = np.nan
        else:
            pa_err_array[i] = np.degrees(pa_sigma_rad)
            ea_err_array[i] = np.degrees(ea_sigma_rad)

        # Store polarisation angle at reference frequency
        ref_idx = len(freq_fit) // 2
        pol_angle_ref_array[i] = fitter.pol_angle[ref_idx]

        # ----------------------------------------------------------------
        # RM fitting
        # ----------------------------------------------------------------
        if method in ('simple', 'rm_synthesis'):
            result = fitter._fit_rm_with_rmtools(
                rm_range=rm_range, n_rm=n_rm,
                noise_i=noise_i_perchan / np.sqrt(n_time_in_bin),
                noise_q=noise_q_perchan / np.sqrt(n_time_in_bin),
                noise_u=noise_u_perchan / np.sqrt(n_time_in_bin),
            )
            rm_array[i] = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
            rm_err_array[i] = result.get('rm_clean_err', result.get('dphi_peak_pi_fit', result.get('noise_estimate', 0) * 2))
            snr_array[i] = result.get('rm_peak_snr', np.nan)

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
                sampler=rmnest_sampler,
            )
            median = result['median']
            low = result['low']
            high = result['high']
            rm_array[i] = median
            rm_err_array[i] = max(median - low, high - median)

        # ----------------------------------------------------------------
        # Derotate Q/U using fitted RM, then re-measure PA & L
        # ----------------------------------------------------------------
        if np.isfinite(rm_array[i]):
            c = 299792458.0
            lambda_sq_bin = (c / freq_fit) ** 2

            # Per-bin (time-averaged) corrected scalars
            P_obs = stokes_q + 1j * stokes_u
            P_corr = P_obs * np.exp(-2j * rm_array[i] * lambda_sq_bin)
            q_corr_freq = np.real(P_corr)
            u_corr_freq = np.imag(P_corr)
            q_corr_mn = np.nanmean(q_corr_freq)
            u_corr_mn = np.nanmean(u_corr_freq)

            q_corr_bin[i] = q_corr_mn / (i_val + 1e-10)
            u_corr_bin[i] = u_corr_mn / (i_val + 1e-10)
            l_corr = np.sqrt(q_corr_mn**2 + u_corr_mn**2)
            l_corr_frac[i] = l_corr / (i_val + 1e-10)
            pa_corr_deg[i] = np.degrees(0.5 * np.arctan2(u_corr_mn, q_corr_mn))
            pa_corr_err_deg[i] = np.degrees(
                0.5 * np.sqrt(
                    (u_corr_mn**2 * noise_q_bin**2 + q_corr_mn**2 * noise_u_bin**2)
                    / max(q_corr_mn**2 + u_corr_mn**2, 1e-20)
                )
            )

            # Re-run RM synthesis on the derotated Q/U to check correction
            try:
                fitter_corr = RMFitter(freq_fit, stokes_i, q_corr_freq, u_corr_freq, None)
                result_corr = fitter_corr._fit_rm_with_rmtools(
                    rm_range=rm_range, n_rm=n_rm,
                    noise_i=noise_i_perchan / np.sqrt(n_time_in_bin),
                    noise_q=noise_q_perchan / np.sqrt(n_time_in_bin),
                    noise_u=noise_u_perchan / np.sqrt(n_time_in_bin),
                )
                rm_corr[i] = result_corr.get('rm_clean_peak', result_corr.get('rm_peak', np.nan))
                rm_corr_err[i] = result_corr.get('rm_clean_err',
                                                  result_corr.get('dphi_peak_pi_fit',
                                                                  result_corr.get('noise_estimate', 0) * 2))
            except Exception:
                rm_corr[i] = np.nan
                rm_corr_err[i] = np.nan

            # Full time-resolution corrected PA and L/I from the derotated dspec
            for t in range(bin_start, bin_end):
                if time_axis == 0:
                    q_t = time_series_data['Q'][t, :]
                    u_t = time_series_data['U'][t, :]
                    i_t = time_series_data['I'][t, :]
                else:
                    q_t = time_series_data['Q'][:, t]
                    u_t = time_series_data['U'][:, t]
                    i_t = time_series_data['I'][:, t]

                if n_edge > 0:
                    q_t = q_t[n_edge:-n_edge]
                    u_t = u_t[n_edge:-n_edge]
                    i_t = i_t[n_edge:-n_edge]

                P_obs_t = q_t + 1j * u_t
                P_corr_t = P_obs_t * np.exp(-2j * rm_array[i] * lambda_sq_bin)
                q_corr_t = np.nanmean(np.real(P_corr_t))
                u_corr_t = np.nanmean(np.imag(P_corr_t))
                i_mn_t = np.nanmean(i_t)

                pa_corr_full[t] = np.degrees(0.5 * np.arctan2(u_corr_t, q_corr_t))
                l_corr_full[t] = np.sqrt(q_corr_t**2 + u_corr_t**2) / (i_mn_t + 1e-10)

    # -------------------------------------------------------------------------
    # Masking
    # -------------------------------------------------------------------------
    valid_bins = i_snr_array >= 2.0

    rm_array[~valid_bins] = np.nan
    rm_err_array[~valid_bins] = np.nan
    snr_array[~valid_bins] = np.nan
    pa_corr_deg[~valid_bins] = np.nan
    pa_corr_err_deg[~valid_bins] = np.nan
    l_corr_frac[~valid_bins] = np.nan
    q_corr_bin[~valid_bins] = np.nan
    u_corr_bin[~valid_bins] = np.nan
    rm_corr[~valid_bins] = np.nan
    rm_corr_err[~valid_bins] = np.nan

    return {
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
        'time_bin_start': time_bin_start,
        'time_bin_end': time_bin_end,
        'i_snr': i_snr_array,
        'valid_bins': valid_bins,
        'l_snr': snr_array_L,
        'sigma_l': sigma_L_bins,
        'pa_corr_deg': pa_corr_deg,
        'pa_corr_err_deg': pa_corr_err_deg,
        'l_corr_frac': l_corr_frac,
        'q_corr_bin': q_corr_bin,
        'u_corr_bin': u_corr_bin,
        'pa_corr_full': pa_corr_full,
        'l_corr_full': l_corr_full,
        'rm_corr': rm_corr,
        'rm_corr_err': rm_corr_err,
        'time_full': np.asarray(times, dtype=float),
    }