"""
RMFitter class and the fit_rm_time_series driver.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from scipy.optimize import curve_fit

from .diagnostics import summarize_posterior

warnings.filterwarnings('ignore')


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
                       exclude_edge_bins: int = 0) -> Dict:
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

    # Compute off-pulse-based Q/U noise estimates using the same fraction of
    # samples used elsewhere for I noise estimation.
    if time_axis == 0:
        I_full_for_noise = np.nanmean(time_series_data['I'], axis=1)
        Q_time = time_series_data['Q']
        U_time = time_series_data['U']
    else:
        I_full_for_noise = np.nanmean(time_series_data['I'], axis=0)
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
        bin_size = 1
        n_bins_actual = n_time
    else:
        n_bins_actual = min(n_time_bins, n_time)
        bin_size = int(np.ceil(n_time / n_bins_actual))
        n_bins_actual = (n_time + bin_size - 1) // bin_size
        n_bins_actual = min(n_bins_actual, n_time_bins)

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

    # prepare noise estimate for Stokes V if available
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

    for i in range(n_bins_actual):
        bin_start = i * bin_size
        bin_end = min((i + 1) * bin_size, n_time)
        if bin_end <= bin_start:
            continue

        time_binned[i] = np.nanmean(times[bin_start:bin_end])

        # Extract data for this time bin and average in time
        if time_axis == 0:
            stokes_i = np.nanmean(time_series_data['I'][bin_start:bin_end, :], axis=0)
            stokes_q = np.nanmean(time_series_data['Q'][bin_start:bin_end, :], axis=0)
            stokes_u = np.nanmean(time_series_data['U'][bin_start:bin_end, :], axis=0)
            if 'V' in time_series_data:
                stokes_v = np.nanmean(time_series_data['V'][bin_start:bin_end, :], axis=0)
            else:
                stokes_v = None
        else:
            stokes_i = np.nanmean(time_series_data['I'][:, bin_start:bin_end], axis=1)
            stokes_q = np.nanmean(time_series_data['Q'][:, bin_start:bin_end], axis=1)
            stokes_u = np.nanmean(time_series_data['U'][:, bin_start:bin_end], axis=1)
            if 'V' in time_series_data:
                stokes_v = np.nanmean(time_series_data['V'][:, bin_start:bin_end], axis=1)
            else:
                stokes_v = None

        if n_edge > 0:
            stokes_i = stokes_i[n_edge:-n_edge]
            stokes_q = stokes_q[n_edge:-n_edge]
            stokes_u = stokes_u[n_edge:-n_edge]
            if stokes_v is not None:
                stokes_v = stokes_v[n_edge:-n_edge]

        # Initialize fitter
        fitter = RMFitter(freq_fit, stokes_i, stokes_q, stokes_u, stokes_v)

        q_val = np.nanmean(stokes_q)
        u_val = np.nanmean(stokes_u)
        i_val = np.nanmean(stokes_i)
        v_val = np.nanmean(stokes_v) if stokes_v is not None else 0.0
        i_snr_array[i] = i_val / (noise_i + 1e-10)
        q_bin[i] = q_val / (i_val + 1e-10)
        u_bin[i] = u_val / (i_val + 1e-10)
        v_bin[i] = v_val / (i_val + 1e-10)
        P_amp = np.sqrt(q_val**2 + u_val**2 + v_val**2) + 1e-10
        pa_array[i] = np.degrees(0.5 * np.arctan2(u_val, q_val))
        ea_array[i] = np.degrees(0.5 * np.arcsin(v_val / P_amp))

        P_lin_sq = q_val**2 + u_val**2 + 1e-20
        pa_sigma_rad = 0.5 * np.sqrt((u_val**2 * noise_q**2 + q_val**2 * noise_u**2) / (P_lin_sq**2))
        pa_err_array[i] = np.degrees(pa_sigma_rad)
        sigma_P = np.sqrt((q_val**2 * noise_q**2 + u_val**2 * noise_u**2)) / (P_amp + 1e-10)
        sigma_VoverP = np.sqrt((noise_v**2 / (P_amp**2)) + ((v_val**2) * (sigma_P**2) / (P_amp**2 + 1e-20)))
        denom = np.sqrt(1.0 - (v_val / P_amp)**2 + 1e-20)
        ea_sigma_rad = 0.5 * (sigma_VoverP / denom)
        ea_err_array[i] = np.degrees(ea_sigma_rad)

        P_frac_bins[i] = P_amp / (i_val + 1e-10)
        L_frac_bins[i] = np.sqrt(q_val**2 + u_val**2) / (i_val + 1e-10)
        V_frac_bins[i] = v_val / (i_val + 1e-10)

        # Store polarisation angle at reference frequency
        ref_idx = len(freq_fit) // 2
        pol_angle_ref_array[i] = fitter.pol_angle[ref_idx]

        # Fit based on method
        if method == 'simple':
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

    valid_bins = i_snr_array >= 2.0

    if valid_bins.size == rm_array.size:
        rm_array[~valid_bins] = np.nan
        rm_err_array[~valid_bins] = np.nan
        snr_array[~valid_bins] = np.nan

    bad_pa = pa_err_array > 50.0
    bad_ea = ea_err_array > 50.0
    bad_bins = (~valid_bins) | bad_pa | bad_ea
    pa_ea_valid = ~bad_bins

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
        'pa_ea_valid': pa_ea_valid,
    }

    return results
