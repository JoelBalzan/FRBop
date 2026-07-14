"""Scattering-index estimation utilities."""

import numpy as np
from scipy.optimize import curve_fit

from frbop.scop.models import scattered_gaussian, linear

try:
    from lmfit import Model, Parameters
    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False


def fit_scattering_index_from_frequencies(
    ds,
    freq,
    time,
    onpulse_mask,
    ref_freq=None,
    band_regions=None,
    return_details=False,
):
    """Fit the scattering index by fitting tau at each frequency or frequency band.

    Parameters
    ----------
    ds : ndarray, shape (nfreq, ntime)
        Dynamic spectrum (on-pulse only).
    freq : ndarray, shape (nfreq,)
        Frequency axis.
    time : ndarray, shape (ntime,)
        Time axis.
    onpulse_mask : ndarray, shape (ntime,), bool
        Mask for on-pulse region.
    ref_freq : float, optional
        Reference frequency. Defaults to min frequency.
    band_regions : list of (int, int), optional
        List of (start, stop) index pairs defining frequency bands.
        When provided, profiles within each band are averaged before fitting.
        When None (default), each frequency channel is fitted independently.

    Returns (fitted_index, tau_at_ref, index_err, n_freq_fitted) or
    (None, None, None, 0) if the fit fails.
    """
    nfreq, ntime_burst = ds.shape
    t_burst = time[onpulse_mask]
    tau_values = []
    freq_values = []

    dt_ms = float(np.abs(time[1] - time[0])) if time.size > 1 else 1e-3
    burst_duration = float(t_burst[-1] - t_burst[0])

    if band_regions is not None:
        iterations = [(np.nanmean(ds[s:e, :], axis=0), float(np.nanmean(freq[s:e])))
                      for s, e in band_regions if e > s]
    else:
        iterations = [(ds[i, :], float(freq[i])) for i in range(nfreq)]

    fit_details = None
    if return_details:
        fit_details = {'freq': [], 'tau': [], 'profile': [], 'popt': []}

    for profile, freq_val in iterations:
        if profile.size < 5:
            continue

        prof_max_idx = int(np.argmax(profile))
        mu0 = float(t_burst[prof_max_idx])
        p_low = np.percentile(profile, 5)
        p_high = np.percentile(profile, 95)
        amp0 = max(1e-6, float(p_high - p_low))
        offset0 = float(p_low)
        width_guess = max(
            float(np.abs(t_burst[-1] - t_burst[0])) / 20.0, dt_ms
        )
        sigma0 = width_guess
        tau0 = width_guess

        # Bounds: [amp, mu, sigma, tau, offset]
        lower = [0.0, float(t_burst[0]), dt_ms * 0.5, dt_ms * 0.5, -np.inf]
        upper = [np.inf, float(t_burst[-1]), burst_duration * 0.5, burst_duration * 0.5, np.inf]

        try:
            popt, _ = curve_fit(
                scattered_gaussian,
                t_burst,
                profile,
                p0=[amp0, mu0, sigma0, tau0, offset0],
                bounds=(lower, upper),
                maxfev=5000,
            )
            tau_fit = float(popt[3])
            if tau_fit > 0:
                tau_values.append(tau_fit)
                freq_values.append(freq_val)
                if return_details and fit_details is not None:
                    fit_details['freq'].append(freq_val)
                    fit_details['tau'].append(tau_fit)
                    fit_details['profile'].append(profile.copy())
                    fit_details['popt'].append(popt.copy())
        except Exception:
            continue

    if len(tau_values) < 3:
        print("Warning: could not fit tau at enough frequencies to estimate index")
        if return_details:
            return None, None, None, 0, None
        return None, None, None, 0

    if ref_freq is None:
        ref_freq = float(np.nanmin(freq_values))

    tau_values = np.array(tau_values)
    freq_values = np.array(freq_values)

    # Compute geometric mean frequency for stable evaluation
    freq_geom_mean = float(np.exp(np.mean(np.log(freq_values))))

    # Fit power law using lmfit with MCMC if available, otherwise use polyfit
    log_freq = np.log(freq_values)
    log_tau = np.log(tau_values)

    # Estimate uncertainties in log space
    log_err = np.ones_like(log_freq) * 0.2

    try:
        # from https://github.com/fjankowsk/scatfit/tree/master
        if not HAS_LMFIT:
            raise ImportError("lmfit not available, using polyfit fallback")

        # Use lmfit for robust fitting with MCMC
        model = Model(linear)
        params = Parameters()
        params.add('slope', value=-4.0, vary=True)
        params.add('intercept', value=np.mean(log_tau), vary=True)

        # Least-squares fit first (initialization)
        fitresult_ls = model.fit(
            data=log_tau,
            x=log_freq,
            weights=1.0 / log_err,
            params=params,
            method="leastsq",
        )

        if not fitresult_ls.success:
            print("Warning: least-squares fit did not converge, using simpler approach")
            raise RuntimeError("LS fit failed")

        # MCMC fit with proper uncertainty estimation
        emcee_kws = dict(steps=2000, burn=700, thin=10, is_weighted=True, progress=False)
        emcee_params = fitresult_ls.params.copy()

        fitresult_mcmc = model.fit(
            data=log_tau,
            x=log_freq,
            weights=1.0 / log_err,
            params=emcee_params,
            method="emcee",
            fit_kws=emcee_kws,
        )

        # Extract results from MCMC
        alpha_fitted = fitresult_mcmc.best_values['slope']
        intercept_fitted = fitresult_mcmc.best_values['intercept']

        # Compute uncertainties from flatchain
        samples = fitresult_mcmc.flatchain
        slope_samples = samples['slope']

        # 16th and 84th percentiles for 1-sigma errors
        slope_err = np.std(slope_samples)

        # Warn if alpha is positive (unphysical for scattering)
        if alpha_fitted > 0:
            print(
                f"Warning: fitted alpha = {alpha_fitted:.3f} is positive (unphysical). "
                "Scattering index should be negative."
            )

        # Evaluate at geometric mean (most stable), then scale to ref_freq
        # linear model: log_tau = slope * log_freq + intercept
        log_tau_at_geom_mean = alpha_fitted * np.log(freq_geom_mean) + intercept_fitted
        tau_at_geom_mean = np.exp(np.clip(log_tau_at_geom_mean, -700, 700))
        tau_at_ref = tau_at_geom_mean * (ref_freq / freq_geom_mean) ** alpha_fitted

        if return_details:
            return alpha_fitted, tau_at_ref, slope_err, len(tau_values), fit_details
        return alpha_fitted, tau_at_ref, slope_err, len(tau_values)

    except Exception as e:
        # Fallback to simple polyfit if lmfit not available or MCMC fails
        print(f"Warning: MCMC fit issue ({e}), falling back to polyfit")
        try:
            coeffs = np.polyfit(log_freq, log_tau, 1)
            alpha_fitted = coeffs[0]
            c_fitted = coeffs[1]

            # Warn if alpha is positive (unphysical for scattering)
            if alpha_fitted > 0:
                print(
                    f"Warning: fitted alpha = {alpha_fitted:.3f} is positive (unphysical). "
                    "Scattering index should be negative."
                )

            # Evaluate at geometric mean (most stable), then scale to ref_freq
            # polyfit returns coeffs[0] * x + coeffs[1], i.e. alpha * log_freq + c
            log_tau_at_geom_mean = alpha_fitted * np.log(freq_geom_mean) + c_fitted
            tau_at_geom_mean = np.exp(np.clip(log_tau_at_geom_mean, -700, 700))
            tau_at_ref = tau_at_geom_mean * (ref_freq / freq_geom_mean) ** alpha_fitted

            # Simple error estimate
            log_tau_fit = np.polyval(coeffs, log_freq)
            residuals = log_tau - log_tau_fit
            rms_residual = np.sqrt(np.mean(residuals ** 2))
            alpha_err = rms_residual * np.sqrt(len(log_freq)) / np.sqrt(
                np.sum((log_freq - np.mean(log_freq)) ** 2)
            )

            if return_details:
                return alpha_fitted, tau_at_ref, alpha_err, len(tau_values), fit_details
            return alpha_fitted, tau_at_ref, alpha_err, len(tau_values)
        except Exception as e2:
            print(f"Warning: polyfit fallback also failed: {e2}")
            if return_details:
                return None, None, None, 0, None
            return None, None, None, 0
