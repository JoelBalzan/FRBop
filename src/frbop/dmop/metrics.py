"""DM optimisation metric functions."""

from typing import Optional, Tuple

import numpy as np

class MetricsMixin:
	@staticmethod
	def _fwhm_window(profile: np.ndarray, width_factor: float = 1.0) -> tuple:
		n = len(profile)
		if n == 0:
			return 0, 0
		peak_idx = int(np.nanargmax(profile))
		peak_val = float(profile[peak_idx])
		if not np.isfinite(peak_val) or peak_val <= 0:
			return 0, n - 1
		half = peak_val / 2.0
		left = 0
		for i in range(peak_idx, -1, -1):
			if not np.isfinite(profile[i]) or profile[i] <= half:
				left = i
				break
		right = n - 1
		for i in range(peak_idx, n):
			if not np.isfinite(profile[i]) or profile[i] <= half:
				right = i
				break
		if width_factor != 1.0:
			fwhm = max(1, right - left)
			half_extent = int(round(fwhm * width_factor / 2.0))
			left = max(0, peak_idx - half_extent)
			right = min(n - 1, peak_idx + half_extent)
		return left, right

	@staticmethod
	def _debiased_linear_error_sum(L_debias: np.ndarray, sigma_L: np.ndarray,
								   mask: np.ndarray) -> Tuple[float, float]:
		valid = mask & np.isfinite(L_debias) & np.isfinite(sigma_L)
		if not np.any(valid):
			return 0.0, 0.0
		total = float(np.sum(L_debias[valid]))
		err = float(np.sqrt(np.sum(sigma_L[valid]**2)))
		return total, err

	@staticmethod
	def _mean_linear_error(L_debias: np.ndarray, sigma_L: np.ndarray,
						   mask: np.ndarray) -> Tuple[float, float]:
		valid = mask & np.isfinite(L_debias) & np.isfinite(sigma_L)
		n = int(np.sum(valid))
		if n == 0:
			return 0.0, 0.0
		mean_val = float(np.mean(L_debias[valid]))
		err = float(np.sqrt(np.sum(sigma_L[valid]**2)) / n)
		return mean_val, err

	def _build_dm_values(self, dm_range: Tuple[float, float], n_points: int = 200,
						 dm_step: Optional[float] = None) -> np.ndarray:
		if dm_step is not None:
			if dm_step <= 0:
				raise ValueError("dm_step must be positive")
			span = dm_range[1] - dm_range[0]
			n_points = max(2, int(np.floor(span / dm_step)) + 1)
			dm_values = dm_range[0] + np.arange(n_points) * dm_step
			if dm_values[-1] < dm_range[1] and (dm_range[1] - dm_values[-1]) > 0.5 * dm_step:
				dm_values = np.append(dm_values, dm_range[1])
		else:
			dm_values = np.linspace(dm_range[0], dm_range[1], n_points)
		return dm_values
	
	def snr_metric(self, data: np.ndarray) -> float:
		"""
		Calculate signal-to-noise ratio metric.
		
		Parameters:
		-----------
		data : np.ndarray
			Dedispersed data (freq x time)
			
		Returns:
		--------
		snr : float
			Signal-to-noise ratio (higher is better)
		"""
		# Collapse to time series
		time_series = np.nansum(data, axis=0)
		
		# Estimate noise from the first 5% of the full time series
		noise_std = self.full_i_noise_std
		
		if noise_std == 0:
			return 0.0
		
		# Signal is the peak
		signal = np.nanmax(time_series) - self.full_i_noise_median
		
		snr = signal / noise_std
		return snr
	
	def pa_slope_metric(self, data_q: np.ndarray, data_u: np.ndarray,
						time_ms: Optional[np.ndarray] = None,
						data_i: Optional[np.ndarray] = None,
						return_error: bool = False):
		"""
		Calculate weighted position angle (PA) slope magnitude metric.
		This requires Stokes Q and U for polarisation analysis.

		The metric uses a weighted polynomial fit (degree specified by pa_fit_degree)
		to PA vs time within a 2-sigma debiased-L mask (with min-run filtering), and
		returns the magnitude of the fitted slope coefficient.
		
		If data_i is provided, only fit to data >= the peak of the Stokes I profile.

		Parameters
		----------
		return_error : bool, optional
			If True, returns (slope_magnitude, slope_standard_error).
		"""
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

		mask = L_debias >= (2.0 * sigma_L)
		min_run = 5
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)

		if data_i is not None:
			i_ts = np.nansum(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.nanstd(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.nanmedian(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				if np.any(np.isfinite(i_ts)):
					peak_idx = int(np.nanargmax(i_ts))
				else:
					peak_idx = 0
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask

		valid = mask & np.isfinite(pa_deg)
		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.nanmedian(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms

		min_points = self.pa_fit_degree + 1
		if np.sum(valid) < min_points:
			return (0.0, 0.0) if return_error else 0.0

		min_contiguous = max(12, min_points)
		if self._longest_true_run(valid) < min_contiguous:
			return (0.0, 0.0) if return_error else 0.0

		weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
		positive = valid & (weights > 0)
		if np.sum(positive) < min_points:
			return (0.0, 0.0) if return_error else 0.0

		x = time_axis[positive]
		y = pa_deg[positive]
		w = weights[positive]

		try:
			coeffs, cov = np.polyfit(x, y, self.pa_fit_degree, w=w, cov='unscaled')
		except Exception:
			return (0.0, 0.0) if return_error else 0.0

		slope_magnitude = float(coeffs[0])
		if not np.isfinite(slope_magnitude):
			return (0.0, 0.0) if return_error else 0.0

		if return_error:
			slope_error = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) and cov[0, 0] > 0 else 0.0
			return slope_magnitude, slope_error
		return slope_magnitude
	
	def linear_to_stokes_i_metric(self, data_q: np.ndarray, data_u: np.ndarray, 
							   data_i: np.ndarray, mode: str = 'peak',
							   use_fwhm_window: bool = False,
							   return_error: bool = False):
		"""
		Calculate debiased L/I ratio metric.
		Uses noise-debiased linear polarisation and calculates fractional
		polarisation using different criteria.

		When mode='mean' and use_fwhm_window=True, the mean is computed over a
		fixed-width window given by the FWHM of the Stokes I profile, centred on
		the Stokes I peak.  This avoids the "flip-flop" that occurs when the
		set of above-threshold points changes between DM trials.

		When return_error=True and mode='mean', also returns the propagated
		uncertainty on the mean L/I from sigma_L.

		Parameters:
		-----------
		data_q : np.ndarray
			Stokes Q data (freq x time)
		data_u : np.ndarray
			Stokes U data (freq x time)
		data_i : np.ndarray
			Stokes I data (freq x time)
		mode : str, optional
			Calculation mode.
		use_fwhm_window : bool, optional
			If True (and mode='mean'), uses a FWHM-based window.
		return_error : bool, optional
			If True, returns (metric, error).
			
		Returns:
		--------
		metric : float or tuple
			Debiased L/I metric, or (metric, error) if return_error.
		"""
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		I_ts = np.nansum(data_i, axis=0)

		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		if mode == 'peak' or mode == 'max' or (mode == 'mean' and not use_fwhm_window):
			sigma_I = float(self.full_i_noise_std)
			i_peak = np.nanmax(I_ts)
			abs_threshold = self.li_i_peak_fraction * i_peak
			threshold = max(
				float(self.full_i_noise_median) + self.li_i_sigma_cut * sigma_I,
				abs_threshold
			)
			mask = I_ts > threshold
		elif mode == 'mean' and use_fwhm_window:
			left, right = self._fwhm_window(I_ts, width_factor=1.0)
			fwhm_mask = np.zeros_like(I_ts, dtype=bool)
			fwhm_mask[left:right + 1] = True
			sigma_I = float(self.full_i_noise_std)
			i_peak = np.nanmax(I_ts)
			abs_threshold = self.li_i_peak_fraction * i_peak
			threshold = max(
				float(self.full_i_noise_median) + self.li_i_sigma_cut * sigma_I,
				abs_threshold
			)
			mask = fwhm_mask #& (I_ts > threshold)
		else:
			raise ValueError(f"Unknown mode '{mode}'. Must be 'peak', 'mean', or 'max'.")

		L_over_I = np.divide(
			L_debias,
			I_ts,
			out=np.zeros_like(I_ts, dtype=float),
			where=mask,
		)
		np.clip(L_over_I, 0.0, 1.0, out=L_over_I)
			
		if mode == 'peak':
			if np.any(mask):
				masked_i = np.where(mask, I_ts, -np.inf)
				peak_idx = int(np.nanargmax(masked_i))
				metric = float(L_over_I[peak_idx])
				if return_error:
					err = float(sigma_L[peak_idx] / I_ts[peak_idx]) if I_ts[peak_idx] > 0 else 0.0
					return metric, err
				return metric
			return (0.0, 0.0) if return_error else 0.0
		elif mode == 'mean':
			if np.any(mask):
				metric = float(np.nanmean(L_over_I[mask]))
				if return_error:
					sigma_ratio = np.divide(
						sigma_L, I_ts,
						out=np.zeros_like(sigma_L, dtype=float),
						where=mask & (I_ts > 0),
					)
					n = int(np.sum(mask & np.isfinite(sigma_ratio)))
					err = float(np.sqrt(np.nansum(sigma_ratio[mask]**2)) / n) if n > 0 else 0.0
					return metric, err
				return metric
			return (0.0, 0.0) if return_error else 0.0
		elif mode == 'max':
			if np.any(mask):
				masked_l = np.where(mask, L_debias, -np.inf)
				max_L_idx = int(np.nanargmax(masked_l))
				metric = float(L_over_I[max_L_idx])
				if return_error:
					err = float(sigma_L[max_L_idx] / I_ts[max_L_idx]) if I_ts[max_L_idx] > 0 else 0.0
					return metric, err
				return metric
			return (0.0, 0.0) if return_error else 0.0
		else:
			raise ValueError(f"Unknown mode '{mode}'. Must be 'peak', 'mean', or 'max'.")

	def linear_polarisation_sum_metric(self, data_q: np.ndarray, data_u: np.ndarray,
									   return_error: bool = False):
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)
		mask = np.ones_like(L_debias, dtype=bool)
		total, err = self._debiased_linear_error_sum(L_debias, sigma_L, mask)
		if return_error:
			return total, err
		return total

