"""DM optimisation metric functions."""

from typing import Optional, Tuple

import numpy as np

class MetricsMixin:
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
	
	def pa_slope_metric(self, data_q: np.ndarray, data_u: np.ndarray, time_ms: Optional[np.ndarray] = None, data_i: Optional[np.ndarray] = None) -> float:
		"""
		Calculate weighted position angle (PA) slope magnitude metric.
		This requires Stokes Q and U for polarisation analysis.

		The metric uses a weighted polynomial fit (degree specified by pa_fit_degree)
		to PA vs time within a 2-sigma debiased-L mask (with min-run filtering), and
		returns the magnitude of the fitted slope coefficient.
		
		If data_i is provided, only fit to data >= the peak of the Stokes I profile.
		"""
		q_ts = np.nanmean(data_q, axis=0)
		u_ts = np.nanmean(data_u, axis=0)
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
			return 0.0

		min_contiguous = max(12, min_points)
		if self._longest_true_run(valid) < min_contiguous:
			return 0.0

		weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
		positive = valid & (weights > 0)
		if np.sum(positive) < min_points:
			return 0.0

		x = time_axis[positive]
		y = pa_deg[positive]
		w = weights[positive]

		try:
			coeffs = np.polyfit(x, y, self.pa_fit_degree, w=w)
		except Exception:
			return 0.0

		slope_magnitude = float(np.abs(coeffs[0]))
		if not np.isfinite(slope_magnitude):
			return 0.0
		return slope_magnitude
	
	def linear_to_stokes_i_metric(self, data_q: np.ndarray, data_u: np.ndarray, 
							   data_i: np.ndarray, mode: str = 'peak') -> float:
		"""
		Calculate debiased L/I ratio metric.
		Uses noise-debiased linear polarisation and calculates fractional
		polarisation using different criteria.

		Only samples with Stokes I above the significance threshold are used in the
		ratio average, which prevents shift-created empty bins from dominating the
		result via clipped low-denominator ratios.

		Parameters:
		-----------
		data_q : np.ndarray
			Stokes Q data (freq x time)
		data_u : np.ndarray
			Stokes U data (freq x time)
		data_i : np.ndarray
			Stokes I data (freq x time)
		mode : str, optional
			Calculation mode: 'peak' (L/I at Stokes I peak), 'mean' (mean L/I across pulse),
			or 'max' (L/I at the point where L is maximum). Default is 'peak'.
			
		Returns:
		--------
		metric : float
			Debiased L/I metric (higher is better)
		"""
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		I_ts = np.nansum(data_i, axis=0)

		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, _, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		sigma_I = float(self.full_i_noise_std)
		threshold = float(self.full_i_noise_median) + self.li_i_sigma_cut * sigma_I
		mask = I_ts > threshold

		L_over_I = np.divide(
			L_debias,
			I_ts,
			out=np.zeros_like(I_ts, dtype=float),
			where=mask,
		)
		#np.clip(L_over_I, 0.0, 1.0, out=L_over_I)

		if mode == 'peak':
			if np.any(mask):
				masked_i = np.where(mask, I_ts, -np.inf)
				peak_idx = int(np.nanargmax(masked_i))
				return float(L_over_I[peak_idx])
			return 0.0
		elif mode == 'mean':
			if np.any(mask):
				return float(np.nanmean(L_over_I[mask]))
			else:
				return 0.0
		elif mode == 'max':
			if np.any(mask):
				masked_l = np.where(mask, L_debias, -np.inf)
				max_L_idx = int(np.nanargmax(masked_l))
				return float(L_over_I[max_L_idx])
			return 0.0
		else:
			raise ValueError(f"Unknown mode '{mode}'. Must be 'peak', 'mean', or 'max'.")

