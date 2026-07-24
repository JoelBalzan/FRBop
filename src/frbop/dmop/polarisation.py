"""Polarisation angle and Stokes Q/U analysis."""

from typing import Optional, Tuple

import numpy as np


class PolarisationMixin:
	@staticmethod
	def _noise_stats_from_series(series: np.ndarray) -> Tuple[float, float]:
		n_edge = max(1, int(0.05 * len(series)))
		noise_region = series[:n_edge]
		return float(np.median(noise_region)), float(np.std(noise_region))

	@staticmethod
	def _apply_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
		valid = mask.astype(int)
		dv = np.diff(np.concatenate(([0], valid, [0])))
		starts = np.where(dv == 1)[0]
		ends = np.where(dv == -1)[0]
		keep = np.zeros_like(mask, dtype=bool)
		for s, e in zip(starts, ends):
			if (e - s) >= min_run:
				keep[s:e] = True
		return keep

	@staticmethod
	def _longest_true_run(mask: np.ndarray) -> int:
		valid = mask.astype(int)
		dv = np.diff(np.concatenate(([0], valid, [0])))
		starts = np.where(dv == 1)[0]
		ends = np.where(dv == -1)[0]
		if len(starts) == 0:
			return 0
		return int(np.max(ends - starts))

	def _pa_slope_metric_shrine(self, data_q: np.ndarray, data_u: np.ndarray,
								  time_ms: Optional[np.ndarray] = None,
								  data_i: Optional[np.ndarray] = None,
								  return_error: bool = False):
		"""
		PA slope metric where PA is SHRINE-smoothed before fitting.
		"""
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		pa_shrine_smooth, _, time_axis = self._get_pa_shrine_smoothed_and_fit(data_q, data_u, data_i, time_ms)
		valid = np.isfinite(pa_shrine_smooth)
		if np.sum(valid) < (self.pa_fit_degree + 1):
			return (0.0, 0.0) if return_error else 0.0

		weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
		w = weights[valid]
		if np.sum(w > 0) < (self.pa_fit_degree + 1):
			return (0.0, 0.0) if return_error else 0.0

		x = time_axis[valid]
		y = pa_shrine_smooth[valid]
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

	def _pa_fit_weights(self, L_debias: np.ndarray, sigma_L: float,
						data_i: Optional[np.ndarray], valid: np.ndarray) -> np.ndarray:
		"""
		Build per-time-sample PA fit weights so lower-S/N regions contribute less.
		"""
		eps = 1e-12
		w_l = np.maximum(L_debias / np.maximum(sigma_L, eps), 0.0)

		weights = w_l

		#if data_i is not None:
		#	i_ts = np.nansum(data_i, axis=0)
		#	i_noise_std = float(self.full_i_noise_std)
		#	i_noise_med = float(self.full_i_noise_median)
		#	if i_noise_std > 0:
		#		w_i = np.maximum((i_ts - i_noise_med) / i_noise_std, 0.0)
		#		weights = weights * w_i

		weights = np.where(valid, weights, 0.0)
		max_w = float(np.max(weights)) if np.any(valid) else 0.0
		if max_w > 0:
			weights = weights / max_w
			if self.pa_weight_strength != 1.0:
				weights = np.power(weights, self.pa_weight_strength)

		return weights

	def _qu_noise_rms_from_full(self, data_q: np.ndarray, data_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Get Q/U noise RMS from full dynamic-spectrum statistics.
		"""
		if data_q.ndim == 1:
			if self.full_q_time_noise_std is not None and self.full_u_time_noise_std is not None:
				return float(self.full_q_time_noise_std), float(self.full_u_time_noise_std)
			return self._qu_noise_rms(data_q, data_u)

		if (
			self.full_q_noise_rms is not None
			and self.full_u_noise_rms is not None
			and self.full_q_noise_rms.shape[0] == data_q.shape[0]
		):
			return self.full_q_noise_rms, self.full_u_noise_rms

		if self.full_q_time_noise_std is not None and self.full_u_time_noise_std is not None:
			q_rms = np.full((data_q.shape[0], 1), float(self.full_q_time_noise_std), dtype=float)
			u_rms = np.full((data_u.shape[0], 1), float(self.full_u_time_noise_std), dtype=float)
			return q_rms, u_rms

		return self._qu_noise_rms(data_q, data_u)

	@staticmethod
	def _qu_noise_rms(data_q: np.ndarray, data_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Estimate Q/U RMS from the first 5% of time samples.
		"""
		if data_q.ndim == 1:
			n_edge = max(1, int(0.05 * len(data_q)))
			q_rms = float(np.std(data_q[:n_edge]))
			u_rms = float(np.std(data_u[:n_edge]))
			return q_rms, u_rms

		# Expect shape (freq, time)
		n_edge = max(1, int(0.05 * data_q.shape[1]))
		q_rms = np.std(data_q[:, :n_edge], axis=1, keepdims=True)
		u_rms = np.std(data_u[:, :n_edge], axis=1, keepdims=True)
		return q_rms, u_rms

	def _linear_dspec_from_qu(self, data_q: np.ndarray, data_u: np.ndarray) -> np.ndarray:
		"""
		Build a debiased linear-polarisation dynamic spectrum (freq x time).
		"""
		q_rms, u_rms = self._qu_noise_rms_from_full(data_q, data_u)
		L_dspec, _, _ = self._debiased_linear_from_qu(data_q, data_u, q_rms, u_rms)
		return L_dspec

	def _linear_time_profile_from_qu(self, data_q: np.ndarray, data_u: np.ndarray) -> np.ndarray:
		"""
		Integrate the linear-polarisation dynamic spectrum over frequency.
		"""
		return np.nansum(self._linear_dspec_from_qu(data_q, data_u), axis=0)

	def _debiased_linear_from_qu(self, data_q: np.ndarray, data_u: np.ndarray,
						   q_rms: np.ndarray, u_rms: np.ndarray,
						   cutoff: float = 1.57, eps: float = 1e-12,
						   debias: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Optionally debias linear polarisation using propagated sigma_L and detection cutoff.
		"""
		if debias is None:
			debias = self.debias_linear
		L_meas = np.sqrt(data_q**2 + data_u**2)
		sigma_L = np.sqrt(data_q**2 * q_rms**2 + data_u**2 * u_rms**2) / np.maximum(L_meas, eps)
		r = L_meas / np.maximum(sigma_L, eps)
		det = r >= cutoff

		if debias:
			L_out = np.zeros_like(L_meas)
			L_out[det] = np.sqrt(np.maximum(L_meas[det]**2 - sigma_L[det]**2, 0.0))
		else:
			L_out = L_meas
		return L_out, sigma_L, det

	def _pa_series_deg(self, data_q: np.ndarray, data_u: np.ndarray, data_i: Optional[np.ndarray] = None) -> np.ndarray:
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)
		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0
		mask = L_debias >= (2.0 * sigma_L)
		
		# Apply Stokes I significance cutoff; optionally also restrict to >= peak
		if data_i is not None:
			i_ts = np.nansum(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask
		
		pa_deg = np.where(mask, pa_deg, np.nan)

		# Drop short valid runs
		min_run = int(getattr(self, "pa_min_run", 3))
		if np.any(mask):
			keep_run = self._apply_min_run(mask, min_run)
			pa_deg = np.where(keep_run, pa_deg, np.nan)

		return pa_deg
	
	def _get_pa_smoothed_and_fit(self, data_q: np.ndarray, data_u: np.ndarray, data_i: Optional[np.ndarray] = None, time_ms: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Get masked PA profile and best fit line for plotting.
		
		Parameters:
		-----------
		data_q : np.ndarray
			Stokes Q data
		data_u : np.ndarray
			Stokes U data
		data_i : np.ndarray, optional
			Stokes I data for S/N masking
		time_ms : np.ndarray, optional
			Time axis
		
		Returns:
		--------
		pa_smooth : np.ndarray
			Masked PA values (no Gaussian smoothing)
		fit_line : np.ndarray
			Best fit line values
		time_axis : np.ndarray
			Time axis for the data
		"""
		q_ts = np.nansum(data_q, axis=0)
		u_ts = np.nansum(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)
		
		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0
		
		# Mask based on L significance
		mask = L_debias >= (2.0 * sigma_L)
		
		# Apply Stokes I significance cutoff; optionally also restrict to >= peak
		if data_i is not None:
			i_ts = np.nansum(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask
		
		min_run = int(getattr(self, "pa_min_run", 3))
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)
		
		valid = mask & np.isfinite(pa_deg)
		
		# Create time axis
		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms
		
		# Apply mask to PA for plotting/fit diagnostics.
		pa_deg_masked = np.where(mask, pa_deg, np.nan)

		# Regular PA method now uses unsmoothed masked PA.
		pa_smooth = pa_deg_masked.copy()
		
		# Compute best fit line
		fit_line = np.full_like(pa_deg, np.nan)
		min_points = self.pa_fit_degree + 1
		if np.sum(valid) >= min_points:
			weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
			weights_valid = weights[valid]
			if np.sum(weights_valid > 0) >= min_points:
				coeffs = np.polyfit(time_axis[valid], pa_deg[valid], self.pa_fit_degree, w=weights_valid)
				fit_line = np.polyval(coeffs, time_axis)
				# Only show fit line where mask is valid
				fit_line = np.where(mask, fit_line, np.nan)
		
		return pa_smooth, fit_line, time_axis

	def _get_pa_shrine_smoothed_and_fit(self, data_q: np.ndarray, data_u: np.ndarray,
									 data_i: Optional[np.ndarray] = None,
									 time_ms: Optional[np.ndarray] = None,
									 force_kc: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Get SHRINE-smoothed PA profile and best fit line for plotting.
		The smoothing is applied to the PA time series itself, then fitting is done on the SHRINE-smoothed PA.
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
		if data_i is not None:
			i_ts = np.nansum(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask

		min_run = int(getattr(self, "pa_min_run", 3))
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)

		valid = mask & np.isfinite(pa_deg)

		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms

		pa_deg_masked = np.where(mask, pa_deg, np.nan)
		kc_source_mask = (L_debias >= (2.0 * sigma_L)) & np.isfinite(pa_deg)
		pa_deg_for_kc = np.where(kc_source_mask, pa_deg, np.nan)
		if np.sum(np.isfinite(pa_deg_for_kc)) < 2:
			pa_deg_for_kc = np.where(np.isfinite(pa_deg), pa_deg, np.nan)
		pa_shrine_smooth = np.full_like(pa_deg, np.nan)
		fit_line = np.full_like(pa_deg, np.nan)
		min_points = self.pa_fit_degree + 1
		if np.sum(valid) >= min_points:
			valid_idx = np.where(np.isfinite(pa_deg_for_kc))[0]
			if valid_idx.size == 1:
				pa_fill = np.full_like(pa_deg, pa_deg_for_kc[valid_idx[0]], dtype=float)
			elif valid_idx.size >= 2:
				all_idx = np.arange(len(pa_deg))
				pa_fill = np.interp(all_idx, valid_idx, pa_deg_for_kc[valid_idx])
			else:
				pa_fill = np.zeros_like(pa_deg, dtype=float)

			if force_kc is not None:
				kc = int(force_kc)
			else:
				kc = self.resolve_nonshrine_kc(pa_fill[np.newaxis, :])
			pa_shrine = self.apply_kc_lowpass_2d(pa_fill[np.newaxis, :], kc)[0]
			pa_shrine_smooth = np.where(np.isfinite(pa_deg_masked), pa_shrine, np.nan)

			weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
			weights_valid = weights[valid]
			if np.sum(weights_valid > 0) >= min_points:
				coeffs = np.polyfit(time_axis[valid], pa_shrine_smooth[valid], self.pa_fit_degree, w=weights_valid)
				fit_line = np.polyval(coeffs, time_axis)
				fit_line = np.where(mask, fit_line, np.nan)

		return pa_shrine_smooth, fit_line, time_axis
	

