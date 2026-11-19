"""DM uncertainty estimation utilities."""

from typing import Dict, Optional, Tuple

import numpy as np

class UncertaintyMixin:
	@staticmethod
	def _robust_vmin_vmax(data: np.ndarray, low: float = 5.0, high: float = 99.0) -> Tuple[float, float]:
		values = np.asarray(data, dtype=float)
		finite = values[np.isfinite(values)]
		if finite.size == 0:
			return 0.0, 1.0
		vmin = float(np.percentile(finite, low))
		vmax = float(np.percentile(finite, high))
		if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
			vmin = float(np.min(finite))
			vmax = float(np.max(finite))
			if vmin == vmax:
				pad = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
				vmin -= pad
				vmax += pad
		return vmin, vmax

	@staticmethod
	def _format_dm(dm: float, precision: int = 6) -> str:
		return f"{dm:.{precision}f}".rstrip('0').rstrip('.')

	@staticmethod
	def _uncertainty_dict(best_dm: float,
						 low_dm: Optional[float],
						 high_dm: Optional[float],
						 method: str) -> Dict[str, Optional[float]]:
		minus = None if low_dm is None else float(best_dm - low_dm)
		plus = None if high_dm is None else float(high_dm - best_dm)
		return {
			'uncertainty_low_dm': None if low_dm is None else float(low_dm),
			'uncertainty_high_dm': None if high_dm is None else float(high_dm),
			'uncertainty_minus': minus,
			'uncertainty_plus': plus,
			'uncertainty_method': method,
		}

	@staticmethod
	def _format_error(err: float) -> str:
		"""2 dp, unless that rounds to zero; then 1 significant figure."""
		err = abs(float(err))
		if round(err, 2) != 0.0 or err == 0.0:
			return f"{err:.2f}"
		return f"{float(f'{err:.1g}'):.{max(0, -int(np.floor(np.log10(err))))}f}"

	@staticmethod
	def _format_uncertainty(best_dm: float,
						 minus: Optional[float],
						 plus: Optional[float],
						 precision: int = 6,
						 compact_errors: bool = False) -> str:
		fmt_err = (UncertaintyMixin._format_error if compact_errors
				   else (lambda v: UncertaintyMixin._format_dm(v, precision)))
		best = UncertaintyMixin._format_dm(best_dm, precision)
		m = "?" if minus is None else fmt_err(minus)
		p = "?" if plus is None else fmt_err(plus)
		return f"{best} (-{m}/+{p})"

	@staticmethod
	def _dm_crossing(values: np.ndarray,
					 dm_values: np.ndarray,
					 start_idx: int,
					 threshold: float,
					 direction: int) -> Optional[float]:
		n = len(values)
		i = int(start_idx)
		if direction < 0:
			while i > 0 and values[i] >= threshold:
				i -= 1
			if values[i] >= threshold:
				return None
			i_below = i
			i_above = i + 1
		else:
			while i < n - 1 and values[i] >= threshold:
				i += 1
			if values[i] >= threshold:
				return None
			i_below = i
			i_above = i - 1

		x1 = float(dm_values[i_below])
		x2 = float(dm_values[i_above])
		y1 = float(values[i_below])
		y2 = float(values[i_above])
		if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
			return float(dm_values[i_above])
		frac = (threshold - y1) / (y2 - y1)
		frac = float(np.clip(frac, 0.0, 1.0))
		return x1 + frac * (x2 - x1)

	@staticmethod
	def _finite_metric_arrays(dm_values: np.ndarray, metric_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		dm = np.asarray(dm_values, dtype=float)
		metric = np.asarray(metric_values, dtype=float)
		valid = np.isfinite(dm) & np.isfinite(metric)
		if np.sum(valid) < 2:
			raise ValueError("Need at least two finite DM/metric points for uncertainty estimation")
		return dm[valid], metric[valid]

	@staticmethod
	def _clamp_uncertainty_to_dm_bounds(
							best_dm: float,
							uncertainty: Dict[str, Optional[float]],
							dm_values: np.ndarray,
							fill_missing_with_bounds: bool = False) -> Dict[str, Optional[float]]:
		"""
		Clamp uncertainty DM bounds to the available DM sample range.
		"""
		dm = np.asarray(dm_values, dtype=float)
		finite = dm[np.isfinite(dm)]
		if finite.size == 0:
			return uncertainty

		dm_min = float(np.min(finite))
		dm_max = float(np.max(finite))
		best_dm = float(best_dm)

		low_dm = uncertainty.get('uncertainty_low_dm')
		high_dm = uncertainty.get('uncertainty_high_dm')
		if fill_missing_with_bounds:
			if low_dm is None:
				low_dm = dm_min
			if high_dm is None:
				high_dm = dm_max

		low_clamped = None if low_dm is None else float(np.clip(float(low_dm), dm_min, dm_max))
		high_clamped = None if high_dm is None else float(np.clip(float(high_dm), dm_min, dm_max))

		if low_clamped is not None:
			low_clamped = min(low_clamped, best_dm)
		if high_clamped is not None:
			high_clamped = max(high_clamped, best_dm)

		clamped = dict(uncertainty)
		clamped['uncertainty_low_dm'] = low_clamped
		clamped['uncertainty_high_dm'] = high_clamped
		clamped['uncertainty_minus'] = None if low_clamped is None else float(best_dm - low_clamped)
		clamped['uncertainty_plus'] = None if high_clamped is None else float(high_clamped - best_dm)
		return clamped

	def _uncertainty_from_half_prominence(self,
							 dm_values: np.ndarray,
							 metric_values: np.ndarray,
							 best_idx: int) -> Dict[str, Optional[float]]:
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx_eff = int(np.argmax(metric))
		best_dm = float(dm[best_idx_eff])
		metric_max = float(np.max(metric))
		metric_min = float(np.min(metric))
		if not np.isfinite(metric_max) or not np.isfinite(metric_min):
			return self._uncertainty_dict(best_dm, None, None, "half-prominence")

		if metric_max <= metric_min:
			step = float(np.median(np.diff(dm))) if len(dm) > 1 else 0.0
			return self._uncertainty_dict(
				best_dm,
				best_dm - 0.5 * step if step > 0 else None,
				best_dm + 0.5 * step if step > 0 else None,
				"half-prominence",
			)

		threshold = metric_min + 0.5 * (metric_max - metric_min)
		low_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=-1)
		high_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=1)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "half-prominence")

	def _uncertainty_from_local_quadratic(self,
						 dm_values: np.ndarray,
						 metric_values: np.ndarray,
						 best_idx: int,
						 target_points: int = 11) -> Dict[str, Optional[float]]:
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx_eff = int(np.argmax(metric))
		best_dm = float(dm[best_idx_eff])
		n = len(dm)
		if n < 5:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		points = int(max(5, min(target_points, n)))
		half = points // 2
		start = max(0, best_idx_eff - half)
		end = min(n, start + points)
		start = max(0, end - points)
		x = dm[start:end]
		y = metric[start:end]
		if x.size < 5:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		try:
			coeffs = np.polyfit(x, y, 2)
		except Exception:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		a = float(coeffs[0])
		if not np.isfinite(a) or a >= 0:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		y_fit = np.polyval(coeffs, x)
		resid = y - y_fit
		med = float(np.median(resid))
		mad = float(np.median(np.abs(resid - med)))
		sigma = 1.4826 * mad if mad > 0 else float(np.std(resid))
		if not np.isfinite(sigma) or sigma <= 0:
			step = float(np.median(np.diff(dm))) if n > 1 else 0.0
			if step > 0:
				return self._uncertainty_dict(
					best_dm,
					best_dm - 0.5 * step,
					best_dm + 0.5 * step,
					"local quadratic",
				)
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		width = float(np.sqrt(sigma / -a))
		if not np.isfinite(width) or width <= 0:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")
		max_width = 0.5 * float(x[-1] - x[0]) if x.size > 1 else 0.0
		if max_width > 0:
			width = min(width, max_width)

		low_dm = float(best_dm - width)
		high_dm = float(best_dm + width)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "local quadratic (1-sigma)")

	def _uncertainty_from_peak_fwhm(self,
						 dm_values: np.ndarray,
						 metric_values: np.ndarray,
						 frac: float = 0.001) -> Dict[str, Optional[float]]:
		"""
		Estimate uncertainty as the width of the region within `frac` of the peak value.
		Appropriate for broad, smooth metric profiles (e.g. L/I) where the peak is
		not sharp enough for quadratic or SHRINE-based methods.

		Parameters
		----------
		frac : float
			Fractional threshold below peak, e.g. 0.001 = within 0.1% of peak.
		"""
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx = int(np.argmax(metric))
		best_dm = float(dm[best_idx])
		frac_use = float(np.clip(frac, 0.0, 1.0))
		threshold = float(metric[best_idx]) * (1.0 - frac_use)

		low_idx = best_idx
		while low_idx > 0 and metric[low_idx - 1] >= threshold:
			low_idx -= 1

		high_idx = best_idx
		while high_idx < len(metric) - 1 and metric[high_idx + 1] >= threshold:
			high_idx += 1

		low_dm = float(dm[low_idx])
		high_dm = float(dm[high_idx])
		return self._uncertainty_dict(best_dm, low_dm, high_dm, f"peak width ({frac_use*100:.1f}% threshold)")

	def _uncertainty_from_snr_drop(self,
						 dm_values: np.ndarray,
						 snr_values: np.ndarray,
						 best_idx: int,
						 drop: float = 1.0) -> Dict[str, Optional[float]]:
		return self._uncertainty_from_metric_drop(
			dm_values,
			snr_values,
			drop=drop,
			method_label=f"S/N drop = {float(drop):g}",
		)

	def _uncertainty_from_metric_drop(self,
						 dm_values: np.ndarray,
						 metric_values: np.ndarray,
						 drop: float,
						 method_label: str = "metric drop") -> Dict[str, Optional[float]]:
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx_eff = int(np.argmax(metric))
		best_dm = float(dm[best_idx_eff])
		metric_max = float(metric[best_idx_eff])
		threshold = metric_max - float(drop)
		low_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=-1)
		high_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=1)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, method_label)

	def _uncertainty_from_metric_errors(self,
							   dm_values: np.ndarray,
							   metric_values: np.ndarray,
							   metric_errors: np.ndarray,
							   best_idx: int) -> Dict[str, Optional[float]]:
		"""
		Estimate DM uncertainty from propagated metric errors.

		For each DM, the metric is compared with the peak value: if
		|metric_max - metric_i| <= sqrt(err_max^2 + err_i^2)
		the DM is considered consistent with the best-fit DM.
		The uncertainty bounds are the outermost DM values where this
		condition holds.
		"""
		dm = np.asarray(dm_values, dtype=float)
		metric = np.asarray(metric_values, dtype=float)
		err = np.asarray(metric_errors, dtype=float)
		best_dm = float(dm[int(best_idx)])
		best_metric = float(metric[int(best_idx)])
		best_err = float(err[int(best_idx)])

		valid = np.isfinite(metric) & np.isfinite(err) & (err >= 0)
		if np.sum(valid) < 2:
			return self._uncertainty_dict(best_dm, None, None, "metric error")

		combined = np.sqrt(best_err**2 + err**2)
		consistent = valid & (np.abs(metric - best_metric) <= combined)

		if not np.any(consistent):
			return self._uncertainty_dict(best_dm, None, None, "metric error")

		consistent_idx = np.where(consistent)[0]
		low_dm = float(dm[int(consistent_idx[0])])
		high_dm = float(dm[int(consistent_idx[-1])])
		return self._uncertainty_dict(
			best_dm,
			low_dm,
			high_dm,
			"metric error",
		)

	def _uncertainty_from_pa_slope_errors(self,
							   dm_values: np.ndarray,
							   slope_values: np.ndarray,
							   slope_errors: np.ndarray,
							   best_idx: int) -> Dict[str, Optional[float]]:
		"""Deprecated: use _uncertainty_from_metric_errors instead."""
		result = self._uncertainty_from_metric_errors(
			dm_values, slope_values, slope_errors, best_idx,
		)
		if result.get('uncertainty_method') == 'metric error':
			result['uncertainty_method'] = 'PA slope error'
		return result

	def _uncertainty_from_sigma_dm(self,
								   dm_best: float,
								   sigma_dm: Optional[float],
								   why: Optional[str] = None) -> Dict[str, Optional[float]]:
		"""
		The DM uncertainty of structure maximisation: dm_best +/- sigma_DM, the 1-sigma
		uncertainty of SHRINE-II (Balzan et al., in prep., Eq. final). None on both sides when
		there is no sigma_DM (why says why).
		"""
		if sigma_dm is None:
			method = "SHRINE-II sigma_DM: none"
			if why:
				method = f"{method} ({why})"
			return self._uncertainty_dict(dm_best, None, None, method)
		return self._uncertainty_dict(dm_best, dm_best - sigma_dm, dm_best + sigma_dm,
									  "SHRINE-II sigma_DM (1 sigma)")
