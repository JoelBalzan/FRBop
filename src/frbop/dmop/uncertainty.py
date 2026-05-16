"""DM uncertainty estimation utilities."""

import contextlib
import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fftpack import dct

from .common import (
    shrine_get_kc,
    shrine_get_ranges_above_max,
    shrine_lowpass_smooth,
    shrine_uncertainty_calc,
)

class UncertaintyMixin:
	@staticmethod
	def _robust_vmin_vmax(data: np.ndarray, low: float = 5.0, high: float = 99.0) -> Tuple[float, float]:
		vmin = float(np.percentile(data, low))
		vmax = float(np.percentile(data, high))
		if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
			vmin = float(np.min(data))
			vmax = float(np.max(data))
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
	def _format_uncertainty(best_dm: float,
						 minus: Optional[float],
						 plus: Optional[float],
						 precision: int = 6) -> str:
		best = UncertaintyMixin._format_dm(best_dm, precision)
		if minus is None and plus is None:
			return f"{best} (-?/+?)"
		if minus is None:
			return f"{best} (-?/+{UncertaintyMixin._format_dm(plus, precision)})"
		if plus is None:
			return f"{best} (-{UncertaintyMixin._format_dm(minus, precision)}/+?)"
		return (
			f"{best} "
			f"(-{UncertaintyMixin._format_dm(minus, precision)}/+{UncertaintyMixin._format_dm(plus, precision)})"
		)

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

	def _uncertainty_from_snr_drop(self,
						 dm_values: np.ndarray,
						 snr_values: np.ndarray,
						 best_idx: int,
						 drop: float = 1.0) -> Dict[str, Optional[float]]:
		dm, sn = self._finite_metric_arrays(dm_values, snr_values)
		best_idx_eff = int(np.argmax(sn))
		best_dm = float(dm[best_idx_eff])
		max_sn = float(sn[best_idx_eff])
		threshold = max_sn - float(drop)
		low_dm = self._dm_crossing(sn, dm, best_idx_eff, threshold, direction=-1)
		high_dm = self._dm_crossing(sn, dm, best_idx_eff, threshold, direction=1)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "S/N drop = 1")

	def _structure_uncertainty_from_shrine_outputs(self,
									 dm_values: np.ndarray,
									 structure_values: np.ndarray,
									 run_dir: Path,
									 run_prefix: str,
									 best_idx: int) -> Dict[str, Optional[float]]:
		rel_path = run_dir / f"{run_prefix}_Relative_Uncertainties.dat"
		if not rel_path.exists():
			best_dm = float(dm_values[int(best_idx)])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		rel = np.asarray(np.loadtxt(rel_path), dtype=float)
		sp = np.asarray(structure_values, dtype=float)
		dm = np.asarray(dm_values, dtype=float)
		if rel.shape != sp.shape:
			best_dm = float(dm[int(best_idx)])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		max_index = int(best_idx)
		max_structure_parameter = float(sp[max_index])
		adjusted_sps = sp + (sp * rel)
		possible_max_ranges = shrine_get_ranges_above_max(max_structure_parameter, adjusted_sps)

		if len(possible_max_ranges) < 1:
			best_dm = float(dm[max_index])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		low_idx = int(possible_max_ranges[0][0])
		low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
		high_dm = None
		if len(possible_max_ranges[-1]) == 2:
			high_idx = int(possible_max_ranges[-1][1])
			if 0 <= high_idx < len(dm):
				high_dm = float(dm[high_idx])

		return self._uncertainty_dict(float(dm[max_index]), low_dm, high_dm, "SHRINE relative uncertainty")

	def _nonshrine_uncertainty_reference_profile(self,
										 data_i: Optional[np.ndarray],
										 data_q: Optional[np.ndarray],
										 data_u: Optional[np.ndarray]) -> np.ndarray:
		"""
		Build a per-DM reference time profile for SHRINE-style uncertainty.
		"""
		if data_i is not None:
			return np.nansum(data_i, axis=0)
		if data_q is not None and data_u is not None:
			linear = np.sqrt(data_q**2 + data_u**2)
			return np.nansum(linear, axis=0)
		if data_q is not None:
			return np.nansum(data_q, axis=0)
		if data_u is not None:
			return np.nansum(data_u, axis=0)
		raise ValueError("At least one of data_i/data_q/data_u must be provided")

	def _li_uncertainty_reference_profile(self,
								 data_q: np.ndarray,
								 data_u: np.ndarray,
								 data_i: np.ndarray) -> np.ndarray:
		"""
		Build a SHRINE-compatible time profile for L/I uncertainty.

		Using the Stokes I mean profile matches SHRINE's uncertainty recipe and
		avoids flat L/I profiles inflating the error range.
		"""
		profile = self._nonshrine_uncertainty_reference_profile(data_i, data_q, data_u)
		profile = np.asarray(profile, dtype=float)
		profile[~np.isfinite(profile)] = 0.0
		return profile

	def _uncertainty_from_shrine_relative(self,
								 dm_values: np.ndarray,
								 metric_values: np.ndarray,
								 reference_profiles: np.ndarray,
								 kc: Optional[int] = None) -> Dict[str, Optional[float]]:
		"""
		Estimate uncertainty using the same relative-uncertainty recipe as SHRINE.
		"""
		dm = np.asarray(dm_values, dtype=float)
		metric = np.asarray(metric_values, dtype=float)
		profiles = np.asarray(reference_profiles, dtype=float)

		if profiles.ndim != 2 or profiles.shape[0] != dm.shape[0] or metric.shape[0] != dm.shape[0]:
			best_idx = int(np.argmax(metric))
			return self._uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

		# Replace non-finite values row-wise so DCT/filtering remains stable.
		profiles_finite = profiles.copy()
		for row_idx in range(profiles_finite.shape[0]):
			row = profiles_finite[row_idx]
			finite = np.isfinite(row)
			if np.any(finite):
				fill_value = float(np.nanmean(row[finite]))
				profiles_finite[row_idx, ~finite] = fill_value
			else:
				profiles_finite[row_idx, :] = 0.0

		ci_data = dct(profiles_finite, norm='ortho')
		k_len = ci_data.shape[1]
		if k_len < 2:
			best_idx = int(np.argmax(metric))
			return self._uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

		if kc is None:
			with contextlib.redirect_stdout(io.StringIO()):
				kc_use = int(shrine_get_kc(ci_data))
		else:
			kc_use = int(kc)
		kc_use = max(1, min(kc_use, k_len))

		i_smooth, lpf_data, _, f_l = shrine_lowpass_smooth(ci_data, kc_use, order=3)
		k = np.linspace(1, k_len, k_len)
		hp = np.sqrt(2 - 2 * np.cos((k - 1) * np.pi / k_len))
		filter_diag = np.diag(hp * f_l)

		delta_i = profiles_finite - i_smooth
		max_idx = int(np.argmax(metric))
		delta_delta_i = delta_i - delta_i[max_idx]

		relative_uncertainty = shrine_uncertainty_calc(delta_delta_i, lpf_data, filter_diag)
		relative_uncertainty = np.asarray(relative_uncertainty, dtype=float)
		relative_uncertainty[~np.isfinite(relative_uncertainty)] = 0.0

		max_metric = float(metric[max_idx])
		adjusted_metrics = metric + (metric * relative_uncertainty)
		possible_max_ranges = shrine_get_ranges_above_max(max_metric, adjusted_metrics)

		if len(possible_max_ranges) < 1:
			return self._uncertainty_dict(float(dm[max_idx]), None, None, "SHRINE relative uncertainty")

		low_idx = int(possible_max_ranges[0][0])
		low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
		high_dm = None
		if len(possible_max_ranges[-1]) == 2:
			high_idx = int(possible_max_ranges[-1][1])
			if 0 <= high_idx < len(dm):
				high_dm = float(dm[high_idx])

		return self._uncertainty_dict(float(dm[max_idx]), low_dm, high_dm, "SHRINE relative uncertainty")


