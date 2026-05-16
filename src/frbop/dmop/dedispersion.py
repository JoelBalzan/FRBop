"""Dedispersion and delay-sample utilities."""

from typing import Optional, Tuple

import numpy as np

from .common import _NUMBA_AVAILABLE
from .numba_kernels import _apply_shifts_numba

class DedispersionMixin:
	def _full_noise_reference_data(self, data: np.ndarray) -> np.ndarray:
		"""
		Select the full dynamic-spectrum array that matches the provided data.
		"""
		if np.shares_memory(data, self.stokes_i):
			return self.stokes_i
		if self.stokes_q is not None and np.shares_memory(data, self.stokes_q):
			return self.stokes_q
		if self.stokes_u is not None and np.shares_memory(data, self.stokes_u):
			return self.stokes_u
		return self.stokes_i
	
	def _get_common_valid_region(self, data: np.ndarray, delay_samples: np.ndarray) -> Tuple[int, int]:
		"""
		Calculate the common valid region after dedispersion (for crop mode).
		
		Returns:
		--------
		start_idx : int
			Start index of common valid region
		end_idx : int
			End index of common valid region (exclusive)
		"""
		max_shift = int(np.max(delay_samples))
		min_shift = int(np.min(delay_samples))
		
		# Common valid region starts where the most-shifted channel begins
		# and ends where the least-shifted channel ends
		start_idx = max_shift
		end_idx = data.shape[1] + min_shift
		
		return start_idx, end_idx

	def _get_delay_samples(self, dm: float, reference_freq: Optional[float] = None) -> np.ndarray:
		"""
		Return per-channel delay in integer samples for a trial DM.
		"""
		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)

		effective_dm = self.input_dm - dm
		delays_ms = self.DM_CONSTANT * effective_dm * (
			1.0 / self.freq_mhz**2 - 1.0 / reference_freq**2
		)
		dt = np.median(np.diff(self.time_ms))
		return np.round(delays_ms / dt).astype(int)

	def recommend_lowest_dm_step(self, reference_freq: Optional[float] = None,
							   samples_per_step: float = 1.0) -> float:
		"""
		Recommend the minimum useful DM grid step from dynamic-spectrum resolution.

		The estimate is based on the DM increment that produces a delay of
		``samples_per_step`` time samples across the most-dispersed channel relative
		to the reference frequency.
		"""
		if samples_per_step <= 0:
			raise ValueError("samples_per_step must be positive")

		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)

		dt_ms = float(np.median(np.diff(self.time_ms)))
		if dt_ms <= 0:
			raise ValueError("time axis must be strictly increasing to estimate DM step")

		delta_inv_f2 = np.abs(1.0 / self.freq_mhz**2 - 1.0 / float(reference_freq)**2)
		max_delta_inv_f2 = float(np.max(delta_inv_f2))
		if max_delta_inv_f2 <= 0:
			raise ValueError("frequency axis does not span reference-frequency delays")

		return (samples_per_step * dt_ms) / (self.DM_CONSTANT * max_delta_inv_f2)
		
	def dedisperse(self, data: np.ndarray, dm: float, reference_freq: Optional[float] = None, 
				   output_size: Optional[int] = None, mode: str = 'expand') -> np.ndarray:
		"""
		Apply dispersion correction to data.
		
		Parameters:
		-----------
		data : np.ndarray
			2D array to dedisperse (freq x time)
		dm : float
			Dispersion measure in pc cm^-3
		reference_freq : float, optional
			Reference frequency in MHz. If None, uses highest frequency.
		output_size : int, optional
			Desired output time axis size. If None, calculated based on mode.
		mode : str, optional
			Dedispersion mode: 'expand' (fill edges with noise, default),
			'expand_zero' (expand with zero fill), or 'crop' (trim to common valid region)
			
		Returns:
		--------
		dedispersed : np.ndarray
			Dedispersed data array
		"""
		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)
		
		delay_samples = self._get_delay_samples(dm, reference_freq=reference_freq)
		valid_modes = {'expand', 'expand_zero', 'crop'}
		if mode not in valid_modes:
			raise ValueError(f"Unknown dedispersion mode '{mode}'. Expected one of {sorted(valid_modes)}")
		
		if mode == 'crop':
			# Crop mode: return only the common valid region
			start_idx, end_idx = self._get_common_valid_region(data, delay_samples)
			if output_size is None:
				n_time_out = end_idx - start_idx
			else:
				n_time_out = output_size

			# Generate noise fill using full dynamic spectrum noise statistics
			noise_ref = self._full_noise_reference_data(data)
			# Always use per-channel mean/std from the early (off-pulse) samples
			n_edge_full = max(1, int(0.05 * noise_ref.shape[1]))
			# Estimate per-channel mean/std from data edges and draw vectorised over time
			noise_std = np.std(noise_ref[:, :n_edge_full], axis=1)
			noise_mean = np.mean(noise_ref[:, :n_edge_full], axis=1)
			noise_fill = self.rng.normal(noise_mean[:, None], noise_std[:, None], size=(data.shape[0], n_time_out))

			# Apply shifts and crop to common region
			if _NUMBA_AVAILABLE:
				dedispersed = _apply_shifts_numba(data, delay_samples, noise_fill, start_idx)
			else:
				dedispersed = noise_fill.copy()
				for i, shift in enumerate(delay_samples):
					for t in range(data.shape[1]):
						t_out = t + shift - start_idx
						if 0 <= t_out < n_time_out:
							dedispersed[i, t_out] = data[i, t]
		else:
			# Expand-like modes: extend time axis and fill with noise (expand)
			# or zeros (expand_zero).
			min_shift = int(np.min(delay_samples))
			max_shift = int(np.max(delay_samples))
			if output_size is None:
				n_time_out = data.shape[1] + max_shift - min_shift
			else:
				n_time_out = output_size

			if mode == 'expand_zero':
				noise_fill = np.zeros((data.shape[0], n_time_out), dtype=float)
			else:
				# Use per-channel mean/std from the early (off-pulse) samples to generate noise
				noise_ref = self._full_noise_reference_data(data)
				n_edge_full = max(1, int(0.05 * noise_ref.shape[1]))
				noise_std = np.std(noise_ref[:, :n_edge_full], axis=1)
				noise_mean = np.mean(noise_ref[:, :n_edge_full], axis=1)
				noise_fill = self.rng.normal(noise_mean[:, None], noise_std[:, None], size=(data.shape[0], n_time_out))

			# Apply dedispersion by shifting each frequency channel
			if _NUMBA_AVAILABLE:
				dedispersed = _apply_shifts_numba(data, delay_samples, noise_fill, min_shift)
			else:
				dedispersed = noise_fill.copy()
				for i, shift in enumerate(delay_samples):
					for t in range(data.shape[1]):
						t_out = t + shift - min_shift
						if 0 <= t_out < n_time_out:
							dedispersed[i, t_out] = data[i, t]
			
		return dedispersed
	def _max_output_size_for_dm_range(self, data: np.ndarray, dm_range: Tuple[float, float]) -> int:
		"""
		Calculate the maximum output size needed for dedispersion across a DM range.
		"""
		# Check both endpoints of the range
		max_size = data.shape[1]
		for dm in dm_range:
			delay_samples = self._get_delay_samples(dm, reference_freq=self.reference_freq)
			max_shift = int(np.max(delay_samples))
			min_shift = int(np.min(delay_samples))
			max_size = max(max_size, data.shape[1] + max_shift - min_shift)
		return max_size


