"""Numba-accelerated kernels for dedispersion and S/N."""

import numpy as np
from .common import _NUMBA_AVAILABLE, njit

if _NUMBA_AVAILABLE:
	@njit
	def _max_snr_for_series(series: np.ndarray, noise_std: float,
							min_window_size: int, max_window_size: int) -> float:
		n = series.shape[0]
		csum = np.empty(n + 1, dtype=np.float64)
		csum[0] = 0.0
		for i in range(n):
			csum[i + 1] = csum[i] + series[i]

		max_sn = -1e30
		for start in range(0, n - min_window_size):
			max_len = max_window_size
			if start + max_len > n:
				max_len = n - start
			for length in range(min_window_size, max_len):
				s = csum[start + length] - csum[start]
				sn = s / (noise_std * np.sqrt(length))
				if sn > max_sn:
					max_sn = sn
		return max_sn

	@njit
	def _apply_shifts_numba(data: np.ndarray, delay_samples: np.ndarray, 
							 noise_fill: np.ndarray, start_idx: int = 0) -> np.ndarray:
		n_freq, n_time = data.shape
		n_freq_out, n_time_out = noise_fill.shape
		out = noise_fill.copy()
		# Place shifted data
		for i in range(n_freq):
			shift = delay_samples[i]
			for t in range(n_time):
				t_out = t + shift - start_idx
				if 0 <= t_out < n_time_out:
					out[i, t_out] = data[i, t]
		return out

