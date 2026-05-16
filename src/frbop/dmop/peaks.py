"""Peak detection and manual peak selection."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from frbop.utils.plotting import savefig_rasterized
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual

class PeaksMixin:
	def separate_peaks(self, min_separation_ms: float = 1.0,
					   diagnostics_path: Optional[str] = None) -> List[Tuple[int, int]]:
		"""
		Identify and separate peaks in the time series.
		
		Parameters:
		-----------
		min_separation_ms : float
			Minimum separation between peaks in milliseconds
			
		Returns:
		--------
		peak_regions : list of tuples
			List of (start_idx, end_idx) for each peak region
		"""
		# Collapse to time series
		time_series = np.nansum(self.stokes_i, axis=0)
		
		# Smooth to find peaks
		smoothed = gaussian_filter1d(time_series, sigma=4)
		
		# Find peaks
		dt = np.median(np.diff(self.time_ms))
		min_distance = int(min_separation_ms / dt)
		n_edge = max(1, int(0.05 * len(smoothed)))
		peaks, properties = find_peaks(smoothed, distance=min_distance, prominence=2*np.std(smoothed[:n_edge]))

		if diagnostics_path:
			plt.figure(figsize=(10, 4))
			plt.plot(self.time_ms, time_series, color='0.6', linewidth=1, label='Raw')
			plt.plot(self.time_ms, smoothed, color='k', linewidth=1.5, label='Smoothed')
			if len(peaks) > 0:
				plt.scatter(self.time_ms[peaks], smoothed[peaks], color='red', s=20, label='Peaks')
			plt.xlabel('Time (ms)')
			plt.ylabel('Flux (arb.)')
			plt.title('Peak Finding Diagnostics')
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			savefig_rasterized(diagnostics_path, dpi=150, bbox_inches='tight')
			plt.close()
		
		if len(peaks) == 0:
			# No peaks found, return entire range
			return [(0, self.n_time)]
		
		# Define regions around each peak
		peak_regions = []
		for peak in peaks:
			# Find boundaries (go to minima on either side)
			# Simple approach: go to half-max on either side
			half_max = (smoothed[peak] - np.min(smoothed)) / 2 + np.min(smoothed)
			
			# Search left
			start = peak
			while start > 0 and smoothed[start] > half_max:
				start -= 1
			
			# Search right
			end = peak
			while end < len(smoothed) - 1 and smoothed[end] > half_max:
				end += 1
			
			peak_regions.append((max(0, start - 20), min(self.n_time, end + 80)))
		
		return peak_regions

	def select_peaks_manual(self) -> List[Tuple[int, int]]:
		"""Manually select peak bounds by clicking on the pulse profile."""
		time_series = np.nansum(self.stokes_i, axis=0)
		return shared_select_peaks_manual(
			self.time_ms,
			time_series,
			title='Click start/end bounds for each peak (close window to finish)',
			x_label='Time (ms)',
			y_label='Flux',
			exclusive_end=True,
		)
	

