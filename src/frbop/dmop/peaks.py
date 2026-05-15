"""
Peak detection and selection utilities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as _select_peaks_manual
from frbop.utils.plotting import savefig_rasterized


def separate_peaks(
    stokes_i: np.ndarray,
    time_ms: np.ndarray,
    min_separation_ms: float = 1.0,
    diagnostics_path: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """
    Identify peaks in the Stokes-I time series and return half-max regions.

    Parameters
    ----------
    stokes_i:
        2-D array (freq × time).
    time_ms:
        1-D time axis in milliseconds.
    min_separation_ms:
        Minimum peak-to-peak separation in milliseconds.
    diagnostics_path:
        If given, save a diagnostic plot to this path.

    Returns
    -------
    list of ``(start_idx, end_idx)`` tuples.
    """
    import matplotlib.pyplot as plt

    n_time = stokes_i.shape[1]
    time_series = np.mean(stokes_i, axis=0)
    smoothed = gaussian_filter1d(time_series, sigma=4)

    dt = float(np.median(np.diff(time_ms)))
    min_distance = int(min_separation_ms / dt)
    n_edge = max(1, int(0.05 * len(smoothed)))
    peaks, _ = find_peaks(
        smoothed,
        distance=min_distance,
        prominence=2 * np.std(smoothed[:n_edge]),
    )

    if diagnostics_path:
        plt.figure(figsize=(10, 4))
        plt.plot(time_ms, time_series, color="0.6", linewidth=1, label="Raw")
        plt.plot(time_ms, smoothed, color="k", linewidth=1.5, label="Smoothed")
        if len(peaks) > 0:
            plt.scatter(time_ms[peaks], smoothed[peaks], color="red", s=20, label="Peaks")
        plt.xlabel("Time (ms)")
        plt.ylabel("Flux (arb.)")
        plt.title("Peak Finding Diagnostics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        savefig_rasterized(diagnostics_path, dpi=150, bbox_inches="tight")
        plt.close()

    if len(peaks) == 0:
        return [(0, n_time)]

    baseline = float(np.min(smoothed))
    peak_regions = []
    for peak in peaks:
        half_max = (smoothed[peak] - baseline) / 2.0 + baseline

        start = peak
        while start > 0 and smoothed[start] > half_max:
            start -= 1

        end = peak
        while end < len(smoothed) - 1 and smoothed[end] > half_max:
            end += 1

        peak_regions.append((max(0, start - 20), min(n_time, end + 80)))

    return peak_regions


def select_peaks_manual(
    stokes_i: np.ndarray,
    time_ms: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Interactively select peak bounds by clicking on the pulse profile.
    """
    time_series = np.nanmean(stokes_i, axis=0)
    return _select_peaks_manual(
        time_ms,
        time_series,
        title="Click start/end bounds for each peak (close window to finish)",
        x_label="Time (ms)",
        y_label="Flux",
        exclusive_end=True,
    )
