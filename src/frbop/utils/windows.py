"""Shared burst and on-pulse window helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def find_burst_window(ts, peak_idx, smooth_win=5, threshold_sigma=3.0, pad=50, fallback_window=200):
    """Find contiguous burst window around peak using robust thresholding.

    Returns (tmin, tmax) inclusive-exclusive indices.
    """
    ts = np.asarray(ts)
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        ts_smooth = np.convolve(ts, kernel, mode="same")
    else:
        ts_smooth = ts

    med = np.median(ts_smooth)
    mad = np.median(np.abs(ts_smooth - med))
    sigma_est = 1.4826 * mad if mad > 0 else np.std(ts_smooth)
    thresh = med + threshold_sigma * sigma_est

    above = np.where(ts_smooth > thresh)[0]
    if above.size > 0:
        breaks = np.where(np.diff(above) > 1)[0]
        segments = []
        start = 0
        for b in breaks:
            segments.append(above[start : b + 1])
            start = b + 1
        segments.append(above[start:])

        chosen = None
        for seg in segments:
            if peak_idx in seg:
                chosen = seg
                break
        if chosen is None:
            chosen = max(segments, key=lambda s: s.size)

        tmin = max(0, chosen[0] - pad)
        tmax = chosen[-1] + 1 + pad
        return tmin, tmax

    tmin = max(0, peak_idx - fallback_window)
    tmax = peak_idx + fallback_window
    return tmin, tmax


def find_onpulse_window(time_profile: np.ndarray, flux_fraction: float = 0.95) -> Tuple[int, int]:
    """Find the smallest contiguous window containing a fraction of total flux."""
    time_profile = np.asarray(time_profile, dtype=float)
    total_flux = np.sum(time_profile)
    target_flux = total_flux * flux_fraction
    n_bins = len(time_profile)

    best_window = None
    best_width = n_bins + 1

    for width in range(1, n_bins + 1):
        for start in range(n_bins - width + 1):
            end = start + width
            window_flux = np.sum(time_profile[start:end])

            if window_flux >= target_flux:
                if width < best_width:
                    best_width = width
                    best_window = (start, end - 1)
                break

    if best_window is None:
        best_window = (0, n_bins - 1)

    return best_window


def find_peak_regions(time_profile: np.ndarray, snr_array: Optional[np.ndarray] = None,
                      min_gap_bins: int = 3, min_peak_bins: int = 3,
                      max_merge_gap: int = 0, snr_threshold: float = 5.0) -> list:
    """Identify separate peak regions in time series data."""
    time_profile = np.asarray(time_profile, dtype=float)
    n_bins = len(time_profile)

    if snr_array is not None and len(snr_array) == n_bins:
        significant = snr_array >= snr_threshold
    else:
        threshold = np.median(time_profile) + 2.0 * np.std(time_profile)
        significant = time_profile >= threshold

    peak_regions = []
    in_peak = False
    start_idx = 0
    gap_count = 0

    for i in range(n_bins):
        if significant[i]:
            if not in_peak:
                start_idx = i
                in_peak = True
                gap_count = 0
            else:
                gap_count = 0
        else:
            if in_peak:
                gap_count += 1
                if gap_count >= min_gap_bins:
                    end_idx = i - gap_count
                    peak_width = end_idx - start_idx + 1
                    if peak_width >= min_peak_bins:
                        peak_regions.append((start_idx, end_idx))
                    in_peak = False
                    gap_count = 0

    if in_peak:
        end_idx = n_bins - 1 - gap_count
        peak_width = end_idx - start_idx + 1
        if peak_width >= min_peak_bins:
            peak_regions.append((start_idx, end_idx))

    if max_merge_gap > 0 and len(peak_regions) > 1:
        merged_regions = []
        current_start, current_end = peak_regions[0]

        for i in range(1, len(peak_regions)):
            next_start, next_end = peak_regions[i]
            gap_size = next_start - current_end - 1

            if gap_size < max_merge_gap:
                current_end = next_end
            else:
                merged_regions.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged_regions.append((current_start, current_end))
        peak_regions = merged_regions

    if len(peak_regions) == 0:
        peak_regions = [(0, n_bins - 1)]

    return peak_regions