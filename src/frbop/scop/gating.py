"""Gating and peak-selection helpers."""

import numpy as np

from frbop.utils.peaks import (
    select_peaks_manual as shared_select_peaks_manual,
    select_peak_fwhm_manual as shared_select_peak_fwhm_manual,
)


def find_burst_window(ts, peak_idx, smooth_win=5, threshold_sigma=3.0, pad=50, fallback_window=200):
    """Find contiguous burst window around peak using robust thresholding.

    Returns (tmin, tmax) inclusive-exclusive indices.
    """
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


def select_peaks_manual(
    time_axis, profile_or_stokes, *,
    title='Click start/end bounds for each peak (close window when done)',
    x_label='Time (ms)', y_label='Flux', exclusive_end=True,
):
    ts = np.nanmean(profile_or_stokes, axis=0) if profile_or_stokes.ndim == 2 else profile_or_stokes
    return shared_select_peaks_manual(
        time_axis,
        ts,
        title=title,
        x_label=x_label,
        y_label=y_label,
        exclusive_end=exclusive_end,
    )


def select_peak_fwhm_manual(
    time_axis, profile_or_stokes, *,
    title="Click peak to measure FWHM (close window when done)",
    x_label="Time (ms)", y_label="Flux",
    baseline_percentile=10.0, local_max_window=3, exclusive_end=True,
):
    ts = np.nanmean(profile_or_stokes, axis=0) if profile_or_stokes.ndim == 2 else profile_or_stokes
    return shared_select_peak_fwhm_manual(
        time_axis,
        ts,
        title=title,
        x_label=x_label,
        y_label=y_label,
        baseline_percentile=baseline_percentile,
        local_max_window=local_max_window,
        exclusive_end=exclusive_end,
    )
