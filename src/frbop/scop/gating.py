"""Gating and peak-selection helpers."""

import numpy as np

from frbop.utils.peaks import \
    select_peak_fwhm_manual as shared_select_peak_fwhm_manual
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual
from frbop.utils.windows import find_burst_window


def select_peaks_manual(
    time_axis, profile_or_stokes, *,
    title='Click start/end bounds for each peak (close window when done)',
    x_label='Time [ms]', y_label='Flux', exclusive_end=True,
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
    x_label="Time [ms]", y_label="Flux",
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
