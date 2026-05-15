"""
Dedispersion utilities.

Handles per-channel delay computation, data shifting, and output-size estimation.
Numba acceleration is used when available.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

# DM constant: k = 4.148808e6 ms MHz^2 pc^-1 cm^3
# From pulsar handbook: dt = 4.15e6 ms × (f1^-2 - f2^-2) × DM
DM_CONSTANT: float = 4.148808e6


if _NUMBA_AVAILABLE:
    @njit
    def _apply_shifts_numba(
        data: np.ndarray,
        delay_samples: np.ndarray,
        noise_fill: np.ndarray,
        start_idx: int = 0,
    ) -> np.ndarray:
        n_freq, n_time = data.shape
        n_freq_out, n_time_out = noise_fill.shape
        out = noise_fill.copy()
        for i in range(n_freq):
            shift = delay_samples[i]
            for t in range(n_time):
                t_out = t + shift - start_idx
                if 0 <= t_out < n_time_out:
                    out[i, t_out] = data[i, t]
        return out


def _apply_shifts_python(
    data: np.ndarray,
    delay_samples: np.ndarray,
    noise_fill: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    """Pure-Python fallback for channel shifting."""
    n_time_out = noise_fill.shape[1]
    out = noise_fill.copy()
    for i, shift in enumerate(delay_samples):
        for t in range(data.shape[1]):
            t_out = t + shift - start_idx
            if 0 <= t_out < n_time_out:
                out[i, t_out] = data[i, t]
    return out


def get_delay_samples(
    dm: float,
    freq_mhz: np.ndarray,
    time_ms: np.ndarray,
    reference_freq: float,
    input_dm: float = 0.0,
) -> np.ndarray:
    """
    Return per-channel delay in integer samples for a trial DM.

    Parameters
    ----------
    dm:
        Trial dispersion measure (pc cm⁻³).
    freq_mhz:
        Channel centre frequencies (MHz).
    time_ms:
        Time axis (ms).  Used only to extract the time resolution dt.
    reference_freq:
        Reference frequency (MHz) at which delay is zero.
    input_dm:
        DM already applied to the input data (pc cm⁻³).  The effective
        shift applied is ``input_dm - dm``.
    """
    effective_dm = input_dm - dm
    delays_ms = DM_CONSTANT * effective_dm * (
        1.0 / freq_mhz ** 2 - 1.0 / reference_freq ** 2
    )
    dt = float(np.median(np.diff(time_ms)))
    return np.round(delays_ms / dt).astype(int)


def get_common_valid_region(
    n_time: int,
    delay_samples: np.ndarray,
) -> Tuple[int, int]:
    """
    Return the (start, end) index of the time window common to all channels
    after dedispersion (crop mode).
    """
    start_idx = int(np.max(delay_samples))
    end_idx = n_time + int(np.min(delay_samples))
    return start_idx, end_idx


def max_output_size_for_dm_range(
    n_time: int,
    freq_mhz: np.ndarray,
    time_ms: np.ndarray,
    reference_freq: float,
    dm_range: Tuple[float, float],
    input_dm: float = 0.0,
) -> int:
    """
    Maximum time-axis length needed across both DM endpoints (expand mode).
    """
    max_size = n_time
    for dm in dm_range:
        delay_samples = get_delay_samples(dm, freq_mhz, time_ms, reference_freq, input_dm)
        max_shift = int(np.max(delay_samples))
        min_shift = int(np.min(delay_samples))
        max_size = max(max_size, n_time + max_shift - min_shift)
    return max_size


def recommend_lowest_dm_step(
    freq_mhz: np.ndarray,
    time_ms: np.ndarray,
    reference_freq: float,
    samples_per_step: float = 1.0,
) -> float:
    """
    Minimum useful DM grid step: the DM increment that shifts the most-dispersed
    channel by ``samples_per_step`` time samples relative to the reference frequency.
    """
    if samples_per_step <= 0:
        raise ValueError("samples_per_step must be positive")

    dt_ms = float(np.median(np.diff(time_ms)))
    if dt_ms <= 0:
        raise ValueError("time axis must be strictly increasing")

    delta_inv_f2 = np.abs(1.0 / freq_mhz ** 2 - 1.0 / float(reference_freq) ** 2)
    max_delta_inv_f2 = float(np.max(delta_inv_f2))
    if max_delta_inv_f2 <= 0:
        raise ValueError("frequency axis does not span reference-frequency delays")

    return (samples_per_step * dt_ms) / (DM_CONSTANT * max_delta_inv_f2)


def build_dm_values(
    dm_range: Tuple[float, float],
    n_points: int = 200,
    dm_step: Optional[float] = None,
) -> np.ndarray:
    """
    Build an array of DM trial values spanning *dm_range*.

    When *dm_step* is given it overrides *n_points* and the array is built
    with uniform spacing; an extra endpoint is appended if it falls more than
    half a step short of ``dm_range[1]``.
    """
    if dm_step is not None:
        if dm_step <= 0:
            raise ValueError("dm_step must be positive")
        span = dm_range[1] - dm_range[0]
        n = max(2, int(np.floor(span / dm_step)) + 1)
        values = dm_range[0] + np.arange(n) * dm_step
        if values[-1] < dm_range[1] and (dm_range[1] - values[-1]) > 0.5 * dm_step:
            values = np.append(values, dm_range[1])
        return values
    return np.linspace(dm_range[0], dm_range[1], n_points)


def make_noise_fill(
    data: np.ndarray,
    noise_ref: np.ndarray,
    n_time_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a per-channel noise array of shape ``(n_freq, n_time_out)`` drawn
    from the statistics of the first 5 % of *noise_ref* along the time axis.
    """
    n_edge = max(1, int(0.05 * noise_ref.shape[1]))
    noise_fill = np.empty((data.shape[0], n_time_out), dtype=data.dtype)
    for i in range(data.shape[0]):
        noise_std = float(np.std(noise_ref[i, :n_edge]))
        noise_mean = float(np.mean(noise_ref[i, :n_edge]))
        noise_fill[i] = rng.normal(noise_mean, noise_std, n_time_out)
    return noise_fill


def dedisperse(
    data: np.ndarray,
    dm: float,
    freq_mhz: np.ndarray,
    time_ms: np.ndarray,
    reference_freq: float,
    rng: np.random.Generator,
    noise_ref: np.ndarray,
    input_dm: float = 0.0,
    output_size: Optional[int] = None,
    mode: str = "expand",
) -> np.ndarray:
    """
    Apply dispersion correction to *data* (freq × time).

    Parameters
    ----------
    data:
        2-D array to dedisperse (freq × time).
    dm:
        Trial DM (pc cm⁻³).
    noise_ref:
        Full dynamic-spectrum array used to estimate per-channel noise
        statistics for the noise fill.  Typically the unsliced Stokes-I cube.
    mode:
        ``'expand'`` — extend the time axis and fill edges with noise (default).
        ``'crop'``   — return only the time window common to all channels.
    """
    delay_samples = get_delay_samples(dm, freq_mhz, time_ms, reference_freq, input_dm)

    if mode == "crop":
        start_idx, end_idx = get_common_valid_region(data.shape[1], delay_samples)
        n_time_out = output_size if output_size is not None else (end_idx - start_idx)
        noise_fill = make_noise_fill(data, noise_ref, n_time_out, rng)
        if _NUMBA_AVAILABLE:
            return _apply_shifts_numba(data, delay_samples, noise_fill, start_idx)
        return _apply_shifts_python(data, delay_samples, noise_fill, start_idx)

    # expand mode (default)
    min_shift = int(np.min(delay_samples))
    max_shift = int(np.max(delay_samples))
    n_time_out = output_size if output_size is not None else (data.shape[1] + max_shift - min_shift)
    noise_fill = make_noise_fill(data, noise_ref, n_time_out, rng)
    if _NUMBA_AVAILABLE:
        return _apply_shifts_numba(data, delay_samples, noise_fill, min_shift)
    return _apply_shifts_python(data, delay_samples, noise_fill, min_shift)
