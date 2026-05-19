"""
Data I/O and on-pulse / peak detection utilities.
"""

from typing import List, Optional, Tuple

import numpy as np

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual


def load_stokes_data(i_file: Optional[str] = None,
                     q_file: Optional[str] = None,
                     u_file: Optional[str] = None,
                     v_file: Optional[str] = None,
                     cube_file: Optional[str] = None,
                     stokes_axis: int = 0,
                     freq_file: Optional[str] = None,
                     time_file: Optional[str] = None,
                     time_axis: int = 1,
                     freq_axis: int = 0,
                     freq_unit: str = 'Hz',
                     time_unit: str = 's') -> Tuple:
    """
    Load Stokes parameter data from separate files or an IQUV cube.

    Parameters:
    -----------
    i_file : str, optional
        Path to Stokes I file
    q_file : str, optional
        Path to Stokes Q file
    u_file : str, optional
        Path to Stokes U file
    v_file : str, optional
        Path to Stokes V file
    cube_file : str, optional
        Path to Stokes cube file containing I,Q,U,(V).
    stokes_axis : int
        Axis index of the Stokes dimension in ``cube_file``.
    freq_file : str, optional
        Path to frequency file (Hz). If None, generates placeholder frequencies.
    time_file : str, optional
        Path to time file (seconds or other units).
    time_axis : int
        Axis for time dimension in 2D arrays (default: 1)
    freq_axis : int
        Axis for frequency dimension in 2D arrays (default: 0)
    freq_unit : str
        Unit of frequency file: 'Hz', 'MHz', 'GHz' (default: 'Hz')
    time_unit : str
        Unit of time file: 's', 'ms', 'us' (default: 's')

    Returns:
    --------
    freq_hz : array
        Frequency array in Hz
    stokes_i : array
        Stokes I data
    stokes_q : array
        Stokes Q data
    stokes_u : array
        Stokes U data
    stokes_v : array or None
        Stokes V data
    time_array : array or None
        Time array if provided
    """

    def load_file(filename):
        """Load data from .npy or text file."""
        if filename.endswith('.npy'):
            return np.load(filename)
        else:
            return np.loadtxt(filename)

    # Load data either from a cube or separate I/Q/U(/V) files
    if cube_file:
        cube = np.asarray(load_file(cube_file))
        if cube.ndim < 2:
            raise ValueError("Stokes cube must have at least 2 dimensions")

        cube = np.moveaxis(cube, stokes_axis, 0)
        n_stokes = cube.shape[0]
        if n_stokes < 3:
            raise ValueError("Stokes cube must contain at least I, Q, U components")

        stokes_i = cube[0]
        stokes_q = cube[1]
        stokes_u = cube[2]
        stokes_v = cube[3] if n_stokes >= 4 else None

        print(f"  Loaded Stokes cube: {cube_file}")
        print(f"    Cube shape (after moveaxis): {cube.shape}")
    else:
        if i_file is None or q_file is None or u_file is None:
            raise ValueError("Provide --stokes-cube or all of --stokes-i/--stokes-q/--stokes-u")
        stokes_i = load_file(i_file)
        stokes_q = load_file(q_file)
        stokes_u = load_file(u_file)
        stokes_v = load_file(v_file) if v_file else None

    print(f"  Loaded data shapes:")
    print(f"    Stokes I: {stokes_i.shape}")
    print(f"    Stokes Q: {stokes_q.shape}")
    print(f"    Stokes U: {stokes_u.shape}")
    if stokes_v is not None:
        print(f"    Stokes V: {stokes_v.shape}")

    # Handle frequency
    if freq_file:
        freq_hz = load_file(freq_file)
        print(f"  Loaded frequency array: {len(freq_hz)} channels")

        if freq_unit.lower() == 'mhz':
            freq_hz = freq_hz * 1e6
            print(f"  Converted from MHz to Hz")
        elif freq_unit.lower() == 'ghz':
            freq_hz = freq_hz * 1e9
            print(f"  Converted from GHz to Hz")
        elif freq_unit.lower() != 'hz':
            print(f"  Warning: Unknown frequency unit '{freq_unit}', assuming Hz")
    else:
        if stokes_i.ndim == 2:
            n_freq = stokes_i.shape[freq_axis]
            print(f"  Warning: No frequency file provided. Generating {n_freq} placeholder frequencies.")
            freq_hz = np.linspace(1e9, 2e9, n_freq)
        else:
            print("  Warning: No frequency file provided. Generating placeholder frequencies.")
            freq_hz = np.linspace(1e9, 2e9, len(stokes_i))

    # Handle time
    time_array = None
    if time_file:
        time_array = load_file(time_file)
        print(f"  Loaded time array: {len(time_array)} samples")

        if time_unit.lower() == 'ms':
            time_array = time_array * 1e-3
            print(f"  Converted from ms to seconds")
        elif time_unit.lower() == 'us':
            time_array = time_array * 1e-6
            print(f"  Converted from μs to seconds")
        elif time_unit.lower() != 's':
            print(f"  Warning: Unknown time unit '{time_unit}', assuming seconds")
    elif stokes_i.ndim == 2:
        n_time = stokes_i.shape[time_axis]
        time_array = np.arange(n_time)

    return freq_hz, stokes_i, stokes_q, stokes_u, stokes_v, time_array


def find_onpulse_window(time_profile: np.ndarray, flux_fraction: float = 0.95) -> Tuple[int, int]:
    """
    Find the smallest contiguous window that contains a given fraction of the total flux.

    Parameters:
    -----------
    time_profile : array
        1D array of flux as function of time
    flux_fraction : float
        Fraction of total flux to contain (default: 0.95 for 95%)

    Returns:
    --------
    start_idx : int
        Start index of on-pulse window
    end_idx : int
        End index of on-pulse window (inclusive)
    """
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
    """
    Identify separate peak regions in time series data by detecting gaps of low signal.

    Parameters:
    -----------
    time_profile : array
        1D array of flux as function of time
    snr_array : array, optional
        1D array of SNR values. If provided, uses SNR threshold to identify gaps.
    min_gap_bins : int
        Minimum number of consecutive low-signal bins to separate peaks (default: 3)
    min_peak_bins : int
        Minimum number of consecutive significant bins required for a valid peak (default: 3)
    max_merge_gap : int
        Maximum gap size for merging nearby peaks (default: 0, no merging)
    snr_threshold : float
        SNR threshold below which signal is considered low (default: 5.0)

    Returns:
    --------
    peak_regions : list of tuples
        List of (start_idx, end_idx) for each peak region (inclusive)
    """
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


def select_peaks_manual(time_ms: np.ndarray, stokes_i: np.ndarray) -> List[Tuple[int, int]]:
    """Manually select peak bounds by clicking on the pulse profile."""
    if stokes_i.ndim == 2:
        if stokes_i.shape[0] == len(time_ms):
            time_series = np.nanmean(stokes_i, axis=1)
        elif stokes_i.shape[1] == len(time_ms):
            time_series = np.nanmean(stokes_i, axis=0)
        else:
            time_series = np.nanmean(stokes_i, axis=1)
    else:
        time_series = np.asarray(stokes_i, float)

    display_time_ms = np.asarray(time_ms, float) * 1e3
    return shared_select_peaks_manual(
        display_time_ms,
        time_series,
        title='Click start/end bounds for each peak (close window to finish)',
        x_label='Time (ms)',
        y_label='Flux',
        exclusive_end=True,
    )
