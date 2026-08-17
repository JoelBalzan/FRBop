"""
Data I/O and on-pulse / peak detection utilities.
"""

from typing import List, Optional, Tuple

import numpy as np

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual
from frbop.utils.windows import find_onpulse_window, find_peak_regions


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
        x_label='Time [ms]',
        y_label='Flux',
        exclusive_end=True,
    )
