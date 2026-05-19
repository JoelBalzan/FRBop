from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def select_peaks_manual(
    time_axis: np.ndarray,
    profile: np.ndarray,
    *,
    title: str = "Click start/end bounds for each peak (close window when done)",
    x_label: str = "Time (ms)",
    y_label: str = rf"S (arb.)",
    exclusive_end: bool = True,
) -> List[Tuple[int, int]]:
    """Interactively select peak regions from a 1D profile."""
    time_axis = np.asarray(time_axis, float)
    profile = np.asarray(profile, float)

    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1D, got shape={time_axis.shape}")
    if profile.ndim != 1:
        raise ValueError(f"profile must be 1D, got shape={profile.shape}")
    if time_axis.size != profile.size:
        raise ValueError(
            f"time_axis length ({time_axis.size}) does not match profile length ({profile.size})"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, profile, color='k', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    cursor_line = ax.axvline(time_axis[0] if time_axis.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)

    clicks: List[float] = []

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_line.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x = float(event.xdata)
        clicks.append(x)
        ax.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
        if len(clicks) % 2 == 0:
            start_t, end_t = sorted((clicks[-2], clicks[-1]))
            ax.axvspan(start_t, end_t, color='tab:orange', alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    if not clicks:
        return [(0, time_axis.size)]

    if len(clicks) % 2 != 0:
        clicks = clicks[:-1]

    regions: List[Tuple[int, int]] = []
    for i in range(0, len(clicks), 2):
        start_t, end_t = sorted((clicks[i], clicks[i + 1]))
        start_idx = int(np.argmin(np.abs(time_axis - start_t)))
        end_idx = int(np.argmin(np.abs(time_axis - end_t)))
        start = min(start_idx, end_idx)
        stop = max(start_idx, end_idx) + (1 if exclusive_end else 0)
        stop = min(time_axis.size, stop)
        if stop <= start:
            stop = min(time_axis.size, start + 1)
        regions.append((start, stop))
    print(f"Parsed {len(regions)} peak regions: {regions}")

    return regions if regions else [(0, time_axis.size)]


def select_peak_fwhm_manual(
    time_axis: np.ndarray,
    profile: np.ndarray,
    *,
    title: str = "Click peak to measure FWHM (close window when done)",
    x_label: str = "Time (ms)",
    y_label: str = "Flux",
    baseline_percentile: float = 10.0,
    local_max_window: int = 3,
    exclusive_end: bool = True,
) -> tuple[tuple[int, int], float]:
    """Select a peak with one click and return (start, stop) at its FWHM."""
    time_axis = np.asarray(time_axis, float)
    profile = np.asarray(profile, float)

    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1D, got shape={time_axis.shape}")
    if profile.ndim != 1:
        raise ValueError(f"profile must be 1D, got shape={profile.shape}")
    if time_axis.size != profile.size:
        raise ValueError(
            f"time_axis length ({time_axis.size}) does not match profile length ({profile.size})"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, profile, color="k", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    cursor_line = ax.axvline(time_axis[0] if time_axis.size else 0.0, color="tab:blue", alpha=0.4, linewidth=1)

    click_x: list[float] = []
    fwhm_region: list[int] = []
    fwhm_ms: list[float] = []

    def _find_peak_index(idx: int) -> int:
        if local_max_window <= 0:
            return idx
        lo = max(0, idx - local_max_window)
        hi = min(profile.size, idx + local_max_window + 1)
        local = profile[lo:hi]
        if local.size == 0:
            return idx
        return int(lo + np.nanargmax(local))

    def _measure_fwhm(idx: int) -> tuple[int, int, float, float]:
        (start, stop), width_ms, half_level = measure_fwhm_region(
            time_axis,
            profile,
            idx,
            baseline_percentile=baseline_percentile,
            exclusive_end=False,
        )
        return start, stop, half_level, width_ms

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_line.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or click_x:
            return
        x = float(event.xdata)
        click_x.append(x)
        idx = int(np.argmin(np.abs(time_axis - x)))
        peak_idx = _find_peak_index(idx)
        start, stop, half_level, width_ms = _measure_fwhm(peak_idx)

        ax.axvline(time_axis[peak_idx], color="tab:red", alpha=0.7, linewidth=1)
        ax.axhline(half_level, color="tab:orange", alpha=0.8, linewidth=1)
        ax.axvspan(time_axis[start], time_axis[stop], color="tab:orange", alpha=0.2)
        fig.canvas.draw_idle()

        fwhm_region[:] = [start, stop]
        fwhm_ms[:] = [width_ms]

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

    if not fwhm_region:
        return (0, time_axis.size), float("nan")

    start, stop = fwhm_region
    if exclusive_end:
        stop = min(time_axis.size, stop + 1)
    if stop <= start:
        stop = min(time_axis.size, start + 1)

    print(f"Selected FWHM region: ({start}, {stop}) width={fwhm_ms[0]:.4f} ms")
    return (start, stop), fwhm_ms[0]


def measure_fwhm_region(
    time_axis: np.ndarray,
    profile: np.ndarray,
    peak_idx: int,
    *,
    baseline_percentile: float = 10.0,
    exclusive_end: bool = True,
) -> tuple[tuple[int, int], float, float]:
    """Return ((start, stop), fwhm_ms, half_level) around peak_idx."""
    time_axis = np.asarray(time_axis, float)
    profile = np.asarray(profile, float)

    if time_axis.ndim != 1:
        raise ValueError(f"time_axis must be 1D, got shape={time_axis.shape}")
    if profile.ndim != 1:
        raise ValueError(f"profile must be 1D, got shape={profile.shape}")
    if time_axis.size != profile.size:
        raise ValueError(
            f"time_axis length ({time_axis.size}) does not match profile length ({profile.size})"
        )
    if time_axis.size == 0:
        return (0, 0), float("nan"), float("nan")

    peak_idx = int(np.clip(peak_idx, 0, profile.size - 1))
    baseline = float(np.nanpercentile(profile, baseline_percentile))
    peak_val = float(profile[peak_idx])
    half_level = baseline + 0.5 * (peak_val - baseline)

    left = peak_idx
    while left > 0 and np.isfinite(profile[left]) and profile[left] > half_level:
        left -= 1
    right = peak_idx
    while right < profile.size - 1 and np.isfinite(profile[right]) and profile[right] > half_level:
        right += 1

    start = max(0, left)
    stop = min(profile.size - 1, right)
    if stop <= start:
        stop = min(profile.size - 1, start + 1)

    start_t = float(time_axis[start])
    stop_t = float(time_axis[stop])
    width_ms = abs(stop_t - start_t)

    if exclusive_end:
        stop = min(time_axis.size, stop + 1)
    if stop <= start:
        stop = min(time_axis.size, start + 1)

    return (start, stop), width_ms, half_level


def select_frequency_bands_manual(
    freq_axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    title: str = "Click start/end bounds for each frequency band (close window when done)",
    x_label: str = "Frequency (MHz)",
    y_label: str = "Flux",
    exclusive_end: bool = True,
) -> List[Tuple[int, int]]:
    """Interactively select frequency bands from a 1D spectrum."""
    freq_axis = np.asarray(freq_axis, float)
    spectrum = np.asarray(spectrum, float)

    if freq_axis.ndim != 1:
        raise ValueError(f"freq_axis must be 1D, got shape={freq_axis.shape}")
    if spectrum.ndim != 1:
        raise ValueError(f"spectrum must be 1D, got shape={spectrum.shape}")
    if freq_axis.size != spectrum.size:
        raise ValueError(
            f"freq_axis length ({freq_axis.size}) does not match spectrum length ({spectrum.size})"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_axis, spectrum, color='k', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    cursor_line = ax.axvline(freq_axis[0] if freq_axis.size else 0.0, color='tab:blue', alpha=0.4, linewidth=1)

    clicks: List[float] = []

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_line.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x = float(event.xdata)
        clicks.append(x)
        ax.axvline(x, color='tab:red', alpha=0.7, linewidth=1)
        if len(clicks) % 2 == 0:
            start_f, end_f = sorted((clicks[-2], clicks[-1]))
            ax.axvspan(start_f, end_f, color='tab:orange', alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    if not clicks:
        return [(0, freq_axis.size)]

    if len(clicks) % 2 != 0:
        clicks = clicks[:-1]

    regions: List[Tuple[int, int]] = []
    for i in range(0, len(clicks), 2):
        start_f, end_f = sorted((clicks[i], clicks[i + 1]))
        start_idx = int(np.argmin(np.abs(freq_axis - start_f)))
        end_idx = int(np.argmin(np.abs(freq_axis - end_f)))
        start = min(start_idx, end_idx)
        stop = max(start_idx, end_idx) + (1 if exclusive_end else 0)
        stop = min(freq_axis.size, stop)
        if stop <= start:
            stop = min(freq_axis.size, start + 1)
        regions.append((start, stop))

    return regions if regions else [(0, freq_axis.size)]


def parse_peak_index_pairs(
    peak_indices: Sequence[int] | None,
    n_time: int,
    *,
    label: str = "--peak-indices",
) -> List[Tuple[int, int]]:
    """Normalize a flat list of indices into clipped start/end regions."""
    if peak_indices is None:
        return []

    values = list(peak_indices)
    if len(values) == 0:
        raise ValueError(f"{label} requires at least one pair of start/end indices")
    if len(values) % 2 != 0:
        raise ValueError(f"{label} requires an even number of values (pairs of start/end indices)")
    if n_time <= 0:
        raise ValueError("Cannot normalize peak indices for an empty time axis")

    regions: List[Tuple[int, int]] = []
    for i in range(0, len(values), 2):
        start_idx = int(values[i])
        end_idx = int(values[i + 1])
        start = int(np.clip(min(start_idx, end_idx), 0, n_time - 1))
        stop = int(np.clip(max(start_idx, end_idx), 0, n_time))
        if stop <= start:
            stop = min(n_time, start + 1)
        regions.append((start, stop))

    return regions
