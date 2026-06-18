"""Stokes cube visualization helpers."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from frbop.utils.plotting import (pub_grid_figsize, savefig_rasterized,
                                  set_pub_col, set_pub_style)


def plot_stokes_cube_summary(
    cube_file: str,
    *,
    freq_file: Optional[str] = None,
    time_file: Optional[str] = None,
    stokes_axis: int = 0,
    freq_axis: int = 0,
    time_axis: int = 1,
    freq_unit: str = "MHz",
    time_unit: str = "ms",
    output_file: Optional[str] = None,
    show: bool = False,
    use_pub_style: bool = True,
) -> plt.Figure:
    """Plot I/Q/U/V dynamic spectra with per-Stokes profiles and spectra.

    Parameters
    ----------
    cube_file : str
        Path to Stokes cube with components ordered I,Q,U,(V).
    freq_file : str, optional
        Path to frequency axis file. Defaults to channel index.
    time_file : str, optional
        Path to time axis file. Defaults to sample index.
    stokes_axis : int
        Axis index of Stokes dimension in the cube.
    freq_axis : int
        Axis index for frequency in the cube after removing stokes axis.
    time_axis : int
        Axis index for time in the cube after removing stokes axis.
    freq_unit : str
        Label for frequency axis.
    time_unit : str
        Label for time axis.
    output_file : str, optional
        Output path for saving the figure. If None, does not save.
    show : bool
        Whether to display the figure interactively.
    use_pub_style : bool
        Apply publication style from utils.plotting.
    """
    if use_pub_style:
        set_pub_style(use_latex=False)

    def _load_array(path: str) -> np.ndarray:
        if path.endswith(".npy"):
            return np.load(path)
        return np.loadtxt(path)

    cube = np.asarray(_load_array(cube_file))
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

    if stokes_i.ndim != 2:
        raise ValueError("Stokes arrays must be 2D (freq x time or time x freq)")

    if {freq_axis, time_axis} != {0, 1}:
        raise ValueError("freq_axis/time_axis must be 0/1 for 2D data")

    if freq_file is not None:
        freq_axis_vals = np.asarray(_load_array(freq_file), dtype=float)
    else:
        freq_axis_vals = np.arange(stokes_i.shape[freq_axis], dtype=float)

    if time_file is not None:
        time_axis_vals = np.asarray(_load_array(time_file), dtype=float)
    else:
        time_axis_vals = np.arange(stokes_i.shape[time_axis], dtype=float)

    if freq_axis_vals.size > 1 and freq_axis_vals[0] > freq_axis_vals[-1]:
        freq_axis_vals = freq_axis_vals[::-1]
        stokes_i = np.flip(stokes_i, axis=freq_axis)
        stokes_q = np.flip(stokes_q, axis=freq_axis)
        stokes_u = np.flip(stokes_u, axis=freq_axis)
        if stokes_v is not None:
            stokes_v = np.flip(stokes_v, axis=freq_axis)

    def _as_freq_time(data: np.ndarray) -> np.ndarray:
        if freq_axis == 0 and time_axis == 1:
            return data
        return np.swapaxes(data, 0, 1)

    i_ft = _as_freq_time(stokes_i)
    q_ft = _as_freq_time(stokes_q)
    u_ft = _as_freq_time(stokes_u)
    v_ft = _as_freq_time(stokes_v) if stokes_v is not None else None

    def _safe_limits(arr: np.ndarray) -> Tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return -1.0, 1.0
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -1.0, 1.0
        return float(vmin), float(vmax)

    stokes_list = [
        ("I", stokes_i),
        ("Q", stokes_q),
        ("U", stokes_u),
    ]
    if stokes_v is not None:
        stokes_list.append(("V", stokes_v))

    n_rows = len(stokes_list)
    fig = plt.figure(figsize=pub_grid_figsize(n_rows, single_column=False, row_height=2.6))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[3.2, 1.2], wspace=0.28, hspace=0.35)

    def _plot_dyn(ax, data: np.ndarray, title: str) -> None:
        vmin, vmax = _safe_limits(data)
        extent = [
            float(time_axis_vals[0]),
            float(time_axis_vals[-1]),
            float(freq_axis_vals[0]),
            float(freq_axis_vals[-1]),
        ]
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(f"Freq ({freq_unit})")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)

    def _plot_profile(ax, label: str, data: np.ndarray) -> None:
        profile = np.nanmean(data, axis=freq_axis)
        ax.plot(time_axis_vals, profile, color="black", linewidth=1.2)
        ax.set_title(f"Stokes {label} profile")
        ax.set_ylabel("Flux (arb.)")
        ax.tick_params(axis="x", labelbottom=False)

    def _plot_spectrum(ax, label: str, data: np.ndarray) -> None:
        spectrum = np.nanmean(data, axis=time_axis)
        ax.plot(freq_axis_vals, spectrum, color="black", linewidth=1.2)
        ax.set_title(f"Stokes {label} spectrum")
        ax.set_xlabel(f"Freq ({freq_unit})")
        ax.set_ylabel("Flux (arb.)")

    ft_map = {
        "I": i_ft,
        "Q": q_ft,
        "U": u_ft,
        "V": v_ft,
    }

    for row_idx, (label, data) in enumerate(stokes_list):
        left_gs = gs[row_idx, 0].subgridspec(2, 1, height_ratios=[1.0, 4.0], hspace=0.05)
        ax_prof = fig.add_subplot(left_gs[0, 0])
        ax_dyn = fig.add_subplot(left_gs[1, 0])
        ax_spec = fig.add_subplot(gs[row_idx, 1])

        _plot_profile(ax_prof, label, data)
        _plot_dyn(ax_dyn, ft_map[label], f"Stokes {label} dspec")
        _plot_spectrum(ax_spec, label, data)

        if row_idx == (n_rows - 1):
            ax_dyn.set_xlabel(f"Time ({time_unit})")
        else:
            ax_dyn.tick_params(axis="x", labelbottom=False)
            ax_spec.tick_params(axis="x", labelbottom=False)

    if output_file:
        savefig_rasterized(output_file, dpi=300, bbox_inches="tight", fig=fig)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main() -> None:
    """CLI entry point for Stokes cube summary plotting."""
    parser = argparse.ArgumentParser(
        description="Plot Stokes I/Q/U/V dynamic spectra with pulse profile and spectrum",
    )
    parser.add_argument("cube", help="Path to Stokes cube (I,Q,U,(V))")
    parser.add_argument("--freq", dest="freq_file", default=None, help="Frequency axis file")
    parser.add_argument("--time", dest="time_file", default=None, help="Time axis file")
    parser.add_argument("--stokes-axis", type=int, default=0, help="Stokes axis index (default: 0)")
    parser.add_argument("--freq-axis", type=int, default=0, help="Frequency axis index (default: 0)")
    parser.add_argument("--time-axis", type=int, default=1, help="Time axis index (default: 1)")
    parser.add_argument("--freq-unit", default="MHz", help="Frequency unit label")
    parser.add_argument("--time-unit", default="ms", help="Time unit label")
    parser.add_argument("-o", "--output", default=None, help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    parser.add_argument('--pub-col', type=int, default=1, help='Publication figure column count (1, 2, 3, ...). Default: 1')
    args = parser.parse_args()
    set_pub_col(args.pub_col)

    plot_stokes_cube_summary(
        args.cube,
        freq_file=args.freq_file,
        time_file=args.time_file,
        stokes_axis=args.stokes_axis,
        freq_axis=args.freq_axis,
        time_axis=args.time_axis,
        freq_unit=args.freq_unit,
        time_unit=args.time_unit,
        output_file=args.output,
        show=args.show,
        use_pub_style=True,
    )


if __name__ == "__main__":
    main()
