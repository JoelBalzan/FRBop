"""Stokes cube visualization helpers."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    pub_col: int = 1,
) -> plt.Figure:
    """Plot I/Q/U/V dynamic spectra with attached pulse profile and spectrum.

    Each Stokes parameter gets one row: a square dynamic spectrum panel,
    with its time profile directly above (sharing the time axis) and its
    frequency spectrum directly to the right (sharing the frequency axis).
    Only the dynamic spectrum panel is labeled per row (with the Stokes
    letter); axis labels for frequency/time only appear on the outer edges
    of the figure to avoid clutter.

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
        set_pub_col(pub_col)

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
    ]
    stokes_list += [("Q", stokes_q), ("U", stokes_u)]
    if stokes_v is not None:
        stokes_list.append(("V", stokes_v))

    ft_map = {"I": i_ft, "Q": q_ft, "U": u_ft, "V": v_ft}

    n_rows = len(stokes_list)

    # ------------------------------------------------------------------
    # Build the whole figure on a single GridSpec whose row/column sizes
    # are specified directly in inches. Because we also set the figure's
    # overall size (and margins) from those same inch values, each
    # GridSpec unit maps to exactly one inch on the page. That guarantees
    # the dspec panel is a perfect square and that the profile (above)
    # and spectrum (right) panels are flush against it -- no aspect
    # tricks, no post-hoc axis repositioning required.
    # ------------------------------------------------------------------
    main_size = 2.0     # dspec panel: width == height (inches) -> square
    prof_h = 0.55        # time-profile strip height
    spec_w = 0.55        # frequency-spectrum strip width
    col_gap = 0.05       # gap between dspec and spectrum panel
    row_gap = 0.35        # gap between one Stokes block and the next
    left_margin = 0.85    # room for "Stokes X" / freq tick labels
    right_margin = 0.1
    top_margin = 0.1
    bottom_margin = 0.55  # room for time tick labels / x-axis label

    fig_w = left_margin + main_size + col_gap + spec_w + right_margin
    fig_h = (
        top_margin
        + n_rows * (prof_h + main_size)
        + (n_rows - 1) * row_gap
        + bottom_margin
    )
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Interleave [profile, dspec] rows for each Stokes parameter, with a
    # blank spacer row (row_gap) between consecutive blocks only.
    height_ratios = []
    block_rows = []  # (prof_row_idx, dyn_row_idx) per Stokes parameter
    grid_row = 0
    for row_idx in range(n_rows):
        if row_idx > 0:
            height_ratios.append(row_gap)
            grid_row += 1
        height_ratios.append(prof_h)
        prof_row = grid_row
        grid_row += 1
        height_ratios.append(main_size)
        dyn_row = grid_row
        grid_row += 1
        block_rows.append((prof_row, dyn_row))

    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=3,
        height_ratios=height_ratios,
        width_ratios=[main_size, col_gap, spec_w],
        left=left_margin / fig_w,
        right=1 - right_margin / fig_w,
        top=1 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        hspace=0,
        wspace=0,
    )

    for row_idx, (label, data) in enumerate(stokes_list):
        prof_row, dyn_row = block_rows[row_idx]

        ax_prof = fig.add_subplot(gs[prof_row, 0])
        ax_dyn = fig.add_subplot(gs[dyn_row, 0])
        ax_spec = fig.add_subplot(gs[dyn_row, 2])

        # --- dynamic spectrum (square by construction) ---
        dyn_data = ft_map[label]
        vmin, vmax = _safe_limits(dyn_data)
        extent = [
            float(time_axis_vals[0]),
            float(time_axis_vals[-1]),
            float(freq_axis_vals[0]),
            float(freq_axis_vals[-1]),
        ]
        ax_dyn.imshow(
            dyn_data,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax_dyn.tick_params(axis="both", which="major", colors="white", labelcolor="black", direction="in")
        ax_dyn.tick_params(axis="both", which="minor", colors="white", direction="in")

        # Stokes letter labels the whole row (placed left of the dspec panel).
        # Only the bottom row also carries the shared frequency-axis label.
        row_label = f"Freq ({freq_unit})\n\nStokes {label}" if row_idx == n_rows - 1 else f"Stokes {label}"
        ax_dyn.set_ylabel(row_label, labelpad=8)

        if row_idx == n_rows - 1:
            ax_dyn.set_xlabel(f"Time ({time_unit})")
        else:
            ax_dyn.tick_params(axis="x", labelbottom=False)

        # --- time profile, attached above dspec, sharing x-axis ---
        profile = np.nanmean(data, axis=freq_axis)
        ax_prof.plot(time_axis_vals, profile, color="black", linewidth=1.0)
        ax_prof.set_xlim(time_axis_vals[0], time_axis_vals[-1])
        ax_prof.sharex(ax_dyn)
        ax_prof.tick_params(axis="x", labelbottom=False)
        ax_prof.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="both"))
        ax_prof.tick_params(axis="y", labelsize="small")
        ax_prof.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right", "left"):
            ax_prof.spines[spine].set_visible(False)

        # --- frequency spectrum, attached to the right of dspec, sharing y-axis ---
        spectrum = np.nanmean(data, axis=time_axis)
        ax_spec.plot(spectrum, freq_axis_vals, color="black", linewidth=1.0)
        ax_spec.set_ylim(freq_axis_vals[0], freq_axis_vals[-1])
        ax_spec.sharey(ax_dyn)
        ax_spec.tick_params(axis="y", labelleft=False)
        ax_spec.set_xticks([])
        ax_spec.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right", "bottom"):
            ax_spec.spines[spine].set_visible(False)

    if output_file:
        savefig_rasterized(output_file, dpi=300, fig=fig)

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
        pub_col=args.pub_col,
    )


if __name__ == "__main__":
    main()