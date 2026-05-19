"""Shared plotting helpers used across FRBop scripts."""

from __future__ import annotations

import contextlib
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE_COLUMN_WIDTH_IN = 4.8
TWO_COLUMN_WIDTH_IN = 7.1


def pub_figsize(*,
                single_column: bool = True,
                height_ratio: float = 0.62,
                min_height: float = 3.0) -> tuple[float, float]:
    """Return a publication-friendly figure size in inches."""
    width = SINGLE_COLUMN_WIDTH_IN if single_column else TWO_COLUMN_WIDTH_IN
    height = max(min_height, width * height_ratio)
    return width, height


def set_pub_style(use_latex: bool = True) -> None:
    """Publication-oriented Matplotlib defaults."""
    fig_w, fig_h = pub_figsize(single_column=True)
    mpl.rcParams.update(
        {
            "figure.figsize": (fig_w, fig_h),
            "figure.dpi": 150,
            "savefig.dpi": 600,

            # Fonts
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["TeX Gyre Pagella"],
            "mathtext.fontset": "stix",

            # Font sizes
            "font.size": 12.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 10,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,

            # Lines/ticks
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",

            # Vector font embedding
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # TeX rendering
            "text.usetex": use_latex,
            "text.latex.preamble": r"""
                \usepackage{amsmath}
                \usepackage{amssymb}
                \usepackage{lmodern}
            """,
        }
    )


def savefig_rasterized(save_path: str,
                       dpi: int = 300,
                       bbox_inches: str = "tight",
                       fig: Optional[plt.Figure] = None) -> None:
    """Save figure with artists rasterized to keep vector outputs lightweight."""
    out_fig = fig if fig is not None else plt.gcf()
    for ax in out_fig.axes:
        for artist in ax.get_children():
            if hasattr(artist, "set_rasterized"):
                with contextlib.suppress(Exception):
                    artist.set_rasterized(True)
    out_fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
