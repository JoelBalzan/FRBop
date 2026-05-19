"""Shared plotting helpers used across FRBop scripts."""

from __future__ import annotations

import contextlib
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from itertools import cycle


SINGLE_COLUMN_WIDTH_IN = 4.8
TWO_COLUMN_WIDTH_IN = 7.1


class ColorManager:
    """
    Stable color assignment for method/label names across all figures.
    """
    def __init__(self, palette):
        self.palette = palette
        self._map = {}
        self._cycle = cycle(palette)

    def color(self, key: str):
        if key not in self._map:
            self._map[key] = next(self._cycle)
        return self._map[key]

    def reset(self):
        """Optional: reset mapping (useful for reproducible figures)."""
        self._map = {}
        self._cycle = cycle(self.palette)


IBM_PALETTE = [
    "#648fff",
    "#785ef0",
    "#dc267f",
    "#fe6100",
    "#ffb000",
]
colour_manager = ColorManager(IBM_PALETTE)


def pub_width(single_column=True):
    return SINGLE_COLUMN_WIDTH_IN if single_column else TWO_COLUMN_WIDTH_IN


def pub_figsize(
    *,
    single_column=True,
    height_ratio=0.62,
    min_height=3.0,
):
    width = pub_width(single_column)
    height = max(min_height, width * height_ratio)
    return width, height


def pub_grid_figsize(
    n_rows,
    *,
    single_column=False,
    row_height=2.5,
    width_scale=1.0,
    min_height=4.0,
):
    width = pub_width(single_column) * width_scale
    height = max(min_height, n_rows * row_height)
    return width, height


def set_pub_style(use_latex=False):

    fig_w, fig_h = pub_figsize()

    rc = {
        "figure.figsize": (fig_w, fig_h),
        "figure.dpi": 150,
        "savefig.dpi": 600,

        # Fonts
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella"],
        "mathtext.fontset": "stix",

        # Font sizes
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        # Lines/ticks
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # Minor ticks
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # Layout
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,

        # Legend
        "legend.frameon": True,
        "legend.handlelength": 1.4,

        # Colours
        "axes.prop_cycle": cycler(color=IBM_PALETTE),

        # Vector embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # TeX
        "text.usetex": use_latex,
    }

    if use_latex:
        rc["text.latex.preamble"] = r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{lmodern}
        """

    mpl.rcParams.update(rc)


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
