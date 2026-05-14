"""Shared plotting helpers used across FRBop scripts."""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
DEFAULT_PUBLICATION_STYLE: Dict[str, float] = {
    "title": 11,
    "label": 10,
    "tick": 11,
    "legend": 10,
    "annotation": 7,
    "line": 1.4,
}


def publication_plot_style() -> Dict[str, float]:
    """Return publication-style font/line sizes used by multiple pipelines."""
    return dict(DEFAULT_PUBLICATION_STYLE)


def apply_cm_math_style(font_size: float = 14) -> None:
    """Apply a Computer Modern-style font setup for math-heavy plots."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'CMU Serif'
    plt.rcParams['mathtext.it'] = 'CMU Serif:italic'
    plt.rcParams['mathtext.bf'] = 'CMU Serif:bold'
    plt.rcParams['mathtext.sf'] = 'CMU Sans Serif'
    plt.rcParams['mathtext.tt'] = 'CMU Typewriter Text'
    plt.rcParams['mathtext.cal'] = 'CMU Serif:italic'  # Or use 'stix:italic'
    plt.rcParams["font.size"] = font_size


def set_pub_style(use_latex: bool = True) -> None:
    """Use conservative publication-oriented Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": use_latex,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
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
