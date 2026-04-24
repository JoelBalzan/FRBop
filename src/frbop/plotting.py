"""Shared plotting helpers used across FRBop scripts."""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import matplotlib.pyplot as plt


DEFAULT_PUBLICATION_STYLE: Dict[str, float] = {
    "title": 11,
    "label": 10,
    "tick": 8,
    "legend": 8,
    "annotation": 7,
    "line": 1.4,
}


def publication_plot_style() -> Dict[str, float]:
    """Return publication-style font/line sizes used by multiple pipelines."""
    return dict(DEFAULT_PUBLICATION_STYLE)


def apply_cm_math_style(font_size: float = 14) -> None:
    """Apply a Computer Modern-style font setup for math-heavy plots."""
    plt.rcParams["font.family"] = "cm"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = font_size


def apply_dark_background() -> None:
    """Apply a dark background style used by some diagnostic scripts."""
    plt.style.use("dark_background")


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
