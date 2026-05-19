"""
Shared layout constants and plot-style helpers.
"""

from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

TWO_COLUMN_WIDTH_IN = 7.1
SINGLE_COLUMN_WIDTH_IN = 4.8


def pub_figsize(height_ratio: float = 0.62, min_height: float = 3.0) -> Tuple[float, float]:
    """Return a publication-friendly figure size for a two-column layout."""
    height = max(min_height, TWO_COLUMN_WIDTH_IN * height_ratio)
    return TWO_COLUMN_WIDTH_IN, height


def plot_style() -> Dict[str, float]:
    """Compatibility wrapper for plotting style used by RM plotting functions."""
    base_font_size = float(plt.rcParams.get("font.size", 10))
    font_scalings = mpl.font_manager.font_scalings

    def _resolve_font_size(value: object, fallback: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            scale = font_scalings.get(value.lower())
            if scale is not None:
                return float(base_font_size * scale)
        return float(fallback)

    return {
        "title": _resolve_font_size(plt.rcParams.get("axes.titlesize", 11), 11.0),
        "label": _resolve_font_size(plt.rcParams.get("axes.labelsize", 10), 10.0),
        "tick": _resolve_font_size(plt.rcParams.get("xtick.labelsize", 10), 10.0),
        "legend": _resolve_font_size(plt.rcParams.get("legend.fontsize", 10), 10.0),
        "annotation": base_font_size,
        "line": float(plt.rcParams.get("lines.linewidth", 1.4)),
    }
