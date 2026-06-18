"""
Shared layout constants and plot-style helpers.
"""

from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

from frbop.utils.plotting import pub_figsize as _pub_figsize, set_pub_col as _set_pub_col_utils


def set_pub_col(n: Optional[int]) -> None:
    """Set publication column count, delegating to the global utils setting."""
    _set_pub_col_utils(n)


def pub_figsize(height_ratio: float = 0.62, ncol: Optional[int] = None) -> Tuple[float, float]:
    """Return a publication-friendly figure size for the given column count."""
    return _pub_figsize(ncol=ncol, height_ratio=height_ratio)


def _pub_scale() -> float:
    """Stub: always returns 1.0 (font/line scaling removed)."""
    return 1.0


def plot_style() -> Dict[str, float]:
    """Compatibility wrapper for plotting style used by RM plotting functions."""
    base_font_size = float(plt.rcParams.get("font.size", 10))
    font_scalings = mpl.font_manager.font_scalings
    sc = _pub_scale()

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
        "marker": float(plt.rcParams.get("lines.markersize", 6)),
        "cap": float(plt.rcParams.get("errorbar.capsize", 3)),
        "scale": sc,
    }
