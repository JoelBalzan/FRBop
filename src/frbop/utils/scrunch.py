"""Shared scrunching helpers for time/frequency axes and peak indices."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def tscrunch_array(arr: np.ndarray, factor: int, axis: int = -1) -> np.ndarray:
    """Average an array along one axis in groups of ``factor`` consecutive bins.

    Trailing bins beyond a whole number of factors are dropped. NaN entries
    are ignored (NaN-safe mean).
    """
    data = np.asarray(arr)
    if factor <= 1:
        return data
    if data.shape[axis] < factor:
        raise ValueError(
            f"--tscrunch factor {factor} is larger than the time axis length {data.shape[axis]}"
        )

    axis_norm = axis % data.ndim
    n_keep = data.shape[axis] // factor
    if n_keep <= 0:
        raise ValueError(f"--tscrunch factor {factor} leaves no complete time bins to analyse")

    kept = np.take(data, np.arange(n_keep * factor), axis=axis)
    new_shape = kept.shape[:axis_norm] + (n_keep, factor) + kept.shape[axis_norm + 1 :]
    return np.nanmean(kept.reshape(new_shape), axis=axis_norm + 1)


def rescale_peak_indices(indices: Sequence[int] | None, factor: int) -> list[int]:
    """Map original-resolution peak bounds onto a scrunched time axis."""
    if factor <= 1 or not indices:
        return list(indices) if indices is not None else []

    values = list(indices)
    if len(values) % 2 != 0:
        raise ValueError("--peak-indices requires an even number of values (pairs of start/end indices)")

    scaled: list[int] = []
    for i in range(0, len(values), 2):
        start = int(values[i])
        end = int(values[i + 1])
        scaled.append(start // factor)
        scaled.append(-(-end // factor))
    return scaled