"""Shared imports and SHRINE setup for dm_optimisation."""

import argparse
import contextlib
import importlib.util
import io
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from frbop.utils.plotting import savefig_rasterized, set_pub_style
from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual

try:
	from numba import njit
	_NUMBA_AVAILABLE = True
except Exception:
	njit = None
	_NUMBA_AVAILABLE = False

warnings.filterwarnings('ignore')


def _resolve_shrine_python_dir() -> Path:
	"""Locate SHRINE/python for FRBop (dmop/SHRINE) and standalone (repo/SHRINE) layouts."""
	here = Path(__file__).resolve().parent
	for candidate in (here / "SHRINE" / "python", here.parent / "SHRINE" / "python"):
		if (candidate / "dm_processing.py").is_file():
			return candidate
	raise FileNotFoundError(
		"Could not find SHRINE/python/dm_processing.py. "
		f"Tried: {here / 'SHRINE' / 'python'} and {here.parent / 'SHRINE' / 'python'}"
	)


_SHRINE_PATH = _resolve_shrine_python_dir()
sys.path.insert(0, str(_SHRINE_PATH))
dm_processing_path = _SHRINE_PATH / "dm_processing.py"
spec = importlib.util.spec_from_file_location("shrine_dm_processing", dm_processing_path)
assert spec is not None and spec.loader is not None
dm_processing_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm_processing_mod)
shrine_get_kc = dm_processing_mod.get_kc
shrine_lowpass_smooth = dm_processing_mod.lowpass_smooth
shrine_get_ranges_above_max = dm_processing_mod.get_ranges_above_max
shrine_uncertainty_calc = dm_processing_mod.uncertainty_calc
