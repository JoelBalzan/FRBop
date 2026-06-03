"""Shared imports and SHRINE setup for dm_optimisation."""

import importlib.util
import sys
import warnings
from pathlib import Path


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
