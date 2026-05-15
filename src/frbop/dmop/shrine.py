"""
SHRINE integration layer.

Handles:
- Loading SHRINE Python helpers from the embedded subpackage.
- kc low-pass filtering via DCT.
- Running SHRINE scripts (maximise_structure.py, maximise_sn.py, etc.) in
  isolated working directories.
- Auto/manual kc resolution for non-SHRINE methods.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.fftpack import dct

# ---------------------------------------------------------------------------
# Load SHRINE helpers from the embedded subpackage at import time
# ---------------------------------------------------------------------------

_SHRINE_PATH = Path(__file__).resolve().parent / "SHRINE" / "python"
sys.path.insert(0, str(_SHRINE_PATH))

_dm_processing_path = _SHRINE_PATH / "dm_processing.py"
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("shrine_dm_processing", _dm_processing_path)
assert _spec is not None and _spec.loader is not None
_dm_processing_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dm_processing_mod)

get_kc = _dm_processing_mod.get_kc
lowpass_smooth = _dm_processing_mod.lowpass_smooth
get_ranges_above_max = _dm_processing_mod.get_ranges_above_max
uncertainty_calc = _dm_processing_mod.uncertainty_calc


# ---------------------------------------------------------------------------
# kc low-pass filter
# ---------------------------------------------------------------------------

def apply_kc_lowpass_2d(data_2d: np.ndarray, kc: int) -> np.ndarray:
    """
    Apply the SHRINE DCT low-pass filter to *data_2d* (freq × time or 1 × time).

    Returns a smoothed array of the same shape.
    """
    if data_2d.ndim != 2:
        raise ValueError("data_2d must be 2D (freq × time)")
    if kc <= 0:
        return data_2d.copy()

    ci_data = dct(data_2d, norm="ortho")
    k_length = ci_data.shape[1]
    kc_eff = max(1, min(int(kc), k_length))
    i_smooth, _, _, _ = lowpass_smooth(ci_data, kc_eff, order=3)
    return i_smooth


# ---------------------------------------------------------------------------
# kc resolution (non-SHRINE methods)
# ---------------------------------------------------------------------------

class KcResolver:
    """
    Manages kc resolution state for a single DM optimisation run.

    One instance per ``compare_methods`` / ``optimise_dm_*`` call so that kc
    is computed once and reused, matching the original behaviour.
    """

    def __init__(
        self,
        fixed_kc: Optional[int] = None,
        use_minimise_uncertainty: bool = False,
    ) -> None:
        self._fixed_kc = fixed_kc
        self._use_minimise_uncertainty = use_minimise_uncertainty
        self._resolved: Optional[int] = None
        self._printed = False

    def reset(self) -> None:
        self._resolved = None
        self._printed = False

    @property
    def resolved(self) -> Optional[int]:
        return self._resolved

    def resolve(self, reference_data_2d: np.ndarray) -> int:
        """
        Return the kc value to use, computing it if necessary.
        """
        if self._resolved is not None:
            if not self._printed:
                print(f"Found kc of: {self._resolved}")
                self._printed = True
            return self._resolved

        if self._fixed_kc is not None:
            self._resolved = int(self._fixed_kc)
            if not self._printed:
                print(f"Found kc of: {self._resolved}")
                self._printed = True
            return self._resolved

        ci_data = dct(reference_data_2d, norm="ortho")
        with contextlib.redirect_stdout(io.StringIO()):
            kc = int(get_kc(ci_data))
        self._resolved = kc
        if not self._printed:
            print(f"Found kc of: {self._resolved}")
            self._printed = True
        return self._resolved


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_shrine_script(
    script_name: str,
    run_prefix: str,
    dm_values: np.ndarray,
    i_data: np.ndarray,
    time_ms: np.ndarray,
    input_dm: float = 0.0,
    include_input_dm: bool = False,
    force_kc: Optional[int] = None,
    save_all: bool = True,
) -> Path:
    """
    Run a SHRINE script in an isolated working directory.

    Writes ``{run_prefix}_DMs.npy`` and ``{run_prefix}_I_{dt_us}us.npy``
    before invoking the script via ``python -m``.

    Returns the run directory path.
    """
    dt_ms = float(np.median(np.diff(time_ms))) if len(time_ms) > 1 else 1.0
    dt_us = max(1, int(round(dt_ms * 1000.0)))

    run_dir = Path("shrine_logs") / run_prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
    np.save(run_dir / f"{run_prefix}_I_{dt_us}us.npy", i_data)

    module_name = script_name[:-3] if script_name.endswith(".py") else script_name
    cmd = [
        sys.executable,
        "-m",
        f"frbop.dmop.SHRINE.python.{module_name}",
        "-l", run_prefix,
        "-t", str(dt_us),
    ]
    if include_input_dm:
        cmd.extend(["-d", str(input_dm)])
    if save_all:
        cmd.append("-s")
    if force_kc is not None:
        cmd.extend(["-kc", str(force_kc)])

    subprocess.run(cmd, cwd=str(run_dir), check=True)
    return run_dir


# ---------------------------------------------------------------------------
# Convenience: maybe apply kc smoothing to a set of Stokes arrays
# ---------------------------------------------------------------------------

def maybe_kc_smooth(
    data_i: Optional[np.ndarray],
    data_q: Optional[np.ndarray],
    data_u: Optional[np.ndarray],
    kc_resolver: KcResolver,
    enabled: bool,
) -> tuple:
    """
    Return ``(sm_i, sm_q, sm_u)``, applying the kc low-pass filter when
    *enabled* is True.  If disabled, returns the inputs unchanged.
    """
    if not enabled:
        return data_i, data_q, data_u

    reference = data_i if data_i is not None else data_q
    if reference is None:
        return data_i, data_q, data_u

    kc = kc_resolver.resolve(reference)
    sm_i = apply_kc_lowpass_2d(data_i, kc) if data_i is not None else None
    sm_q = apply_kc_lowpass_2d(data_q, kc) if data_q is not None else None
    sm_u = apply_kc_lowpass_2d(data_u, kc) if data_u is not None else None
    return sm_i, sm_q, sm_u
