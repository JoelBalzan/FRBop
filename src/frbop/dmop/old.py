#!/usr/bin/env python3
"""
DM Correction Optimisation Methods Comparison

This script implements and compares different methods for optimising 
dispersion measure (DM) correction for Fast Radio Bursts (FRBs).

Methods implemented:
1. Structure Maximising (SHRINE; Sutinjo et al. 2023)
2. S/N Maximising
3. PA Slope Maximising
4. L/I Maximising

Author: Joel
Date: February 2026
"""

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

from frbop.utils.plotting import pub_figsize, savefig_rasterized, set_pub_style
from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.peaks import select_peaks_manual as shared_select_peaks_manual

try:
	from numba import njit
	_NUMBA_AVAILABLE = True
except Exception:
	njit = None
	_NUMBA_AVAILABLE = False

warnings.filterwarnings('ignore')

_SHRINE_PATH = Path(__file__).resolve().parent / "SHRINE" / "python"
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


if _NUMBA_AVAILABLE:
	@njit
	def _max_snr_for_series(series: np.ndarray, noise_std: float,
							min_window_size: int, max_window_size: int) -> float:
		n = series.shape[0]
		csum = np.empty(n + 1, dtype=np.float64)
		csum[0] = 0.0
		for i in range(n):
			csum[i + 1] = csum[i] + series[i]

		max_sn = -1e30
		for start in range(0, n - min_window_size):
			max_len = max_window_size
			if start + max_len > n:
				max_len = n - start
			for length in range(min_window_size, max_len):
				s = csum[start + length] - csum[start]
				sn = s / (noise_std * np.sqrt(length))
				if sn > max_sn:
					max_sn = sn
		return max_sn

	@njit
	def _apply_shifts_numba(data: np.ndarray, delay_samples: np.ndarray, 
							 noise_fill: np.ndarray, start_idx: int = 0) -> np.ndarray:
		n_freq, n_time = data.shape
		n_freq_out, n_time_out = noise_fill.shape
		out = noise_fill.copy()
		# Place shifted data
		for i in range(n_freq):
			shift = delay_samples[i]
			for t in range(n_time):
				t_out = t + shift - start_idx
				if 0 <= t_out < n_time_out:
					out[i, t_out] = data[i, t]
		return out


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="DM correction optimisation with configurable inputs",
	)
	parser.add_argument(
		"-d", "--stokes-cube",
		default=None,
		help="Optional path to a single numpy file containing Stokes I/Q/U(/V) cube. Accepted shapes: (4,freq,time), (3,freq,time), (freq,time,4) or (freq,time,3)",
	)
	parser.add_argument(
		"--stokes-i",
		default=None,
		help="Path to Stokes I numpy file (freq x time)",
	)
	parser.add_argument(
		"--stokes-q",
		default=None,
		help="Optional path to Stokes Q numpy file (freq x time)",
	)
	parser.add_argument(
		"--stokes-u",
		default=None,
		help="Optional path to Stokes U numpy file (freq x time)",
	)
	parser.add_argument(
		"--freq",
		default="freq.npy",
		help="Path to frequency array numpy file (MHz)",
	)
	parser.add_argument(
		"--time",
		default="time.npy",
		help="Path to time array numpy file (ms)",
	)
	parser.add_argument(
		"--dm-min",
		type=float,
		default=None,
		help="Minimum DM to search (pc cm^-3). Use with --dm-max",
	)
	parser.add_argument(
		"--dm-max",
		type=float,
		default=None,
		help="Maximum DM to search (pc cm^-3). Use with --dm-min",
	)
	parser.add_argument(
		"--dm-guess",
		type=float,
		default=None,
		help="Starting DM guess (pc cm^-3) used to build a default range",
	)
	parser.add_argument(
		"--dm-step",
		type=float,
		default=None,
		help="Optional DM step for DM-space scan (pc cm^-3). Overrides default sampling.",
	)
	parser.add_argument(
		"--ref-freq",
		type=float,
		default=None,
		help="Reference frequency in MHz for dedispersion (defaults to max frequency in file)",
	)
	parser.add_argument(
		"--input-dm",
		type=float,
		default=0.0,
		help="DM already applied to the input data (pc cm^-3). Optimisation uses input_dm - dm.",
	)
	parser.add_argument(
		"--separate-peaks",
		action="store_true",
		help="Enable peak separation; otherwise, operate on the full dataset",
	)
	parser.add_argument(
		"--manual-peaks",
		action="store_true",
		help="Manually select peak bounds by clicking on the pulse profile",
	)
	parser.add_argument(
		"--peak-indices",
		nargs="*",
		type=int,
		default=None,
		help="Manually specify peak indices as pairs: start1 end1 start2 end2 ...",
	)
	parser.add_argument(
		"--dedisp-mode",
		type=str,
		choices=["expand", "crop"],
		default="expand",
		help="Dedispersion mode: 'expand' (fill edges with noise) or 'crop' (trim to common valid region)",
	)
	parser.add_argument(
		"--fast",
		action="store_true",
		help="Speedy test mode: uses fewer DM samples and skips full DM scan",
	)
	parser.add_argument(
		"--pa-fit-degree",
		type=int,
		default=1,
		help="Polynomial degree for PA profile fitting (default: 1 for linear)",
	)
	parser.add_argument(
		"--pa-weight-strength",
		type=float,
		default=1.0,
		help="Strength of PA fit weighting (power on normalised weights; 1.0 = current behaviour, >1 stronger)",
	)
	parser.add_argument(
		"--pa-fit-post-peak-only",
		action="store_true",
		help="Restrict PA fitting/masking to samples at or after the Stokes-I peak (default uses pre-peak too)",
	)
	parser.add_argument(
		"--pa",
		action="store_true",
		help="Enable PA workflow flag (PA methods already run automatically when Q/U are provided; kept for CLI compatibility)",
	)
	parser.add_argument(
		"--nonshrine-kc-smooth",
		action="store_true",
		help="Apply SHRINE-style kc low-pass smoothing to PA/LI methods",
	)
	parser.add_argument(
		"--nonshrine-shrine-like-errors",
		action="store_true",
		help="Use SHRINE-style relative-uncertainty error bars for non-SHRINE PA/LI methods without requiring kc smoothing",
	)
	parser.add_argument(
		"--nonshrine-kc",
		type=int,
		default=None,
		help="Fixed kc value for non-SHRINE smoothing (default: auto per trial)",
	)
	parser.add_argument(
		"--nonshrine-kc-minimise-uncertainty",
		action="store_true",
		help="Find non-SHRINE kc by running SHRINE minimise_uncertainty.py (writes/reads kc.txt in run dir)",
	)
	parser.add_argument(
		"--li-sig",
		type=float,
		default=2.0,
		help="Stokes I sigma cutoff for L/I mean masking (default: 2.0)",
	)
	parser.add_argument(
		"--debias-linear",
		action="store_true",
		help="Enable linear-polarisation debiasing for PA/L/I metrics.",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		choices=["structure", "snr", "pa", "pa-shrine", "li"],
		default=None,
		help="Methods to run (default: all). Choices: structure, snr, pa, pa-shrine, li",
	)
	parser.add_argument(
		"--exclude-methods",
		nargs="+",
		choices=["structure", "snr", "pa", "pa-shrine", "li"],
		default=None,
		help="Methods to exclude from run/plots/analysis",
	)
	parser.add_argument(
		"--label",
		type=str,
		default="frb",
		help="FRB label for output files (default: 'frb')",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=1234,
		help="Random seed for reproducible noise fill during dedispersion. Default: 1234",
	)
	parser.add_argument(
		"--ext",
		type=str,
		default="png",
		help="Figure extension for saved plots (e.g. png, pdf, svg). Default: png",
	)
	return parser.parse_args()


class DMOptimiser:
	"""
	Class for optimising DM correction using various methods.
	"""
	
	def __init__(self, stokes_i: np.ndarray, freq_mhz: np.ndarray, time_ms: np.ndarray,
				 stokes_q: Optional[np.ndarray] = None, stokes_u: Optional[np.ndarray] = None,
				 reference_freq: Optional[float] = None,
				 input_dm: float = 0.0,
				 dedisp_mode: str = 'expand',
				 pa_fit_degree: int = 1,
				 pa_weight_strength: float = 1.0,
				 pa_fit_post_peak_only: bool = False,
				 nonshrine_kc_smooth: bool = False,
				 nonshrine_shrine_like_errors: bool = False,
				 nonshrine_kc_minimise_uncertainty: bool = False,
				 nonshrine_kc: Optional[int] = None,
				 li_i_sigma_cut: float = 2.0,
				 debias_linear: bool = False,
				 random_seed: Optional[int] = None):
		"""
		Initialize the DM optimiser.
		
		Parameters:
		-----------
		stokes_i : np.ndarray
			2D array of Stokes I data (freq x time)
		freq_mhz : np.ndarray
			Frequency array in MHz
		time_ms : np.ndarray
			Time array in ms
		"""
		self.stokes_i = stokes_i
		self.stokes_q = stokes_q
		self.stokes_u = stokes_u
		self.freq_mhz = freq_mhz
		self.time_ms = time_ms
		self.n_freq, self.n_time = stokes_i.shape
		self.reference_freq = reference_freq if reference_freq is not None else np.max(freq_mhz)
		self.input_dm = float(input_dm)
		self.dedisp_mode = dedisp_mode
		self.pa_fit_degree = int(pa_fit_degree)
		self.pa_weight_strength = float(pa_weight_strength)
		if self.pa_weight_strength <= 0:
			raise ValueError("pa_weight_strength must be positive")
		self.pa_fit_post_peak_only = bool(pa_fit_post_peak_only)
		self.nonshrine_kc_smooth = bool(nonshrine_kc_smooth)
		self.nonshrine_shrine_like_errors = bool(nonshrine_shrine_like_errors)
		self.use_nonshrine_shrine_like_uncertainty = bool(
			self.nonshrine_kc_smooth or self.nonshrine_shrine_like_errors
		)
		self.nonshrine_kc_minimise_uncertainty = bool(nonshrine_kc_minimise_uncertainty)
		self.nonshrine_kc = None if nonshrine_kc is None else int(nonshrine_kc)
		if self.nonshrine_kc is not None and self.nonshrine_kc <= 0:
			raise ValueError("nonshrine_kc must be positive")
		self._nonshrine_resolved_kc: Optional[int] = None
		self._nonshrine_kc_printed = False
		self.li_i_sigma_cut = float(li_i_sigma_cut)
		if self.li_i_sigma_cut <= 0:
			raise ValueError("li_i_sigma_cut must be positive")
		self.debias_linear = bool(debias_linear)
		self.random_seed = random_seed
		self.rng = np.random.default_rng(random_seed)

		self.full_i_time_series = np.mean(self.stokes_i, axis=0)
		self.full_i_noise_median, self.full_i_noise_std = self._noise_stats_from_series(self.full_i_time_series)
		if self.stokes_q is not None and self.stokes_u is not None:
			full_L = np.sqrt(self.stokes_q**2 + self.stokes_u**2)
			self.full_L_time = np.mean(full_L, axis=0)
			self.full_L_noise_median, self.full_L_noise_std = self._noise_stats_from_series(self.full_L_time)
			self.full_q_time_series = np.mean(self.stokes_q, axis=0)
			self.full_u_time_series = np.mean(self.stokes_u, axis=0)
			_, self.full_q_time_noise_std = self._noise_stats_from_series(self.full_q_time_series)
			_, self.full_u_time_noise_std = self._noise_stats_from_series(self.full_u_time_series)
			n_edge_full = max(1, int(0.05 * self.stokes_q.shape[1]))
			self.full_q_noise_rms = np.std(self.stokes_q[:, :n_edge_full], axis=1, keepdims=True)
			self.full_u_noise_rms = np.std(self.stokes_u[:, :n_edge_full], axis=1, keepdims=True)
		else:
			self.full_L_time = None
			self.full_L_noise_median = None
			self.full_L_noise_std = None
			self.full_q_time_series = None
			self.full_u_time_series = None
			self.full_q_time_noise_std = None
			self.full_u_time_noise_std = None
			self.full_q_noise_rms = None
			self.full_u_noise_rms = None
		
		# DM constant: k = 4.148808e6 ms MHz^2 pc^-1 cm^3 (delay between frequencies)
		# From pulsar handbook: dt = 4.15 × 10^6 ms × (f1^-2 - f2^-2) × DM
		self.DM_CONSTANT = 4.148808e6

	def _apply_kc_lowpass_2d(self, data_2d: np.ndarray, kc: int) -> np.ndarray:
		"""
		Apply SHRINE low-pass filter using the shared SHRINE implementation.
		"""
		if data_2d.ndim != 2:
			raise ValueError("data_2d must be 2D (freq x time)")
		if kc <= 0:
			return data_2d.copy()

		ci_data = dct(data_2d, norm='ortho')
		k_length = ci_data.shape[1]
		kc_eff = max(1, min(int(kc), k_length))
		i_smooth, _, _, _ = shrine_lowpass_smooth(ci_data, kc_eff, order=3)
		return i_smooth

	def _resolve_nonshrine_kc(self, reference_data_2d: np.ndarray) -> int:
		if self._nonshrine_resolved_kc is not None:
			if not self._nonshrine_kc_printed:
				print(f"Found kc of: {self._nonshrine_resolved_kc}")
				self._nonshrine_kc_printed = True
			return int(self._nonshrine_resolved_kc)

		if self.nonshrine_kc is not None:
			self._nonshrine_resolved_kc = int(self.nonshrine_kc)
			if not self._nonshrine_kc_printed:
				print(f"Found kc of: {self._nonshrine_resolved_kc}")
				self._nonshrine_kc_printed = True
			return self._nonshrine_resolved_kc

		if self.nonshrine_kc_minimise_uncertainty:
			reference = np.asarray(reference_data_2d, dtype=float)
			if reference.ndim == 1:
				reference = reference[np.newaxis, :]
			ci_data = dct(reference, norm='ortho')
			with contextlib.redirect_stdout(io.StringIO()):
				seed_kc = int(shrine_get_kc(ci_data))
			dm_values = np.arange(reference.shape[0], dtype=float)
			run_prefix = "nonshrine_kc_min_unc"
			run_dir = self._run_shrine_method(
				script_name="minimise_uncertainty.py",
				run_prefix=run_prefix,
				dm_values=dm_values,
				i_data=reference,
				include_input_dm=False,
				force_kc=seed_kc,
				save_all=False,
			)
			kc_path = run_dir / "kc.txt"
			if not kc_path.exists():
				raise RuntimeError(f"Expected kc output not found: {kc_path}")
			kc_text = kc_path.read_text().strip()
			self._nonshrine_resolved_kc = int(float(kc_text))
			if self._nonshrine_resolved_kc <= 0:
				raise RuntimeError(f"Invalid kc from minimise_uncertainty: {self._nonshrine_resolved_kc}")
			if not self._nonshrine_kc_printed:
				print(f"Found kc of: {self._nonshrine_resolved_kc}")
				self._nonshrine_kc_printed = True
			return self._nonshrine_resolved_kc

		ci_data = dct(reference_data_2d, norm='ortho')
		with contextlib.redirect_stdout(io.StringIO()):
			kc = shrine_get_kc(ci_data)
		self._nonshrine_resolved_kc = int(kc)
		if not self._nonshrine_kc_printed:
			print(f"Found kc of: {self._nonshrine_resolved_kc}")
			self._nonshrine_kc_printed = True
		return self._nonshrine_resolved_kc

	def _reset_nonshrine_kc_state(self) -> None:
		self._nonshrine_resolved_kc = None
		self._nonshrine_kc_printed = False

	def _maybe_kc_smooth_nonshrine(self,
									   data_i: Optional[np.ndarray],
									   data_q: Optional[np.ndarray] = None,
									   data_u: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
		"""
		Optionally apply kc smoothing to non-SHRINE method inputs.
		"""
		if not self.nonshrine_kc_smooth:
			return data_i, data_q, data_u

		reference = data_i if data_i is not None else data_q
		if reference is None:
			return data_i, data_q, data_u

		kc = self._resolve_nonshrine_kc(reference)

		sm_i = self._apply_kc_lowpass_2d(data_i, kc) if data_i is not None else None
		sm_q = self._apply_kc_lowpass_2d(data_q, kc) if data_q is not None else None
		sm_u = self._apply_kc_lowpass_2d(data_u, kc) if data_u is not None else None
		return sm_i, sm_q, sm_u

	def _run_shrine_method(self,
					   script_name: str,
					   run_prefix: str,
					   dm_values: np.ndarray,
					   i_data: np.ndarray,
					   include_input_dm: bool = False,
					   force_kc: Optional[int] = None,
					   save_all: bool = True) -> Path:
		"""
		Run a SHRINE script in an isolated working directory using precomputed DM/I arrays.
		"""
		dt_ms = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
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
			cmd.extend(["-d", str(self.input_dm)])
		if save_all:
			cmd.append("-s")
		if force_kc is not None:
			cmd.extend(["-kc", str(force_kc)])

		subprocess.run(cmd, cwd=str(run_dir), check=True)
		return run_dir

	def _save_nonshrine_run_outputs(self,
								   run_prefix: str,
								   method_label: str,
								   dm_values: np.ndarray,
								   metric_values: np.ndarray,
								   metric_name: str,
								   dedispersed_i: np.ndarray,
								   optimal_dm: float,
								   optimal_metric: float,
								   uncertainty: Optional[Dict[str, Optional[float]]] = None) -> Path:
		"""
		Save non-SHRINE method logs/plots in a SHRINE-like run directory.
		"""
		run_dir = Path("shrine_logs") / run_prefix
		run_dir.mkdir(parents=True, exist_ok=True)

		max_idx = int(np.argmax(metric_values))
		np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
		np.savetxt(run_dir / f"{run_prefix}_{metric_name}.dat", np.asarray(metric_values, dtype=float))
		np.save(run_dir / f"{run_prefix}_I_at_max.npy", dedispersed_i)

		# Metric-vs-DM plot
		plt.figure(figsize=pub_figsize(single_column=True, height_ratio=0.6, min_height=3.2))
		plt.plot(dm_values, metric_values, '-', color='tab:blue', linewidth=1.8)
		if uncertainty is not None:
			low_dm = uncertainty.get('uncertainty_low_dm')
			high_dm = uncertainty.get('uncertainty_high_dm')
			dm_left = float(np.min(dm_values))
			dm_right = float(np.max(dm_values))
			shade_low = dm_left if low_dm is None else float(low_dm)
			shade_high = dm_right if high_dm is None else float(high_dm)
			if shade_low <= shade_high:
				plt.axvspan(shade_low, shade_high, color='tab:orange', alpha=0.18, label='DM uncertainty')
		plt.axvline(optimal_dm, color='tab:red', linestyle='--', linewidth=1.2,
					label=f"max DM={optimal_dm:.6f}")
		if uncertainty is not None:
			minus = uncertainty.get('uncertainty_minus')
			plus = uncertainty.get('uncertainty_plus')
			unc_text = self._format_uncertainty(optimal_dm, minus, plus)
			plt.title(rf"{method_label}: {metric_name} vs DM\nDM = {unc_text} pc cm$^{{-3}}$")
		else:
			plt.title(rf"{method_label}: {metric_name} vs DM")
		plt.xlabel(rf"DM (pc cm$^{{-3}}$)")
		plt.ylabel(metric_name)
		plt.grid(True, alpha=0.3)
		plt.legend(loc='best')
		plt.tight_layout()
		savefig_rasterized(run_dir / f"{run_prefix}_{metric_name}_v_DM.png", dpi=150, bbox_inches='tight')
		plt.close()

		# I profile at best DM
		time_series = np.mean(dedispersed_i, axis=0)
		plt.figure(figsize=pub_figsize(single_column=True, height_ratio=0.6, min_height=3.2))
		plt.plot(time_series, color='k', linewidth=1.3)
		plt.xlabel('Time index')
		plt.ylabel('Stokes I (arb.)')
		plt.title(f"{method_label}: I at best DM")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		savefig_rasterized(run_dir / f"{run_prefix}_I_at_max.png", dpi=150, bbox_inches='tight')
		plt.close()

		with open(run_dir / f"{run_prefix}_summaryfile.txt", "w") as summary_file:
			summary_file.write(f"//begin {run_prefix} summary//\n/*\n")
			summary_file.write(f"Method: {method_label}\n")
			summary_file.write(f"Metric name: {metric_name}\n")
			summary_file.write(f"Input DM: {self.input_dm}\n")
			summary_file.write(f"Best metric index: {max_idx}\n")
			summary_file.write(f"Best DM: {optimal_dm}\n")
			summary_file.write(f"Best metric: {optimal_metric}\n")
			if uncertainty is not None:
				summary_file.write(f"Uncertainty method: {uncertainty.get('uncertainty_method', 'unknown')}\n")
				summary_file.write(f"Uncertainty lower DM: {uncertainty.get('uncertainty_low_dm', 'unknown')}\n")
				summary_file.write(f"Uncertainty upper DM: {uncertainty.get('uncertainty_high_dm', 'unknown')}\n")
				summary_file.write(f"Uncertainty -DM: {uncertainty.get('uncertainty_minus', 'unknown')}\n")
				summary_file.write(f"Uncertainty +DM: {uncertainty.get('uncertainty_plus', 'unknown')}\n")
			if self.nonshrine_kc_smooth:
				summary_file.write(f"kc smoothing enabled: True\n")
				if self._nonshrine_resolved_kc is not None:
					summary_file.write(f"kc: {self._nonshrine_resolved_kc}\n")
			else:
				summary_file.write(f"kc smoothing enabled: False\n")
			summary_file.write("*/\n//end summary//\n")

		with open(run_dir / "DM.txt", "w") as dm_file:
			dm_file.write(str(max_idx))

		return run_dir

	@staticmethod
	def _robust_vmin_vmax(data: np.ndarray, low: float = 5.0, high: float = 99.0) -> Tuple[float, float]:
		vmin = float(np.percentile(data, low))
		vmax = float(np.percentile(data, high))
		if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
			vmin = float(np.min(data))
			vmax = float(np.max(data))
		return vmin, vmax

	@staticmethod
	def _format_dm(dm: float, precision: int = 6) -> str:
		return f"{dm:.{precision}f}".rstrip('0').rstrip('.')

	@staticmethod
	def _uncertainty_dict(best_dm: float,
						 low_dm: Optional[float],
						 high_dm: Optional[float],
						 method: str) -> Dict[str, Optional[float]]:
		minus = None if low_dm is None else float(best_dm - low_dm)
		plus = None if high_dm is None else float(high_dm - best_dm)
		return {
			'uncertainty_low_dm': None if low_dm is None else float(low_dm),
			'uncertainty_high_dm': None if high_dm is None else float(high_dm),
			'uncertainty_minus': minus,
			'uncertainty_plus': plus,
			'uncertainty_method': method,
		}

	@staticmethod
	def _format_uncertainty(best_dm: float,
						 minus: Optional[float],
						 plus: Optional[float],
						 precision: int = 6) -> str:
		best = DMOptimiser._format_dm(best_dm, precision)
		if minus is None and plus is None:
			return f"{best} (-?/+?)"
		if minus is None:
			return f"{best} (-?/+{DMOptimiser._format_dm(plus, precision)})"
		if plus is None:
			return f"{best} (-{DMOptimiser._format_dm(minus, precision)}/+?)"
		return (
			f"{best} "
			f"(-{DMOptimiser._format_dm(minus, precision)}/+{DMOptimiser._format_dm(plus, precision)})"
		)

	@staticmethod
	def _dm_crossing(values: np.ndarray,
					 dm_values: np.ndarray,
					 start_idx: int,
					 threshold: float,
					 direction: int) -> Optional[float]:
		n = len(values)
		i = int(start_idx)
		if direction < 0:
			while i > 0 and values[i] >= threshold:
				i -= 1
			if values[i] >= threshold:
				return None
			i_below = i
			i_above = i + 1
		else:
			while i < n - 1 and values[i] >= threshold:
				i += 1
			if values[i] >= threshold:
				return None
			i_below = i
			i_above = i - 1

		x1 = float(dm_values[i_below])
		x2 = float(dm_values[i_above])
		y1 = float(values[i_below])
		y2 = float(values[i_above])
		if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
			return float(dm_values[i_above])
		frac = (threshold - y1) / (y2 - y1)
		frac = float(np.clip(frac, 0.0, 1.0))
		return x1 + frac * (x2 - x1)

	@staticmethod
	def _finite_metric_arrays(dm_values: np.ndarray, metric_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		dm = np.asarray(dm_values, dtype=float)
		metric = np.asarray(metric_values, dtype=float)
		valid = np.isfinite(dm) & np.isfinite(metric)
		if np.sum(valid) < 2:
			raise ValueError("Need at least two finite DM/metric points for uncertainty estimation")
		return dm[valid], metric[valid]

	@staticmethod
	def _clamp_uncertainty_to_dm_bounds(
							best_dm: float,
							uncertainty: Dict[str, Optional[float]],
							dm_values: np.ndarray,
							fill_missing_with_bounds: bool = False) -> Dict[str, Optional[float]]:
		"""
		Clamp uncertainty DM bounds to the available DM sample range.
		"""
		dm = np.asarray(dm_values, dtype=float)
		finite = dm[np.isfinite(dm)]
		if finite.size == 0:
			return uncertainty

		dm_min = float(np.min(finite))
		dm_max = float(np.max(finite))
		best_dm = float(best_dm)

		low_dm = uncertainty.get('uncertainty_low_dm')
		high_dm = uncertainty.get('uncertainty_high_dm')
		if fill_missing_with_bounds:
			if low_dm is None:
				low_dm = dm_min
			if high_dm is None:
				high_dm = dm_max

		low_clamped = None if low_dm is None else float(np.clip(float(low_dm), dm_min, dm_max))
		high_clamped = None if high_dm is None else float(np.clip(float(high_dm), dm_min, dm_max))

		if low_clamped is not None:
			low_clamped = min(low_clamped, best_dm)
		if high_clamped is not None:
			high_clamped = max(high_clamped, best_dm)

		clamped = dict(uncertainty)
		clamped['uncertainty_low_dm'] = low_clamped
		clamped['uncertainty_high_dm'] = high_clamped
		clamped['uncertainty_minus'] = None if low_clamped is None else float(best_dm - low_clamped)
		clamped['uncertainty_plus'] = None if high_clamped is None else float(high_clamped - best_dm)
		return clamped

	def _uncertainty_from_half_prominence(self,
							 dm_values: np.ndarray,
							 metric_values: np.ndarray,
							 best_idx: int) -> Dict[str, Optional[float]]:
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx_eff = int(np.argmax(metric))
		best_dm = float(dm[best_idx_eff])
		metric_max = float(np.max(metric))
		metric_min = float(np.min(metric))
		if not np.isfinite(metric_max) or not np.isfinite(metric_min):
			return self._uncertainty_dict(best_dm, None, None, "half-prominence")

		if metric_max <= metric_min:
			step = float(np.median(np.diff(dm))) if len(dm) > 1 else 0.0
			return self._uncertainty_dict(
				best_dm,
				best_dm - 0.5 * step if step > 0 else None,
				best_dm + 0.5 * step if step > 0 else None,
				"half-prominence",
			)

		threshold = metric_min + 0.5 * (metric_max - metric_min)
		low_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=-1)
		high_dm = self._dm_crossing(metric, dm, best_idx_eff, threshold, direction=1)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "half-prominence")

	def _uncertainty_from_local_quadratic(self,
						 dm_values: np.ndarray,
						 metric_values: np.ndarray,
						 best_idx: int,
						 target_points: int = 11) -> Dict[str, Optional[float]]:
		dm, metric = self._finite_metric_arrays(dm_values, metric_values)
		best_idx_eff = int(np.argmax(metric))
		best_dm = float(dm[best_idx_eff])
		n = len(dm)
		if n < 5:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		points = int(max(5, min(target_points, n)))
		half = points // 2
		start = max(0, best_idx_eff - half)
		end = min(n, start + points)
		start = max(0, end - points)
		x = dm[start:end]
		y = metric[start:end]
		if x.size < 5:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		try:
			coeffs = np.polyfit(x, y, 2)
		except Exception:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		a = float(coeffs[0])
		if not np.isfinite(a) or a >= 0:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		y_fit = np.polyval(coeffs, x)
		resid = y - y_fit
		med = float(np.median(resid))
		mad = float(np.median(np.abs(resid - med)))
		sigma = 1.4826 * mad if mad > 0 else float(np.std(resid))
		if not np.isfinite(sigma) or sigma <= 0:
			step = float(np.median(np.diff(dm))) if n > 1 else 0.0
			if step > 0:
				return self._uncertainty_dict(
					best_dm,
					best_dm - 0.5 * step,
					best_dm + 0.5 * step,
					"local quadratic",
				)
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")

		width = float(np.sqrt(sigma / -a))
		if not np.isfinite(width) or width <= 0:
			return self._uncertainty_dict(best_dm, None, None, "local quadratic")
		max_width = 0.5 * float(x[-1] - x[0]) if x.size > 1 else 0.0
		if max_width > 0:
			width = min(width, max_width)

		low_dm = float(best_dm - width)
		high_dm = float(best_dm + width)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "local quadratic (1-sigma)")

	def _uncertainty_from_snr_drop(self,
						 dm_values: np.ndarray,
						 snr_values: np.ndarray,
						 best_idx: int,
						 drop: float = 1.0) -> Dict[str, Optional[float]]:
		dm, sn = self._finite_metric_arrays(dm_values, snr_values)
		best_idx_eff = int(np.argmax(sn))
		best_dm = float(dm[best_idx_eff])
		max_sn = float(sn[best_idx_eff])
		threshold = max_sn - float(drop)
		low_dm = self._dm_crossing(sn, dm, best_idx_eff, threshold, direction=-1)
		high_dm = self._dm_crossing(sn, dm, best_idx_eff, threshold, direction=1)
		return self._uncertainty_dict(best_dm, low_dm, high_dm, "S/N drop = 1")

	def _structure_uncertainty_from_shrine_outputs(self,
									 dm_values: np.ndarray,
									 structure_values: np.ndarray,
									 run_dir: Path,
									 run_prefix: str,
									 best_idx: int) -> Dict[str, Optional[float]]:
		rel_path = run_dir / f"{run_prefix}_Relative_Uncertainties.dat"
		if not rel_path.exists():
			best_dm = float(dm_values[int(best_idx)])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		rel = np.asarray(np.loadtxt(rel_path), dtype=float)
		sp = np.asarray(structure_values, dtype=float)
		dm = np.asarray(dm_values, dtype=float)
		if rel.shape != sp.shape:
			best_dm = float(dm[int(best_idx)])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		max_index = int(best_idx)
		max_structure_parameter = float(sp[max_index])
		adjusted_sps = sp + (sp * rel)
		possible_max_ranges = shrine_get_ranges_above_max(max_structure_parameter, adjusted_sps)

		if len(possible_max_ranges) < 1:
			best_dm = float(dm[max_index])
			return self._uncertainty_dict(best_dm, None, None, "SHRINE relative uncertainty")

		low_idx = int(possible_max_ranges[0][0])
		low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
		high_dm = None
		if len(possible_max_ranges[-1]) == 2:
			high_idx = int(possible_max_ranges[-1][1])
			if 0 <= high_idx < len(dm):
				high_dm = float(dm[high_idx])

		return self._uncertainty_dict(float(dm[max_index]), low_dm, high_dm, "SHRINE relative uncertainty")

	def _nonshrine_uncertainty_reference_profile(self,
										 data_i: Optional[np.ndarray],
										 data_q: Optional[np.ndarray],
										 data_u: Optional[np.ndarray]) -> np.ndarray:
		"""
		Build a per-DM reference time profile for SHRINE-style uncertainty.
		"""
		if data_i is not None:
			return np.nanmean(data_i, axis=0)
		if data_q is not None and data_u is not None:
			linear = np.sqrt(data_q**2 + data_u**2)
			return np.nanmean(linear, axis=0)
		if data_q is not None:
			return np.nanmean(data_q, axis=0)
		if data_u is not None:
			return np.nanmean(data_u, axis=0)
		raise ValueError("At least one of data_i/data_q/data_u must be provided")

	def _li_uncertainty_reference_profile(self,
								 data_q: np.ndarray,
								 data_u: np.ndarray,
								 data_i: np.ndarray) -> np.ndarray:
		"""
		Build a SHRINE-compatible time profile for L/I uncertainty.

		Using the Stokes I mean profile matches SHRINE's uncertainty recipe and
		avoids flat L/I profiles inflating the error range.
		"""
		profile = self._nonshrine_uncertainty_reference_profile(data_i, data_q, data_u)
		profile = np.asarray(profile, dtype=float)
		profile[~np.isfinite(profile)] = 0.0
		return profile

	def _uncertainty_from_shrine_relative(self,
								 dm_values: np.ndarray,
								 metric_values: np.ndarray,
								 reference_profiles: np.ndarray,
								 kc: Optional[int] = None) -> Dict[str, Optional[float]]:
		"""
		Estimate uncertainty using the same relative-uncertainty recipe as SHRINE.
		"""
		dm = np.asarray(dm_values, dtype=float)
		metric = np.asarray(metric_values, dtype=float)
		profiles = np.asarray(reference_profiles, dtype=float)

		if profiles.ndim != 2 or profiles.shape[0] != dm.shape[0] or metric.shape[0] != dm.shape[0]:
			best_idx = int(np.argmax(metric))
			return self._uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

		# Replace non-finite values row-wise so DCT/filtering remains stable.
		profiles_finite = profiles.copy()
		for row_idx in range(profiles_finite.shape[0]):
			row = profiles_finite[row_idx]
			finite = np.isfinite(row)
			if np.any(finite):
				fill_value = float(np.nanmean(row[finite]))
				profiles_finite[row_idx, ~finite] = fill_value
			else:
				profiles_finite[row_idx, :] = 0.0

		ci_data = dct(profiles_finite, norm='ortho')
		k_len = ci_data.shape[1]
		if k_len < 2:
			best_idx = int(np.argmax(metric))
			return self._uncertainty_dict(float(dm[best_idx]), None, None, "SHRINE relative uncertainty")

		if kc is None:
			with contextlib.redirect_stdout(io.StringIO()):
				kc_use = int(shrine_get_kc(ci_data))
		else:
			kc_use = int(kc)
		kc_use = max(1, min(kc_use, k_len))

		i_smooth, lpf_data, _, f_l = shrine_lowpass_smooth(ci_data, kc_use, order=3)
		k = np.linspace(1, k_len, k_len)
		hp = np.sqrt(2 - 2 * np.cos((k - 1) * np.pi / k_len))
		filter_diag = np.diag(hp * f_l)

		delta_i = profiles_finite - i_smooth
		max_idx = int(np.argmax(metric))
		delta_delta_i = delta_i - delta_i[max_idx]

		relative_uncertainty = shrine_uncertainty_calc(delta_delta_i, lpf_data, filter_diag)
		relative_uncertainty = np.asarray(relative_uncertainty, dtype=float)
		relative_uncertainty[~np.isfinite(relative_uncertainty)] = 0.0

		max_metric = float(metric[max_idx])
		adjusted_metrics = metric + (metric * relative_uncertainty)
		possible_max_ranges = shrine_get_ranges_above_max(max_metric, adjusted_metrics)

		if len(possible_max_ranges) < 1:
			return self._uncertainty_dict(float(dm[max_idx]), None, None, "SHRINE relative uncertainty")

		low_idx = int(possible_max_ranges[0][0])
		low_dm = float(dm[low_idx]) if 0 <= low_idx < len(dm) else None
		high_dm = None
		if len(possible_max_ranges[-1]) == 2:
			high_idx = int(possible_max_ranges[-1][1])
			if 0 <= high_idx < len(dm):
				high_dm = float(dm[high_idx])

		return self._uncertainty_dict(float(dm[max_idx]), low_dm, high_dm, "SHRINE relative uncertainty")

	def _full_noise_reference_data(self, data: np.ndarray) -> np.ndarray:
		"""
		Select the full dynamic-spectrum array that matches the provided data.
		"""
		if np.shares_memory(data, self.stokes_i):
			return self.stokes_i
		if self.stokes_q is not None and np.shares_memory(data, self.stokes_q):
			return self.stokes_q
		if self.stokes_u is not None and np.shares_memory(data, self.stokes_u):
			return self.stokes_u
		return self.stokes_i
	
	def _get_common_valid_region(self, data: np.ndarray, delay_samples: np.ndarray) -> Tuple[int, int]:
		"""
		Calculate the common valid region after dedispersion (for crop mode).
		
		Returns:
		--------
		start_idx : int
			Start index of common valid region
		end_idx : int
			End index of common valid region (exclusive)
		"""
		max_shift = int(np.max(delay_samples))
		min_shift = int(np.min(delay_samples))
		
		# Common valid region starts where the most-shifted channel begins
		# and ends where the least-shifted channel ends
		start_idx = max_shift
		end_idx = data.shape[1] + min_shift
		
		return start_idx, end_idx

	def _get_delay_samples(self, dm: float, reference_freq: Optional[float] = None) -> np.ndarray:
		"""
		Return per-channel delay in integer samples for a trial DM.
		"""
		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)

		effective_dm = self.input_dm - dm
		delays_ms = self.DM_CONSTANT * effective_dm * (
			1.0 / self.freq_mhz**2 - 1.0 / reference_freq**2
		)
		dt = np.median(np.diff(self.time_ms))
		return np.round(delays_ms / dt).astype(int)

	def recommend_lowest_dm_step(self, reference_freq: Optional[float] = None,
							   samples_per_step: float = 1.0) -> float:
		"""
		Recommend the minimum useful DM grid step from dynamic-spectrum resolution.

		The estimate is based on the DM increment that produces a delay of
		``samples_per_step`` time samples across the most-dispersed channel relative
		to the reference frequency.
		"""
		if samples_per_step <= 0:
			raise ValueError("samples_per_step must be positive")

		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)

		dt_ms = float(np.median(np.diff(self.time_ms)))
		if dt_ms <= 0:
			raise ValueError("time axis must be strictly increasing to estimate DM step")

		delta_inv_f2 = np.abs(1.0 / self.freq_mhz**2 - 1.0 / float(reference_freq)**2)
		max_delta_inv_f2 = float(np.max(delta_inv_f2))
		if max_delta_inv_f2 <= 0:
			raise ValueError("frequency axis does not span reference-frequency delays")

		return (samples_per_step * dt_ms) / (self.DM_CONSTANT * max_delta_inv_f2)
		
	def dedisperse(self, data: np.ndarray, dm: float, reference_freq: Optional[float] = None, 
				   output_size: Optional[int] = None, mode: str = 'expand') -> np.ndarray:
		"""
		Apply dispersion correction to data.
		
		Parameters:
		-----------
		data : np.ndarray
			2D array to dedisperse (freq x time)
		dm : float
			Dispersion measure in pc cm^-3
		reference_freq : float, optional
			Reference frequency in MHz. If None, uses highest frequency.
		output_size : int, optional
			Desired output time axis size. If None, calculated based on mode.
		mode : str, optional
			Dedispersion mode: 'expand' (fill edges with noise, default) or 'crop' (trim to common valid region)
			
		Returns:
		--------
		dedispersed : np.ndarray
			Dedispersed data array
		"""
		if reference_freq is None:
			reference_freq = self.reference_freq if self.reference_freq is not None else np.max(self.freq_mhz)
		
		delay_samples = self._get_delay_samples(dm, reference_freq=reference_freq)
		
		if mode == 'crop':
			# Crop mode: return only the common valid region
			start_idx, end_idx = self._get_common_valid_region(data, delay_samples)
			if output_size is None:
				n_time_out = end_idx - start_idx
			else:
				n_time_out = output_size

			# Generate noise fill using full dynamic spectrum noise statistics
			noise_ref = self._full_noise_reference_data(data)
			n_edge_full = max(1, int(0.05 * noise_ref.shape[1]))
			noise_fill = np.empty((data.shape[0], n_time_out), dtype=data.dtype)
			for i in range(data.shape[0]):
				# Use noise from full dynamic spectrum for this frequency channel
				noise_std = np.std(noise_ref[i, :n_edge_full])
				noise_mean = np.mean(noise_ref[i, :n_edge_full])
				noise_fill[i] = self.rng.normal(noise_mean, noise_std, n_time_out)

			# Apply shifts and crop to common region
			if _NUMBA_AVAILABLE:
				dedispersed = _apply_shifts_numba(data, delay_samples, noise_fill, start_idx)
			else:
				dedispersed = noise_fill.copy()
				for i, shift in enumerate(delay_samples):
					for t in range(data.shape[1]):
						t_out = t + shift - start_idx
						if 0 <= t_out < n_time_out:
							dedispersed[i, t_out] = data[i, t]
		else:
			# Expand mode: extend time axis and fill with noise (default)
			min_shift = int(np.min(delay_samples))
			max_shift = int(np.max(delay_samples))
			if output_size is None:
				n_time_out = data.shape[1] + max_shift - min_shift
			else:
				n_time_out = output_size

			# Generate noise to fill expanded regions using full dynamic spectrum noise statistics
			noise_ref = self._full_noise_reference_data(data)
			n_edge_full = max(1, int(0.05 * noise_ref.shape[1]))
			noise_fill = np.empty((data.shape[0], n_time_out), dtype=data.dtype)
			for i in range(data.shape[0]):
				# Use noise from full dynamic spectrum for this frequency channel
				noise_std = np.std(noise_ref[i, :n_edge_full])
				noise_mean = np.mean(noise_ref[i, :n_edge_full])
				noise_fill[i] = self.rng.normal(noise_mean, noise_std, n_time_out)

			# Apply dedispersion by shifting each frequency channel
			if _NUMBA_AVAILABLE:
				dedispersed = _apply_shifts_numba(data, delay_samples, noise_fill, min_shift)
			else:
				dedispersed = noise_fill.copy()
				for i, shift in enumerate(delay_samples):
					for t in range(data.shape[1]):
						t_out = t + shift - min_shift
						if 0 <= t_out < n_time_out:
							dedispersed[i, t_out] = data[i, t]
			
		return dedispersed
	def _max_output_size_for_dm_range(self, data: np.ndarray, dm_range: Tuple[float, float]) -> int:
		"""
		Calculate the maximum output size needed for dedispersion across a DM range.
		"""
		# Check both endpoints of the range
		max_size = data.shape[1]
		for dm in dm_range:
			delay_samples = self._get_delay_samples(dm, reference_freq=self.reference_freq)
			max_shift = int(np.max(delay_samples))
			min_shift = int(np.min(delay_samples))
			max_size = max(max_size, data.shape[1] + max_shift - min_shift)
		return max_size

	@staticmethod
	def _noise_stats_from_series(series: np.ndarray) -> Tuple[float, float]:
		n_edge = max(1, int(0.05 * len(series)))
		noise_region = series[:n_edge]
		return float(np.median(noise_region)), float(np.std(noise_region))

	@staticmethod
	def _apply_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
		valid = mask.astype(int)
		dv = np.diff(np.concatenate(([0], valid, [0])))
		starts = np.where(dv == 1)[0]
		ends = np.where(dv == -1)[0]
		keep = np.zeros_like(mask, dtype=bool)
		for s, e in zip(starts, ends):
			if (e - s) >= min_run:
				keep[s:e] = True
		return keep

	@staticmethod
	def _longest_true_run(mask: np.ndarray) -> int:
		valid = mask.astype(int)
		dv = np.diff(np.concatenate(([0], valid, [0])))
		starts = np.where(dv == 1)[0]
		ends = np.where(dv == -1)[0]
		if len(starts) == 0:
			return 0
		return int(np.max(ends - starts))

	def _pa_slope_metric_shrine(self, data_q: np.ndarray, data_u: np.ndarray,
								  time_ms: Optional[np.ndarray] = None,
								  data_i: Optional[np.ndarray] = None) -> float:
		"""
		PA slope metric where PA is SHRINE-smoothed before fitting.
		"""
		q_ts = np.mean(data_q, axis=0)
		u_ts = np.mean(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		pa_shrine_smooth, _, time_axis = self._get_pa_shrine_smoothed_and_fit(data_q, data_u, data_i, time_ms)
		valid = np.isfinite(pa_shrine_smooth)
		if np.sum(valid) < (self.pa_fit_degree + 1):
			return 0.0

		weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
		w = weights[valid]
		if np.sum(w > 0) < (self.pa_fit_degree + 1):
			return 0.0

		x = time_axis[valid]
		y = pa_shrine_smooth[valid]
		try:
			coeffs = np.polyfit(x, y, self.pa_fit_degree, w=w)
		except Exception:
			return 0.0

		slope_magnitude = float(np.abs(coeffs[0]))
		if not np.isfinite(slope_magnitude):
			return 0.0
		return slope_magnitude

	def _pa_fit_weights(self, L_debias: np.ndarray, sigma_L: float,
						data_i: Optional[np.ndarray], valid: np.ndarray) -> np.ndarray:
		"""
		Build per-time-sample PA fit weights so lower-S/N regions contribute less.
		"""
		eps = 1e-12
		w_l = np.maximum(L_debias / np.maximum(sigma_L, eps), 0.0)

		weights = w_l

		if data_i is not None:
			i_ts = np.mean(data_i, axis=0)
			i_noise_std = float(self.full_i_noise_std)
			i_noise_med = float(self.full_i_noise_median)
			if i_noise_std > 0:
				w_i = np.maximum((i_ts - i_noise_med) / i_noise_std, 0.0)
				weights = weights * w_i

		weights = np.where(valid, weights, 0.0)
		max_w = float(np.max(weights)) if np.any(valid) else 0.0
		if max_w > 0:
			weights = weights / max_w
			if self.pa_weight_strength != 1.0:
				weights = np.power(weights, self.pa_weight_strength)

		return weights

	def _qu_noise_rms_from_full(self, data_q: np.ndarray, data_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Get Q/U noise RMS from full dynamic-spectrum statistics.
		"""
		if data_q.ndim == 1:
			if self.full_q_time_noise_std is not None and self.full_u_time_noise_std is not None:
				return float(self.full_q_time_noise_std), float(self.full_u_time_noise_std)
			return self._qu_noise_rms(data_q, data_u)

		if (
			self.full_q_noise_rms is not None
			and self.full_u_noise_rms is not None
			and self.full_q_noise_rms.shape[0] == data_q.shape[0]
		):
			return self.full_q_noise_rms, self.full_u_noise_rms

		if self.full_q_time_noise_std is not None and self.full_u_time_noise_std is not None:
			q_rms = np.full((data_q.shape[0], 1), float(self.full_q_time_noise_std), dtype=float)
			u_rms = np.full((data_u.shape[0], 1), float(self.full_u_time_noise_std), dtype=float)
			return q_rms, u_rms

		return self._qu_noise_rms(data_q, data_u)

	@staticmethod
	def _qu_noise_rms(data_q: np.ndarray, data_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Estimate Q/U RMS from the first 5% of time samples.
		"""
		if data_q.ndim == 1:
			n_edge = max(1, int(0.05 * len(data_q)))
			q_rms = float(np.std(data_q[:n_edge]))
			u_rms = float(np.std(data_u[:n_edge]))
			return q_rms, u_rms

		# Expect shape (freq, time)
		n_edge = max(1, int(0.05 * data_q.shape[1]))
		q_rms = np.std(data_q[:, :n_edge], axis=1, keepdims=True)
		u_rms = np.std(data_u[:, :n_edge], axis=1, keepdims=True)
		return q_rms, u_rms

	def _debiased_linear_from_qu(self, data_q: np.ndarray, data_u: np.ndarray,
						   q_rms: np.ndarray, u_rms: np.ndarray,
						   cutoff: float = 1.57, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Optionally debias linear polarisation using propagated sigma_L and detection cutoff.
		"""
		L_meas = np.sqrt(data_q**2 + data_u**2)
		sigma_L = np.sqrt(data_q**2 * q_rms**2 + data_u**2 * u_rms**2) / np.maximum(L_meas, eps)
		r = L_meas / np.maximum(sigma_L, eps)
		det = r >= cutoff

		if self.debias_linear:
			L_out = np.zeros_like(L_meas)
			L_out[det] = np.sqrt(np.maximum(L_meas[det]**2 - sigma_L[det]**2, 0.0))
		else:
			L_out = L_meas.copy()
			L_out[~det] = 0.0
		return L_out, sigma_L, det

	def _pa_series_deg(self, data_q: np.ndarray, data_u: np.ndarray, data_i: Optional[np.ndarray] = None) -> np.ndarray:
		q_ts = np.mean(data_q, axis=0)
		u_ts = np.mean(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)
		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0
		mask = L_debias >= (2.0 * sigma_L)
		
		# Apply Stokes I significance cutoff; optionally also restrict to >= peak
		if data_i is not None:
			i_ts = np.mean(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask
		
		pa_deg = np.where(mask, pa_deg, np.nan)

		# Drop short valid runs
		min_run = 10
		if np.any(mask):
			keep_run = self._apply_min_run(mask, min_run)
			pa_deg = np.where(keep_run, pa_deg, np.nan)

		return pa_deg
	
	def _get_pa_smoothed_and_fit(self, data_q: np.ndarray, data_u: np.ndarray, data_i: Optional[np.ndarray] = None, time_ms: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Get masked PA profile and best fit line for plotting.
		
		Parameters:
		-----------
		data_q : np.ndarray
			Stokes Q data
		data_u : np.ndarray
			Stokes U data
		data_i : np.ndarray, optional
			Stokes I data for S/N masking
		time_ms : np.ndarray, optional
			Time axis
		
		Returns:
		--------
		pa_smooth : np.ndarray
			Masked PA values (no Gaussian smoothing)
		fit_line : np.ndarray
			Best fit line values
		time_axis : np.ndarray
			Time axis for the data
		"""
		q_ts = np.mean(data_q, axis=0)
		u_ts = np.mean(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)
		
		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0
		
		# Mask based on L significance
		mask = L_debias >= (2.0 * sigma_L)
		
		# Apply Stokes I significance cutoff; optionally also restrict to >= peak
		if data_i is not None:
			i_ts = np.mean(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask
		
		min_run = 5
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)
		
		valid = mask & np.isfinite(pa_deg)
		
		# Create time axis
		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms
		
		# Apply mask to PA for plotting/fit diagnostics.
		pa_deg_masked = np.where(mask, pa_deg, np.nan)

		# Regular PA method now uses unsmoothed masked PA.
		pa_smooth = pa_deg_masked.copy()
		
		# Compute best fit line
		fit_line = np.full_like(pa_deg, np.nan)
		min_points = self.pa_fit_degree + 1
		if np.sum(valid) >= min_points:
			weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
			weights_valid = weights[valid]
			if np.sum(weights_valid > 0) >= min_points:
				coeffs = np.polyfit(time_axis[valid], pa_deg[valid], self.pa_fit_degree, w=weights_valid)
				fit_line = np.polyval(coeffs, time_axis)
				# Only show fit line where mask is valid
				fit_line = np.where(mask, fit_line, np.nan)
		
		return pa_smooth, fit_line, time_axis

	def _get_pa_shrine_smoothed_and_fit(self, data_q: np.ndarray, data_u: np.ndarray,
									 data_i: Optional[np.ndarray] = None,
									 time_ms: Optional[np.ndarray] = None,
									 force_kc: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Get SHRINE-smoothed PA profile and best fit line for plotting.
		The smoothing is applied to the PA time series itself, then fitting is done on the SHRINE-smoothed PA.
		"""
		q_ts = np.mean(data_q, axis=0)
		u_ts = np.mean(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

		mask = L_debias >= (2.0 * sigma_L)
		if data_i is not None:
			i_ts = np.mean(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask

		min_run = 5
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)

		valid = mask & np.isfinite(pa_deg)

		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms

		pa_deg_masked = np.where(mask, pa_deg, np.nan)
		kc_source_mask = (L_debias >= (2.0 * sigma_L)) & np.isfinite(pa_deg)
		pa_deg_for_kc = np.where(kc_source_mask, pa_deg, np.nan)
		if np.sum(np.isfinite(pa_deg_for_kc)) < 2:
			pa_deg_for_kc = np.where(np.isfinite(pa_deg), pa_deg, np.nan)
		pa_shrine_smooth = np.full_like(pa_deg, np.nan)
		fit_line = np.full_like(pa_deg, np.nan)
		min_points = self.pa_fit_degree + 1
		if np.sum(valid) >= min_points:
			valid_idx = np.where(np.isfinite(pa_deg_for_kc))[0]
			if valid_idx.size == 1:
				pa_fill = np.full_like(pa_deg, pa_deg_for_kc[valid_idx[0]], dtype=float)
			elif valid_idx.size >= 2:
				all_idx = np.arange(len(pa_deg))
				pa_fill = np.interp(all_idx, valid_idx, pa_deg_for_kc[valid_idx])
			else:
				pa_fill = np.zeros_like(pa_deg, dtype=float)

			if force_kc is not None:
				kc = int(force_kc)
			else:
				kc = self._resolve_nonshrine_kc(pa_fill[np.newaxis, :])
			pa_shrine = self._apply_kc_lowpass_2d(pa_fill[np.newaxis, :], kc)[0]
			pa_shrine_smooth = np.where(np.isfinite(pa_deg_masked), pa_shrine, np.nan)

			weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
			weights_valid = weights[valid]
			if np.sum(weights_valid > 0) >= min_points:
				coeffs = np.polyfit(time_axis[valid], pa_shrine_smooth[valid], self.pa_fit_degree, w=weights_valid)
				fit_line = np.polyval(coeffs, time_axis)
				fit_line = np.where(mask, fit_line, np.nan)

		return pa_shrine_smooth, fit_line, time_axis
	
	def _build_dm_values(self, dm_range: Tuple[float, float], n_points: int = 200,
						 dm_step: Optional[float] = None) -> np.ndarray:
		if dm_step is not None:
			if dm_step <= 0:
				raise ValueError("dm_step must be positive")
			span = dm_range[1] - dm_range[0]
			n_points = max(2, int(np.floor(span / dm_step)) + 1)
			dm_values = dm_range[0] + np.arange(n_points) * dm_step
			if dm_values[-1] < dm_range[1] and (dm_range[1] - dm_values[-1]) > 0.5 * dm_step:
				dm_values = np.append(dm_values, dm_range[1])
		else:
			dm_values = np.linspace(dm_range[0], dm_range[1], n_points)
		return dm_values
	
	def snr_metric(self, data: np.ndarray) -> float:
		"""
		Calculate signal-to-noise ratio metric.
		
		Parameters:
		-----------
		data : np.ndarray
			Dedispersed data (freq x time)
			
		Returns:
		--------
		snr : float
			Signal-to-noise ratio (higher is better)
		"""
		# Collapse to time series
		time_series = np.mean(data, axis=0)
		
		# Estimate noise from the first 5% of the full time series
		noise_std = self.full_i_noise_std
		
		if noise_std == 0:
			return 0.0
		
		# Signal is the peak
		signal = np.max(time_series) - self.full_i_noise_median
		
		snr = signal / noise_std
		return snr
	
	def pa_slope_metric(self, data_q: np.ndarray, data_u: np.ndarray, time_ms: Optional[np.ndarray] = None, data_i: Optional[np.ndarray] = None) -> float:
		"""
		Calculate weighted position angle (PA) slope magnitude metric.
		This requires Stokes Q and U for polarisation analysis.

		The metric uses a weighted polynomial fit (degree specified by pa_fit_degree)
		to PA vs time within a 2-sigma debiased-L mask (with min-run filtering), and
		returns the magnitude of the fitted slope coefficient.
		
		If data_i is provided, only fit to data >= the peak of the Stokes I profile.
		"""
		q_ts = np.mean(data_q, axis=0)
		u_ts = np.mean(data_u, axis=0)
		q_rms, u_rms = self._qu_noise_rms_from_full(q_ts, u_ts)
		L_debias, sigma_L, _ = self._debiased_linear_from_qu(q_ts, u_ts, q_rms, u_rms)

		pa = 0.5 * np.arctan2(u_ts, q_ts)
		pa = 0.5 * np.unwrap(2.0 * pa)
		pa_deg = np.degrees(pa)
		pa_deg = ((pa_deg + 90.0) % 180.0) - 90.0

		mask = L_debias >= (2.0 * sigma_L)
		min_run = 5
		if np.any(mask):
			mask = self._apply_min_run(mask, min_run)

		# If Stokes I provided, apply I significance cutoff and optionally restrict to >= peak
		if data_i is not None:
			i_ts = np.mean(data_i, axis=0)
			sigma_i = self.full_i_noise_std if self.full_i_noise_std is not None else np.std(i_ts)
			med_i = self.full_i_noise_median if self.full_i_noise_median is not None else np.median(i_ts)
			threshold_i = med_i + self.li_i_sigma_cut * sigma_i
			i_mask = i_ts >= threshold_i
			if self.pa_fit_post_peak_only:
				peak_idx = int(np.argmax(i_ts))
				peak_mask = np.zeros_like(mask, dtype=bool)
				peak_mask[peak_idx:] = True
				i_mask = i_mask & peak_mask
			mask = mask & i_mask

		valid = mask & np.isfinite(pa_deg)
		if time_ms is None or len(time_ms) != len(pa_deg):
			dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = np.arange(len(pa_deg)) * dt
		else:
			time_axis = time_ms

		min_points = self.pa_fit_degree + 1
		if np.sum(valid) < min_points:
			return 0.0

		min_contiguous = max(12, min_points)
		if self._longest_true_run(valid) < min_contiguous:
			return 0.0

		weights = self._pa_fit_weights(L_debias, sigma_L, data_i, valid)
		positive = valid & (weights > 0)
		if np.sum(positive) < min_points:
			return 0.0

		x = time_axis[positive]
		y = pa_deg[positive]
		w = weights[positive]

		try:
			coeffs = np.polyfit(x, y, self.pa_fit_degree, w=w)
		except Exception:
			return 0.0

		slope_magnitude = float(np.abs(coeffs[0]))
		if not np.isfinite(slope_magnitude):
			return 0.0
		return slope_magnitude
	
	def linear_to_stokes_i_metric(self, data_q: np.ndarray, data_u: np.ndarray, 
							   data_i: np.ndarray, mode: str = 'peak') -> float:
		"""
		Calculate debiased L/I ratio metric.
		Uses noise-debiased linear polarization and calculates fractional
		polarization using different criteria.
		
		Parameters:
		-----------
		data_q : np.ndarray
			Stokes Q data (freq x time)
		data_u : np.ndarray
			Stokes U data (freq x time)
		data_i : np.ndarray
			Stokes I data (freq x time)
		mode : str, optional
			Calculation mode: 'peak' (L/I at Stokes I peak), 'mean' (mean L/I across pulse),
			or 'max' (L/I at the point where L is maximum). Default is 'peak'.
			
		Returns:
		--------
		metric : float
			Debiased L/I metric (higher is better)
		"""
		# Calculate linear polarisation per pixel
		if (
			self.full_q_noise_rms is not None
			and self.full_u_noise_rms is not None
			and data_q.ndim == 2
			and self.full_q_noise_rms.shape[0] == data_q.shape[0]
		):
			q_rms, u_rms = self.full_q_noise_rms, self.full_u_noise_rms
		else:
			q_rms, u_rms = self._qu_noise_rms(data_q, data_u)
		L_debias, _, _ = self._debiased_linear_from_qu(data_q, data_u, q_rms, u_rms)
		
		# Calculate L/I ratio per pixel (avoiding division by zero)
		L_over_I_2d = np.where(data_i > 0, L_debias / data_i, 0.0)
		
		# Clip to valid range [0, 1] to handle any numerical issues
		L_over_I_2d = np.clip(L_over_I_2d, 0.0, 1.0)
		
		# Average L/I across frequency to get time series
		L_over_I = np.mean(L_over_I_2d, axis=0)
		
		I_ts = np.mean(data_i, axis=0)
		sigma_I = float(self.full_i_noise_std)
		threshold = float(self.full_i_noise_median) + self.li_i_sigma_cut * sigma_I
		mask = I_ts > threshold

		if mode == 'peak':
			# L/I at the peak of Stokes I within significant pulse region
			if np.any(mask):
				masked_i = np.where(mask, I_ts, -np.inf)
				peak_idx = int(np.argmax(masked_i))
				return float(L_over_I[peak_idx])
			return 0.0
		elif mode == 'mean':
			# Mean L/I across significant pulse region
			if np.any(mask):
				return float(np.mean(L_over_I[mask]))
			else:
				return 0.0
		elif mode == 'max':
			# L/I at the point where L is maximum within significant pulse region
			L_ts = np.mean(L_debias, axis=0)
			if np.any(mask):
				masked_l = np.where(mask, L_ts, -np.inf)
				max_L_idx = int(np.argmax(masked_l))
				return float(L_over_I[max_L_idx])
			return 0.0
		else:
			raise ValueError(f"Unknown mode '{mode}'. Must be 'peak', 'mean', or 'max'.")
	def optimise_dm_pa_slope(self, dm_range: Tuple[float, float],
							 data_q: Optional[np.ndarray] = None,
							 data_u: Optional[np.ndarray] = None,
							 data_i: Optional[np.ndarray] = None,
							 n_points: int = 200,
							 dm_step: Optional[float] = None,
							 label: str = "frb",
							 segment: Optional[str] = None) -> Dict:
		"""
		Optimise DM using PA slope smoothness (requires Q and U).
		"""
		if data_q is None or data_u is None:
			raise ValueError("Stokes Q and U data required for PA slope optimisation")
		self._reset_nonshrine_kc_state()

		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		pa_values = np.zeros(len(dm_values))
		base_data = data_i if data_i is not None else data_q
		output_size = self._max_output_size_for_dm_range(base_data, dm_range)
		uncertainty_reference_profiles = np.zeros((len(dm_values), output_size), dtype=float)

		if data_i is not None and data_i.shape[1] > 0:
			dt_diag = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = self.time_ms[0] + np.arange(output_size) * dt_diag
		else:
			dt_diag = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
			time_axis = self.time_ms[0] + np.arange(output_size) * dt_diag

		for i, dm in enumerate(dm_values):
			dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_i = self.dedisperse(data_i, dm, output_size=output_size, mode=self.dedisp_mode) if data_i is not None else None
			sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
			uncertainty_reference_profiles[i] = self._nonshrine_uncertainty_reference_profile(sm_i, sm_q, sm_u)
			pa_values[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i)

		max_idx = int(np.argmax(pa_values))
		optimal_dm = float(dm_values[max_idx])
		best_dedisp_q = self.dedisperse(data_q, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		best_dedisp_u = self.dedisperse(data_u, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		best_dedisp_i = self.dedisperse(data_i, optimal_dm, output_size=output_size, mode=self.dedisp_mode) if data_i is not None else None
		best_sm_i, best_sm_q, best_sm_u = self._maybe_kc_smooth_nonshrine(best_dedisp_i, best_dedisp_q, best_dedisp_u)
		best_pa_smooth, best_fit_line, best_time_axis = self._get_pa_smoothed_and_fit(best_sm_q, best_sm_u, best_sm_i, time_axis)
		best_pa_deg = self._pa_series_deg(best_sm_q, best_sm_u, best_sm_i)
		dedispersed_display = self.dedisperse(
			data_i if data_i is not None else data_q,
			optimal_dm,
			output_size=output_size,
			mode=self.dedisp_mode,
		)
		metric = float(pa_values[max_idx])
		if self.use_nonshrine_shrine_like_uncertainty:
			pa_uncertainty = self._uncertainty_from_shrine_relative(
				dm_values,
				pa_values,
				uncertainty_reference_profiles,
				kc=self._nonshrine_resolved_kc,
			)
		else:
			pa_uncertainty = self._uncertainty_from_half_prominence(dm_values, pa_values, max_idx)
		run_prefix = f"{label}_{segment or 'segment1'}_pa_slope"
		dedispersed_i_for_logs = (
			dedispersed_display
			if data_i is not None
			else self.dedisperse(self.stokes_i, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		)
		run_dir = self._save_nonshrine_run_outputs(
			run_prefix=run_prefix,
			method_label='PA Slope Maximising',
			dm_values=dm_values,
			metric_values=pa_values,
			metric_name='PA_Slope',
			dedispersed_i=dedispersed_i_for_logs,
			optimal_dm=optimal_dm,
			optimal_metric=metric,
			uncertainty=pa_uncertainty,
		)

		result = {
			'dm': optimal_dm,
			'metric': metric,
			'dedispersed': dedispersed_display,
			'method': 'PA Slope Maximising',
			'pa_plot_kind': 'raw',
			'pa_plot_time': best_time_axis.copy(),
			'pa_plot_series': best_pa_deg.copy(),
			'pa_plot_smooth': best_pa_smooth.copy(),
			'pa_plot_fit': best_fit_line.copy(),
			'run_dir': str(run_dir),
			'dm_values': dm_values.copy(),
			'metric_values': pa_values.copy(),
			**pa_uncertainty,
		}
		
		return result

	def optimise_dm_pa_slope_shrine(self, dm_range: Tuple[float, float],
									data_q: Optional[np.ndarray] = None,
									data_u: Optional[np.ndarray] = None,
									data_i: Optional[np.ndarray] = None,
									n_points: int = 200,
									dm_step: Optional[float] = None,
									label: str = "frb",
									segment: Optional[str] = None) -> Dict:
		"""
		Optimise DM using PA slope with SHRINE-smoothed PA fitting (requires Q and U).
		"""
		if data_q is None or data_u is None:
			raise ValueError("Stokes Q and U data required for PA slope optimisation")
		self._reset_nonshrine_kc_state()

		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		pa_values = np.zeros(len(dm_values))
		base_data = data_i if data_i is not None else data_q
		output_size = self._max_output_size_for_dm_range(base_data, dm_range)
		uncertainty_reference_profiles = np.zeros((len(dm_values), output_size), dtype=float)
		dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = self.time_ms[0] + np.arange(output_size) * dt

		for i, dm in enumerate(dm_values):
			dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_i = self.dedisperse(data_i, dm, output_size=output_size, mode=self.dedisp_mode) if data_i is not None else None
			sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
			uncertainty_reference_profiles[i] = self._nonshrine_uncertainty_reference_profile(sm_i, sm_q, sm_u)
			pa_values[i] = self._pa_slope_metric_shrine(sm_q, sm_u, time_axis, sm_i)

		max_idx = int(np.argmax(pa_values))
		optimal_dm = float(dm_values[max_idx])
		best_dedisp_q = self.dedisperse(data_q, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		best_dedisp_u = self.dedisperse(data_u, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		best_dedisp_i = self.dedisperse(data_i, optimal_dm, output_size=output_size, mode=self.dedisp_mode) if data_i is not None else None
		best_sm_i, best_sm_q, best_sm_u = self._maybe_kc_smooth_nonshrine(best_dedisp_i, best_dedisp_q, best_dedisp_u)
		best_pa_smooth, best_fit_line, best_time_axis = self._get_pa_shrine_smoothed_and_fit(
			best_sm_q,
			best_sm_u,
			best_sm_i,
			time_axis,
			force_kc=self._nonshrine_resolved_kc,
		)
		best_pa_deg = self._pa_series_deg(best_sm_q, best_sm_u, best_sm_i)
		dedispersed_display = self.dedisperse(
			data_i if data_i is not None else data_q,
			optimal_dm,
			output_size=output_size,
			mode=self.dedisp_mode,
		)
		metric = float(pa_values[max_idx])
		if self.use_nonshrine_shrine_like_uncertainty:
			pa_shrine_uncertainty = self._uncertainty_from_shrine_relative(
				dm_values,
				pa_values,
				uncertainty_reference_profiles,
				kc=self._nonshrine_resolved_kc,
			)
		else:
			pa_shrine_uncertainty = self._uncertainty_from_half_prominence(dm_values, pa_values, max_idx)
		run_prefix = f"{label}_{segment or 'segment1'}_pa_slope_shrine"
		dedispersed_i_for_logs = (
			dedispersed_display
			if data_i is not None
			else self.dedisperse(self.stokes_i, optimal_dm, output_size=output_size, mode=self.dedisp_mode)
		)
		run_dir = self._save_nonshrine_run_outputs(
			run_prefix=run_prefix,
			method_label='PA Slope Maximising (SHRINE PA)',
			dm_values=dm_values,
			metric_values=pa_values,
			metric_name='PA_Slope_SHRINE',
			dedispersed_i=dedispersed_i_for_logs,
			optimal_dm=optimal_dm,
			optimal_metric=metric,
			uncertainty=pa_shrine_uncertainty,
		)

		return {
			'dm': optimal_dm,
			'metric': metric,
			'dedispersed': dedispersed_display,
			'method': 'PA Slope Maximising (SHRINE PA)',
			'kc': None if self._nonshrine_resolved_kc is None else int(self._nonshrine_resolved_kc),
			'pa_plot_kind': 'shrine',
			'pa_plot_time': best_time_axis.copy(),
			'pa_plot_series': best_pa_deg.copy(),
			'pa_plot_smooth': best_pa_smooth.copy(),
			'pa_plot_fit': best_fit_line.copy(),
			'run_dir': str(run_dir),
			'dm_values': dm_values.copy(),
			'metric_values': pa_values.copy(),
			**pa_shrine_uncertainty,
		}

	def optimise_dm_linear_to_stokes_i(self, dm_range: Tuple[float, float],
									   data_q: Optional[np.ndarray] = None,
									   data_u: Optional[np.ndarray] = None,
									   data_i: Optional[np.ndarray] = None,
									   n_points: int = 200,
									   dm_step: Optional[float] = None,
									   mode: str = 'peak',
									   label: str = "frb",
									   segment: Optional[str] = None) -> Dict:
		"""
		Optimise DM using L/I maximisation (requires Q, U, and I).
		
		Parameters:
		-----------
		mode : str, optional
			Calculation mode: 'peak' (L/I at Stokes I peak), 'mean' (mean L/I across pulse),
			or 'max' (absolute maximum L/I). Default is 'peak'.
		"""
		if data_q is None or data_u is None or data_i is None:
			raise ValueError("Stokes I, Q, and U data required for L/I optimisation")
		self._reset_nonshrine_kc_state()

		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		li_values = np.zeros(len(dm_values))
		output_size = self._max_output_size_for_dm_range(data_i, dm_range)
		uncertainty_reference_profiles = np.zeros((len(dm_values), output_size), dtype=float)

		for i, dm in enumerate(dm_values):
			dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_i = self.dedisperse(data_i, dm, output_size=output_size, mode=self.dedisp_mode)
			sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
			uncertainty_reference_profiles[i] = self._li_uncertainty_reference_profile(sm_q, sm_u, sm_i)
			li_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode=mode)

		max_idx = int(np.argmax(li_values))
		optimal_dm = float(dm_values[max_idx])
		dedispersed_i = self.dedisperse(data_i, optimal_dm, mode=self.dedisp_mode)
		metric = float(li_values[max_idx])
		if self.use_nonshrine_shrine_like_uncertainty:
			li_uncertainty = self._uncertainty_from_shrine_relative(
				dm_values,
				li_values,
				uncertainty_reference_profiles,
				kc=self._nonshrine_resolved_kc,
			)
			if (
				li_uncertainty.get('uncertainty_low_dm') is None
				or li_uncertainty.get('uncertainty_high_dm') is None
			):
				li_uncertainty = self._uncertainty_from_local_quadratic(dm_values, li_values, max_idx)
				if (
					li_uncertainty.get('uncertainty_low_dm') is None
					or li_uncertainty.get('uncertainty_high_dm') is None
				):
					li_uncertainty = self._uncertainty_from_half_prominence(dm_values, li_values, max_idx)
		else:
			li_uncertainty = self._uncertainty_from_half_prominence(dm_values, li_values, max_idx)
		li_uncertainty = self._clamp_uncertainty_to_dm_bounds(
			optimal_dm,
			li_uncertainty,
			dm_values,
			fill_missing_with_bounds=not self.use_nonshrine_shrine_like_uncertainty,
		)
		run_prefix = f"{label}_{segment or 'segment'}_l_i_{mode}"
		run_dir = self._save_nonshrine_run_outputs(
			run_prefix=run_prefix,
			method_label=f"L/I Maximising ({mode})",
			dm_values=dm_values,
			metric_values=li_values,
			metric_name=f"L_over_I_{mode}",
			dedispersed_i=dedispersed_i,
			optimal_dm=optimal_dm,
			optimal_metric=metric,
			uncertainty=li_uncertainty,
		)
		
		mode_labels = {
			'peak': "L/I Maximising (peak)",
			'mean': "L/I Maximising (mean)",
			'max': "L/I Maximising (max)"
		}
		
		result = {
			'dm': optimal_dm,
			'metric': metric,
			'dedispersed': dedispersed_i,
			'method': mode_labels.get(mode, f'L/I Maximising ({mode})'),
			'run_dir': str(run_dir),
			'dm_values': dm_values.copy(),
			'metric_values': li_values.copy(),
			**li_uncertainty,
		}
		
		return result
	
	def optimise_dm_structure(self, dm_range: Tuple[float, float],
							 data: Optional[np.ndarray] = None,
							 n_points: int = 200,
							 dm_step: Optional[float] = None,
							 label: str = "frb",
							 segment: Optional[str] = None) -> Dict:
		"""
		Optimise DM using SHRINE structure maximisation.
		
		Parameters:
		-----------
		dm_range : tuple
			(min_dm, max_dm) range to search
		data : np.ndarray, optional
			Data to optimise. If None, uses self.stokes_i
			
		Returns:
		--------
		result : dict
			Dictionary with 'dm', 'metric', and 'dedispersed' keys
		"""
		if data is None:
			data = self.stokes_i

		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		
		# Calculate max output size to ensure consistent shapes
		output_size = self._max_output_size_for_dm_range(data, dm_range)
		i_data = np.zeros((len(dm_values), output_size))

		for i, dm in enumerate(dm_values):
			dedispersed = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nanmean(dedispersed, axis=0)

		run_prefix = f"{label}_{segment or 'segment'}_structure"
		run_dir = self._run_shrine_method(
			script_name="maximise_structure.py",
			run_prefix=run_prefix,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=True,
			save_all=True,
		)
		structure_values = np.loadtxt(run_dir / f"{run_prefix}_SPs.dat")
		summary_path = run_dir / f"{run_prefix}_structure_summaryfile.txt"
		kc = None
		if summary_path.exists():
			with open(summary_path, "r") as f:
				for line in f:
					if line.strip().startswith("kc:"):
						try:
							kc = int(line.split(":", 1)[1].strip())
						except Exception:
							kc = None
						break
		max_idx = int(np.argmax(structure_values))
		optimal_dm = float(dm_values[max_idx])
		dedispersed = self.dedisperse(data, optimal_dm, mode=self.dedisp_mode)
		metric = float(structure_values[max_idx])
		structure_uncertainty = self._structure_uncertainty_from_shrine_outputs(
			dm_values=dm_values,
			structure_values=structure_values,
			run_dir=run_dir,
			run_prefix=run_prefix,
			best_idx=max_idx,
		)
		
		result = {
			'dm': optimal_dm,
			'metric': metric,
			'dedispersed': dedispersed,
			'method': 'Structure Maximising (SHRINE)',
			'kc': kc,
			'dm_values': dm_values.copy(),
			'metric_values': structure_values.copy(),
			**structure_uncertainty,
		}
		
		return result
	
	def optimise_dm_snr(self, dm_range: Tuple[float, float], 
					   data: Optional[np.ndarray] = None,
					   n_points: int = 200,
					   dm_step: Optional[float] = None,
					   label: str = "frb",
					   segment: Optional[str] = None) -> Dict:
		"""
		Optimise DM using SHRINE S/N maximisation method.
		
		Parameters:
		-----------
		dm_range : tuple
			(min_dm, max_dm) range to search
		data : np.ndarray, optional
			Data to optimise. If None, uses self.stokes_i
			
		Returns:
		--------
		result : dict
			Dictionary with 'dm', 'metric', and 'dedispersed' keys
		"""
		if data is None:
			data = self.stokes_i

		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		
		# Calculate max output size to ensure consistent shapes
		output_size = self._max_output_size_for_dm_range(data, dm_range)
		i_data = np.zeros((len(dm_values), output_size))

		for i, dm in enumerate(dm_values):
			dedispersed = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nanmean(dedispersed, axis=0)

		run_prefix = f"{label}_{segment or 'segment'}_snr"
		run_dir = self._run_shrine_method(
			script_name="maximise_sn.py",
			run_prefix=run_prefix,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=False,
			save_all=True,
		)
		sn_path = run_dir / f"{run_prefix}_SNs.dat"
		if not sn_path.exists():
			raise FileNotFoundError(f"Expected SHRINE S/N output not found: {sn_path}")
		snr_values = np.loadtxt(sn_path)
		max_idx = int(np.argmax(snr_values))
		optimal_dm = float(dm_values[max_idx])
		dedispersed = self.dedisperse(data, optimal_dm, mode=self.dedisp_mode)
		metric = float(snr_values[max_idx])
		snr_uncertainty = self._uncertainty_from_snr_drop(dm_values, snr_values, max_idx, drop=1.0)
		
		result = {
			'dm': optimal_dm,
			'metric': metric,
			'dedispersed': dedispersed,
			'method': 'S/N Maximising (SHRINE)',
			'dm_values': dm_values.copy(),
			'metric_values': snr_values.copy(),
			**snr_uncertainty,
		}
		
		return result
	
	def separate_peaks(self, min_separation_ms: float = 1.0,
					   diagnostics_path: Optional[str] = None) -> List[Tuple[int, int]]:
		"""
		Identify and separate peaks in the time series.
		
		Parameters:
		-----------
		min_separation_ms : float
			Minimum separation between peaks in milliseconds
			
		Returns:
		--------
		peak_regions : list of tuples
			List of (start_idx, end_idx) for each peak region
		"""
		# Collapse to time series
		time_series = np.mean(self.stokes_i, axis=0)
		
		# Smooth to find peaks
		smoothed = gaussian_filter1d(time_series, sigma=4)
		
		# Find peaks
		dt = np.median(np.diff(self.time_ms))
		min_distance = int(min_separation_ms / dt)
		n_edge = max(1, int(0.05 * len(smoothed)))
		peaks, properties = find_peaks(smoothed, distance=min_distance, prominence=2*np.std(smoothed[:n_edge]))

		if diagnostics_path:
			plt.figure(figsize=(10, 4))
			plt.plot(self.time_ms, time_series, color='0.6', linewidth=1, label='Raw')
			plt.plot(self.time_ms, smoothed, color='k', linewidth=1.5, label='Smoothed')
			if len(peaks) > 0:
				plt.scatter(self.time_ms[peaks], smoothed[peaks], color='red', s=20, label='Peaks')
			plt.xlabel('Time (ms)')
			plt.ylabel('Flux (arb.)')
			plt.title('Peak Finding Diagnostics')
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			savefig_rasterized(diagnostics_path, dpi=150, bbox_inches='tight')
			plt.close()
		
		if len(peaks) == 0:
			# No peaks found, return entire range
			return [(0, self.n_time)]
		
		# Define regions around each peak
		peak_regions = []
		for peak in peaks:
			# Find boundaries (go to minima on either side)
			# Simple approach: go to half-max on either side
			half_max = (smoothed[peak] - np.min(smoothed)) / 2 + np.min(smoothed)
			
			# Search left
			start = peak
			while start > 0 and smoothed[start] > half_max:
				start -= 1
			
			# Search right
			end = peak
			while end < len(smoothed) - 1 and smoothed[end] > half_max:
				end += 1
			
			peak_regions.append((max(0, start - 20), min(self.n_time, end + 80)))
		
		return peak_regions

	def select_peaks_manual(self) -> List[Tuple[int, int]]:
		"""Manually select peak bounds by clicking on the pulse profile."""
		time_series = np.nanmean(self.stokes_i, axis=0)
		return shared_select_peaks_manual(
			self.time_ms,
			time_series,
			title='Click start/end bounds for each peak (close window to finish)',
			x_label='Time (ms)',
			y_label='Flux',
			exclusive_end=True,
		)
	
	def compare_methods(self, dm_range: Tuple[float, float], 
					   peak_region: Optional[Tuple[int, int]] = None,
					   n_points: int = 200,
					   dm_step: Optional[float] = None,
					   segment_tag: str = "segment1",
					   label: str = "frb",
					   selected_methods: Optional[List[str]] = None) -> Dict:
		"""
		Compare all optimisation methods on the same data.
		
		Parameters:
		-----------
		dm_range : tuple
			(min_dm, max_dm) range to search
		peak_region : tuple, optional
			(start_idx, end_idx) to focus on. If None, uses entire time range.
			
		Returns:
		--------
		results : dict
			Dictionary containing results from each method
		"""
		# Select data region
		if peak_region is not None:
			data = self.stokes_i[:, peak_region[0]:peak_region[1]]
			data_q = None if self.stokes_q is None else self.stokes_q[:, peak_region[0]:peak_region[1]]
			data_u = None if self.stokes_u is None else self.stokes_u[:, peak_region[0]:peak_region[1]]
		else:
			data = self.stokes_i
			data_q = self.stokes_q
			data_u = self.stokes_u

		print(f"Comparing methods on DM range [{dm_range[0]:.2f}, {dm_range[1]:.2f}] pc cm^-3")

		has_qu = data_q is not None and data_u is not None
		results: Dict[str, Dict] = {}
		all_method_keys = ['structure', 'snr', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		if selected_methods is None:
			selected = set(all_method_keys)
		else:
			unknown = sorted(set(selected_methods) - set(all_method_keys))
			if len(unknown) > 0:
				raise ValueError(f"Unknown methods in selected_methods: {unknown}")
			selected = set(selected_methods)

		run_structure = 'structure' in selected
		run_snr = 'snr' in selected
		run_pa = 'pa_slope' in selected
		run_pa_shrine = 'pa_slope_shrine' in selected
		run_li_mean = 'l_i_mean' in selected
		run_qu_methods = run_pa or run_pa_shrine or run_li_mean
		if run_qu_methods and not has_qu:
			print("  - Skipping selected PA/LI methods (no Stokes Q/U provided)")
			run_pa = False
			run_pa_shrine = False
			run_li_mean = False
			run_qu_methods = False

		if not (run_structure or run_snr or run_qu_methods):
			print("  - No methods selected after filtering; returning empty results.")
			return results

		# Shared DM sweep for all methods.
		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		output_size = self._max_output_size_for_dm_range(data, dm_range)
		i_data = np.zeros((len(dm_values), output_size), dtype=float)

		pa_values = np.zeros(len(dm_values), dtype=float) if run_pa else None
		pa_shrine_values = np.zeros(len(dm_values), dtype=float) if run_pa_shrine else None
		li_mean_values = np.zeros(len(dm_values), dtype=float) if run_li_mean else None
		pa_uncertainty_profiles = (
			np.zeros((len(dm_values), output_size), dtype=float)
			if (run_pa or run_pa_shrine)
			else None
		)
		li_uncertainty_profiles = (
			np.zeros((len(dm_values), output_size), dtype=float)
			if run_li_mean
			else None
		)

		self._reset_nonshrine_kc_state()
		dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = self.time_ms[0] + np.arange(output_size) * dt

		print(f"  - Shared DM sweep for all methods ({len(dm_values)} trials)...")
		for i, dm in enumerate(dm_values):
			if i % 25 == 0:
				print(f"\r    Progress: {i}/{len(dm_values)}", end='', flush=True)

			dedisp_i = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nanmean(dedisp_i, axis=0)

			if run_qu_methods:
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
				if pa_values is not None:
					pa_values[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i)
				if pa_shrine_values is not None:
					pa_shrine_values[i] = self._pa_slope_metric_shrine(sm_q, sm_u, time_axis, sm_i)
				if li_mean_values is not None:
					li_mean_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode='mean')
				if pa_uncertainty_profiles is not None:
					pa_uncertainty_profiles[i] = self._nonshrine_uncertainty_reference_profile(sm_i, sm_q, sm_u)
				if li_uncertainty_profiles is not None:
					li_uncertainty_profiles[i] = self._li_uncertainty_reference_profile(sm_q, sm_u, sm_i)
		print(f"\r    Progress: {len(dm_values)}/{len(dm_values)}", flush=True)

		if run_structure:
			print("  - Testing Structure Maximising (SHRINE)...")
			run_prefix_structure = f"{label}_{segment_tag}_structure"
			run_dir_structure = self._run_shrine_method(
				script_name="maximise_structure.py",
				run_prefix=run_prefix_structure,
				dm_values=dm_values,
				i_data=i_data,
				include_input_dm=True,
				save_all=True,
			)
			structure_values = np.loadtxt(run_dir_structure / f"{run_prefix_structure}_SPs.dat")
			summary_path = run_dir_structure / f"{run_prefix_structure}_structure_summaryfile.txt"
			kc = None
			if summary_path.exists():
				with open(summary_path, "r") as f:
					for line in f:
						if line.strip().startswith("kc:"):
							try:
								kc = int(line.split(":", 1)[1].strip())
							except Exception:
								kc = None
							break
			max_idx_structure = int(np.argmax(structure_values))
			optimal_dm_structure = float(dm_values[max_idx_structure])
			dedispersed_structure = self.dedisperse(data, optimal_dm_structure, mode=self.dedisp_mode)
			metric_structure = float(structure_values[max_idx_structure])
			structure_uncertainty = self._structure_uncertainty_from_shrine_outputs(
				dm_values=dm_values,
				structure_values=structure_values,
				run_dir=run_dir_structure,
				run_prefix=run_prefix_structure,
				best_idx=max_idx_structure,
			)
			results['structure'] = {
				'dm': optimal_dm_structure,
				'metric': metric_structure,
				'dedispersed': dedispersed_structure,
				'method': 'Structure Maximising (SHRINE)',
				'kc': kc,
				'run_dir': str(run_dir_structure),
				'dm_values': dm_values.copy(),
				'metric_values': np.asarray(structure_values, dtype=float).copy(),
				**structure_uncertainty,
			}
			if has_qu:
				n_time_out = results['structure']['dedispersed'].shape[1]
				results['structure']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_structure, output_size=n_time_out, mode=self.dedisp_mode)
				results['structure']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_structure, output_size=n_time_out, mode=self.dedisp_mode)

		if run_snr:
			print("  - Testing S/N Maximising (SHRINE)...")
			run_prefix_snr = f"{label}_{segment_tag}_snr"
			run_dir_snr = self._run_shrine_method(
				script_name="maximise_sn.py",
				run_prefix=run_prefix_snr,
				dm_values=dm_values,
				i_data=i_data,
				include_input_dm=False,
				save_all=True,
			)
			sn_path = run_dir_snr / f"{run_prefix_snr}_SNs.dat"
			if not sn_path.exists():
				raise FileNotFoundError(f"Expected SHRINE S/N output not found: {sn_path}")
			snr_values = np.loadtxt(sn_path)
			max_idx_snr = int(np.argmax(snr_values))
			optimal_dm_snr = float(dm_values[max_idx_snr])
			dedispersed_snr = self.dedisperse(data, optimal_dm_snr, mode=self.dedisp_mode)
			metric_snr = float(snr_values[max_idx_snr])
			snr_uncertainty = self._uncertainty_from_snr_drop(dm_values, snr_values, max_idx_snr, drop=1.0)
			results['snr'] = {
				'dm': optimal_dm_snr,
				'metric': metric_snr,
				'dedispersed': dedispersed_snr,
				'method': 'S/N Maximising (SHRINE)',
				'run_dir': str(run_dir_snr),
				'dm_values': dm_values.copy(),
				'metric_values': np.asarray(snr_values, dtype=float).copy(),
				**snr_uncertainty,
			}
			if has_qu:
				n_time_out = results['snr']['dedispersed'].shape[1]
				results['snr']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_snr, output_size=n_time_out, mode=self.dedisp_mode)
				results['snr']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_snr, output_size=n_time_out, mode=self.dedisp_mode)

		if not run_qu_methods:
			return results

		print("  - Testing selected PA slope/L/I methods from shared sweep...")

		# PA slope (raw masked PA)
		if run_pa and pa_values is not None:
			max_idx_pa = int(np.argmax(pa_values))
			optimal_dm_pa = float(dm_values[max_idx_pa])
			best_dedisp_i_pa = self.dedisperse(data, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_q_pa = self.dedisperse(data_q, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_u_pa = self.dedisperse(data_u, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_sm_i_pa, best_sm_q_pa, best_sm_u_pa = self._maybe_kc_smooth_nonshrine(best_dedisp_i_pa, best_dedisp_q_pa, best_dedisp_u_pa)
			best_pa_smooth, best_fit_line, best_time_axis = self._get_pa_smoothed_and_fit(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa, time_axis)
			best_pa_deg = self._pa_series_deg(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa)
			metric_pa = float(pa_values[max_idx_pa])
			if self.use_nonshrine_shrine_like_uncertainty and pa_uncertainty_profiles is not None:
				pa_uncertainty = self._uncertainty_from_shrine_relative(
					dm_values,
					pa_values,
					pa_uncertainty_profiles,
					kc=self._nonshrine_resolved_kc,
				)
			else:
				pa_uncertainty = self._uncertainty_from_half_prominence(dm_values, pa_values, max_idx_pa)
			run_prefix_pa = f"{label}_{segment_tag}_pa_slope"
			run_dir_pa = self._save_nonshrine_run_outputs(
				run_prefix=run_prefix_pa,
				method_label='PA Slope Maximising',
				dm_values=dm_values,
				metric_values=pa_values,
				metric_name='PA_Slope',
				dedispersed_i=best_dedisp_i_pa,
				optimal_dm=optimal_dm_pa,
				optimal_metric=metric_pa,
				uncertainty=pa_uncertainty,
			)
			results['pa_slope'] = {
				'dm': optimal_dm_pa,
				'metric': metric_pa,
				'dedispersed': best_dedisp_i_pa,
				'dedispersed_q': best_dedisp_q_pa,
				'dedispersed_u': best_dedisp_u_pa,
				'method': 'PA Slope Maximising',
				'pa_plot_kind': 'raw',
				'pa_plot_time': best_time_axis.copy(),
				'pa_plot_series': best_pa_deg.copy(),
				'pa_plot_smooth': best_pa_smooth.copy(),
				'pa_plot_fit': best_fit_line.copy(),
				'run_dir': str(run_dir_pa),
				'dm_values': dm_values.copy(),
				'metric_values': pa_values.copy(),
				**pa_uncertainty,
			}

		# PA slope (SHRINE-smoothed PA)
		if run_pa_shrine and pa_shrine_values is not None:
			max_idx_pas = int(np.argmax(pa_shrine_values))
			optimal_dm_pas = float(dm_values[max_idx_pas])
			best_dedisp_i_pas = self.dedisperse(data, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_q_pas = self.dedisperse(data_q, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_u_pas = self.dedisperse(data_u, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_sm_i_pas, best_sm_q_pas, best_sm_u_pas = self._maybe_kc_smooth_nonshrine(best_dedisp_i_pas, best_dedisp_q_pas, best_dedisp_u_pas)
			best_pa_shrine_smooth, best_shrine_fit_line, best_shrine_time_axis = self._get_pa_shrine_smoothed_and_fit(
				best_sm_q_pas,
				best_sm_u_pas,
				best_sm_i_pas,
				time_axis,
				force_kc=self._nonshrine_resolved_kc,
			)
			best_pa_shrine_deg = self._pa_series_deg(best_sm_q_pas, best_sm_u_pas, best_sm_i_pas)
			metric_pas = float(pa_shrine_values[max_idx_pas])
			if self.use_nonshrine_shrine_like_uncertainty and pa_uncertainty_profiles is not None:
				pa_shrine_uncertainty = self._uncertainty_from_shrine_relative(
					dm_values,
					pa_shrine_values,
					pa_uncertainty_profiles,
					kc=self._nonshrine_resolved_kc,
				)
			else:
				pa_shrine_uncertainty = self._uncertainty_from_half_prominence(dm_values, pa_shrine_values, max_idx_pas)
			run_prefix_pas = f"{label}_{segment_tag}_pa_slope_shrine"
			run_dir_pas = self._save_nonshrine_run_outputs(
				run_prefix=run_prefix_pas,
				method_label='PA Slope Maximising (SHRINE PA)',
				dm_values=dm_values,
				metric_values=pa_shrine_values,
				metric_name='PA_Slope_SHRINE',
				dedispersed_i=best_dedisp_i_pas,
				optimal_dm=optimal_dm_pas,
				optimal_metric=metric_pas,
				uncertainty=pa_shrine_uncertainty,
			)
			results['pa_slope_shrine'] = {
				'dm': optimal_dm_pas,
				'metric': metric_pas,
				'dedispersed': best_dedisp_i_pas,
				'dedispersed_q': best_dedisp_q_pas,
				'dedispersed_u': best_dedisp_u_pas,
				'method': 'PA Slope Maximising (SHRINE PA)',
				'kc': None if self._nonshrine_resolved_kc is None else int(self._nonshrine_resolved_kc),
				'pa_plot_kind': 'shrine',
				'pa_plot_time': best_shrine_time_axis.copy(),
				'pa_plot_series': best_pa_shrine_deg.copy(),
				'pa_plot_smooth': best_pa_shrine_smooth.copy(),
				'pa_plot_fit': best_shrine_fit_line.copy(),
				'run_dir': str(run_dir_pas),
				'dm_values': dm_values.copy(),
				'metric_values': pa_shrine_values.copy(),
				**pa_shrine_uncertainty,
			}

		# L/I mean
		if run_li_mean and li_mean_values is not None:
			max_idx_li_mean = int(np.argmax(li_mean_values))
			optimal_dm_li_mean = float(dm_values[max_idx_li_mean])
			dedispersed_li_mean = self.dedisperse(data, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode)
			metric_li_mean = float(li_mean_values[max_idx_li_mean])
			if self.use_nonshrine_shrine_like_uncertainty and li_uncertainty_profiles is not None:
				li_mean_uncertainty = self._uncertainty_from_shrine_relative(
					dm_values,
					li_mean_values,
					li_uncertainty_profiles,
					kc=self._nonshrine_resolved_kc,
				)
				if (
					li_mean_uncertainty.get('uncertainty_low_dm') is None
					or li_mean_uncertainty.get('uncertainty_high_dm') is None
				):
					li_mean_uncertainty = self._uncertainty_from_local_quadratic(
						dm_values, li_mean_values, max_idx_li_mean
					)
					if (
						li_mean_uncertainty.get('uncertainty_low_dm') is None
						or li_mean_uncertainty.get('uncertainty_high_dm') is None
					):
						li_mean_uncertainty = self._uncertainty_from_half_prominence(
							dm_values, li_mean_values, max_idx_li_mean
						)
			else:
				li_mean_uncertainty = self._uncertainty_from_half_prominence(dm_values, li_mean_values, max_idx_li_mean)
			li_mean_uncertainty = self._clamp_uncertainty_to_dm_bounds(
				optimal_dm_li_mean,
				li_mean_uncertainty,
				dm_values,
				fill_missing_with_bounds=not self.use_nonshrine_shrine_like_uncertainty,
			)
			run_prefix_li_mean = f"{label}_{segment_tag}_l_i_mean"
			run_dir_li_mean = self._save_nonshrine_run_outputs(
				run_prefix=run_prefix_li_mean,
				method_label='L/I Maximising (mean)',
				dm_values=dm_values,
				metric_values=li_mean_values,
				metric_name='L_over_I_mean',
				dedispersed_i=dedispersed_li_mean,
				optimal_dm=optimal_dm_li_mean,
				optimal_metric=metric_li_mean,
				uncertainty=li_mean_uncertainty,
			)
			results['l_i_mean'] = {
				'dm': optimal_dm_li_mean,
				'metric': metric_li_mean,
				'dedispersed': dedispersed_li_mean,
				'dedispersed_q': self.dedisperse(data_q, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode),
				'dedispersed_u': self.dedisperse(data_u, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode),
				'method': 'L/I Maximising (mean)',
				'run_dir': str(run_dir_li_mean),
				'dm_values': dm_values.copy(),
				'metric_values': li_mean_values.copy(),
				**li_mean_uncertainty,
			}

		return results
	
	def plot_comparison(self, results: Dict, dm_range: Tuple[float, float],
					   peak_region: Optional[Tuple[int, int]] = None,
					   save_path: Optional[str] = None):
		"""
		Create visualization comparing all methods.
		
		Parameters:
		-----------
		results : dict
			Results dictionary from compare_methods
		dm_range : tuple
			DM range used for optimisation
		peak_region : tuple, optional
			(start_idx, end_idx) of the region analyzed
		save_path : str, optional
			Path to save figure. If None, displays instead.
		"""
		n_methods = len(results)
		has_qu = self.stokes_q is not None and self.stokes_u is not None
		base_width, base_height = pub_figsize(single_column=False, height_ratio=0.9, min_height=5.5)
		fig_width = base_width
		row_height = 2.7
		fig_height = max(base_height, row_height * (n_methods + 1))
		fig, axes = plt.subplots(
			n_methods + 1,
			5,
			figsize=(fig_width, fig_height),
			gridspec_kw={'width_ratios': [0.85, 0.11, 0.85, 0.32, 0.85]},
		)
		if n_methods == 0:
			axes = np.atleast_2d(axes)

		# Remove spacer-column axes and remap remaining columns to [left, middle, right]
		for spacer_ax in axes[:, 1]:
			fig.delaxes(spacer_ax)
		for spacer_ax in axes[:, 3]:
			fig.delaxes(spacer_ax)
		axes = np.stack((axes[:, 0], axes[:, 2], axes[:, 4]), axis=1)
		original_highlight_bg = '#FFF7D6'
		original_highlight_edge = '#C88A00'
		scan_colors = {
			'structure': 'tab:blue',
			'snr': 'tab:red',
			'pa_slope': 'tab:green',
			'pa_slope_shrine': 'tab:cyan',
			'l_i_mean': 'tab:purple',
		}
		scan_labels = {
			'structure': 'Structure',
			'snr': 'S/N',
			'pa_slope': 'PA',
			'pa_slope_shrine': 'PA (SHRINE)',
			'l_i_mean': 'L/I mean',
		}
		fs_title = 16
		fs_label = 14
		fs_tick = 12
		fs_legend = 11
		fs_overlay = 12
		fs_labelpad = 2
		
		# Plot original data
		if peak_region is not None:
			original_data = self.stokes_i[:, peak_region[0]:peak_region[1]]
			time_range = self.time_ms[peak_region[0]:peak_region[1]]
		else:
			original_data = self.stokes_i
			time_range = self.time_ms

		pa_limits = None
		q_region = None
		u_region = None
		if has_qu:
			if peak_region is not None:
				q_region = self.stokes_q[:, peak_region[0]:peak_region[1]]
				u_region = self.stokes_u[:, peak_region[0]:peak_region[1]]
			else:
				q_region = self.stokes_q
				u_region = self.stokes_u
			pa_series = [self._pa_series_deg(q_region, u_region, original_data)]
			for result in results.values():
				n_time_out = result['dedispersed'].shape[1]
				dedisp_q = result.get('dedispersed_q')
				dedisp_u = result.get('dedispersed_u')
				if dedisp_q is None or dedisp_u is None:
					dedisp_q = self.dedisperse(q_region, result['dm'], output_size=n_time_out, mode=self.dedisp_mode)
					dedisp_u = self.dedisperse(u_region, result['dm'], output_size=n_time_out, mode=self.dedisp_mode)
				pa_series.append(self._pa_series_deg(dedisp_q, dedisp_u, result['dedispersed']))
			pa_all = np.concatenate([p[np.isfinite(p)] for p in pa_series if p is not None])
			if pa_all.size > 0:
				pa_min = float(np.nanmin(pa_all)) - 10.0
				pa_max = float(np.nanmax(pa_all)) + 10.0
				pa_limits = (pa_min, pa_max)
		
		# Original waterfall
		vmin0, vmax0 = self._robust_vmin_vmax(original_data)
		im0 = axes[0, 0].imshow(
			original_data,
			aspect='auto',
			extent=[time_range[0], time_range[-1], self.freq_mhz[0], self.freq_mhz[-1]],
			cmap='viridis',
			origin='lower',
			vmin=vmin0,
			vmax=vmax0,
		)
		axes[0, 0].set_title("Original Data (SHRINE structure-maximised)\n"+rf"Input DM = {self._format_dm(self.input_dm, 3)} pc cm$^{{-3}}$")
		axes[0, 0].set_ylabel('Frequency (MHz)')
		axes[0, 0].set_xlabel('Time (ms)')
		axes[0, 0].title.set_fontsize(fs_title)
		axes[0, 0].xaxis.label.set_size(fs_label)
		axes[0, 0].yaxis.label.set_size(fs_label)
		axes[0, 0].xaxis.labelpad = fs_labelpad
		axes[0, 0].yaxis.labelpad = fs_labelpad
		axes[0, 0].tick_params(axis='both', labelsize=fs_tick)

		# Original time series
		time_series_orig = np.mean(original_data, axis=0)
		axes[0, 1].plot(time_range, time_series_orig, 'k-', linewidth=1, label='I')
		ax0r = None
		if has_qu:
			ax0r = axes[0, 1].twinx()
			L_series = np.mean(np.sqrt(q_region**2 + u_region**2), axis=0)
			if self.full_L_noise_median is not None:
				L_baseline = self.full_L_noise_median
			else:
				n_edge = max(1, int(0.05 * len(L_series)))
				L_baseline = np.median(L_series[:n_edge])
			L_plot = L_series - L_baseline
			axes[0, 1].plot(time_range, L_plot, 'r', linewidth=1, label='L')
			pa_deg = self._pa_series_deg(q_region, u_region, original_data)
			pa_smooth, fit_line, _ = self._get_pa_smoothed_and_fit(q_region, u_region, original_data, time_range)
			ax0r.plot(time_range, pa_deg, color='silver', linewidth=1, alpha=0.9)#, label='PA')
			ax0r.plot(time_range, pa_smooth, color='tab:purple', linewidth=2, alpha=0.8, label='PA')
			ax0r.plot(time_range, fit_line, color='tab:orange', linewidth=1.5, linestyle='--', alpha=0.7, label='PA fit')

		if ax0r is not None:
			h1, l1 = axes[0, 1].get_legend_handles_labels()
			h2, l2 = ax0r.get_legend_handles_labels()
			axes[0, 1].legend(h1 + h2, l1 + l2, loc='best', fontsize=fs_legend)
			ax0r.set_ylabel('PA (deg)')
			ax0r.yaxis.label.set_size(fs_label)
			ax0r.yaxis.labelpad = fs_labelpad
			ax0r.tick_params(axis='y', labelsize=fs_tick)
			if pa_limits is not None:
				ax0r.set_ylim(pa_limits)
		else:
			axes[0, 1].legend(loc='best', fontsize=fs_legend)
		axes[0, 1].set_title('Original Time Series')
		axes[0, 1].set_ylabel('Flux')
		axes[0, 1].set_xlabel('Time (ms)')
		axes[0, 1].grid(True, alpha=0.3)
		axes[0, 1].title.set_fontsize(fs_title)
		axes[0, 1].xaxis.label.set_size(fs_label)
		axes[0, 1].yaxis.label.set_size(fs_label)
		axes[0, 1].xaxis.labelpad = fs_labelpad
		axes[0, 1].yaxis.labelpad = fs_labelpad
		axes[0, 1].tick_params(axis='both', labelsize=fs_tick)

		# Top-right panel: compact summary of best DM and uncertainty per method.
		all_scan_ax = axes[0, 2]
		method_names = list(results.keys())
		if len(method_names) > 0:
			y_pos = np.arange(len(method_names), dtype=float)
			y_labels = [scan_labels.get(name, name) for name in method_names]

			for j, method_name in enumerate(method_names):
				result = results[method_name]
				dm_best = float(result['dm'])
				minus = result.get('uncertainty_minus')
				plus = result.get('uncertainty_plus')
				xerr = np.array([
					[0.0 if minus is None else float(minus)],
					[0.0 if plus is None else float(plus)],
				])
				all_scan_ax.errorbar(
					x=[dm_best],
					y=[y_pos[j]],
					xerr=xerr,
					fmt='o',
					markersize=5,
					capsize=3,
					elinewidth=1.8,
					color=scan_colors.get(method_name, 'black'),
				)

			all_scan_ax.axvline(self.input_dm, color='gray', linestyle=':', linewidth=1.5, alpha=0.9)
			all_scan_ax.set_xlim(dm_range[0], dm_range[1])
			all_scan_ax.set_yticks(y_pos)
			all_scan_ax.set_yticklabels(y_labels)
			all_scan_ax.invert_yaxis()
			all_scan_ax.set_title('Best DM Summary')
			all_scan_ax.set_xlabel(rf'DM (pc cm$^{{-3}}$)')
			all_scan_ax.set_ylabel('')
			all_scan_ax.grid(True, axis='x', alpha=0.3)
			all_scan_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
			all_scan_ax.title.set_fontsize(fs_title)
			all_scan_ax.xaxis.label.set_size(fs_label)
			all_scan_ax.yaxis.label.set_size(fs_label)
			all_scan_ax.xaxis.labelpad = fs_labelpad
			all_scan_ax.yaxis.labelpad = fs_labelpad
			all_scan_ax.tick_params(axis='both', labelsize=fs_tick)
		else:
			all_scan_ax.text(0.5, 0.5, 'No method results', ha='center', va='center', transform=all_scan_ax.transAxes)
			all_scan_ax.set_axis_off()

		# Highlight the top row (original data) with a distinct background color and border
		#for ax in (axes[0, 0], axes[0, 1]):
		#	ax.set_facecolor(original_highlight_bg)
		#	for spine in ax.spines.values():
		#		spine.set_edgecolor(original_highlight_edge)
		#		spine.set_linewidth(2.5)

		# Plot each method's result (one row per method)
		show_time_legend = False
		show_scan_legend = True
		for idx, (method_name, result) in enumerate(results.items(), start=1):
			# Create time axis for dedispersed data (may be longer due to expansion)
			n_time_dedisp = result['dedispersed'].shape[1]
			dt = float(np.median(np.diff(time_range))) if len(time_range) > 1 else 1.0
			delay_samples = self._get_delay_samples(result['dm'])
			if self.dedisp_mode == 'crop':
				start_shift = int(np.max(delay_samples))
			else:
				start_shift = int(np.min(delay_samples))
			time_range_dedisp = time_range[0] + start_shift * dt + np.arange(n_time_dedisp) * dt

			vmin, vmax = self._robust_vmin_vmax(result['dedispersed'])
			im = axes[idx, 0].imshow(
				result['dedispersed'],
				aspect='auto',
				extent=[time_range_dedisp[0], time_range_dedisp[-1], self.freq_mhz[0], self.freq_mhz[-1]],
				cmap='viridis',
				origin='lower',
				vmin=vmin,
				vmax=vmax,
			)
			axes[idx, 0].set_title(f"{result['method']}")
			axes[idx, 0].set_ylabel('Frequency (MHz)')
			axes[idx, 0].set_xlabel('Time (ms)')
			axes[idx, 0].title.set_fontsize(fs_title)
			axes[idx, 0].xaxis.label.set_size(fs_label)
			axes[idx, 0].yaxis.label.set_size(fs_label)
			axes[idx, 0].xaxis.labelpad = fs_labelpad
			axes[idx, 0].yaxis.labelpad = fs_labelpad
			axes[idx, 0].tick_params(axis='both', labelsize=fs_tick)
			axes[idx, 0].text(
				0.98,
				0.98,
				(
					"DM="
					+ self._format_uncertainty(
						result['dm'],
						result.get('uncertainty_minus'),
						result.get('uncertainty_plus'),
						precision=3,
					)
					+ rf" pc cm$^{{-3}}$"
				),
				transform=axes[idx, 0].transAxes,
				ha='right',
				va='top',
				color='white',
				fontsize=fs_overlay,
				bbox=dict(facecolor='black', edgecolor='none', alpha=0.35, pad=2.0),
			)

			time_series = np.mean(result['dedispersed'], axis=0)
			axes[idx, 1].plot(time_range_dedisp, time_series, 'k-', linewidth=1, label='I')
			axr = None

			if has_qu:
				axr = axes[idx, 1].twinx()
				dedisp_q = result.get('dedispersed_q')
				dedisp_u = result.get('dedispersed_u')
				if dedisp_q is None or dedisp_u is None:
					dedisp_q = self.dedisperse(q_region, result['dm'], output_size=n_time_dedisp, mode=self.dedisp_mode)
					dedisp_u = self.dedisperse(u_region, result['dm'], output_size=n_time_dedisp, mode=self.dedisp_mode)
				L_series = np.mean(np.sqrt(dedisp_q**2 + dedisp_u**2), axis=0)
				if self.full_L_noise_median is not None:
					L_baseline = self.full_L_noise_median
				else:
					n_edge = max(1, int(0.05 * len(L_series)))
					L_baseline = np.median(L_series[:n_edge])
				L_plot = L_series - L_baseline
				axes[idx, 1].plot(time_range_dedisp, L_plot, 'r', linewidth=1, label='L')

				if (
					method_name in ('pa_slope', 'pa_slope_shrine')
					and result.get('pa_plot_series') is not None
					and result.get('pa_plot_smooth') is not None
					and result.get('pa_plot_fit') is not None
				):
					pa_deg = np.asarray(result['pa_plot_series'])
					pa_smooth_plot = np.asarray(result['pa_plot_smooth'])
					fit_plot = np.asarray(result['pa_plot_fit'])
				else:
					pa_deg = self._pa_series_deg(dedisp_q, dedisp_u, result['dedispersed'])
					if method_name == 'pa_slope_shrine':
						result_kc = result.get('kc', None)
						pa_smooth_plot, fit_plot, _ = self._get_pa_shrine_smoothed_and_fit(
							dedisp_q, dedisp_u, result['dedispersed'], time_range_dedisp, force_kc=result_kc
						)
					else:
						pa_smooth_plot, fit_plot, _ = self._get_pa_smoothed_and_fit(dedisp_q, dedisp_u, result['dedispersed'], time_range_dedisp)

				axr.plot(time_range_dedisp, pa_deg, color='silver', linewidth=1, alpha=0.9, label='PA')
				if method_name == 'pa_slope_shrine':
					text = "(S)"
				else:
					text = None
				axr.plot(time_range_dedisp, pa_smooth_plot, color='tab:purple', linewidth=2, alpha=0.8, label='PA sm' + (text if text else ''))
				axr.plot(time_range_dedisp, fit_plot, color='tab:orange', linewidth=1.5, linestyle='--', alpha=0.7, label='PA fit' + (text if text else ''))

				if show_time_legend:
					h1, l1 = axes[idx, 1].get_legend_handles_labels()
					h2, l2 = axr.get_legend_handles_labels()
					axes[idx, 1].legend(h1 + h2, l1 + l2, loc='best', fontsize=fs_legend)
					show_time_legend = False
				axr.set_ylabel('PA (deg)')
				axr.yaxis.label.set_size(fs_label)
				axr.yaxis.labelpad = fs_labelpad
				axr.tick_params(axis='y', labelsize=fs_tick)
				if pa_limits is not None:
					axr.set_ylim(pa_limits)
			else:
				if show_time_legend:
					axes[idx, 1].legend(loc='best', fontsize=fs_legend)
					show_time_legend = False

			axes[idx, 1].set_title(f"Metric = {result['metric']:.6f}")
			axes[idx, 1].set_ylabel('Flux')
			axes[idx, 1].set_xlabel('Time (ms)')
			axes[idx, 1].grid(True, alpha=0.3)
			axes[idx, 1].title.set_fontsize(fs_title)
			axes[idx, 1].xaxis.label.set_size(fs_label)
			axes[idx, 1].yaxis.label.set_size(fs_label)
			axes[idx, 1].xaxis.labelpad = fs_labelpad
			axes[idx, 1].yaxis.labelpad = fs_labelpad
			axes[idx, 1].tick_params(axis='both', labelsize=fs_tick)

			# Right column: per-method DM scan curve
			scan_ax = axes[idx, 2]
			dm_vals = result.get('dm_values')
			metric_vals = result.get('metric_values')
			if dm_vals is not None and metric_vals is not None:
				low_dm = result.get('uncertainty_low_dm')
				high_dm = result.get('uncertainty_high_dm')
				dm_left = float(dm_range[0])
				dm_right = float(dm_range[1])
				shade_low = dm_left if low_dm is None else float(low_dm)
				shade_high = dm_right if high_dm is None else float(high_dm)
				if shade_low <= shade_high:
					scan_ax.axvspan(
						shade_low,
						shade_high,
						color='tab:orange',
						alpha=0.18,
						label='DM uncertainty' if show_scan_legend else None,
					)
				scan_ax.plot(
					dm_vals,
					metric_vals,
					linewidth=2.0,
					color=scan_colors.get(method_name, 'black'),
				)
				scan_ax.axvline(self.input_dm, color='gray', linestyle=':', linewidth=1.4,
							alpha=0.9,
							label=(f'Input DM'))#={self._format_dm(self.input_dm, 3)}' if show_scan_legend else None))
				scan_ax.axvline(result['dm'], color='red', linestyle='--', linewidth=1.4,
							alpha=0.9,
							label=(
								(
									"Best DM"
									#+ " = "
									#+ self._format_uncertainty(
									#	result['dm'],
									#	result.get('uncertainty_minus'),
									#	result.get('uncertainty_plus'),
									#	precision=3,
									#)
								)
								if show_scan_legend else None
							)
						)
				scan_ax.set_xlim(dm_range[0], dm_range[1])
				scan_ax.set_xlabel(rf'DM (pc cm$^{{-3}}$)')
				scan_ax.set_ylabel('Metric')
				scan_ax.grid(True, alpha=0.3)
				scan_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
				scan_ax.xaxis.label.set_size(fs_label)
				scan_ax.yaxis.label.set_size(fs_label)
				scan_ax.xaxis.labelpad = fs_labelpad
				scan_ax.yaxis.labelpad = fs_labelpad
				scan_ax.tick_params(axis='both', labelsize=fs_tick)
				if show_scan_legend:
					scan_ax.legend(loc='upper left', fontsize=fs_legend)
					show_scan_legend = False
			else:
				scan_ax.text(0.5, 0.5, 'No scan data', ha='center', va='center', transform=scan_ax.transAxes)
				scan_ax.set_axis_off()
		
		plt.tight_layout(rect=[0.02, 0.02, 0.995, 0.995])
		fig.subplots_adjust(wspace=0.04, hspace=0.5)
		
		if save_path:
			savefig_rasterized(save_path, dpi=600, bbox_inches='tight')
			print(f"\nFigure saved to: {save_path}")
		else:
			plt.show()
	
	def scan_dm_space(self, dm_range: Tuple[float, float], n_points: int = 100,
					 data: Optional[np.ndarray] = None, dm_step: Optional[float] = None,
					 data_q: Optional[np.ndarray] = None, data_u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
		"""
		Scan through DM space and calculate metrics.
		
		Parameters:
		-----------
		dm_range : tuple
			(min_dm, max_dm) range to scan
		n_points : int
			Number of DM values to test (ignored if dm_step is provided)
		data : np.ndarray, optional
			Stokes I data to scan. If None, uses self.stokes_i
		dm_step : float, optional
			Step size between DMs. When provided, overrides n_points sampling.
		data_q : np.ndarray, optional
			Stokes Q data aligned with `data`. If None, uses self.stokes_q (if available).
		data_u : np.ndarray, optional
			Stokes U data aligned with `data`. If None, uses self.stokes_u (if available).
			
		Returns:
		--------
		dm_values : np.ndarray
			Array of DM values tested
		metrics : dict
			Dictionary of metric arrays for each method
		"""
		if data is None:
			data = self.stokes_i
		if data_q is None:
			data_q = self.stokes_q
		if data_u is None:
			data_u = self.stokes_u
		self._reset_nonshrine_kc_state()
		
		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		
		# Calculate max output size to ensure consistent shapes
		output_size = self._max_output_size_for_dm_range(data, dm_range)
		
		structure_values = np.zeros(len(dm_values))
		snr_values = np.zeros(len(dm_values))
		pa_slope_values = np.zeros(len(dm_values)) if (data_q is not None and data_u is not None) else None
		l_i_mean_values = np.zeros(len(dm_values)) if (data_q is not None and data_u is not None) else None
		i_data = np.zeros((len(dm_values), output_size))
		
		print(f"Scanning {len(dm_values)} DM values...")
		dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = np.arange(output_size) * dt
		for i, dm in enumerate(dm_values):
			if i % 20 == 0:
				print(f"\r  Progress: {i}/{len(dm_values)}", end='', flush=True)
			
			dedispersed = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nanmean(dedispersed, axis=0)
			if pa_slope_values is not None:
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedispersed, dedisp_q, dedisp_u)
				pa_slope_values[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i)
				if l_i_mean_values is not None:
					l_i_mean_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode='mean')

		run_tag = f"scan_{int(np.round(dm_values[0] * 1000))}_{int(np.round(dm_values[-1] * 1000))}_{len(dm_values)}"

		run_prefix_structure = f"{run_tag}_structure"
		run_dir_structure = self._run_shrine_method(
			script_name="maximise_structure.py",
			run_prefix=run_prefix_structure,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=True,
			save_all=True,
		)
		structure_values = np.loadtxt(run_dir_structure / f"{run_prefix_structure}_SPs.dat")

		run_prefix_snr = f"{run_tag}_snr"
		run_dir_snr = self._run_shrine_method(
			script_name="maximise_sn.py",
			run_prefix=run_prefix_snr,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=False,
			save_all=True,
		)
		sn_path = run_dir_snr / f"{run_prefix_snr}_SNs.dat"
		if not sn_path.exists():
			raise FileNotFoundError(f"Expected SHRINE S/N output not found: {sn_path}")
		snr_values = np.loadtxt(sn_path)
		
		metrics = {
			'structure': structure_values,
			'snr': snr_values
		}
		if pa_slope_values is not None:
			metrics['pa_slope'] = pa_slope_values
			pa_slope_shrine_values = np.zeros(len(dm_values))
			for i, dm in enumerate(dm_values):
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_i = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self._maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
				pa_slope_shrine_values[i] = self._pa_slope_metric_shrine(sm_q, sm_u, time_axis, sm_i)
			metrics['pa_slope_shrine'] = pa_slope_shrine_values
		if l_i_mean_values is not None:
			metrics['l_i_mean'] = l_i_mean_values
		
		return dm_values, metrics
	
	def plot_dm_scan(self, dm_values: np.ndarray, metrics: Dict,
					save_path: Optional[str] = None):
		"""
		Plot metric values across DM space.
		
		Parameters:
		-----------
		dm_values : np.ndarray
			Array of DM values
		metrics : dict
			Dictionary of metric arrays
		save_path : str, optional
			Path to save figure
		"""
		base_width, base_height = pub_figsize(single_column=True, height_ratio=0.75, min_height=3.4)
		row_height = 2.6
		fig_height = max(base_height, row_height * len(metrics))
		fig, axes = plt.subplots(len(metrics), 1, figsize=(base_width, fig_height))
		
		if len(metrics) == 1:
			axes = [axes]
		
		colors = {'structure': 'blue', 'snr': 'red', 'pa_slope': 'green', 'pa_slope_shrine': 'teal',
				  'l_i_mean': 'purple'}
		labels = {
			'structure': 'Structure Metric (SHRINE)',
			'snr': 'S/N',
			'pa_slope': "Weighted PA Slope magnitude",
			'pa_slope_shrine': "Weighted PA Slope magnitude (SHRINE-smoothed PA)",
			'l_i_mean': "L/I (mean)"
		}
		
		for idx, (metric_name, metric_values) in enumerate(metrics.items()):
			axes[idx].plot(dm_values, metric_values, 
						  color=colors.get(metric_name, 'black'),
						  linewidth=2)
			axes[idx].set_xlabel(rf'DM (pc cm$^{{-3}}$)')
			axes[idx].set_ylabel('Metric Value')
			axes[idx].set_title(labels.get(metric_name, metric_name))
			axes[idx].grid(True, alpha=0.3)
			
			# Mark input DM
			axes[idx].axvline(self.input_dm, color='gray', 
							linestyle=':', alpha=1, linewidth=2, label=f'Input DM={self._format_dm(self.input_dm, 3)}')
			
			# Mark maximum
			max_idx = np.argmax(metric_values)
			axes[idx].axvline(dm_values[max_idx], color='red', 
							linestyle='--', alpha=1, label=f'Max at DM={self._format_dm(dm_values[max_idx], 3)}')
			axes[idx].legend()
		
		plt.tight_layout()
		
		if save_path:
			savefig_rasterized(save_path, dpi=150, bbox_inches='tight')
			print(f"DM scan plot saved to: {save_path}")
		else:
			plt.show()

	def plot_component_dm_diagnostics(self,
							 all_results: List[Dict],
							 component_ids: Optional[np.ndarray] = None,
							 label: str = "segment",
							 save_path: Optional[str] = None):
		"""
		Plot DM diagnostics across multiple components.

		Creates a single-panel diagnostic showing absolute best DM per
		component for each method, including asymmetric error bars.
		"""
		n_components = len(all_results)
		if n_components < 2:
			print("Component DM diagnostics skipped (need at least 2 components).")
			return

		if component_ids is None:
			component_ids = np.arange(1, n_components + 1, dtype=int)
		else:
			component_ids = np.asarray(component_ids, dtype=int)
			if component_ids.ndim != 1 or component_ids.shape[0] != n_components:
				raise ValueError("component_ids must be 1D with one value per component")

		preferred_order = ['structure', 'snr', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		first_methods = list(all_results[0].keys())
		common_methods = [m for m in first_methods if all(m in comp for comp in all_results)]
		if len(common_methods) == 0:
			print("Component DM diagnostics skipped (no common methods across components).")
			return

		ordered_methods = [m for m in preferred_order if m in common_methods]
		ordered_methods.extend([m for m in common_methods if m not in ordered_methods])

		method_display = {
			'structure': 'Structure',
			'snr': 'S/N',
			'pa_slope': 'PA slope',
			'pa_slope_shrine': 'PA slope (SHRINE)',
			'l_i_mean': 'L/I mean',
		}
		colors = {
			'structure': 'tab:blue',
			'snr': 'tab:red',
			'pa_slope': 'tab:green',
			'pa_slope_shrine': 'tab:cyan',
			'l_i_mean': 'tab:purple',
		}

		component_idx = np.arange(1, n_components + 1)
		dm_matrix = np.zeros((len(ordered_methods), n_components), dtype=float)
		dm_minus = np.zeros((len(ordered_methods), n_components), dtype=float)
		dm_plus = np.zeros((len(ordered_methods), n_components), dtype=float)
		for i, method_name in enumerate(ordered_methods):
			for j, comp in enumerate(all_results):
				method_result = comp[method_name]
				dm_matrix[i, j] = float(method_result['dm'])
				minus = method_result.get('uncertainty_minus')
				plus = method_result.get('uncertainty_plus')
				dm_minus[i, j] = 0.0 if minus is None else max(float(minus), 0.0)
				dm_plus[i, j] = 0.0 if plus is None else max(float(plus), 0.0)

		# Draw larger-uncertainty methods underneath smaller-uncertainty methods.
		uncertainty_rank_metric = np.nanmean(dm_minus + dm_plus, axis=1)
		draw_order = list(np.argsort(-uncertainty_rank_metric))

		fig, ax = plt.subplots(1, 1)

		for draw_rank, i in enumerate(draw_order):
			method_name = ordered_methods[i]
			disp_name = method_display.get(method_name, method_name)
			color = colors.get(method_name, None)
			ax.errorbar(
				component_idx,
				dm_matrix[i],
				yerr=np.vstack((dm_minus[i], dm_plus[i])),
				fmt='o-',
				label=disp_name,
				color=color,
				zorder=2 + draw_rank,
                capsize=3,
			)

		ax.set_ylabel(rf'Best DM (pc cm$^{{-3}}$)')
		ax.grid(True, alpha=0.3)
		ax.legend()
		#handles, labels = ax.get_legend_handles_labels()
		#if handles:
		#	fig.legend(
		#		handles,
		#		labels,
		#		loc='upper center',
		#		bbox_to_anchor=(0.5, 1.02),
		#		ncol=2,
		#		fontsize=8,
		#		frameon=False,
		#	)

		ax.set_xticks(component_idx)
		component_names = []
		for cid in component_ids:
			cid_int = int(cid)
			if cid_int == 1:
				component_names.append('Main component')
			elif cid_int == 2:
				component_names.append('Precursor')
			else:
				component_names.append(f'Precursor {cid_int - 1}')
		ax.set_xticklabels(component_names)
		x_pad = 0.2
		ax.set_xlim(1 - x_pad, n_components + x_pad)

		plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
		if save_path:
			savefig_rasterized(save_path, dpi=600, bbox_inches='tight')
			print(f"Component DM diagnostics saved to: {save_path}")
		else:
			plt.show()

	def calculate_dn_e_between_components(
		self,
		all_results: List[Dict],
		component_separation_pc: Optional[float] = None,
		component_times_ms: Optional[np.ndarray] = None,
		comparison: str = 'adjacent',
		reference_component: int = 0,
	) -> Dict:
		"""
		Calculate delta-DM and dn_e between components.

		Parameters:
		-----------
		all_results : list of dict
			List of per-component result dictionaries returned by compare_methods.
		component_separation_pc : float, optional
			Physical line-of-sight separation between components in parsec.
			Used for all pairs if component_times_ms is not provided.
		component_times_ms : np.ndarray, optional
			Component arrival times in ms. When provided, pair-specific separation is
			computed as L ~ c*Delta t and converted to parsec.
		comparison : str, optional
			How to compare components:
			- 'adjacent': component i -> i+1
			- 'reference': reference_component -> every other component
		reference_component : int, optional
			Reference component index used when comparison='reference'.

		Returns:
		--------
		diagnostics : dict
			Dictionary containing pair labels, delta-DM values, and dn_e values per
			method, including conservative asymmetric bounds from DM uncertainties.
		"""
		n_components = len(all_results)
		if n_components < 2:
			raise ValueError("Need at least two components to calculate dn_e")

		if component_times_ms is not None:
			component_times_ms = np.asarray(component_times_ms, dtype=float)
			if component_times_ms.ndim != 1 or component_times_ms.shape[0] != n_components:
				raise ValueError("component_times_ms must be 1D with one value per component")
			if not np.all(np.isfinite(component_times_ms)):
				raise ValueError("component_times_ms must be finite")
		elif component_separation_pc is None:
			raise ValueError("Provide either component_times_ms or component_separation_pc")
		elif component_separation_pc <= 0:
			raise ValueError("component_separation_pc must be positive")

		if comparison not in ('adjacent', 'reference'):
			raise ValueError("comparison must be 'adjacent' or 'reference'")

		first_methods = list(all_results[0].keys())
		common_methods = [m for m in first_methods if all(m in comp for comp in all_results)]
		if len(common_methods) == 0:
			raise ValueError("No common methods across components")

		if comparison == 'adjacent':
			pair_indices = [(i, i + 1) for i in range(n_components - 1)]
		else:
			if reference_component < 0 or reference_component >= n_components:
				raise ValueError(
					f"reference_component must be in [0, {n_components - 1}]"
				)
			pair_indices = [
				(reference_component, j)
				for j in range(n_components)
				if j != reference_component
			]

		pair_labels = [f"comp{a + 1}->comp{b + 1}" for a, b in pair_indices]
		c_pc_per_ms = 9.715611890180196e-12
		pair_separations_pc = np.zeros(len(pair_indices), dtype=float)
		for i, (idx_a, idx_b) in enumerate(pair_indices):
			if component_times_ms is not None:
				delta_t_ms = abs(float(component_times_ms[idx_b]) - float(component_times_ms[idx_a]))
				sep_pc = c_pc_per_ms * delta_t_ms
				if sep_pc <= 0:
					raise ValueError("Component times imply zero separation for at least one pair")
				pair_separations_pc[i] = sep_pc
			else:
				pair_separations_pc[i] = float(component_separation_pc)

		method_diagnostics: Dict[str, Dict[str, np.ndarray]] = {}

		for method_name in common_methods:
			delta_dm = np.zeros(len(pair_indices), dtype=float)
			delta_dm_low = np.zeros(len(pair_indices), dtype=float)
			delta_dm_high = np.zeros(len(pair_indices), dtype=float)
			dn_e = np.zeros(len(pair_indices), dtype=float)
			dn_e_low = np.zeros(len(pair_indices), dtype=float)
			dn_e_high = np.zeros(len(pair_indices), dtype=float)

			for i, (idx_a, idx_b) in enumerate(pair_indices):
				sep_pc = float(pair_separations_pc[i])
				res_a = all_results[idx_a][method_name]
				res_b = all_results[idx_b][method_name]

				dm_a = float(res_a['dm'])
				dm_b = float(res_b['dm'])
				minus_a = 0.0 if res_a.get('uncertainty_minus') is None else max(float(res_a.get('uncertainty_minus')), 0.0)
				plus_a = 0.0 if res_a.get('uncertainty_plus') is None else max(float(res_a.get('uncertainty_plus')), 0.0)
				minus_b = 0.0 if res_b.get('uncertainty_minus') is None else max(float(res_b.get('uncertainty_minus')), 0.0)
				plus_b = 0.0 if res_b.get('uncertainty_plus') is None else max(float(res_b.get('uncertainty_plus')), 0.0)

				# Conservative asymmetric interval propagation for delta DM = DM_b - DM_a.
				dm_a_low = dm_a - minus_a
				dm_a_high = dm_a + plus_a
				dm_b_low = dm_b - minus_b
				dm_b_high = dm_b + plus_b

				delta = dm_b - dm_a
				delta_low = dm_b_low - dm_a_high
				delta_high = dm_b_high - dm_a_low

				delta_dm[i] = delta
				delta_dm_low[i] = delta_low
				delta_dm_high[i] = delta_high

				dn_e[i] = delta / sep_pc
				dn_e_low[i] = delta_low / sep_pc
				dn_e_high[i] = delta_high / sep_pc

			method_diagnostics[method_name] = {
				'delta_dm': delta_dm,
				'delta_dm_low': delta_dm_low,
				'delta_dm_high': delta_dm_high,
				'dn_e': dn_e,
				'dn_e_low': dn_e_low,
				'dn_e_high': dn_e_high,
			}

		return {
			'comparison': comparison,
			'component_separation_pc': None if component_separation_pc is None else float(component_separation_pc),
			'component_times_ms': None if component_times_ms is None else component_times_ms.copy(),
			'pair_indices': pair_indices,
			'pair_labels': pair_labels,
			'pair_separations_pc': pair_separations_pc,
			'methods': method_diagnostics,
		}

	def plot_component_dne_diagnostics(
		self,
		dne_diag: Dict,
		label: str = "segment",
		save_path: Optional[str] = None,
	):
		"""
		Plot dn_e diagnostics between component pairs for all methods.

		Parameters:
		-----------
		dne_diag : dict
			Output dictionary from calculate_dn_e_between_components.
		label : str, optional
			Label for plot title context.
		save_path : str, optional
			Path to save figure. If None, displays instead.
		"""
		pair_labels = dne_diag.get('pair_labels', [])
		methods = dne_diag.get('methods', {})
		if len(pair_labels) == 0 or len(methods) == 0:
			print("dn_e diagnostics plot skipped (no dn_e data).")
			return

		preferred_order = ['structure', 'snr', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		method_names = [m for m in preferred_order if m in methods]
		method_names.extend([m for m in methods.keys() if m not in method_names])

		method_display = {
			'structure': 'Structure',
			'snr': 'S/N',
			'pa_slope': 'PA slope',
			'pa_slope_shrine': 'PA slope (SHRINE)',
			'l_i_mean': 'L/I mean',
		}
		colors = {
			'structure': 'tab:blue',
			'snr': 'tab:red',
			'pa_slope': 'tab:green',
			'pa_slope_shrine': 'tab:cyan',
			'l_i_mean': 'tab:purple',
		}

		x = np.arange(len(pair_labels), dtype=float)
		fig, ax = plt.subplots(1, 1)

		# Small x-offset per method so uncertainty bars are readable.
		n_methods = max(1, len(method_names))
		offset_span = 0.18
		if n_methods == 1:
			offsets = np.array([0.0])
		else:
			offsets = np.linspace(-offset_span, offset_span, n_methods)

		all_abs = []
		for i, method_name in enumerate(method_names):
			vals = methods[method_name]
			y = np.asarray(vals.get('dn_e', np.zeros_like(x)), dtype=float)
			y_low = np.asarray(vals.get('dn_e_low', y), dtype=float)
			y_high = np.asarray(vals.get('dn_e_high', y), dtype=float)

			err_minus = np.abs(y - y_low)
			err_plus = np.abs(y_high - y)
			yerr = np.vstack((err_minus, err_plus))
			x_plot = x + offsets[i]

			all_abs.extend(np.abs(y[np.isfinite(y)]).tolist())
			all_abs.extend(np.abs(y_low[np.isfinite(y_low)]).tolist())
			all_abs.extend(np.abs(y_high[np.isfinite(y_high)]).tolist())

			ax.errorbar(
				x_plot,
				y,
				yerr=yerr,
				fmt='o',
				label=method_display.get(method_name, method_name),
				color=colors.get(method_name, None),
                capsize=3,
			)

		ax.axhline(0.0, color='0.35', linestyle='--', alpha=0.8)
		ax.set_xticks(x)
		ax.set_xticklabels([])
		#ax.set_xlabel('Component pair')
		ax.set_ylabel(r'$\Delta n_e (\text{cm}^{-3})$')
		#ax.set_title(f'dn_e between components ({label})')
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		all_abs = np.asarray(all_abs, dtype=float)
		finite_abs = all_abs[np.isfinite(all_abs) & (all_abs > 0)]
		if finite_abs.size > 1:
			dynamic = float(np.max(finite_abs) / np.min(finite_abs))
			if dynamic > 100.0:
				ax.set_yscale('symlog', linthresh=max(1.0, float(np.min(finite_abs))))

		plt.tight_layout()
		if save_path:
			savefig_rasterized(save_path, dpi=600, bbox_inches='tight')
			print(f"Component dn_e diagnostics saved to: {save_path}")
		else:
			plt.show()


def main():
	"""
	Example usage of the DM optimisation comparison.
	"""
	print("="*70)
	print("DM Correction Optimisation Methods Comparison")
	print("="*70)

	args = parse_args()
	
	# Load data
	print("\nLoading data...")
	print("Using files:")
	print(f"  - Stokes I: {args.stokes_i}")
	if args.stokes_q:
		print(f"  - Stokes Q: {args.stokes_q}")
	if args.stokes_u:
		print(f"  - Stokes U: {args.stokes_u}")
	print(f"  - Frequency: {args.freq}")
	print(f"  - Time: {args.time}")
	if args.dm_step is not None:
		print(f"  - DM step for scan: {args.dm_step} pc cm⁻³")
	if args.ref_freq is not None:
		print(f"  - Reference frequency override: {args.ref_freq} MHz")
	if args.input_dm:
		print(f"  - Input data already dedispersed at DM: {args.input_dm} pc cm⁻³")
	print(f"  - PA weight strength: {args.pa_weight_strength}")
	print(f"  - PA fit post-peak only: {args.pa_fit_post_peak_only}")
	print(f"  - Linear debiasing: {args.debias_linear}")
	print(f"  - Non-SHRINE kc smoothing: {args.nonshrine_kc_smooth}")
	print(f"  - Non-SHRINE SHRINE-like errors: {args.nonshrine_shrine_like_errors}")
	if args.nonshrine_kc is not None:
		print(f"  - Non-SHRINE kc value: {args.nonshrine_kc}")
	print(f"  - Non-SHRINE kc via minimise_uncertainty: {args.nonshrine_kc_minimise_uncertainty}")
	print(f"  - L/I sigma cutoff: {args.li_sig}")
	if args.methods is not None:
		print(f"  - Included methods (CLI): {', '.join(args.methods)}")
	if args.exclude_methods is not None:
		print(f"  - Excluded methods (CLI): {', '.join(args.exclude_methods)}")
	if args.seed is not None:
		print(f"  - Random seed: {args.seed}")
	
	# Support a combined Stokes cube input if provided
	if args.stokes_cube:
		cube = np.load(args.stokes_cube)
		# Accept a few common layouts: (4, freq, time), (3, freq, time), (freq, time, 4), (freq, time, 3)
		if cube.ndim != 3:
			raise ValueError(f"stokes-cube must be a 3D numpy array, got shape {cube.shape}")
		# (4, freq, time) or (3, freq, time)
		if cube.shape[0] in (3, 4):
			stokes_i = cube[0]
			stokes_q = cube[1]
			stokes_u = cube[2]
			# ignore V if present
		# (freq, time, 4) or (freq, time, 3)
		elif cube.shape[2] in (3, 4):
			stokes_i = cube[..., 0]
			stokes_q = cube[..., 1]
			stokes_u = cube[..., 2]
		else:
			raise ValueError(f"Unrecognized stokes-cube layout: {cube.shape}. Expected first or last axis length 3 or 4.")
	else:
		stokes_i = np.load(args.stokes_i)
		stokes_q = np.load(args.stokes_q) if args.stokes_q else None
		stokes_u = np.load(args.stokes_u) if args.stokes_u else None
	freq_mhz = np.load(args.freq)
	time_ms = np.load(args.time)

	if freq_mhz.ndim != 1:
		raise ValueError("Frequency array must be 1D")
	if not np.all(np.diff(freq_mhz) >= 0):
		order = np.argsort(freq_mhz)
		freq_mhz = freq_mhz[order]
		stokes_i = stokes_i[order]
		if stokes_q is not None:
			stokes_q = stokes_q[order]
		if stokes_u is not None:
			stokes_u = stokes_u[order]

	if (stokes_q is None) != (stokes_u is None):
		print("\nWarning: both --stokes-q and --stokes-u are required to run Q/U-based metrics. Skipping them.")
		stokes_q = None
		stokes_u = None

	method_alias_to_key = {
		'structure': 'structure',
		'snr': 'snr',
		'pa': 'pa_slope',
		'pa-shrine': 'pa_slope_shrine',
		'li': 'l_i_mean',
	}
	default_method_order = ['structure', 'snr', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
	if args.methods is None:
		selected_method_keys = default_method_order.copy()
	else:
		selected_method_keys = []
		for alias in args.methods:
			key = method_alias_to_key[alias]
			if key not in selected_method_keys:
				selected_method_keys.append(key)

	if args.exclude_methods is not None:
		exclude_keys = {method_alias_to_key[alias] for alias in args.exclude_methods}
		selected_method_keys = [m for m in selected_method_keys if m not in exclude_keys]

	if (stokes_q is None or stokes_u is None):
		qu_methods = {'pa_slope', 'pa_slope_shrine', 'l_i_mean'}
		removed_qu = [m for m in selected_method_keys if m in qu_methods]
		if len(removed_qu) > 0:
			selected_method_keys = [m for m in selected_method_keys if m not in qu_methods]
			print("\nWarning: removed Q/U-based methods from selection because Stokes Q/U are unavailable.")

	if len(selected_method_keys) == 0:
		raise ValueError("No methods selected to run after applying include/exclude filters")

	method_key_to_name = {
		'structure': 'structure',
		'snr': 'snr',
		'pa_slope': 'pa',
		'pa_slope_shrine': 'pa-shrine',
		'l_i_mean': 'li',
	}
	selected_method_labels = [method_key_to_name[m] for m in selected_method_keys]
	print(f"  Active methods: {', '.join(selected_method_labels)}")
	
	print(f"\nData loaded successfully!")
	print(f"  Shape: {stokes_i.shape} (freq x time)")
	print(f"  Frequency range: {freq_mhz[0]:.1f} - {freq_mhz[-1]:.1f} MHz")
	print(f"  Time range: {time_ms[0]:.3f} - {time_ms[-1]:.3f} ms")
		
	
	# Initialize optimiser
	optimiser = DMOptimiser(
		stokes_i,
		freq_mhz,
		time_ms,
		stokes_q=stokes_q,
		stokes_u=stokes_u,
		reference_freq=args.ref_freq,
		input_dm=args.input_dm,
		dedisp_mode=args.dedisp_mode,
		pa_fit_degree=args.pa_fit_degree,
		pa_weight_strength=args.pa_weight_strength,
		pa_fit_post_peak_only=args.pa_fit_post_peak_only,
		nonshrine_kc_smooth=args.nonshrine_kc_smooth,
		nonshrine_shrine_like_errors=args.nonshrine_shrine_like_errors,
		nonshrine_kc_minimise_uncertainty=args.nonshrine_kc_minimise_uncertainty,
		nonshrine_kc=args.nonshrine_kc,
		li_i_sigma_cut=args.li_sig,
		debias_linear=args.debias_linear,
		random_seed=args.seed,
	)

	# Inform final reference frequency used
	print(f"  Using reference frequency: {optimiser.reference_freq:.3f} MHz")
	recommended_dm_step = optimiser.recommend_lowest_dm_step()
	print(
		"  Recommended lowest DM step from dspec resolution "
		f"(~1 time sample across band): {recommended_dm_step:.6g} pc cm⁻³"
	)
	if args.dm_step is None:
		print(f"  Using recommended DM step: {recommended_dm_step:.6g} pc cm⁻³")
		args.dm_step = recommended_dm_step

	# Resolve DM search range from flags
	dm_range: Tuple[float, float]
	if (args.dm_min is None) != (args.dm_max is None):
		print("\nWarning: both --dm-min and --dm-max are required; ignoring partial input.")
	if args.dm_min is not None and args.dm_max is not None:
		dm_range = (args.dm_min, args.dm_max)
		print(f"\nDM search range set from flags: {dm_range[0]:.1f} - {dm_range[1]:.1f} pc cm⁻³")
	elif args.dm_guess is not None:
		span = 50.0
		dm_range = (args.dm_guess - span, args.dm_guess + span)
		print(f"\nDM search range built around guess {args.dm_guess:.1f} ± {span:.1f} pc cm⁻³")
	else:
		dm_range = (300, 400)
		print(f"\nDM search range defaulting to {dm_range[0]} - {dm_range[1]} pc cm⁻³")
	
	# Peak handling
	print("\n" + "="*70)
	if args.peak_indices is not None:
		print("Using manually specified peak indices...")
		peak_regions = parse_peak_index_pairs(args.peak_indices, stokes_i.shape[1])
		print(f"  Specified {len(peak_regions)} peak region(s)")
		for i, (start, end) in enumerate(peak_regions):
			end_disp = min(end - 1, len(time_ms) - 1)
			print(f"    Peak {i+1}: time indices {start}-{end} "
				  f"({time_ms[start]:.2f} - {time_ms[end_disp]:.2f} ms)")
	elif args.manual_peaks:
		print("Manual peak selection enabled...")
		peak_regions = optimiser.select_peaks_manual()
		print(f"  Selected {len(peak_regions)} peak region(s)")
		for i, (start, end) in enumerate(peak_regions):
			end_disp = min(end - 1, len(time_ms) - 1)
			print(f"    Peak {i+1}: time indices {start}-{end} "
				  f"({time_ms[start]:.2f} - {time_ms[end_disp]:.2f} ms)")
	elif args.separate_peaks:
		print("Separating peaks...")
		peak_regions = optimiser.separate_peaks(min_separation_ms=5.0, diagnostics_path=None)
		print(f"  Found {len(peak_regions)} peak(s)")
		for i, (start, end) in enumerate(peak_regions):
			print(f"    Peak {i+1}: time indices {start}-{end} "
				  f"({time_ms[start]:.2f} - {time_ms[end]:.2f} ms)")
	else:
		print("Skipping peak separation (processing full dataset).")
		peak_regions = [(0, stokes_i.shape[1])]
	
	label = "Peak" if args.separate_peaks else "Segment"
	fig_ext = args.ext.strip().lower().lstrip('.') or 'png'

	set_pub_style(use_latex=False)

	# Analyze each segment (peak or full dataset)
	all_results = []
	grid_n_points = 100
	if args.fast and args.dm_step is None:
		grid_n_points = 50
	for i, peak_region in enumerate(peak_regions):
		print("\n" + "="*70)
		print(f"Analyzing {label} {i+1}")
		print("="*70)
		
		# Compare methods
		results = optimiser.compare_methods(dm_range, peak_region, n_points=grid_n_points, dm_step=args.dm_step, 
											segment_tag=f"{label.lower()}{i+1}",
											label=args.label,
											selected_methods=selected_method_keys)
		all_results.append(results)
		
		# Print results
		print(f"\nResults for {label} {i+1}:")
		for method_name, result in results.items():
			print(f"  {result['method']}:")
			print(
				"    Optimal DM: "
				+ optimiser._format_uncertainty(
					result['dm'],
					result.get('uncertainty_minus'),
					result.get('uncertainty_plus'),
				)
				+ " pc cm⁻³"
			)
			print(f"    Metric value: {result['metric']:.6f}")
			if result.get('uncertainty_method') is not None:
				print(f"    Uncertainty method: {result.get('uncertainty_method')}")
		
		# Plot comparison
		print(f"\nGenerating comparison plot for {label} {i+1}...")
		optimiser.plot_comparison(results, dm_range, peak_region, 
								 save_path=f'dm_comparison_{label.lower()}{i+1}.{fig_ext}')

	if len(all_results) > 1:
		# Compute component peak times for physical time ordering and L~c*Delta t.
		component_peak_times_ms = np.zeros(len(peak_regions), dtype=float)
		for i, (start_idx, end_idx) in enumerate(peak_regions):
			start_clamped = min(max(start_idx, 0), stokes_i.shape[1] - 1)
			end_exclusive = min(max(end_idx, start_clamped + 1), stokes_i.shape[1])
			window_i = stokes_i[:, start_clamped:end_exclusive]
			window_profile = np.mean(window_i, axis=0)
			peak_local = int(np.argmax(window_profile))
			peak_global = start_clamped + peak_local
			component_peak_times_ms[i] = float(time_ms[peak_global])

		# Enforce physical ordering for dn_e: earliest component peak -> latest.
		sort_idx = np.argsort(component_peak_times_ms)
		sorted_all_results = [all_results[int(i)] for i in sort_idx]
		sorted_peak_times_ms = component_peak_times_ms[sort_idx]
		sorted_component_ids = (sort_idx + 1).astype(int)

		print(f"\nGenerating multi-{label.lower()} DM diagnostics plot (time-ordered)...")
		optimiser.plot_component_dm_diagnostics(
			sorted_all_results,
			component_ids=sorted_component_ids,
			label=label.lower(),
			save_path=f'dm_component_dm_diagnostics.{fig_ext}',
		)

		dne_diag = optimiser.calculate_dn_e_between_components(
			sorted_all_results,
			component_times_ms=sorted_peak_times_ms,
			comparison='adjacent',
		)

		# Re-label pairs with original segment/component IDs after time-order sorting.
		pair_labels_time = []
		for idx_a, idx_b in dne_diag['pair_indices']:
			pair_labels_time.append(
				f"comp{int(sorted_component_ids[idx_a])}->comp{int(sorted_component_ids[idx_b])}"
			)
		dne_diag['pair_labels'] = pair_labels_time
		print("\nComponent-to-component dn_e diagnostics (adjacent pairs, L~cΔt):")
		for i, pair_label in enumerate(dne_diag['pair_labels']):
			sep_pc = float(dne_diag['pair_separations_pc'][i])
			print(f"  {pair_label}: L = {sep_pc:.6e} pc")
			for method_name, method_vals in dne_diag['methods'].items():
				delta_dm = float(method_vals['delta_dm'][i])
				delta_dm_low = float(method_vals['delta_dm_low'][i])
				delta_dm_high = float(method_vals['delta_dm_high'][i])
				dn_e = float(method_vals['dn_e'][i])
				dn_e_low = float(method_vals['dn_e_low'][i])
				dn_e_high = float(method_vals['dn_e_high'][i])
				print(
					f"    {method_name}: "
					f"ΔDM={delta_dm:.6f} [{delta_dm_low:.6f}, {delta_dm_high:.6f}] pc cm⁻³, "
					f"dn_e={dn_e:.6e} [{dn_e_low:.6e}, {dn_e_high:.6e}] cm⁻³"
				)

		#dne_path = Path(f'dm_component_dne_diagnostics_{label.lower()}.txt')
		#with open(dne_path, 'w') as f:
		#	f.write("# dn_e diagnostics between components\n")
		#	f.write("# Assumption: L ~ c * Delta t using component peak arrival times\n")
		#	f.write("# Columns: pair method separation_pc delta_dm delta_dm_low delta_dm_high dn_e dn_e_low dn_e_high\n")
		#	for i, pair_label in enumerate(dne_diag['pair_labels']):
		#		sep_pc = float(dne_diag['pair_separations_pc'][i])
		#		for method_name, method_vals in dne_diag['methods'].items():
		#			f.write(
		#				f"{pair_label} {method_name} {sep_pc:.10e} "
		#				f"{float(method_vals['delta_dm'][i]):.10e} "
		#				f"{float(method_vals['delta_dm_low'][i]):.10e} "
		#				f"{float(method_vals['delta_dm_high'][i]):.10e} "
		#				f"{float(method_vals['dn_e'][i]):.10e} "
		#				f"{float(method_vals['dn_e_low'][i]):.10e} "
		#				f"{float(method_vals['dn_e_high'][i]):.10e}\n"
		#			)
		#print(f"Saved dn_e diagnostics to: {dne_path}")

		dne_plot_path = f'dm_component_dne_diagnostics_{label.lower()}.{fig_ext}'
		optimiser.plot_component_dne_diagnostics(
			dne_diag,
			label=label.lower(),
			save_path=dne_plot_path,
		)
	
	print("\n" + "="*70)
	print("Analysis complete!")
	print("="*70)
	print("\nGenerated files:")
	print(f"  - dm_comparison_{label.lower()}*.{fig_ext}: Comparison of methods for each {label.lower()}")
	if len(all_results) > 1:
		print(f"  - dm_component_dm_diagnostics.{fig_ext}: Multi-{label.lower()} DM diagnostics")
		#print(f"  - dm_component_dne_diagnostics_{label.lower()}.txt: dn_e diagnostics (L~cΔt)")
		print(f"  - dm_component_dne_diagnostics_{label.lower()}.{fig_ext}: dn_e plot between components")


if __name__ == "__main__":
	main()
