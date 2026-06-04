"""SHRINE integration and kc smoothing helpers."""

import contextlib
import io
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct

from frbop.utils.plotting import pub_figsize, savefig_rasterized

from .common import shrine_get_kc, shrine_lowpass_smooth

class ShrineMixin:
	def apply_kc_lowpass_2d(self, data_2d: np.ndarray, kc: int) -> np.ndarray:
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

	def build_nonshrine_L_dm_reference(self,
									   dm_values: np.ndarray,
									   data_q: np.ndarray,
									   data_u: np.ndarray,
									   output_size: int) -> np.ndarray:
		"""
		Build L(t, DM) from dedispersed linear-polarisation dynamic spectra.

		Rows are DM trials; columns are time samples (frequency-summed L).
		This matches the (delta_DM, time) layout expected by SHRINE get_kc.
		"""
		dm_values = np.asarray(dm_values, dtype=float)
		n_dm = dm_values.shape[0]
		L_dm = np.zeros((n_dm, output_size), dtype=float)
		for i, dm in enumerate(dm_values):
			dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
			L_dm[i] = self._linear_time_profile_from_qu(dedisp_q, dedisp_u)
		self._nonshrine_L_dm_reference = L_dm
		return L_dm

	def _maybe_prepare_nonshrine_L_dm_reference(self,
												dm_values: np.ndarray,
												data_q: Optional[np.ndarray],
												data_u: Optional[np.ndarray],
												output_size: int) -> None:
		"""
		Precompute L(t, DM) for polarisation kc selection and structure-style uncertainties.
		"""
		if data_q is None or data_u is None:
			return
		if self._nonshrine_L_dm_reference is not None:
			return
		self.build_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)

	def resolve_nonshrine_kc(self, reference_data_2d: np.ndarray) -> int:
		if self._nonshrine_L_dm_reference is not None:
			reference_data_2d = self._nonshrine_L_dm_reference
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
			run_dir = self.run_shrine_method(
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
		self._nonshrine_L_dm_reference = None

	def maybe_kc_smooth_nonshrine(self,
									   data_i: Optional[np.ndarray],
									   data_q: Optional[np.ndarray] = None,
									   data_u: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
		"""
		Optionally apply kc smoothing to non-SHRINE method inputs.
		"""
		if not self.nonshrine_kc_smooth:
			return data_i, data_q, data_u

		if self._nonshrine_L_dm_reference is not None:
			kc = self.resolve_nonshrine_kc(self._nonshrine_L_dm_reference)
		elif data_q is not None and data_u is not None:
			l_profile = self._linear_time_profile_from_qu(data_q, data_u)
			kc = self.resolve_nonshrine_kc(l_profile[np.newaxis, :])
		else:
			reference = data_i if data_i is not None else data_q
			if reference is None:
				return data_i, data_q, data_u
			kc = self.resolve_nonshrine_kc(reference)

		sm_i = self.apply_kc_lowpass_2d(data_i, kc) if data_i is not None else None
		sm_q = self.apply_kc_lowpass_2d(data_q, kc) if data_q is not None else None
		sm_u = self.apply_kc_lowpass_2d(data_u, kc) if data_u is not None else None
		return sm_i, sm_q, sm_u

	def run_shrine_method(self,
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

	def save_nonshrine_run_outputs(self,
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
		time_series = np.nansum(dedispersed_i, axis=0)
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


