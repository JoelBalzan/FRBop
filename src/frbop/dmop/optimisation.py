"""DM optimisation drivers for each method."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

class OptimisationMixin:
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
		pa_uncertainty = self._uncertainty_from_metric_shrine(dm_values, pa_values)
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
		pa_shrine_uncertainty = self._uncertainty_from_metric_shrine(dm_values, pa_values)
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
		li_uncertainty = self._uncertainty_from_metric_shrine(dm_values, li_values)
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
			i_data[i] = np.nansum(dedispersed, axis=0)

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
			i_data[i] = np.nansum(dedispersed, axis=0)

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
	

