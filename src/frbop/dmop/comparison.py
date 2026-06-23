"""Multi-method DM comparison workflow."""

from typing import Dict, List, Optional, Tuple

import numpy as np


class ComparisonMixin:
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
		all_method_keys = ['structure', 'snr', 'min_uncertainty', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		if selected_methods is None:
			selected = set(all_method_keys)
		else:
			unknown = sorted(set(selected_methods) - set(all_method_keys))
			if len(unknown) > 0:
				raise ValueError(f"Unknown methods in selected_methods: {unknown}")
			selected = set(selected_methods)

		run_structure = 'structure' in selected
		run_snr = 'snr' in selected
		run_min_uncertainty = 'min_uncertainty' in selected
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

		if not (run_structure or run_snr or run_min_uncertainty or run_qu_methods):
			print("  - No methods selected after filtering; returning empty results.")
			return results

		# Shared DM sweep for all methods.
		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		output_size = self._max_output_size_for_dm_range(data, dm_range)
		i_data = np.zeros((len(dm_values), output_size), dtype=float)

		pa_values = np.zeros(len(dm_values), dtype=float) if run_pa else None
		pa_shrine_values = np.zeros(len(dm_values), dtype=float) if run_pa_shrine else None
		li_mean_values = np.zeros(len(dm_values), dtype=float) if run_li_mean else None

		self._reset_nonshrine_kc_state()
		dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = self.time_ms[0] + np.arange(output_size) * dt
		if run_qu_methods:
			self._maybe_prepare_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)

		print(f"  - Shared DM sweep for all methods ({len(dm_values)} trials)...")
		for i, dm in enumerate(dm_values):
			if i % 25 == 0:
				print(f"\r    Progress: {i}/{len(dm_values)}", end='', flush=True)

			dedisp_i = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nansum(dedisp_i, axis=0)

			if run_qu_methods:
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self.maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
				if pa_values is not None:
					pa_values[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i)
				if pa_shrine_values is not None:
					pa_shrine_values[i] = self._pa_slope_metric_shrine(sm_q, sm_u, time_axis, sm_i)
				if li_mean_values is not None:
					li_mean_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode='mean')
		print(f"\r    Progress: {len(dm_values)}/{len(dm_values)}", flush=True)

		if run_structure:
			print("  - Testing Structure Maximising (SHRINE)...")
			run_prefix_structure = f"{label}_{segment_tag}_structure"
			run_dir_structure = self.run_shrine_method(
				script_name="maximise_structure.py",
				run_prefix=run_prefix_structure,
				dm_values=dm_values,
				i_data=i_data,
				include_input_dm=True,
				save_all=True,
				force_kc=self.shrine_kc,
			)
			structure_values = np.loadtxt(run_dir_structure / f"{run_prefix_structure}_SPs.dat")
			summary_path = run_dir_structure / f"{run_prefix_structure}_structure_summaryfile.txt"
			kc = self.shrine_kc
			if kc is None and summary_path.exists():
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
			run_dir_snr = self.run_shrine_method(
				script_name="maximise_sn.py",
				run_prefix=run_prefix_snr,
				dm_values=dm_values,
				i_data=i_data,
				include_input_dm=False,
				save_all=True,
				force_kc=self.shrine_kc,
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

		if run_min_uncertainty:
			print("  - Testing Minimise Uncertainty (SHRINE)...")
			run_prefix_unc = f"{label}_{segment_tag}_min_uncertainty"
			run_dir_unc = self.run_shrine_method(
				script_name="minimise_uncertainty.py",
				run_prefix=run_prefix_unc,
				dm_values=dm_values,
				i_data=i_data,
				include_input_dm=False,
				save_all=True,
			)
			unc_path = run_dir_unc / f"{run_prefix_unc}_uncertainly_kc.dat"
			if not unc_path.exists():
				raise FileNotFoundError(f"Expected SHRINE uncertainty output not found: {unc_path}")
			unc_table = np.loadtxt(unc_path)
			if unc_table.ndim == 1:
				unc_table = unc_table[np.newaxis, :]
			if unc_table.shape[1] < 4:
				raise ValueError(
					"Unexpected minimise_uncertainty output format; expected 4 columns (kc, low, dm, high)"
				)
			kc_vals = np.asarray(unc_table[:, 0], dtype=float)
			low_dm = np.asarray(unc_table[:, 1], dtype=float)
			best_dm = np.asarray(unc_table[:, 2], dtype=float)
			high_dm = np.asarray(unc_table[:, 3], dtype=float)
			unc_ranges = high_dm - low_dm
			finite_mask = np.isfinite(unc_ranges)
			if not np.any(finite_mask):
				raise ValueError("No finite uncertainty ranges returned by minimise_uncertainty")
			min_unc = float(np.nanmin(unc_ranges))
			min_indices = np.where(np.isclose(unc_ranges, min_unc))[0]
			if min_indices.size == 0:
				min_indices = np.array([int(np.nanargmin(unc_ranges))])
			best_idx = int(min_indices[len(min_indices) // 2])
			optimal_dm_unc = float(best_dm[best_idx])
			unc_low_dm = float(low_dm[best_idx])
			unc_high_dm = float(high_dm[best_idx])
			unc_minus = max(0.0, optimal_dm_unc - unc_low_dm)
			unc_plus = max(0.0, unc_high_dm - optimal_dm_unc)
			dedispersed_unc = self.dedisperse(data, optimal_dm_unc, mode=self.dedisp_mode)
			results['min_uncertainty'] = {
				'dm': optimal_dm_unc,
				'metric': float(unc_ranges[best_idx]),
				'dedispersed': dedispersed_unc,
				'method': 'Minimise Uncertainty (SHRINE)',
				'kc': int(round(float(kc_vals[best_idx]))),
				'run_dir': str(run_dir_unc),
				'uncertainty_method': 'minimise_uncertainty',
				'uncertainty_low_dm': unc_low_dm,
				'uncertainty_high_dm': unc_high_dm,
				'uncertainty_minus': unc_minus,
				'uncertainty_plus': unc_plus,
			}
			if has_qu:
				n_time_out = results['min_uncertainty']['dedispersed'].shape[1]
				results['min_uncertainty']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_unc, output_size=n_time_out, mode=self.dedisp_mode)
				results['min_uncertainty']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_unc, output_size=n_time_out, mode=self.dedisp_mode)

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
			best_sm_i_pa, best_sm_q_pa, best_sm_u_pa = self.maybe_kc_smooth_nonshrine(best_dedisp_i_pa, best_dedisp_q_pa, best_dedisp_u_pa)
			best_pa_smooth, best_fit_line, best_time_axis = self._get_pa_smoothed_and_fit(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa, time_axis)
			best_pa_deg = self._pa_series_deg(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa)
			metric_pa = float(pa_values[max_idx_pa])
			pa_uncertainty = self._uncertainty_from_polarisation_L_dm(
				dm_values, pa_values, kc=self._nonshrine_resolved_kc,
			)
			run_prefix_pa = f"{label}_{segment_tag}_pa_slope"
			run_dir_pa = self.save_nonshrine_run_outputs(
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
			best_sm_i_pas, best_sm_q_pas, best_sm_u_pas = self.maybe_kc_smooth_nonshrine(best_dedisp_i_pas, best_dedisp_q_pas, best_dedisp_u_pas)
			best_pa_shrine_smooth, best_shrine_fit_line, best_shrine_time_axis = self._get_pa_shrine_smoothed_and_fit(
				best_sm_q_pas,
				best_sm_u_pas,
				best_sm_i_pas,
				time_axis,
				force_kc=self._nonshrine_resolved_kc,
			)
			best_pa_shrine_deg = self._pa_series_deg(best_sm_q_pas, best_sm_u_pas, best_sm_i_pas)
			metric_pas = float(pa_shrine_values[max_idx_pas])
			pa_shrine_uncertainty = self._uncertainty_from_polarisation_L_dm(
				dm_values, pa_shrine_values, kc=self._nonshrine_resolved_kc,
			)
			run_prefix_pas = f"{label}_{segment_tag}_pa_slope_shrine"
			run_dir_pas = self.save_nonshrine_run_outputs(
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
			li_mean_uncertainty = self._uncertainty_from_polarisation_L_dm(
				dm_values, li_mean_values, kc=self._nonshrine_resolved_kc,
			)
			li_mean_uncertainty = self._clamp_uncertainty_to_dm_bounds(
				optimal_dm_li_mean,
				li_mean_uncertainty,
				dm_values,
				fill_missing_with_bounds=False,
			)
			run_prefix_li_mean = f"{label}_{segment_tag}_l_i_mean"
			run_dir_li_mean = self.save_nonshrine_run_outputs(
				run_prefix=run_prefix_li_mean,
				method_label=r'$\Pi_L$ Maximising (mean)',
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
				'method': r'$\Pi_L$ Maximising (mean)',
				'run_dir': str(run_dir_li_mean),
				'dm_values': dm_values.copy(),
				'metric_values': li_mean_values.copy(),
				**li_mean_uncertainty,
			}

		return results
	

