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
					   selected_methods: Optional[List[str]] = None,
					   segment_index: int = 0) -> Dict:
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
			data_v = None if self.stokes_v is None else self.stokes_v[:, peak_region[0]:peak_region[1]]
		else:
			data = self.stokes_i
			data_q = self.stokes_q
			data_u = self.stokes_u
			data_v = self.stokes_v

		print(f"Comparing methods on DM range [{dm_range[0]:.2f}, {dm_range[1]:.2f}] pc cm^-3")

		has_qu = data_q is not None and data_u is not None
		_shrine_kc = self.shrine_kc[segment_index] if self.shrine_kc is not None else None
		_nonshrine_kc = self.nonshrine_kc[segment_index] if self.nonshrine_kc is not None else None
		results: Dict[str, Dict] = {}
		all_method_keys = ['structure', 'snr', 'pa_slope', 'pa_slope_shrine', 'l_i_mean', 'structure_L']
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
		run_structure_L = 'structure_L' in selected
		run_qu_methods = run_pa or run_pa_shrine or run_li_mean
		if (run_qu_methods or run_structure_L) and not has_qu:
			print("  - Skipping selected PA/LI/structure_L methods (no Stokes Q/U provided)")
			run_pa = False
			run_pa_shrine = False
			run_li_mean = False
			run_qu_methods = False
			run_structure_L = False

		if not (run_structure or run_snr or run_qu_methods or run_structure_L):
			print("  - No methods selected after filtering; returning empty results.")
			return results

		# Shared DM sweep for all methods.
		dm_values = self._build_dm_values(dm_range, n_points=n_points, dm_step=dm_step)
		output_size = self._max_output_size_for_dm_range(data, dm_range)

		pa_values = np.zeros(len(dm_values), dtype=float) if run_pa else None
		pa_errors = np.zeros(len(dm_values), dtype=float) if run_pa else None
		pa_shrine_values = np.zeros(len(dm_values), dtype=float) if run_pa_shrine else None
		pa_shrine_errors = np.zeros(len(dm_values), dtype=float) if run_pa_shrine else None
		li_mean_values = np.zeros(len(dm_values), dtype=float) if run_li_mean else None
		li_mean_errors = np.zeros(len(dm_values), dtype=float) if run_li_mean else None

		self._reset_nonshrine_kc_state()
		if _nonshrine_kc is not None:
			self._nonshrine_resolved_kc = _nonshrine_kc

		# If sync_kc and no explicit nonshrine_kc, resolve kc from I data now
		# so the sweep loop uses the synced kc for kc smoothing.
		if self.sync_kc and _nonshrine_kc is None:
			if _shrine_kc is not None:
				_nonshrine_kc = _shrine_kc
				self._nonshrine_resolved_kc = _shrine_kc
				if not self._nonshrine_kc_printed:
					print(f"\nNon-SHRINE kc: {self._nonshrine_resolved_kc} (synced from SHRINE kc)")
					self._nonshrine_kc_printed = True
			else:
				# SHRINE-II's automatic k_c, on the profiles its structure and S/N methods use.
				print(f"  - Finding the SHRINE-II kc for kc sync ({len(dm_values)} trials)...")
				_i_for_kc = self.shrine2_profiles(data, dm_values)[0]
				_nonshrine_kc = self._shrine2_kc(_i_for_kc)[0]
				self._nonshrine_resolved_kc = _nonshrine_kc
				print(f"\nNon-SHRINE kc: {self._nonshrine_resolved_kc} (synced from SHRINE-II kc)")
				self._nonshrine_kc_printed = True

		dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = self.time_ms[0] + np.arange(output_size) * dt
		if run_qu_methods:
			self._maybe_prepare_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)

		if run_qu_methods:
			print(f"  - Shared DM sweep for the PA/LI methods ({len(dm_values)} trials)...")
			for i, dm in enumerate(dm_values):
				if i % 25 == 0:
					print(f"\r    Progress: {i}/{len(dm_values)}", end='', flush=True)
				dedisp_i = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self.maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
				if pa_values is not None:
					pa_values[i], pa_errors[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i, return_error=True)
				if pa_shrine_values is not None:
					pa_shrine_values[i], pa_shrine_errors[i] = self._pa_slope_metric_shrine(sm_q, sm_u, time_axis, sm_i, return_error=True)
				if li_mean_values is not None:
					li_mean_values[i], li_mean_errors[i] = self.linear_to_stokes_i_metric(
						sm_q, sm_u, sm_i, mode='mean', use_fwhm_window=True, return_error=True,
					)
			print(f"\r    Progress: {len(dm_values)}/{len(dm_values)}", flush=True)

		shrine2_I = None
		shrine2_kc = _shrine_kc
		if run_structure:
			print("  - Testing Structure Maximising (SHRINE-II)...")
			run_prefix_structure = f"{label}_{segment_tag}_structure"
			structure_run = self.run_shrine2_structure(
				run_prefix=run_prefix_structure,
				dm_values=dm_values,
				data=data,
				force_kc=_shrine_kc,
				save_all=True,
			)
			res = structure_run["res"]
			shrine2_I = structure_run["I"]
			shrine2_kc = structure_run["kc"]
			optimal_dm_structure = structure_run["dm_best"]
			if res["ok"]:
				structure_uncertainty = self._uncertainty_from_sigma_dm(optimal_dm_structure, float(res["sig_DM"]))
			else:
				structure_uncertainty = self._uncertainty_from_sigma_dm(optimal_dm_structure, None, why=res["why"])
			dedispersed_structure = self.dedisperse(data, optimal_dm_structure, mode=self.dedisp_mode)
			results['structure'] = {
				'dm': optimal_dm_structure,
				'metric': float(res["Sp_max"]),
				'dedispersed': dedispersed_structure,
				'method': 'Structure Maximising (SHRINE-II)',
				'kc': structure_run["kc"],
				'run_dir': str(structure_run["run_dir"]),
				'dm_values': dm_values.copy(),
				'metric_values': np.asarray(res["Sp"], dtype=float).copy(),
				'sigma_dm': float(res["sig_DM"]) if res["ok"] else None,
				'sigma_S': float(res["sig_S"]),
				'c2_err': float(res["c2_err"]) if res["ok"] else None,
				'e_ratio': float(res["e_ratio"]),
				'shrine2_warnings': list(structure_run["warnings"]),
				**structure_uncertainty,
			}
			if has_qu:
				n_time_out = results['structure']['dedispersed'].shape[1]
				results['structure']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_structure, output_size=n_time_out, mode=self.dedisp_mode)
				results['structure']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_structure, output_size=n_time_out, mode=self.dedisp_mode)
			if data_v is not None:
				n_time_out = results['structure']['dedispersed'].shape[1]
				results['structure']['dedispersed_v'] = self.dedisperse(data_v, optimal_dm_structure, output_size=n_time_out, mode=self.dedisp_mode)

		if run_snr:
			print("  - Testing S/N Maximising (SHRINE-II)...")
			run_prefix_snr = f"{label}_{segment_tag}_snr"
			# The profiles and k_c of the structure method, if it ran; otherwise made here.
			if shrine2_I is None:
				shrine2_I = self.shrine2_profiles(data, dm_values)[0]
			if shrine2_kc is None:
				shrine2_kc = self._shrine2_kc(shrine2_I)[0]
			sn = self.run_shrine2_sn(
				run_prefix=run_prefix_snr,
				dm_values=dm_values,
				I=shrine2_I,
				kc=shrine2_kc,
				save_all=True,
			)
			optimal_dm_snr = sn["dm_best"]
			snr_values = np.asarray(sn["SN"], dtype=float)
			dedispersed_snr = self.dedisperse(data, optimal_dm_snr, mode=self.dedisp_mode)
			snr_uncertainty = self._uncertainty_dict(optimal_dm_snr, sn["dm_low"], sn["dm_high"],
													 "S/N drop = 1 (SHRINE-II)")
			results['snr'] = {
				'dm': optimal_dm_snr,
				'metric': float(sn["SN_max"]),
				'dedispersed': dedispersed_snr,
				'method': 'S/N Maximising (SHRINE-II)',
				'kc': int(sn["kc"]),
				'run_dir': str(sn["run_dir"]),
				'dm_values': dm_values.copy(),
				'metric_values': snr_values.copy(),
				**snr_uncertainty,
			}
			if has_qu:
				n_time_out = results['snr']['dedispersed'].shape[1]
				results['snr']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_snr, output_size=n_time_out, mode=self.dedisp_mode)
				results['snr']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_snr, output_size=n_time_out, mode=self.dedisp_mode)
			if data_v is not None:
				n_time_out = results['snr']['dedispersed'].shape[1]
				results['snr']['dedispersed_v'] = self.dedisperse(data_v, optimal_dm_snr, output_size=n_time_out, mode=self.dedisp_mode)

		if not (run_qu_methods or run_structure_L):
			return results

		if run_qu_methods:
			print("  - Testing selected PA slope/L/I methods from shared sweep...")

		# PA slope (raw masked PA)
		if run_pa and pa_values is not None:
			max_idx_pa = int(np.argmax(np.abs(pa_values)))
			optimal_dm_pa = float(dm_values[max_idx_pa])
			best_dedisp_i_pa = self.dedisperse(data, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_q_pa = self.dedisperse(data_q, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_u_pa = self.dedisperse(data_u, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_v_pa = self.dedisperse(data_v, optimal_dm_pa, output_size=output_size, mode=self.dedisp_mode) if data_v is not None else None
			best_sm_i_pa, best_sm_q_pa, best_sm_u_pa = self.maybe_kc_smooth_nonshrine(best_dedisp_i_pa, best_dedisp_q_pa, best_dedisp_u_pa)
			best_pa_smooth, best_fit_line, best_time_axis = self._get_pa_smoothed_and_fit(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa, time_axis)
			best_pa_deg = self._pa_series_deg(best_sm_q_pa, best_sm_u_pa, best_sm_i_pa)
			metric_pa = float(pa_values[max_idx_pa])
			pa_uncertainty = self._uncertainty_from_metric_errors(
				dm_values, pa_values, pa_errors, max_idx_pa,
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
				'dedispersed_v': best_dedisp_v_pa,
				'method': 'PA Slope Maximising',
				'kc': self._nonshrine_resolved_kc,
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
			max_idx_pas = int(np.argmax(np.abs(pa_shrine_values)))
			optimal_dm_pas = float(dm_values[max_idx_pas])
			best_dedisp_i_pas = self.dedisperse(data, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_q_pas = self.dedisperse(data_q, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_u_pas = self.dedisperse(data_u, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode)
			best_dedisp_v_pas = self.dedisperse(data_v, optimal_dm_pas, output_size=output_size, mode=self.dedisp_mode) if data_v is not None else None
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
			pa_shrine_uncertainty = self._uncertainty_from_metric_errors(
				dm_values, pa_shrine_values, pa_shrine_errors, max_idx_pas,
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
				'dedispersed_v': best_dedisp_v_pas,
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
			li_mean_uncertainty = self._uncertainty_from_metric_errors(
				dm_values, li_mean_values, li_mean_errors, max_idx_li_mean,
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
			dedisp_v_li = self.dedisperse(data_v, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode) if data_v is not None else None
			results['l_i_mean'] = {
				'dm': optimal_dm_li_mean,
				'metric': metric_li_mean,
				'dedispersed': dedispersed_li_mean,
				'dedispersed_q': self.dedisperse(data_q, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode),
				'dedispersed_u': self.dedisperse(data_u, optimal_dm_li_mean, output_size=output_size, mode=self.dedisp_mode),
				'dedispersed_v': dedisp_v_li,
				'method': r'$\Pi_L$ Maximising (mean)',
				'kc': self._nonshrine_resolved_kc,
				'run_dir': str(run_dir_li_mean),
				'dm_values': dm_values.copy(),
				'metric_values': li_mean_values.copy(),
				**li_mean_uncertainty,
			}

		if run_structure_L:
			print("  - Testing Structure Maximising on L polarisation...")
			l_dm_data = self.build_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)
			run_prefix_structure_L = f"{label}_{segment_tag}_structure_L"
			kc_L_force = _nonshrine_kc if _nonshrine_kc is not None else self._nonshrine_resolved_kc
			scan_L = self.structure_parameter_scan(
				run_prefix=run_prefix_structure_L,
				dm_values=dm_values,
				profiles=l_dm_data,
				kc=kc_L_force,
				save_all=True,
			)
			structure_L_values = scan_L["Sp"]
			kc_L = scan_L["kc"]
			run_dir_structure_L = scan_L["run_dir"]
			max_idx_structure_L = scan_L["j_max"]
			optimal_dm_structure_L = float(dm_values[max_idx_structure_L])
			dedispersed_L_i = self.dedisperse(data, optimal_dm_structure_L, mode=self.dedisp_mode)
			metric_structure_L = float(structure_L_values[max_idx_structure_L])
			# No sigma_DM: the noise of L is not the zero-mean noise it assumes.
			structure_L_uncertainty = self._uncertainty_dict(
				optimal_dm_structure_L, None, None, "none (sigma_DM does not apply to L)",
			)
			results['structure_L'] = {
				'dm': optimal_dm_structure_L,
				'metric': metric_structure_L,
				'dedispersed': dedispersed_L_i,
				'method': 'Structure Maximising (L)',
				'kc': kc_L,
				'run_dir': str(run_dir_structure_L),
				'dm_values': dm_values.copy(),
				'metric_values': np.asarray(structure_L_values, dtype=float).copy(),
				**structure_L_uncertainty,
			}
			if has_qu:
				n_time_out = results['structure_L']['dedispersed'].shape[1]
				results['structure_L']['dedispersed_q'] = self.dedisperse(data_q, optimal_dm_structure_L, output_size=n_time_out, mode=self.dedisp_mode)
				results['structure_L']['dedispersed_u'] = self.dedisperse(data_u, optimal_dm_structure_L, output_size=n_time_out, mode=self.dedisp_mode)
			if data_v is not None:
				n_time_out = results['structure_L']['dedispersed'].shape[1]
				results['structure_L']['dedispersed_v'] = self.dedisperse(data_v, optimal_dm_structure_L, output_size=n_time_out, mode=self.dedisp_mode)

		return results