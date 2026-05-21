"""Plotting and DM-space scanning utilities."""

from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from frbop.utils.plotting import pub_figsize, savefig_rasterized, pub_grid_figsize, colour_manager

class PlottingMixin:
	def plot_comparison(self, results: Dict, dm_range: Tuple[float, float],
					   peak_region: Optional[Tuple[int, int]] = None,
					   save_path: Optional[str] = None,
					   show_summary_errors: bool = True,
					   show_scan_uncertainty: bool = True,
					   show_overlay_uncertainty: bool = True,
					   disabled_error_methods: Optional[Set[str]] = None):
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
		colour_manager.reset()
		n_methods = len(results)
		has_qu = self.stokes_q is not None and self.stokes_u is not None

		figsize = pub_grid_figsize(
		    n_rows=n_methods + 1,
		    single_column=False,
		    row_height=2.7,
		    width_scale=1.8,
		)
		fig, axes = plt.subplots(
			n_methods + 1,
			5,
			figsize=figsize,
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
			cmap='plasma',
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
		time_series_orig = np.nansum(original_data, axis=0)
		axes[0, 1].plot(time_range, time_series_orig, 'k-', linewidth=1, label='I')
		ax0r = None
		if has_qu:
			ax0r = axes[0, 1].twinx()
			q_time = np.nansum(q_region, axis=0)
			u_time = np.nansum(u_region, axis=0)
			L_series = np.sqrt(q_time**2 + u_time**2)
			axes[0, 1].plot(time_range, L_series, 'r', linewidth=1, label='L')
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
		axes[0, 1].set_ylabel(rf'S (arb.)')
		axes[0, 1].set_xlabel('Time (ms)')
		axes[0, 1].grid(True, alpha=0.3)
		axes[0, 1].title.set_fontsize(fs_title)
		axes[0, 1].xaxis.label.set_size(fs_label)
		axes[0, 1].yaxis.label.set_size(fs_label)
		axes[0, 1].xaxis.labelpad = fs_labelpad
		axes[0, 1].yaxis.labelpad = fs_labelpad
		axes[0, 1].set_yticklabels([])
		axes[0, 1].tick_params(axis='both', labelsize=fs_tick)

		# Top-right panel: compact summary of best DM and uncertainty per method.
		all_scan_ax = axes[0, 2]
		method_names = list(results.keys())
		if len(method_names) > 0:
			y_pos = np.arange(len(method_names), dtype=float)
			y_labels = [scan_labels.get(name, name) for name in method_names]

			disabled = disabled_error_methods or set()
			for j, method_name in enumerate(method_names):
				result = results[method_name]
				dm_best = float(result['dm'])
				show_summary_for_method = show_summary_errors and (method_name not in disabled)
				if show_summary_for_method:
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
						color=colour_manager.color(method_name),
					)
				else:
					all_scan_ax.plot(
						dm_best,
						y_pos[j],
						'o',
						markersize=5,
						color=colour_manager.color(method_name),
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
			dt = float(np.nanmedian(np.diff(time_range))) if len(time_range) > 1 else 1.0
			delay_samples = self._get_delay_samples(result['dm'])
			if self.dedisp_mode == 'crop':
				start_shift = int(np.max(delay_samples))
			else:
				start_shift = int(np.min(delay_samples))
			time_range_dedisp = time_range[0] + start_shift * dt + np.arange(n_time_dedisp) * dt

			# In expand_zero mode, trim all-zero padding from the display only.
			display_slice = slice(None)
			if self.dedisp_mode == 'expand_zero':
				nonzero_cols = np.any(np.isfinite(result['dedispersed']) & (result['dedispersed'] != 0.0), axis=0)
				if np.any(nonzero_cols):
					first_valid = int(np.argmax(nonzero_cols))
					last_valid = int(len(nonzero_cols) - np.argmax(nonzero_cols[::-1]))
					display_slice = slice(first_valid, last_valid)
					time_range_dedisp = time_range_dedisp[display_slice]

			plot_dedispersed = result['dedispersed'][:, display_slice]
			vmin, vmax = self._robust_vmin_vmax(plot_dedispersed)
			im = axes[idx, 0].imshow(
				plot_dedispersed,
				aspect='auto',
				extent=[time_range_dedisp[0], time_range_dedisp[-1], self.freq_mhz[0], self.freq_mhz[-1]],
				cmap='plasma',
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
			show_overlay_for_method = show_overlay_uncertainty and (method_name not in disabled)
			if show_overlay_for_method:
				dm_text = self._format_uncertainty(
					result['dm'],
					result.get('uncertainty_minus'),
					result.get('uncertainty_plus'),
					precision=3,
				)
			else:
				dm_text = self._format_dm(result['dm'], 3)

			axes[idx, 0].text(
				0.98,
				0.98,
				rf"DM={dm_text} pc cm$^{{-3}}$",
				transform=axes[idx, 0].transAxes,
				ha='right',
				va='top',
				color='white',
				fontsize=fs_overlay,
				bbox=dict(facecolor='black', edgecolor='none', alpha=0.35, pad=2.0),
			)

			time_series = np.nansum(result['dedispersed'], axis=0)[display_slice]
			axes[idx, 1].plot(time_range_dedisp, time_series, 'k-', linewidth=1, label='I')
			axr = None

			if has_qu:
				axr = axes[idx, 1].twinx()
				dedisp_q = result.get('dedispersed_q')
				dedisp_u = result.get('dedispersed_u')
				if dedisp_q is None or dedisp_u is None:
					dedisp_q = self.dedisperse(q_region, result['dm'], output_size=n_time_dedisp, mode=self.dedisp_mode)
					dedisp_u = self.dedisperse(u_region, result['dm'], output_size=n_time_dedisp, mode=self.dedisp_mode)
				q_time = np.nansum(dedisp_q, axis=0)[display_slice]
				u_time = np.nansum(dedisp_u, axis=0)[display_slice]
				L_series = np.sqrt(q_time**2 + u_time**2)
				axes[idx, 1].plot(time_range_dedisp, L_series, 'r', linewidth=1, label='L')

				if (
					method_name in ('pa_slope', 'pa_slope_shrine')
					and result.get('pa_plot_series') is not None
					and result.get('pa_plot_smooth') is not None
					and result.get('pa_plot_fit') is not None
				):
					pa_deg = np.asarray(result['pa_plot_series'])[display_slice]
					pa_smooth_plot = np.asarray(result['pa_plot_smooth'])[display_slice]
					fit_plot = np.asarray(result['pa_plot_fit'])[display_slice]
				else:
					pa_deg = self._pa_series_deg(dedisp_q[:, display_slice], dedisp_u[:, display_slice], plot_dedispersed)
					if method_name == 'pa_slope_shrine':
						result_kc = result.get('kc', None)
						pa_smooth_plot, fit_plot, _ = self._get_pa_shrine_smoothed_and_fit(
							dedisp_q[:, display_slice], dedisp_u[:, display_slice], plot_dedispersed, time_range_dedisp, force_kc=result_kc
						)
					else:
						pa_smooth_plot, fit_plot, _ = self._get_pa_smoothed_and_fit(dedisp_q[:, display_slice], dedisp_u[:, display_slice], plot_dedispersed, time_range_dedisp)

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
				axr.set_ylabel('PA (deg.)')
				axr.yaxis.label.set_size(fs_label)
				axr.yaxis.labelpad = fs_labelpad
				axr.tick_params(axis='y', labelsize=fs_tick)
				if pa_limits is not None:
					axr.set_ylim(pa_limits)
			else:
				if show_time_legend:
					axes[idx, 1].legend(loc='best', fontsize=fs_legend)
					show_time_legend = False

			axes[idx, 1].set_title(f"Metric = {result['metric']:.3f}")
			axes[idx, 1].set_ylabel(rf'S (arb.)')
			axes[idx, 1].set_xlabel('Time (ms)')
			axes[idx, 1].grid(True, alpha=0.3)
			axes[idx, 1].title.set_fontsize(fs_title)
			axes[idx, 1].xaxis.label.set_size(fs_label)
			axes[idx, 1].yaxis.label.set_size(fs_label)
			axes[idx, 1].xaxis.labelpad = fs_labelpad
			axes[idx, 1].yaxis.labelpad = fs_labelpad
			axes[idx, 1].tick_params(axis='both', labelsize=fs_tick)
			axes[idx, 1].set_yticklabels([])

			# Right column: per-method DM scan curve
			scan_ax = axes[idx, 2]
			dm_vals = result.get('dm_values')
			metric_vals = result.get('metric_values')
			if dm_vals is not None and metric_vals is not None:
				show_scan_for_method = show_scan_uncertainty and (method_name not in disabled)
				if show_scan_for_method:
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
					color=colour_manager.color(method_name),
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
		dt = float(np.nanmedian(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		time_axis = np.arange(output_size) * dt
		for i, dm in enumerate(dm_values):
			if i % 20 == 0:
				print(f"\r  Progress: {i}/{len(dm_values)}", end='', flush=True)
			
			dedispersed = self.dedisperse(data, dm, output_size=output_size, mode=self.dedisp_mode)
			i_data[i] = np.nansum(dedispersed, axis=0)
			if pa_slope_values is not None:
				dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
				dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
				sm_i, sm_q, sm_u = self.maybe_kc_smooth_nonshrine(dedispersed, dedisp_q, dedisp_u)
				pa_slope_values[i] = self.pa_slope_metric(sm_q, sm_u, time_axis, sm_i)
				if l_i_mean_values is not None:
					l_i_mean_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode='mean')

		run_tag = f"scan_{int(np.round(dm_values[0] * 1000))}_{int(np.round(dm_values[-1] * 1000))}_{len(dm_values)}"

		run_prefix_structure = f"{run_tag}_structure"
		run_dir_structure = self.run_shrine_method(
			script_name="maximise_structure.py",
			run_prefix=run_prefix_structure,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=True,
			save_all=True,
		)
		structure_values = np.loadtxt(run_dir_structure / f"{run_prefix_structure}_SPs.dat")

		run_prefix_snr = f"{run_tag}_snr"
		run_dir_snr = self.run_shrine_method(
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
				sm_i, sm_q, sm_u = self.maybe_kc_smooth_nonshrine(dedisp_i, dedisp_q, dedisp_u)
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
			if np.any(np.isfinite(metric_values)):
				max_idx = int(np.nanargmax(metric_values))
			else:
				max_idx = 0
			axes[idx].axvline(dm_values[max_idx], color='red', 
							linestyle='--', alpha=1, label=f'Max at DM={self._format_dm(dm_values[max_idx], 3)}')
			axes[idx].legend()
		
		plt.tight_layout()
		
		if save_path:
			savefig_rasterized(save_path, dpi=150, bbox_inches='tight')
			print(f"DM scan plot saved to: {save_path}")
		else:
			plt.show()


