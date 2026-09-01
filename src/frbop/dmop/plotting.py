"""Plotting and DM-space scanning utilities."""

from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from frbop.utils.plotting import (colour_manager, pub_figsize,
                                  pub_grid_figsize, savefig)


class PlottingMixin:
	def _style_imshow_ticks(self, ax, labelsize: Optional[float] = None,
						   always_white: bool = False) -> None:
		"""
		White inward major+minor ticks for imshow axes (visible over the dark
		waterfall). In expand_nan mode the NaN-padded edges are transparent, so
		the y-axis ticks fall on the white figure background and are drawn black,
		unless ``always_white`` (e.g. the original dspec, which is never NaN-filled).
		"""
		kw = {'axis': 'both', 'which': 'both', 'direction': 'in',
			  'color': 'white', 'labelcolor': 'black'}
		if labelsize is not None:
			kw['labelsize'] = labelsize
		ax.tick_params(**kw)
		if self.dedisp_mode == 'expand_nan' and not always_white:
			ax.tick_params(axis='y', which='both', color='black', labelcolor='black')

	def plot_comparison(self, results: Dict, dm_range: Tuple[float, float],
					   peak_region: Optional[Tuple[int, int]] = None,
					   label: str = "frb",
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

		_, target_h = pub_figsize(ncol=1)
		baseline_rows = 5
		figsize = pub_grid_figsize(
		    ncol=1,
		    n_rows=n_methods + 1,
		    row_height=target_h / baseline_rows,
			width_scale=1.2,
		)
		fig, axes = plt.subplots(
			n_methods + 1,
			5,
			figsize=figsize,
			gridspec_kw={'width_ratios': [0.85, 0.06, 0.85, 0.32, 0.8]},
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
			'min_uncertainty': 'Min. Uncertainty',
			'pa_slope': 'PA',
			'pa_slope_shrine': 'PA (SHRINE)',
			'l_i_mean': r'$\Pi_L$ mean',
			'structure_L': 'Structure (L)',
		}
		fs_title = plt.rcParams.get('axes.titlesize', 11)
		fs_label = plt.rcParams.get('axes.labelsize', 11)
		fs_tick = plt.rcParams.get('xtick.labelsize', 10)
		fs_legend = plt.rcParams.get('legend.fontsize', 9)
		fs_overlay = plt.rcParams.get('axes.labelsize', 11)
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
		axes[0, 0].set_title(f"Original (SHRINE structure-maximised)\n" + rf"Input DM = {self._format_dm(self.input_dm, 3)} $\mathrm{{pc\,cm^{{-3}}}}$")
		axes[0, 0].set_ylabel('Frequency [MHz]')
		axes[0, 0].set_xlabel('Time [ms]')
		axes[0, 0].title.set_fontsize(fs_title)
		axes[0, 0].xaxis.label.set_size(fs_label)
		axes[0, 0].yaxis.label.set_size(fs_label)
		axes[0, 0].xaxis.labelpad = fs_labelpad
		axes[0, 0].yaxis.labelpad = fs_labelpad
		self._style_imshow_ticks(axes[0, 0], fs_tick, always_white=True)

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
			ax0r.set_ylabel('PA [deg.]')
			ax0r.yaxis.label.set_size(fs_label)
			ax0r.yaxis.labelpad = fs_labelpad
			ax0r.tick_params(axis='y', labelsize=fs_tick)
			if pa_limits is not None:
				ax0r.set_ylim(pa_limits)
		else:
			axes[0, 1].legend(loc='best', fontsize=fs_legend)
		axes[0, 1].set_title('Original Time Series')
		axes[0, 1].set_ylabel(r'S [arb.]')
		axes[0, 1].set_xlabel('Time [ms]')
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
			#all_scan_ax.set_title('Best DM Summary')
			all_scan_ax.set_xlabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
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

			# In expand_zero / expand_nan mode, trim all-padding columns from the
			# display only (zeros are excluded via the != 0.0 check; NaN via isfinite).
			display_slice = slice(None)
			if self.dedisp_mode in ('expand_zero', 'expand_nan'):
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
			axes[idx, 0].set_ylabel('Frequency [MHz]')
			axes[idx, 0].set_xlabel('Time [ms]')
			axes[idx, 0].title.set_fontsize(fs_title)
			axes[idx, 0].xaxis.label.set_size(fs_label)
			axes[idx, 0].yaxis.label.set_size(fs_label)
			axes[idx, 0].xaxis.labelpad = fs_labelpad
			axes[idx, 0].yaxis.labelpad = fs_labelpad
			self._style_imshow_ticks(axes[idx, 0], fs_tick)
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
				rf"DM={dm_text} $\mathrm{{pc\,cm^{{-3}}}}$",
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
				axr.set_ylabel('PA [deg.]')
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
			axes[idx, 1].set_ylabel(r'S [arb.]')
			axes[idx, 1].set_xlabel('Time [ms]')
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
			kc_vals = result.get('kc_values')
			if method_name == 'min_uncertainty' and kc_vals is not None and metric_vals is not None:
				kc_low = result.get('kc_uncertainty_low')
				kc_high = result.get('kc_uncertainty_high')
				kc_opt_idx = result.get('kc_optimal_idx')
				if kc_low is not None and kc_high is not None:
					kc_low = np.asarray(kc_low, dtype=float)
					kc_high = np.asarray(kc_high, dtype=float)
					scan_ax.fill_between(
						kc_vals, kc_low, kc_high,
						color='tab:orange', alpha=0.18,
						label=r'$\delta$ DM' if show_scan_legend else None,
					)
					scan_ax.plot(kc_vals, kc_low, '--', color='tab:orange', linewidth=1.0, alpha=0.7)
					scan_ax.plot(kc_vals, kc_high, '--', color='tab:orange', linewidth=1.0, alpha=0.7)
				scan_ax.plot(
					kc_vals, dm_vals,
					linewidth=2.0, color='black',
					label='Best DM' if show_scan_legend else None,
				)
				if kc_opt_idx is not None:
					scan_ax.axvline(
						kc_vals[int(kc_opt_idx)], color='red', linestyle='--', linewidth=1.4, alpha=0.9,
						label='Optimal kc' if show_scan_legend else None,
					)
				scan_ax.set_xlabel(r'$k_c$')
				scan_ax.set_ylabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
				scan_ax.set_xlim(kc_vals[0], kc_vals[-1])
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
			elif dm_vals is not None and metric_vals is not None:
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
							label=r'$\delta$ DM' if show_scan_legend else None,
						)
				scan_ax.plot(
					dm_vals,
					metric_vals,
					linewidth=2.0,
					color=colour_manager.color(method_name),
				)
				if method_name in ('pa_slope', 'pa_slope_shrine'):
					y_lo, y_hi = scan_ax.get_ylim()
					if y_lo <= 0.0 <= y_hi:
						scan_ax.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)
						scan_ax.set_ylim(y_lo, y_hi)
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
				scan_ax.set_xlabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
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
			savefig(save_path, dpi=600, bbox_inches='tight')
			print(f"\nFigure saved to: {save_path}")
		else:
			plt.show()
	
	def plot_single_method_comparison(self,
									   method_key: str,
									   all_results: List[Dict],
									   dm_range: Tuple[float, float],
									   peak_regions: List[Tuple[int, int]],
									   label: str = "frb",
									   save_path: Optional[str] = None,
									   show_errors: bool = True):
		"""
		Single-method component comparison with a max-range overview row.

		Layout (one method):
		  Row 1 (max range): the full envelope dedispersed dynamic spectrum and
		  Stokes I time profile with the component regions shaded and labelled
		  (no PA/L), plus a per-component best-DM comparison against the input
		  DM (DM on the x-axis).
		  Rows 2..n (components): each component's dedispersed dynamic spectrum,
		  corrected Stokes I profile (no PA/L), and the metric-vs-DM curve.
		"""
		n_segments = len(all_results)
		if n_segments == 0:
			print("  Single-method comparison skipped (no segment results).")
			return

		max_start = int(min(r[0] for r in peak_regions))
		max_end = int(max(r[1] for r in peak_regions))

		n_rows = 1 + n_segments
		_, target_h = pub_figsize(ncol=1)
		baseline_rows = 5
		figsize = pub_grid_figsize(
			ncol=1,
			n_rows=n_rows,
			row_height=target_h / baseline_rows,
			width_scale=1.2,
		)
		fig, axes = plt.subplots(
			n_rows, 5, figsize=figsize,
			gridspec_kw={'width_ratios': [0.85, 0.1, 0.85, 0.15, 0.8]},
		)
		if n_rows == 1:
			axes = np.atleast_2d(axes)

		for spacer_ax in axes[:, 1]:
			fig.delaxes(spacer_ax)
		for spacer_ax in axes[:, 3]:
			fig.delaxes(spacer_ax)
		axes = np.stack((axes[:, 0], axes[:, 2], axes[:, 4]), axis=1)

		seg_colours = ['#ffb000', '#785ef0', '#00b0ff', '#00d16b', '#ff5b5b',
					   '#d000d0', '#3b3b3b']

		method_label = {
			'structure': 'Structure',
			'snr': 'S/N',
			'min_uncertainty': 'Min. Uncertainty',
			'pa_slope': 'PA',
			'pa_slope_shrine': 'PA (SHRINE)',
			'l_i_mean': r'$\Pi_L$ mean',
			'structure_L': 'Structure (L)',
		}
		fs_title = plt.rcParams.get('axes.titlesize', 11)
		fs_label = plt.rcParams.get('axes.labelsize', 11)
		fs_tick = plt.rcParams.get('xtick.labelsize', 10)
		fs_legend = plt.rcParams.get('legend.fontsize', 9)
		fs_labelpad = 2

		# --- Row 0: max range overview ----------------------------------------
		max_time = self.time_ms[max_start:max_end]
		max_dspec = self.stokes_i[:, max_start:max_end]
		display_slice = slice(None)
		nonzero = np.any(np.isfinite(max_dspec) & (max_dspec != 0.0), axis=0)
		if np.any(nonzero):
			first_valid = int(np.argmax(nonzero))
			last_valid = int(len(nonzero) - np.argmax(nonzero[::-1]))
			if first_valid < last_valid:
				display_slice = slice(first_valid, last_valid)
		max_dspec = max_dspec[:, display_slice]
		max_time = max_time[display_slice]
		vmin0, vmax0 = self._robust_vmin_vmax(max_dspec)
		dax = axes[0, 0]
		dax.imshow(
			max_dspec,
			aspect='auto',
			extent=[max_time[0], max_time[-1], self.freq_mhz[0], self.freq_mhz[-1]],
			cmap='plasma',
			origin='lower',
			vmin=vmin0,
			vmax=vmax0,
		)
		dax.set_title(
			#f"{method_label.get(method_key, method_key)} - Max range\n"
					   rf"DM$_{{\rm C}}$ = {self._format_dm(self.input_dm, 3)} $\mathrm{{pc\,cm^{{-3}}}}$")
		dax.set_ylabel('Frequency [MHz]')
		dax.set_xlabel('Time [ms]')
		dax.title.set_fontsize(fs_title)
		dax.xaxis.label.set_size(fs_label)
		dax.yaxis.label.set_size(fs_label)
		dax.xaxis.labelpad = fs_labelpad
		dax.yaxis.labelpad = fs_labelpad
		self._style_imshow_ticks(dax, fs_tick, always_white=True)

		pax = axes[0, 1]
		max_series = np.nansum(max_dspec, axis=0)
		pax.plot(max_time, max_series, 'k-', linewidth=1)
		pax.set_ylabel(r'S [arb.]')
		pax.set_xlabel('Time [ms]')
		pax.grid(True, alpha=0.3)
		pax.title.set_fontsize(fs_title)
		pax.xaxis.label.set_size(fs_label)
		pax.yaxis.label.set_size(fs_label)
		pax.xaxis.labelpad = fs_labelpad
		pax.yaxis.labelpad = fs_labelpad
		pax.tick_params(axis='both', labelsize=fs_tick)
		pax.set_yticklabels([])

		# Shade each component region on the max-range profile; label the profile.
		p_max = float(np.nanmax(max_series)) if len(max_series) > 0 else 0.0
		x_lo = float(max_time[0])
		x_hi = float(max_time[-1])
		clamped_bounds = []
		for r_idx, (r0, r1) in enumerate(peak_regions):
			x0 = float(self.time_ms[min(max(r0, 0), len(self.time_ms) - 1)])
			x1 = float(self.time_ms[min(max(r1, 0), len(self.time_ms) - 1)])
			clamped_bounds.append((min(max(x0, x_lo), x_hi), min(max(x1, x_lo), x_hi)))
		flat_bounds = [b for pair in clamped_bounds for b in pair]
		extreme_min = min(flat_bounds) if flat_bounds else None
		extreme_max = max(flat_bounds) if flat_bounds else None
		for r_idx, ((x0, x1), (r0, r1)) in enumerate(zip(clamped_bounds, peak_regions)):
			seg_col = seg_colours[r_idx % len(seg_colours)]
			for bx in (x0, x1):
				if extreme_min is not None and extreme_max is not None and extreme_min < bx < extreme_max:
					dax.axvline(bx, color='white', linewidth=0.8, alpha=0.9)
			pax.axvspan(x0, x1, color=seg_col, alpha=0.15, zorder=0)
			pax.text(x0, 0.8 * p_max, rf'$C_{{{r_idx + 1}}}$', fontsize=fs_label,
					 fontweight='bold', color='black', ha='left', va='bottom',
					 transform=pax.transData)

		# Per-component DM vs input DM (right column of row 0); DM on the x-axis.
		cmp_ax = axes[0, 2]
		seg_pos = np.arange(1, n_segments + 1, dtype=float)
		seg_dm = np.array([all_results[i][method_key]['dm'] for i in range(n_segments)])
		for i in range(n_segments):
			seg_col = seg_colours[i % len(seg_colours)]
			if show_errors:
				minus = float(all_results[i][method_key].get('uncertainty_minus') or 0.0)
				plus = float(all_results[i][method_key].get('uncertainty_plus') or 0.0)
				cmp_ax.errorbar([seg_dm[i]], [seg_pos[i]], xerr=[[minus], [plus]],
								fmt='o', color=seg_col, capsize=3, elinewidth=1.8, markersize=5)
			else:
				cmp_ax.plot([seg_dm[i]], [seg_pos[i]], 'o', color=seg_col, markersize=5)
		cmp_ax.axvline(self.input_dm, color='gray', linestyle=':', linewidth=1.4, alpha=0.9,
					   label=rf"DM$_{{\rm C}}$ = {self._format_dm(self.input_dm, 3)}")
		cmp_ax.set_ylim(0.5, n_segments + 0.5)
		cmp_ax.invert_yaxis()
		cmp_ax.set_yticks(seg_pos)
		cmp_ax.set_yticklabels([rf'$C_{{{i + 1}}}$' for i in range(n_segments)])
		cmp_ax.set_title('Component DM comparison')
		cmp_ax.set_xlabel(r'Best DM [$\mathrm{pc\,cm}^{{-3}}$]')
		cmp_ax.grid(True, axis='x', alpha=0.3)
		cmp_ax.legend(loc='best', fontsize=fs_legend)
		cmp_ax.title.set_fontsize(fs_title)
		cmp_ax.xaxis.label.set_size(fs_label)
		cmp_ax.yaxis.label.set_size(fs_label)
		cmp_ax.xaxis.labelpad = fs_labelpad
		cmp_ax.yaxis.labelpad = fs_labelpad
		cmp_ax.tick_params(axis='both', labelsize=fs_tick)

		# --- Component rows -------------------------------------------------------
		for seg in range(n_segments):
			res = all_results[seg][method_key]
			r0, r1 = peak_regions[seg]
			row = seg + 1
			seg_col = seg_colours[seg % len(seg_colours)]

			n_time_out = res['dedispersed'].shape[1]
			stored_time = res.get('time_ms')
			if stored_time is not None and len(stored_time) == n_time_out:
				time_disp = np.asarray(stored_time, dtype=float)
			else:
				if len(self.time_ms) > 1:
					dt_val = float(np.nanmedian(np.diff(self.time_ms)))
				else:
					dt_val = 1.0
				delay_samples = self._get_delay_samples(res['dm'])
				if self.dedisp_mode == 'crop':
					start_shift = int(np.max(delay_samples))
				else:
					start_shift = int(np.min(delay_samples))
				base_start = self.time_ms[min(r0, len(self.time_ms) - 1)]
				time_disp = base_start + start_shift * dt_val + np.arange(n_time_out) * dt_val

			display_slice = slice(None)
			if self.dedisp_mode in ('expand_zero', 'expand_nan'):
				nonzero = np.any(np.isfinite(res['dedispersed']) & (res['dedispersed'] != 0.0), axis=0)
				if np.any(nonzero):
					first_valid = int(np.argmax(nonzero))
					last_valid = int(len(nonzero) - np.argmax(nonzero[::-1]))
					display_slice = slice(first_valid, last_valid)

			plot_dedisp = res['dedispersed'][:, display_slice]
			time_disp = time_disp[display_slice]
			sax = axes[row, 0]
			vmin, vmax = self._robust_vmin_vmax(plot_dedisp)
			sax.imshow(
				plot_dedisp,
				aspect='auto',
				extent=[time_disp[0], time_disp[-1], self.freq_mhz[0], self.freq_mhz[-1]],
				cmap='plasma',
				origin='lower',
				vmin=vmin,
				vmax=vmax,
			)
			sax.set_title(rf'$C_{{{seg + 1}}}$')
			sax.set_ylabel('Frequency [MHz]')
			sax.set_xlabel('Time [ms]')
			sax.title.set_fontsize(fs_title)
			sax.xaxis.label.set_size(fs_label)
			sax.yaxis.label.set_size(fs_label)
			sax.xaxis.labelpad = fs_labelpad
			sax.yaxis.labelpad = fs_labelpad
			self._style_imshow_ticks(sax, fs_tick)

			iax = axes[row, 1]
			prof = np.nansum(plot_dedisp, axis=0)
			iax.plot(time_disp, prof, 'k-', linewidth=1)
			dm_val = self._format_dm(res['dm'], 3)
			minus = res.get('uncertainty_minus')
			plus = res.get('uncertainty_plus')
			if minus is not None or plus is not None:
				sup = f"+{self._format_dm(plus, 2)}" if plus is not None else ""
				sub = f"-{self._format_dm(minus, 2)}" if minus is not None else ""
				dm_label = f"DM = {dm_val}" + rf"$^{{{sup}}}_{{{sub}}}$"
			else:
				dm_label = f"DM = {dm_val}"
			iax.set_title(rf"{dm_label} $\mathrm{{pc\,cm^{{-3}}}}$")
			iax.set_ylabel(r'S [arb.]')
			iax.set_xlabel('Time [ms]')
			iax.grid(True, alpha=0.3)
			iax.title.set_fontsize(fs_title)
			iax.xaxis.label.set_size(fs_label)
			iax.yaxis.label.set_size(fs_label)
			iax.xaxis.labelpad = fs_labelpad
			iax.yaxis.labelpad = fs_labelpad
			iax.tick_params(axis='both', labelsize=fs_tick)
			iax.set_yticklabels([])

			m_ax = axes[row, 2]
			dm_vals = res.get('dm_values')
			met_vals = res.get('metric_values')
			kc_vals = res.get('kc_values')
			if method_key == 'min_uncertainty' and kc_vals is not None and met_vals is not None and len(met_vals) > 0:
				kc_low = res.get('kc_uncertainty_low')
				kc_high = res.get('kc_uncertainty_high')
				kc_opt_idx = res.get('kc_optimal_idx')
				if kc_low is not None and kc_high is not None:
					kc_low = np.asarray(kc_low, dtype=float)
					kc_high = np.asarray(kc_high, dtype=float)
					m_ax.fill_between(kc_vals, kc_low, kc_high, color='tab:orange', alpha=0.18, label=r'$\delta$ DM')
					m_ax.plot(kc_vals, kc_low, '--', color='tab:orange', linewidth=1.0, alpha=0.7)
					m_ax.plot(kc_vals, kc_high, '--', color='tab:orange', linewidth=1.0, alpha=0.7)
				m_ax.plot(kc_vals, dm_vals, linewidth=2.0, color='black', label='Best DM')
				if kc_opt_idx is not None:
					m_ax.axvline(kc_vals[int(kc_opt_idx)], color='red', linestyle='--', linewidth=1.4,
								 alpha=0.9, label='Optimal kc')
				m_ax.set_xlabel(r'$k_c$')
				m_ax.set_ylabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
				m_ax.set_xlim(kc_vals[0], kc_vals[-1])
				m_ax.grid(True, alpha=0.3)
				m_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
				if seg == 0:
					m_ax.legend(loc='best', fontsize=fs_legend)
			elif dm_vals is not None and met_vals is not None and len(met_vals) > 0:
				if show_errors:
					low_dm = res.get('uncertainty_low_dm')
					high_dm = res.get('uncertainty_high_dm')
					shade_low = float(dm_range[0]) if low_dm is None else float(low_dm)
					shade_high = float(dm_range[1]) if high_dm is None else float(high_dm)
					if shade_low <= shade_high:
						m_ax.axvspan(shade_low, shade_high, color='tab:orange', alpha=0.18,
									 label=r'$\delta$ DM')
				m_ax.plot(dm_vals, met_vals, linewidth=2.0, color=seg_col)
				m_ax.axvline(self.input_dm, color='gray', linestyle=':', linewidth=1.4,
							 alpha=0.9, label=r'DM$_{\rm C}$')
				m_ax.axvline(res['dm'], color='red', linestyle='--', linewidth=1.4,
							 alpha=0.9, label='Best DM')
				m_ax.set_title(f"Metric = {res['metric']:.3f}")
				m_ax.set_xlim(dm_range[0], dm_range[1])
				m_ax.set_xlabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
				m_ax.set_ylabel('Metric')
				m_ax.grid(True, alpha=0.3)
				m_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
				if seg == 0:
					m_ax.legend(loc='best', fontsize=fs_legend)
			else:
				m_ax.text(0.5, 0.5, 'No scan data', ha='center', va='center', transform=m_ax.transAxes)
				m_ax.set_axis_off()
			m_ax.title.set_fontsize(fs_title)
			m_ax.xaxis.label.set_size(fs_label)
			m_ax.yaxis.label.set_size(fs_label)
			m_ax.xaxis.labelpad = fs_labelpad
			m_ax.yaxis.labelpad = fs_labelpad
			m_ax.tick_params(axis='both', labelsize=fs_tick)

		plt.tight_layout(rect=[0.02, 0.02, 0.995, 0.995])
		fig.subplots_adjust(wspace=0.04, hspace=0.5)

		if save_path:
			savefig(save_path, dpi=600, bbox_inches='tight')
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
		_scan_shrine_kc = self.shrine_kc[0] if isinstance(self.shrine_kc, list) else self.shrine_kc
		
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
		if pa_slope_values is not None:
			self._maybe_prepare_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)
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
					l_i_mean_values[i] = self.linear_to_stokes_i_metric(sm_q, sm_u, sm_i, mode='mean', use_fwhm_window=True)

		run_tag = f"scan_{int(np.round(dm_values[0] * 1000))}_{int(np.round(dm_values[-1] * 1000))}_{len(dm_values)}"

		run_prefix_structure = f"{run_tag}_structure"
		run_dir_structure = self.run_shrine_method(
			script_name="maximise_structure.py",
			run_prefix=run_prefix_structure,
			dm_values=dm_values,
			i_data=i_data,
			include_input_dm=True,
			save_all=True,
			force_kc=_scan_shrine_kc,
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
			force_kc=_scan_shrine_kc,
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
		base_width, base_height = pub_figsize(height_ratio=0.75)
		row_height = 2.6
		fig_height = max(base_height, row_height * len(metrics))
		fig, axes = plt.subplots(len(metrics), 1, figsize=(base_width, fig_height))
		
		if len(metrics) == 1:
			axes = [axes]
		
		colors = {
			'structure': 'blue',
			'snr': 'red',
			'min_uncertainty': 'brown',
			'pa_slope': 'green',
			'pa_slope_shrine': 'teal',
			'l_i_mean': 'purple',
			'structure_L': 'orange',
		}
		labels = {
			'structure': 'Structure Metric (SHRINE)',
			'snr': 'S/N',
			'pa_slope': "Weighted PA Slope magnitude",
			'pa_slope_shrine': "Weighted PA Slope magnitude (SHRINE-smoothed PA)",
			'l_i_mean': "L/I (mean)",
			'structure_L': "Structure Metric (L)"
		}
		
		for idx, (metric_name, metric_values) in enumerate(metrics.items()):
			axes[idx].plot(dm_values, metric_values, 
						  color=colors.get(metric_name, 'black'),
						  linewidth=2)
			axes[idx].set_xlabel(r'DM [$\mathrm{pc\,cm}^{{-3}}$]')
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
			savefig(save_path, dpi=150, bbox_inches='tight')
			print(f"DM scan plot saved to: {save_path}")
		else:
			plt.show()


	def plot_range(
		self,
		structure_result: Dict,
		peak_region: Optional[Tuple[int, int]] = None,
		label: str = "frb",
		save_path: Optional[str] = None,
	):
		"""
		Plot the burst dedispersed to the lower-bound, optimum, and upper-bound DM
		for the structure method in a 1×3 waterfall layout.

		Parameters:
		-----------
		structure_result : dict
			Result dict for the 'structure' method, containing 'dm',
			'uncertainty_low_dm', and 'uncertainty_high_dm'.
		peak_region : tuple, optional
			(start, end) indices into the time axis. If None, uses full data.
		label : str
			FRB label, used for title.
		save_path : str, optional
			Path to save the figure.
		"""
		dm_low = structure_result.get('uncertainty_low_dm')
		dm_opt = structure_result.get('dm')
		dm_high = structure_result.get('uncertainty_high_dm')
		if dm_low is None or dm_high is None or dm_opt is None:
			print("Range plot skipped — uncertainty bounds not available for structure method.")
			return

		data = self.stokes_i
		if peak_region is not None:
			data = data[:, peak_region[0]:peak_region[1]]
			time_range = self.time_ms[peak_region[0]:peak_region[1]]
		else:
			time_range = self.time_ms

		dms = [dm_low, dm_opt, dm_high]
		titles = ['Lower bound', 'Optimum', 'Upper bound']
		dedispersed_list = []
		time_axes = []

		for dm in dms:
			dedisp = self.dedisperse(data, dm, mode=self.dedisp_mode)
			n_time = dedisp.shape[1]
			dt = float(np.nanmedian(np.diff(time_range))) if len(time_range) > 1 else 1.0
			delay_samples = self._get_delay_samples(dm)
			if self.dedisp_mode == 'crop':
				start_shift = int(np.max(delay_samples))
			else:
				start_shift = int(np.min(delay_samples))
			time_axis = time_range[0] + start_shift * dt + np.arange(n_time) * dt
			dedispersed_list.append(dedisp)
			time_axes.append(time_axis)

		vmin, vmax = self._robust_vmin_vmax(np.concatenate(
			[d.ravel() for d in dedispersed_list]
		))

		# Leading edge of the pulse in the optimised DM panel, measured from the
		# top quarter of the band (highest-frequency channels).
		opt_idx = 1
		fmin_b = float(self.freq_mhz[0])
		fmax_b = float(self.freq_mhz[-1])
		fmin_top = fmin_b + 0.75 * (fmax_b - fmin_b)
		top_mask = self.freq_mhz >= fmin_top
		if np.count_nonzero(top_mask) < 3:
			top_mask = np.zeros_like(top_mask, dtype=bool)
			top_mask[-max(3, len(self.freq_mhz) // 4):] = True
		f_top = float(np.mean(self.freq_mhz[top_mask]))

		prof_opt = np.nansum(dedispersed_list[opt_idx][top_mask], axis=0)
		kernel = np.ones(5) / 5.0
		smoothed = np.convolve(prof_opt, kernel, mode='same')
		peak = float(np.max(smoothed))
		thresh = 0.1 * peak
		peak_idx = int(np.argmax(smoothed))
		left_idx = peak_idx
		while left_idx > 0 and smoothed[left_idx - 1] >= thresh:
			left_idx -= 1
		t_left_ms = float(time_axes[opt_idx][left_idx])

		# Residual dispersion delay of the top band relative to the optimised DM:
		# the reference channel anchors absolute time, so the panel-to-panel shift is
		# K * (dm_opt - dm_panel) * (1/f_top^2 - 1/f_ref^2).
		def _topband_shift(dm: float) -> float:
			return self.DM_CONSTANT * (float(dm_opt) - float(dm)) * (
				1.0 / f_top**2 - 1.0 / float(self.reference_freq)**2
			)

		fig, axes = plt.subplots(
			1, 3,
			figsize=pub_figsize(ncol=1, height_ratio=0.3),
		)

		for i, (dedisp, taxis, title, dm) in enumerate(zip(
			dedispersed_list, time_axes, titles, dms
		)):
			ax = axes[i]
			ax.imshow(
				dedisp,
				aspect='auto',
				extent=[taxis[0], taxis[-1], self.freq_mhz[0], self.freq_mhz[-1]],
				cmap='plasma',
				origin='lower',
				vmin=vmin,
				vmax=vmax,
			)
			self._style_imshow_ticks(ax)
			ax.axvline(t_left_ms + _topband_shift(dm), color='white',
					   linewidth=.7, alpha=0.8)
			#ax.set_title(title)
			dm_text = self._format_dm(dm, 3)
			ax.text(
				0.98, 0.98,
				rf"DM={dm_text} $\mathrm{{pc\,cm^{{-3}}}}$",
				transform=ax.transAxes,
				ha='right', va='top',
				color='white',
				fontsize=8,
				bbox=dict(facecolor='black', edgecolor='none', alpha=0.35, pad=2.0),
			)
			ax.set_xlabel('Time [ms]')
			if i == 0:
				ax.set_ylabel('Frequency [MHz]')

		#fig.suptitle(f'{label} — Structure method')
		plt.tight_layout()
		if save_path:
			savefig(save_path, dpi=600, bbox_inches='tight')
			print(f"Range plot saved to: {save_path}")
		else:
			plt.show()
