"""Multi-component DM and dn_e diagnostics."""

from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np

from frbop.dmop.common import add_asym
from frbop.utils.plotting import colour_manager, savefig_rasterized, pub_figsize


class ComponentsMixin:
	def plot_component_dm_diagnostics(self,
						 all_results: List[Dict],
						 component_ids: Optional[np.ndarray] = None,
						 component_times_ms: Optional[np.ndarray] = None,
						 label: str = "segment",
						 frb_label: str = "frb",
						 save_path: Optional[str] = None,
						 show_errors: bool = True,
						 excluded_methods: Optional[Set[str]] = None):
		"""
		Plot DM diagnostics across multiple components.

		Creates a single-panel diagnostic showing absolute best DM per
		component for each method, including asymmetric error bars.
		"""
		excluded = excluded_methods or set()
		n_components = len(all_results)
		if n_components < 2:
			print("Component DM diagnostics skipped (need at least 2 components).")
			return

		if component_times_ms is not None:
			component_times_ms = np.asarray(component_times_ms, dtype=float)
			if component_times_ms.ndim != 1 or component_times_ms.shape[0] != n_components:
				raise ValueError("component_times_ms must be 1D with one value per component")
			if not np.all(np.isfinite(component_times_ms)):
				raise ValueError("component_times_ms must be finite")
			sort_idx = np.argsort(component_times_ms)
			all_results = [all_results[int(i)] for i in sort_idx]
			component_ids = np.arange(1, n_components + 1, dtype=int)

		if component_ids is None:
			component_ids = np.arange(1, n_components + 1, dtype=int)
		else:
			component_ids = np.asarray(component_ids, dtype=int)
			if component_ids.ndim != 1 or component_ids.shape[0] != n_components:
				raise ValueError("component_ids must be 1D with one value per component")

		preferred_order = ['structure', 'snr', 'min_uncertainty', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		first_methods = list(all_results[0].keys())
		common_methods = [m for m in first_methods if all(m in comp for comp in all_results)]
		if len(common_methods) == 0:
			print("Component DM diagnostics skipped (no common methods across components).")
			return

		ordered_methods = [m for m in preferred_order if m in common_methods]
		ordered_methods.extend([m for m in common_methods if m not in ordered_methods])
		ordered_methods = [m for m in ordered_methods if m not in excluded]
		if len(ordered_methods) == 0:
			print("Component DM diagnostics skipped (no methods left after exclusions).")
			return

		method_display = {
			'structure': 'Structure',
			'snr': 'S/N',
			'min_uncertainty': 'Min. uncertainty',
			'pa_slope': 'PA slope',
			'pa_slope_shrine': 'PA slope (SHRINE)',
			'l_i_mean': 'L/I mean',
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

		fig, ax = plt.subplots(
			1,
			1,
			figsize=pub_figsize(
				ncol=2,
				height_ratio=0.6,
			),
		)

		for draw_rank, i in enumerate(draw_order):
			method_name = ordered_methods[i]
			disp_name = method_display.get(method_name, method_name)
			color = colour_manager.color(method_name)
			if show_errors:
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
			else:
				ax.plot(
					component_idx,
					dm_matrix[i],
					'o-',
					label=disp_name,
					color=color,
					zorder=2 + draw_rank,
				)

		ax.set_ylabel(r'Best DM [$\mathrm{pc\,cm}^{{-3}}$]')
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

		# Draw horizontal line at input DM if available
		if hasattr(self, 'input_dm') and np.isfinite(getattr(self, 'input_dm', np.nan)):
			try:
				_input_dm = float(self.input_dm)
				ax.axhline(
					_input_dm,
					color='gray',
					linestyle=':',
					linewidth=1.4,
					alpha=0.9,
					label=(f'Input DM={_input_dm}' if not hasattr(self, '_format_dm') else f'Input DM={self._format_dm(_input_dm, 3)}'),
				)
			except Exception:
				pass

		ax.set_xticks(component_idx)
		component_names = []
		for cid in component_ids:
			cid_int = int(cid)
			component_names.append(f'Component {cid_int}')
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
		excluded_methods: Optional[Set[str]] = None,
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
		excluded = excluded_methods or set()
		common_methods = [m for m in common_methods if m not in excluded]

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

		pair_labels = [f"{b + 1}-{a + 1}" for a, b in pair_indices]
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
		if len(common_methods) == 0:
			return {
				'comparison': comparison,
				'component_separation_pc': None if component_separation_pc is None else float(component_separation_pc),
				'component_times_ms': None if component_times_ms is None else component_times_ms.copy(),
				'pair_indices': pair_indices,
				'pair_labels': pair_labels,
				'pair_separations_pc': pair_separations_pc,
				'methods': method_diagnostics,
			}

		for method_name in common_methods:
			delta_dm = np.zeros(len(pair_indices), dtype=float)
			delta_dm_sigma_minus = np.zeros(len(pair_indices), dtype=float)
			delta_dm_sigma_plus = np.zeros(len(pair_indices), dtype=float)
			dn_e = np.zeros(len(pair_indices), dtype=float)
			dn_e_sigma_minus = np.zeros(len(pair_indices), dtype=float)
			dn_e_sigma_plus = np.zeros(len(pair_indices), dtype=float)

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

				# Delta DM = DM_b - DM_a
				delta, sigma_minus, sigma_plus = add_asym(
					x0s=[dm_b, -dm_a],
					siglos=[minus_b, plus_a],  # note swap for -DM_a
					sighis=[plus_b, minus_a],  # note swap for -DM_a
					order=2
				)
				delta_dm[i] = delta
				delta_dm_sigma_minus[i] = sigma_minus
				delta_dm_sigma_plus[i] = sigma_plus

				dn_e[i] = delta / sep_pc
				dn_e_sigma_minus[i] = sigma_minus / sep_pc
				dn_e_sigma_plus[i] = sigma_plus / sep_pc

			method_diagnostics[method_name] = {
				'delta_dm': delta_dm,
				'delta_dm_sigma_minus': delta_dm_sigma_minus,
				'delta_dm_sigma_plus': delta_dm_sigma_plus,
				'dn_e': dn_e,
				'dn_e_sigma_minus': dn_e_sigma_minus,
				'dn_e_sigma_plus': dn_e_sigma_plus,
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
		frb_label: str = "frb",
		save_path: Optional[str] = None,
		show_errors: bool = True,
		excluded_methods: Optional[Set[str]] = None,
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
		excluded = excluded_methods or set()
		if len(pair_labels) == 0 or len(methods) == 0:
			print("dn_e diagnostics plot skipped (no dn_e data).")
			return

		preferred_order = ['structure', 'snr', 'min_uncertainty', 'pa_slope', 'pa_slope_shrine', 'l_i_mean']
		method_names = [m for m in preferred_order if m in methods and m not in excluded]
		method_names.extend([m for m in methods.keys() if m not in method_names and m not in excluded])
		if len(method_names) == 0:
			print("dn_e diagnostics plot skipped (no methods left after exclusions).")
			return

		method_display = {
			'structure': 'Structure',
			'snr': 'S/N',
			'min_uncertainty': 'Min. uncertainty',
			'pa_slope': 'PA slope',
			'pa_slope_shrine': 'PA slope (SHRINE)',
			'l_i_mean': 'L/I mean',
		}

		x = np.arange(len(pair_labels), dtype=float)
		fig, ax = plt.subplots(
			1,
			1,
			figsize=pub_figsize(
				ncol=2,
				height_ratio=0.6,
			),
		)

		# Small x-offset per method so uncertainty bars are readable.
		n_methods = max(1, len(method_names))
		offset_span = 0.18 * (n_methods / max(n_methods, 4))
		if n_methods == 1:
			offsets = np.array([0.0])
		else:
			offsets = np.linspace(-offset_span, offset_span, n_methods)

		all_abs = []
		for i, method_name in enumerate(method_names):
			vals = methods[method_name]
			y = np.asarray(vals.get('dn_e', np.zeros_like(x)), dtype=float)
			err_minus = np.asarray(vals.get('dn_e_sigma_minus', np.zeros_like(y)), dtype=float)
			err_plus = np.asarray(vals.get('dn_e_sigma_plus', np.zeros_like(y)), dtype=float)
			yerr = np.vstack((err_minus, err_plus))
			x_plot = x + offsets[i]

			all_abs.extend(np.abs(y[np.isfinite(y)]).tolist())
			all_abs.extend(np.abs(y - err_minus)[np.isfinite(err_minus)].tolist())
			all_abs.extend(np.abs(y + err_plus)[np.isfinite(err_plus)].tolist())

			if show_errors:
				ax.errorbar(
					x_plot,
					y,
					yerr=yerr,
					fmt='o',
					label=method_display.get(method_name, method_name),
					color=colour_manager.color(method_name),
					capsize=3,
				)
			else:
				ax.plot(
					x_plot,
					y,
					'o',
					label=method_display.get(method_name, method_name),
					color=colour_manager.color(method_name),
				)

		ax.axhline(0.0, color='0.35', linestyle='--', alpha=0.8)
		ax.set_xticks(x)
		x_pad = 0.2
		ax.set_xlim(x[0] - x_pad, x[-1] + x_pad)
		ax.set_xticklabels([])
		#ax.set_xlabel('Component pair')
		ax.set_ylabel(r'$\Delta n_e [\mathrm{cm}^{-3}]$')
		ax.grid(True, alpha=0.3)
		#ax.legend(loc='best')

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



