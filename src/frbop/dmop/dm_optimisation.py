"""Command-line interface for dm_optimisation."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.plotting import set_pub_style

from .optimiser import DMOptimiser

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
		choices=["expand", "expand_zero", "crop"],
		default="expand",
		help="Dedispersion mode: 'expand' (fill edges with noise), 'expand_zero' (zero-fill new bins), or 'crop' (trim to common valid region)",
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
			window_profile = np.nansum(window_i, axis=0)
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
