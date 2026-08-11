"""Command-line interface for dm_optimisation."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy import units as u

from frbop.utils.peaks import parse_peak_index_pairs
from frbop.utils.plotting import set_pub_col, set_pub_style

from .optimiser import DMOptimiser


def tscrunch_arrays(arr, factor: int, axis: int = -1) -> np.ndarray:
    """Average ``arr`` along ``axis`` in groups of ``factor`` consecutive bins.

    Trailing bins beyond a whole number of factors are dropped. NaN entries
    are ignored (NaN-safe mean), matching the optimiser's NaN handling.
    """
    n = arr.shape[axis]
    axis_norm = axis % arr.ndim
    n_keep = n // factor
    kept = np.take(arr, np.arange(n_keep * factor), axis=axis)
    new_shape = (
        kept.shape[:axis_norm]
        + (n_keep, factor)
        + kept.shape[axis_norm + 1:]
    )
    scrunched = np.nanmean(kept.reshape(new_shape), axis=axis_norm + 1)
    return scrunched


def build_optimiser(stokes_i, freq_mhz, time_ms, stokes_q, stokes_u, stokes_v, args) -> DMOptimiser:
	"""Construct a DMOptimiser from CLI arguments and (possibly scrunched) data."""
	return DMOptimiser(
		stokes_i,
		freq_mhz,
		time_ms,
		stokes_q=stokes_q,
		stokes_u=stokes_u,
		stokes_v=stokes_v,
		reference_freq=args.ref_freq,
		input_dm=args.input_dm,
		dedisp_mode=args.dedisp_mode,
		pa_fit_degree=args.pa_fit_degree,
		pa_min_run=args.pa_min_run,
		pa_weight_strength=args.pa_weight_strength,
		pa_fit_post_peak_only=args.pa_fit_post_peak_only,
		nonshrine_kc_smooth=args.nonshrine_kc_smooth,
		nonshrine_shrine_like_errors=args.nonshrine_shrine_like_errors,
		nonshrine_kc_minimise_uncertainty=args.nonshrine_kc_minimise_uncertainty,
		nonshrine_kc=args.nonshrine_kc,
		shrine_kc=args.shrine_kc,
		sync_kc=args.sync_kc,
		li_i_sigma_cut=args.li_sig,
		random_seed=args.seed,
	)


def rescale_peak_indices(indices, factor: int) -> list:
    """Map original-resolution start/end peak index pairs to scrunched bins.

    Original bin ``i`` falls into scrunched bin ``i // factor``, so a region
    ``[start, end)`` covers the same time span as ``[start // factor, ceil(end / factor))``.
    Returns the input unchanged for ``factor <= 1`` or empty input.
    """
    if factor <= 1 or not indices:
        return indices
    values = list(indices)
    if len(values) % 2 != 0:
        raise ValueError(
            "--peak-indices requires an even number of values (pairs of start/end indices)"
        )
    scaled = []
    for i in range(0, len(values), 2):
        start = int(values[i])
        end = int(values[i + 1])
        scaled.append(start // factor)
        scaled.append(-(-end // factor))
    return scaled


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
		"--stokes-v",
		default=None,
		help="Optional path to Stokes V numpy file (freq x time)",
	)
	parser.add_argument(
		"--freq",
		default="freq.npy",
		help="Path to frequency array numpy file [MHz]",
	)
	parser.add_argument(
		"--time",
		default="time.npy",
		help="Path to time array numpy file [ms]",
	)
	parser.add_argument(
		"--tscrunch",
		nargs="*",
		type=int,
		default=None,
		help="Time scrunch factor(s) applied before optimisation: average every N time bins. "
		"Give one value for a global scrunch of the full dataset (default: 1 = none), or one "
		"value per component (must equal the number of peak regions) to apply different factors. "
		"--peak-indices are in the original time resolution and are scaled automatically.",
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
		help="Manually specify peak indices as pairs: start1 end1 start2 end2 ... "
		"(indices in the original time resolution; scaled to the scrunched axis when --tscrunch is used)",
	)
	parser.add_argument(
		"--dedisp-mode",
		type=str,
		choices=["expand", "expand_zero", "expand_nan", "crop"],
		default="expand",
		help="Dedispersion mode: 'expand' (fill edges with noise), 'expand_zero' (zero-fill new bins), 'expand_nan' (NaN-fill new bins), or 'crop' (trim to common valid region)",
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
		"--pa-min-run",
		type=int,
		default=3,
		help="Minimum consecutive PA samples required to keep a run (default: 3)",
	)
	parser.add_argument(
		"--pa-weight-strength", "--pa-weight",
		type=float,
		default=1.0,
		help="Strength of PA fit weighting (power on normalised weights; 1.0 = current behaviour, >1 stronger)",
	)
	parser.add_argument(
		"--pa-fit-post-peak-only", "--pa-post-peak",
		action="store_true",
		help="Restrict PA fitting/masking to samples at or after the Stokes-I peak (default uses pre-peak too)",
	)
	parser.add_argument(
		"--pa",
		action="store_true",
		help="Enable PA workflow flag (PA methods already run automatically when Q/U are provided; kept for CLI compatibility)",
	)
	parser.add_argument(
		"--nonshrine-kc-smooth", "--nonshrine-smooth",
		action="store_true",
		help="Apply SHRINE-style kc low-pass smoothing to PA/LI methods",
	)
	parser.add_argument(
		"--nonshrine-shrine-like-errors", "--nonshrine-errors",
		action="store_true",
		help="Use SHRINE-style relative-uncertainty error bars for non-SHRINE PA/LI methods without requiring kc smoothing",
	)
	parser.add_argument(
		"--nonshrine-kc",
		nargs='*',
		type=int,
		default=None,
		help="Fixed kc value(s) for non-SHRINE smoothing, one per segment (default: auto per segment)",
	)
	parser.add_argument(
		"--shrine-kc",
		nargs='*',
		type=int,
		default=None,
		help="Fixed kc value(s) for SHRINE structure/S/N methods, one per segment (default: auto per segment)",
	)
	parser.add_argument(
		"--sync-kc",
		action="store_true",
		help="Set non-SHRINE kc to match SHRINE kc (overrides --nonshrine-kc)",
	)
	parser.add_argument(
		"--nonshrine-kc-minimise-uncertainty", "--nonshrine-min-uncert",
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
		"--methods",
		nargs="+",
		choices=["structure", "structure-l", "snr", "min-uncertainty", "minimise-uncertainty", "pa", "pa-shrine", "li"],
		default=None,
		help="Methods to run (default: all). Choices: structure, structure-l, snr, min-uncertainty, minimise-uncertainty, pa, pa-shrine, li",
	)
	parser.add_argument(
		"--single-method",
		type=str,
		choices=["structure", "structure-l", "snr", "min-uncertainty", "minimise-uncertainty", "pa", "pa-shrine", "li"],
		default=None,
		help="Run only this one method and produce a single-method segment comparison plot "
		"(max-range overview row + one row per segment). Overrides --methods.",
	)
	parser.add_argument(
		"--exclude-methods",
		nargs="+",
		choices=["structure", "structure-l", "snr", "min-uncertainty", "minimise-uncertainty", "pa", "pa-shrine", "li"],
		default=None,
		help="Methods to exclude from run/plots/analysis",
	)
	parser.add_argument(
		"--disable-method-errors", "--no-method-err",
		nargs="+",
		choices=["structure", "structure-l", "snr", "min_uncertainty", "minimise_uncertainty", "pa", "pa-shrine", "li"],
		default=None,
		help="Exclude specified methods from component DM/dn_e comparison plots and disable "
		"their uncertainty overlays in per-segment comparison plots "
		"(aliases: structure, snr, min-uncertainty, minimise-uncertainty, pa, pa-shrine, li)",
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
	parser.add_argument(
		"--error-plots",
		nargs="+",
		choices=[
			"comparison-summary",
			"comparison-scan",
			"comparison-overlay",
			"component-dm",
			"component-dne",
			"none",
		],
		default=None,
		help=(
			"Select which plots show uncertainty/errors. "
			"Choices: comparison-summary, comparison-scan, comparison-overlay, "
			"component-dm, component-dne, none. Default: all enabled."
		),
	)
	parser.add_argument(
		"--structure-max-cubes-dir", "--struct-dir",
		default=None,
		help=(
			"Optional output directory for saving per-component Stokes cubes at the "
			"SHRINE structure-max DM. Files are saved as .npy arrays with shape "
			"(n_stokes, freq, time)."
		),
	)
	parser.add_argument('--pub-col', type=float, default=1, help='Publication figure column count (1, 2, 3, ...). Default: 1')
	parser.add_argument(
		"--plot-range",
		action="store_true",
		help="Save a 1×3 waterfall plot showing the burst dedispersed to the lower-bound, optimum, "
		"and upper-bound DM for the structure method",
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

	factors = [1] if not args.tscrunch else [int(f) for f in args.tscrunch]
	for f in factors:
		if f < 1:
			raise ValueError(f"--tscrunch values must be >= 1, got {f}")
	per_component_scrunch = len(factors) > 1

	all_error_plot_targets = {
		"comparison-summary",
		"comparison-scan",
		"comparison-overlay",
		"component-dm",
		"component-dne",
	}
	if args.error_plots is None:
		error_plot_targets = set(all_error_plot_targets)
	else:
		requested_error_targets = set(args.error_plots)
		if "none" in requested_error_targets:
			error_plot_targets = set()
		else:
			error_plot_targets = requested_error_targets

	show_comparison_summary_errors = "comparison-summary" in error_plot_targets
	show_comparison_scan_uncertainty = "comparison-scan" in error_plot_targets
	show_comparison_overlay_uncertainty = "comparison-overlay" in error_plot_targets
	show_component_dm_errors = "component-dm" in error_plot_targets
	show_component_dne_errors = "component-dne" in error_plot_targets
	
	# Load data
	print("\nLoading data...")
	print("Using files:")
	print(f"  - Stokes I: {args.stokes_i}")
	if args.stokes_q:
		print(f"  - Stokes Q: {args.stokes_q}")
	if args.stokes_u:
		print(f"  - Stokes U: {args.stokes_u}")
	if args.stokes_v:
		print(f"  - Stokes V: {args.stokes_v}")
	print(f"  - Frequency: {args.freq}")
	print(f"  - Time: {args.time}")
	if args.dm_step is not None:
		print(f"  - DM step for scan: {args.dm_step} pc cm⁻³")
	if args.ref_freq is not None:
		print(f"  - Reference frequency override: {args.ref_freq} MHz")
	if args.input_dm:
		print(f"  - Input data already dedispersed at DM: {args.input_dm} pc cm⁻³")
	if per_component_scrunch:
		print(f"  - Time scrunch factors (per component): {factors}")
	elif factors[0] > 1:
		print(f"  - Time scrunch factor: {factors[0]}")
	print(f"  - PA weight strength: {args.pa_weight_strength}")
	print(f"  - PA fit post-peak only: {args.pa_fit_post_peak_only}")
	print(f"  - PA min run: {args.pa_min_run}")
	print(f"  - Non-SHRINE kc smoothing: {args.nonshrine_kc_smooth}")
	print(f"  - Non-SHRINE SHRINE-like errors: {args.nonshrine_shrine_like_errors}")
	if args.sync_kc:
		if args.shrine_kc is not None:
			if args.nonshrine_kc is not None:
				print("  - --sync-kc overrides --nonshrine-kc")
			args.nonshrine_kc = args.shrine_kc.copy()
			print(f"  - Non-SHRINE kc synced to SHRINE kc: {args.nonshrine_kc}")
		else:
			print("  - --sync-kc set: non-SHRINE kc will be synced from SHRINE auto-detected kc")
	elif args.nonshrine_kc is not None:
		print(f"  - Non-SHRINE kc value: {args.nonshrine_kc}")
	if args.shrine_kc is not None:
		print(f"  - SHRINE structure/S/N kc value: {args.shrine_kc}")
	print(f"  - Non-SHRINE kc via minimise_uncertainty: {args.nonshrine_kc_minimise_uncertainty}")
	print(f"  - L/I sigma cutoff: {args.li_sig}")
	if args.methods is not None:
		print(f"  - Included methods (CLI): {', '.join(args.methods)}")
	if args.exclude_methods is not None:
		print(f"  - Excluded methods (CLI): {', '.join(args.exclude_methods)}")
	if args.seed is not None:
		print(f"  - Random seed: {args.seed}")
	if args.structure_max_cubes_dir:
		print(f"  - Structure-max Stokes cube output dir: {args.structure_max_cubes_dir}")
	if len(error_plot_targets) == 0:
		print("  - Error/uncertainty overlays on plots: none")
	else:
		print(f"  - Error/uncertainty overlays on plots: {', '.join(sorted(error_plot_targets))}")
	
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
			stokes_v = cube[3] if cube.shape[0] == 4 else None
		# (freq, time, 4) or (freq, time, 3)
		elif cube.shape[2] in (3, 4):
			stokes_i = cube[..., 0]
			stokes_q = cube[..., 1]
			stokes_u = cube[..., 2]
			stokes_v = cube[..., 3] if cube.shape[2] == 4 else None
		else:
			raise ValueError(f"Unrecognized stokes-cube layout: {cube.shape}. Expected first or last axis length 3 or 4.")
	else:
		stokes_i = np.load(args.stokes_i)
		stokes_q = np.load(args.stokes_q) if args.stokes_q else None
		stokes_u = np.load(args.stokes_u) if args.stokes_u else None
		stokes_v = np.load(args.stokes_v) if args.stokes_v else None
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
		if stokes_v is not None:
			stokes_v = stokes_v[order]

	if (stokes_q is None) != (stokes_u is None):
		print("\nWarning: both --stokes-q and --stokes-u are required to run Q/U-based metrics. Skipping them.")
		stokes_q = None
		stokes_u = None

	if not per_component_scrunch and factors[0] > 1:
		print(f"\nScrunching in time by factor {factors[0]} (pre-optimisation)...")
		n_time_orig = time_ms.size
		stokes_i = tscrunch_arrays(stokes_i, factors[0])
		if stokes_q is not None:
			stokes_q = tscrunch_arrays(stokes_q, factors[0])
		if stokes_u is not None:
			stokes_u = tscrunch_arrays(stokes_u, factors[0])
		if stokes_v is not None:
			stokes_v = tscrunch_arrays(stokes_v, factors[0])
		time_ms = tscrunch_arrays(time_ms, factors[0])
		print(f"  Time samples: {n_time_orig} -> {time_ms.size}")
		if args.peak_indices is not None:
			orig_indices = list(args.peak_indices)
			args.peak_indices = rescale_peak_indices(orig_indices, factors[0])
			print(f"  Peak indices scaled {orig_indices} -> {args.peak_indices} (scrunched axis)")

	method_alias_to_key = {
		'structure': 'structure',
		'structure-l': 'structure_L',
		'snr': 'snr',
		'min-uncertainty': 'min_uncertainty',
		'minimise-uncertainty': 'min_uncertainty',
		'pa': 'pa_slope',
		'pa-shrine': 'pa_slope_shrine',
		'li': 'l_i_mean',
	}
	default_method_order = ['structure', 'snr', 'min_uncertainty', 'pa_slope', 'pa_slope_shrine', 'l_i_mean', 'structure_L']
	if args.single_method is not None:
		single_method_key = method_alias_to_key[args.single_method]
		selected_method_keys = [single_method_key]
	elif args.methods is None:
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

	# Map disable-method-errors aliases (if provided) to internal method keys
	if args.disable_method_errors is None:
		disabled_method_keys = set()
	else:
		disabled_method_keys = {method_alias_to_key[alias] for alias in args.disable_method_errors}

	if (stokes_q is None or stokes_u is None):
		qu_methods = {'pa_slope', 'pa_slope_shrine', 'l_i_mean', 'structure_L'}
		removed_qu = [m for m in selected_method_keys if m in qu_methods]
		if len(removed_qu) > 0:
			selected_method_keys = [m for m in selected_method_keys if m not in qu_methods]
			print("\nWarning: removed Q/U-based methods from selection because Stokes Q/U are unavailable.")

	if len(selected_method_keys) == 0:
		raise ValueError("No methods selected to run after applying include/exclude filters")

	method_key_to_name = {
		'structure': 'structure',
		'structure_L': 'structure-l',
		'snr': 'snr',
		'min_uncertainty': 'min-uncertainty',
		'pa_slope': 'pa',
		'pa_slope_shrine': 'pa-shrine',
		'l_i_mean': 'li',
	}
	selected_method_labels = [method_key_to_name[m] for m in selected_method_keys]
	print(f"  Active methods: {', '.join(selected_method_labels)}")
	if disabled_method_keys:
		disabled_labels = [method_key_to_name.get(k, k) for k in sorted(disabled_method_keys)]
		print(f"  - Omitted from component DM/dn_e plots: {', '.join(disabled_labels)}")

	print(f"\nData loaded successfully!")
	print(f"  Shape: {stokes_i.shape} (freq x time)")
	print(f"  Frequency range: {freq_mhz[0]:.1f} - {freq_mhz[-1]:.1f} MHz")
	print(f"  Time range: {time_ms[0]:.3f} - {time_ms[-1]:.3f} ms")
		
	
	# Initialize optimiser
	optimiser = build_optimiser(stokes_i, freq_mhz, time_ms, stokes_q, stokes_u, stokes_v, args)

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

	if per_component_scrunch and len(factors) != len(peak_regions):
		raise ValueError(
			f"--tscrunch provides {len(factors)} scrunch factor(s) but "
			f"{len(peak_regions)} peak region(s) were found; "
			f"provide one factor per component."
		)

	label = "Peak" if args.separate_peaks else "Segment"
	fig_ext = args.ext.strip().lower().lstrip('.') or 'png'

	set_pub_col(args.pub_col)
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
		segment_tag = f"{label.lower()}{i+1}"

		# Per-component scrunch: build a dedicated optimiser for this component.
		if per_component_scrunch:
			f = factors[i]
			if f > 1:
				seg_stokes_i = tscrunch_arrays(stokes_i, f)
				seg_stokes_q = None if stokes_q is None else tscrunch_arrays(stokes_q, f)
				seg_stokes_u = None if stokes_u is None else tscrunch_arrays(stokes_u, f)
				seg_stokes_v = None if stokes_v is None else tscrunch_arrays(stokes_v, f)
				seg_time = tscrunch_arrays(time_ms, f)
				seg_opt = build_optimiser(seg_stokes_i, freq_mhz, seg_time,
										  seg_stokes_q, seg_stokes_u, seg_stokes_v, args)
				seg_region = (peak_region[0] // f, -(-peak_region[1] // f))
				print(f"  Time scrunch factor {f}: region {peak_region} -> {seg_region}")
			else:
				seg_opt, seg_region = optimiser, peak_region
		else:
			seg_opt, seg_region = optimiser, peak_region

		# Compare methods
		results = seg_opt.compare_methods(dm_range, seg_region, n_points=grid_n_points, dm_step=args.dm_step, 
										segment_tag=segment_tag,
											label=args.label,
											selected_methods=selected_method_keys,
											segment_index=i)
		if per_component_scrunch and factors[i] > 1:
			dt_seg = float(np.nanmedian(np.diff(seg_opt.time_ms)))
			for mkey, mres in results.items():
				if mres.get('dedispersed') is None:
					continue
				n_time_out = mres['dedispersed'].shape[1]
				delay_samples = seg_opt._get_delay_samples(mres['dm'])
				if seg_opt.dedisp_mode == 'crop':
					start_shift = int(np.max(delay_samples))
				else:
					start_shift = int(np.min(delay_samples))
				base_start = seg_opt.time_ms[min(seg_region[0], len(seg_opt.time_ms) - 1)]
				mres['time_ms'] = base_start + start_shift * dt_seg + np.arange(n_time_out) * dt_seg
		all_results.append(results)
		
		# Print results
		print(f"\nResults for {label} {i+1}:")
		for method_name, result in results.items():
			print(f"  {result['method']}:")
			print(
				"    Optimal DM: "
				+ seg_opt._format_uncertainty(
					result['dm'],
					result.get('uncertainty_minus'),
					result.get('uncertainty_plus'),
				)
				+ " pc cm⁻³"
			)
			print(f"    Metric value: {result['metric']:.6f}")
			if result.get('kc') is not None:
				print(f"    kc: {result['kc']}")
			if result.get('uncertainty_method') is not None:
				print(f"    Uncertainty method: {result.get('uncertainty_method')}")
		
		# Plot comparison (skipped in single-method mode; one combined plot is
		# generated after the segment loop).
		if args.single_method is None:
			print(f"\nGenerating comparison plot for {label} {i+1}...")
			seg_opt.plot_comparison(
				results,
				dm_range,
				seg_region,
				label=args.label,
				save_path=f'{args.label}_dm_comparison_{label.lower()}{i+1}.{fig_ext}',
				show_summary_errors=show_comparison_summary_errors,
				show_scan_uncertainty=show_comparison_scan_uncertainty,
				show_overlay_uncertainty=show_comparison_overlay_uncertainty,
				disabled_error_methods=disabled_method_keys,
			)

		if args.plot_range and 'structure' in results:
			range_path = f'{args.label}_dm_range_{segment_tag}.{fig_ext}'
			seg_opt.plot_range(
				results['structure'],
				peak_region=seg_region,
				label=args.label,
				save_path=range_path,
			)

		seg_opt.save_nonshrine_L_dm_diagnostics(
			label=args.label,
			segment_tag=segment_tag,
		)

		if args.structure_max_cubes_dir:
			output_dir = Path(args.structure_max_cubes_dir)
			output_dir.mkdir(parents=True, exist_ok=True)
			structure_result = results.get('structure')
			if structure_result is None:
				print("  - Skipping structure-max cube save (structure method not run).")
			else:
				dedisp_i = structure_result.get('dedispersed')
				# Save freq and time arrays for this component
				freq_out = output_dir / f"{args.label}_{segment_tag}_structure_max_freq.npy"
				np.save(freq_out, np.asarray(freq_mhz, dtype=float))
				print(f"  - Saved structure-max freq: {freq_out}")
				if dedisp_i is not None:
					n_time_out = dedisp_i.shape[1]
					dt = float(np.median(np.diff(seg_opt.time_ms)))
					delay_samples = seg_opt._get_delay_samples(structure_result['dm'])
					dedisp_mode = args.dedisp_mode
					if dedisp_mode == 'crop':
						start_idx = int(np.max(delay_samples))
						end_idx = int(seg_region[1] - seg_region[0] + np.min(delay_samples))
						time_out = seg_opt.time_ms[seg_region[0] + start_idx : seg_region[0] + start_idx + n_time_out]
					else:
						min_shift = int(np.min(delay_samples))
						time_out = seg_opt.time_ms[seg_region[0]] + (np.arange(n_time_out) + min_shift) * dt
					time_path = output_dir / f"{args.label}_{segment_tag}_structure_max_time.npy"
					np.save(time_path, np.asarray(time_out, dtype=float))
					print(f"  - Saved structure-max time: {time_path}")
				dedisp_q = structure_result.get('dedispersed_q')
				dedisp_u = structure_result.get('dedispersed_u')
				dedisp_v = structure_result.get('dedispersed_v')
				if dedisp_i is not None:
					out_path_i = output_dir / f"{args.label}_{segment_tag}_structure_max_I.npy"
					flipped_i = np.flip(np.asarray(dedisp_i, dtype=float), axis=0)
					np.save(out_path_i, flipped_i)
					print(f"  - Saved structure-max Stokes I: {out_path_i}")
				if dedisp_q is not None:
					out_path_q = output_dir / f"{args.label}_{segment_tag}_structure_max_Q.npy"
					flipped_q = np.flip(np.asarray(dedisp_q, dtype=float), axis=0)
					np.save(out_path_q, flipped_q)
					print(f"  - Saved structure-max Stokes Q: {out_path_q}")
				if dedisp_u is not None:
					out_path_u = output_dir / f"{args.label}_{segment_tag}_structure_max_U.npy"
					flipped_u = np.flip(np.asarray(dedisp_u, dtype=float), axis=0)
					np.save(out_path_u, flipped_u)
					print(f"  - Saved structure-max Stokes U: {out_path_u}")
				if dedisp_v is not None:
					out_path_v = output_dir / f"{args.label}_{segment_tag}_structure_max_V.npy"
					flipped_v = np.flip(np.asarray(dedisp_v, dtype=float), axis=0)
					np.save(out_path_v, flipped_v)
					print(f"  - Saved structure-max Stokes V: {out_path_v}")
				cube_parts = []
				if dedisp_i is not None:
					cube_parts.append(np.flip(np.asarray(dedisp_i, dtype=float), axis=0))
				if dedisp_q is not None:
					cube_parts.append(np.flip(np.asarray(dedisp_q, dtype=float), axis=0))
				if dedisp_u is not None:
					cube_parts.append(np.flip(np.asarray(dedisp_u, dtype=float), axis=0))
				if dedisp_v is not None:
					cube_parts.append(np.flip(np.asarray(dedisp_v, dtype=float), axis=0))
				if len(cube_parts) == 0:
					print("  - Skipping structure-max cube save (no dedispersed Stokes data available).")
				else:
					stokes_cube = np.stack(cube_parts, axis=0)
					out_path = output_dir / f"{args.label}_{segment_tag}_structure_max_stokes.npy"
					np.save(out_path, stokes_cube)
					print(f"  - Saved structure-max Stokes cube: {out_path}")

	if args.single_method is not None and len(all_results) > 0:
		print(f"\nGenerating single-method comparison plot for {args.single_method}...")
		optimiser.plot_single_method_comparison(
			single_method_key,
			all_results,
			dm_range,
			peak_regions,
			label=args.label,
			save_path=f'{args.label}_dm_comparison_single_{method_key_to_name[single_method_key]}.{fig_ext}',
			show_errors=show_comparison_summary_errors,
		)

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

		print(f"\nGenerating multi-{label.lower()} DM diagnostics plot (time-ordered)...")
		optimiser.plot_component_dm_diagnostics(
			sorted_all_results,
			component_times_ms=sorted_peak_times_ms,
			label=label.lower(),
			frb_label=args.label,
			save_path=f'{args.label}_dm_component_dm_diagnostics.{fig_ext}',
			show_errors=show_component_dm_errors,
			excluded_methods=disabled_method_keys,
		)

		dne_diag = optimiser.calculate_dn_e_between_components(
			sorted_all_results,
			component_times_ms=sorted_peak_times_ms,
			comparison='adjacent',
			excluded_methods=disabled_method_keys,
		)

		# Re-label pairs with original segment/component IDs after time-order sorting.
		pair_labels_time = [
			f"{idx_b + 1}-{idx_a + 1}"
			for idx_a, idx_b in dne_diag['pair_indices']
		]
		dne_diag['pair_labels'] = pair_labels_time
		print("\nComponent-to-component dn_e diagnostics (adjacent pairs, L~cΔt):")
		for i, pair_label in enumerate(dne_diag['pair_labels']):
			sep_pc = float(dne_diag['pair_separations_pc'][i])
			print(f"  {pair_label}: L = {sep_pc:.6e} pc")
			print(f"  {pair_label}: L = {(sep_pc * u.pc).to(u.km).value:.6e} km")
			for method_name, method_vals in dne_diag['methods'].items():
				delta_dm = float(method_vals['delta_dm'][i])
				dm_err_minus = float(method_vals['delta_dm_sigma_minus'][i])
				dm_err_plus = float(method_vals['delta_dm_sigma_plus'][i])
				dn_e = float(method_vals['dn_e'][i])
				dne_err_minus = float(method_vals['dn_e_sigma_minus'][i])
				dne_err_plus = float(method_vals['dn_e_sigma_plus'][i])
				print(
					f"    {method_name}: "
					f"ΔDM={delta_dm:.6f} (-{dm_err_minus:.6f}, +{dm_err_plus:.6f}) pc cm⁻³, "
					f"dn_e={dn_e:.6e} (-{dne_err_minus:.6e}, +{dne_err_plus:.6e}) cm⁻³"
				)

		dne_path = Path(f'{args.label}_dm_component_dne_diagnostics_{label.lower()}.txt')
		with open(dne_path, 'w') as f:
			f.write("# dn_e diagnostics between components\n")
			f.write("# Assumption: L ~ c * Delta t using component peak arrival times\n")
			f.write("# Columns: pair method separation_pc delta_dm delta_dm_low delta_dm_high dn_e dn_e_low dn_e_high\n")
			for i, pair_label in enumerate(dne_diag['pair_labels']):
				sep_pc = float(dne_diag['pair_separations_pc'][i])
				for method_name, method_vals in dne_diag['methods'].items():
					f.write(
						f"{pair_label} {method_name} {sep_pc:.10e} "
						f"{float(method_vals['delta_dm'][i]):.10e} "
						f"{float(method_vals['delta_dm_sigma_minus'][i]):.10e} "
						f"{float(method_vals['delta_dm_sigma_plus'][i]):.10e} "
						f"{float(method_vals['dn_e'][i]):.10e} "
						f"{float(method_vals['dn_e_sigma_minus'][i]):.10e} "
						f"{float(method_vals['dn_e_sigma_plus'][i]):.10e}\n"
					)
		print(f"Saved dn_e diagnostics to: {dne_path}")

		dne_plot_path = f'{args.label}_dm_component_dne_diagnostics_{label.lower()}.{fig_ext}'
		optimiser.plot_component_dne_diagnostics(
			dne_diag,
			label=label.lower(),
			frb_label=args.label,
			save_path=dne_plot_path,
			show_errors=show_component_dne_errors,
			excluded_methods=disabled_method_keys,
		)
	
	print("\n" + "="*70)
	print("Analysis complete!")
	print("="*70)
	print("\nGenerated files:")
	print(f"  - {args.label}_dm_comparison_{label.lower()}*.{fig_ext}: Comparison of methods for each {label.lower()}")
	if len(all_results) > 1:
		print(f"  - {args.label}_dm_component_dm_diagnostics.{fig_ext}: Multi-{label.lower()} DM diagnostics")
		#print(f"  - {args.label}_dm_component_dne_diagnostics_{label.lower()}.txt: dn_e diagnostics (L~cΔt)")
		print(f"  - {args.label}_dm_component_dne_diagnostics_{label.lower()}.{fig_ext}: dn_e plot between components")


if __name__ == "__main__":
	main()
