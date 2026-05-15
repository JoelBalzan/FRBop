"""
Command-line interface for DM correction optimisation.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DM correction optimisation with configurable inputs",
    )
    parser.add_argument(
        "-d", "--stokes-cube", default=None,
        help=(
            "Optional path to a single numpy file containing Stokes I/Q/U(/V) cube. "
            "Accepted shapes: (4,freq,time), (3,freq,time), (freq,time,4) or (freq,time,3)"
        ),
    )
    parser.add_argument("--stokes-i", default=None, help="Path to Stokes I numpy file (freq x time)")
    parser.add_argument("--stokes-q", default=None, help="Optional path to Stokes Q numpy file (freq x time)")
    parser.add_argument("--stokes-u", default=None, help="Optional path to Stokes U numpy file (freq x time)")
    parser.add_argument("--freq", default="freq.npy", help="Path to frequency array numpy file (MHz)")
    parser.add_argument("--time", default="time.npy", help="Path to time array numpy file (ms)")
    parser.add_argument("--dm-min", type=float, default=None, help="Minimum DM to search (pc cm⁻³)")
    parser.add_argument("--dm-max", type=float, default=None, help="Maximum DM to search (pc cm⁻³)")
    parser.add_argument("--dm-guess", type=float, default=None,
                        help="Starting DM guess (pc cm⁻³) used to build a default range")
    parser.add_argument("--dm-step", type=float, default=None,
                        help="DM step for DM-space scan (pc cm⁻³). Overrides default sampling.")
    parser.add_argument("--ref-freq", type=float, default=None,
                        help="Reference frequency in MHz for dedispersion (defaults to max frequency in file)")
    parser.add_argument("--input-dm", type=float, default=0.0,
                        help="DM already applied to the input data (pc cm⁻³)")
    parser.add_argument("--separate-peaks", action="store_true",
                        help="Enable peak separation; otherwise, operate on the full dataset")
    parser.add_argument("--manual-peaks", action="store_true",
                        help="Manually select peak bounds by clicking on the pulse profile")
    parser.add_argument("--peak-indices", nargs="*", type=int, default=None,
                        help="Manually specify peak indices as pairs: start1 end1 start2 end2 ...")
    parser.add_argument("--dedisp-mode", type=str, choices=["expand", "crop"], default="expand",
                        help="Dedispersion mode: 'expand' (fill edges with noise) or 'crop'")
    parser.add_argument("--fast", action="store_true",
                        help="Speedy test mode: uses fewer DM samples")
    parser.add_argument("--pa-fit-degree", type=int, default=1,
                        help="Polynomial degree for PA profile fitting (default: 1)")
    parser.add_argument("--pa-weight-strength", type=float, default=1.0,
                        help="Strength of PA fit weighting (power on normalised weights)")
    parser.add_argument("--pa-fit-post-peak-only", action="store_true",
                        help="Restrict PA fitting to samples at or after the Stokes-I peak")
    parser.add_argument("--pa", action="store_true",
                        help="Enable PA workflow flag (kept for CLI compatibility)")
    parser.add_argument("--nonshrine-kc-smooth", action="store_true",
                        help="Apply SHRINE-style kc low-pass smoothing to PA/LI methods")
    parser.add_argument("--nonshrine-shrine-like-errors", action="store_true",
                        help="Use SHRINE-style relative-uncertainty error bars for non-SHRINE methods")
    parser.add_argument("--nonshrine-kc", type=int, default=None,
                        help="Fixed kc value for non-SHRINE smoothing (default: auto)")
    parser.add_argument("--nonshrine-kc-minimise-uncertainty", action="store_true",
                        help="Find non-SHRINE kc via minimise_uncertainty.py")
    parser.add_argument("--li-sig", type=float, default=2.0,
                        help="Stokes I sigma cutoff for L/I mean masking (default: 2.0)")
    parser.add_argument("--debias-linear", action="store_true",
                        help="Enable linear-polarisation debiasing for PA/L/I metrics.")
    parser.add_argument(
        "--methods", nargs="+",
        choices=["structure", "snr", "pa", "pa-shrine", "li"],
        default=None,
        help="Methods to run (default: all). Choices: structure, snr, pa, pa-shrine, li",
    )
    parser.add_argument(
        "--exclude-methods", nargs="+",
        choices=["structure", "snr", "pa", "pa-shrine", "li"],
        default=None,
        help="Methods to exclude from run/plots/analysis",
    )
    parser.add_argument("--label", type=str, default="frb",
                        help="FRB label for output files (default: 'frb')")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for reproducible noise fill. Default: 1234")
    parser.add_argument("--ext", type=str, default="png",
                        help="Figure extension for saved plots (e.g. png, pdf, svg). Default: png")
    return parser.parse_args()


def _load_stokes(args: argparse.Namespace):
    """
    Load Stokes arrays from CLI arguments.

    Returns ``(stokes_i, stokes_q, stokes_u, freq_mhz, time_ms)``.
    """
    if args.stokes_cube:
        cube = np.load(args.stokes_cube)
        if cube.ndim != 3:
            raise ValueError(f"stokes-cube must be a 3-D numpy array, got shape {cube.shape}")
        if cube.shape[0] in (3, 4):
            stokes_i, stokes_q, stokes_u = cube[0], cube[1], cube[2]
        elif cube.shape[2] in (3, 4):
            stokes_i, stokes_q, stokes_u = cube[..., 0], cube[..., 1], cube[..., 2]
        else:
            raise ValueError(
                f"Unrecognized stokes-cube layout: {cube.shape}. "
                "Expected first or last axis length 3 or 4."
            )
    else:
        stokes_i = np.load(args.stokes_i)
        stokes_q = np.load(args.stokes_q) if args.stokes_q else None
        stokes_u = np.load(args.stokes_u) if args.stokes_u else None

    freq_mhz = np.load(args.freq)
    time_ms = np.load(args.time)

    if freq_mhz.ndim != 1:
        raise ValueError("Frequency array must be 1-D")

    # Ensure frequency axis is sorted ascending
    if not np.all(np.diff(freq_mhz) >= 0):
        order = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[order]
        stokes_i = stokes_i[order]
        if stokes_q is not None:
            stokes_q = stokes_q[order]
        if stokes_u is not None:
            stokes_u = stokes_u[order]

    # Both Q and U must be present together or not at all
    if (stokes_q is None) != (stokes_u is None):
        print("\nWarning: both --stokes-q and --stokes-u are required. Skipping Q/U methods.")
        stokes_q = None
        stokes_u = None

    return stokes_i, stokes_q, stokes_u, freq_mhz, time_ms


def _resolve_methods(args: argparse.Namespace, has_qu: bool) -> List[str]:
    """Return the ordered list of internal method keys to run."""
    alias_to_key = {
        "structure": "structure",
        "snr": "snr",
        "pa": "pa_slope",
        "pa-shrine": "pa_slope_shrine",
        "li": "l_i_mean",
    }
    default_order = ["structure", "snr", "pa_slope", "pa_slope_shrine", "l_i_mean"]

    if args.methods is None:
        selected = default_order.copy()
    else:
        selected = []
        for alias in args.methods:
            key = alias_to_key[alias]
            if key not in selected:
                selected.append(key)

    if args.exclude_methods:
        exclude = {alias_to_key[a] for a in args.exclude_methods}
        selected = [m for m in selected if m not in exclude]

    if not has_qu:
        qu_methods = {"pa_slope", "pa_slope_shrine", "l_i_mean"}
        removed = [m for m in selected if m in qu_methods]
        if removed:
            print("\nWarning: removed Q/U-based methods (Stokes Q/U unavailable):", removed)
        selected = [m for m in selected if m not in qu_methods]

    if not selected:
        raise ValueError("No methods selected after applying include/exclude filters")

    return selected


def _resolve_dm_range(args: argparse.Namespace) -> Tuple[float, float]:
    if args.dm_min is not None and args.dm_max is not None:
        dm_range = (args.dm_min, args.dm_max)
        print(f"\nDM search range set from flags: {dm_range[0]:.1f} – {dm_range[1]:.1f} pc cm⁻³")
    elif args.dm_guess is not None:
        span = 50.0
        dm_range = (args.dm_guess - span, args.dm_guess + span)
        print(f"\nDM search range built around guess {args.dm_guess:.1f} ± {span:.1f} pc cm⁻³")
    else:
        dm_range = (300.0, 400.0)
        print(f"\nDM search range defaulting to {dm_range[0]} – {dm_range[1]} pc cm⁻³")
    if args.dm_min is not None and args.dm_max is None:
        print("Warning: --dm-min provided without --dm-max; ignoring.")
    if args.dm_max is not None and args.dm_min is None:
        print("Warning: --dm-max provided without --dm-min; ignoring.")
    return dm_range


def main() -> None:
    """Entry point: parse arguments, run optimisation, save outputs."""
    # Imports deferred to keep startup fast
    from frbop.utils.plotting import set_pub_style
    from frbop.utils.peaks import parse_peak_index_pairs
    import matplotlib.pyplot as plt

    from .optimiser import DMOptimiser
    from .uncertainty import format_uncertainty

    print("=" * 70)
    print("DM Correction Optimisation Methods Comparison")
    print("=" * 70)

    args = parse_args()

    # ---- Load data ----
    print("\nLoading data...")
    stokes_i, stokes_q, stokes_u, freq_mhz, time_ms = _load_stokes(args)

    print(f"\nData loaded successfully!")
    print(f"  Shape: {stokes_i.shape} (freq × time)")
    print(f"  Frequency range: {freq_mhz[0]:.1f} – {freq_mhz[-1]:.1f} MHz")
    print(f"  Time range: {time_ms[0]:.3f} – {time_ms[-1]:.3f} ms")

    has_qu = stokes_q is not None and stokes_u is not None
    selected_method_keys = _resolve_methods(args, has_qu)

    key_to_label = {
        "structure": "structure", "snr": "snr",
        "pa_slope": "pa", "pa_slope_shrine": "pa-shrine", "l_i_mean": "li",
    }
    print(f"  Active methods: {', '.join(key_to_label[m] for m in selected_method_keys)}")
    print(f"  PA weight strength: {args.pa_weight_strength}")
    print(f"  PA fit post-peak only: {args.pa_fit_post_peak_only}")
    print(f"  Linear debiasing: {args.debias_linear}")
    print(f"  Non-SHRINE kc smoothing: {args.nonshrine_kc_smooth}")
    print(f"  Non-SHRINE SHRINE-like errors: {args.nonshrine_shrine_like_errors}")
    if args.nonshrine_kc is not None:
        print(f"  Non-SHRINE kc value: {args.nonshrine_kc}")
    print(f"  L/I sigma cutoff: {args.li_sig}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")

    # ---- Initialise optimiser ----
    optimiser = DMOptimiser(
        stokes_i, freq_mhz, time_ms,
        stokes_q=stokes_q, stokes_u=stokes_u,
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
    print(f"  Reference frequency: {optimiser.reference_freq:.3f} MHz")

    recommended_dm_step = optimiser.recommend_lowest_dm_step()
    print(f"  Recommended lowest DM step (~1 sample): {recommended_dm_step:.6g} pc cm⁻³")
    if args.dm_step is None:
        print(f"  Using recommended DM step: {recommended_dm_step:.6g} pc cm⁻³")
        args.dm_step = recommended_dm_step

    dm_range = _resolve_dm_range(args)

    # ---- Peak handling ----
    print("\n" + "=" * 70)
    if args.peak_indices is not None:
        print("Using manually specified peak indices...")
        peak_regions = parse_peak_index_pairs(args.peak_indices, stokes_i.shape[1])
        print(f"  Specified {len(peak_regions)} peak region(s)")
        for i, (start, end) in enumerate(peak_regions):
            end_d = min(end - 1, len(time_ms) - 1)
            print(f"    Peak {i+1}: indices {start}–{end} ({time_ms[start]:.2f}–{time_ms[end_d]:.2f} ms)")
    elif args.manual_peaks:
        print("Manual peak selection enabled...")
        peak_regions = optimiser.select_peaks_manual()
        print(f"  Selected {len(peak_regions)} peak region(s)")
        for i, (start, end) in enumerate(peak_regions):
            end_d = min(end - 1, len(time_ms) - 1)
            print(f"    Peak {i+1}: indices {start}–{end} ({time_ms[start]:.2f}–{time_ms[end_d]:.2f} ms)")
    elif args.separate_peaks:
        print("Separating peaks automatically...")
        peak_regions = optimiser.separate_peaks(min_separation_ms=5.0)
        print(f"  Found {len(peak_regions)} peak(s)")
        for i, (start, end) in enumerate(peak_regions):
            print(f"    Peak {i+1}: indices {start}–{end} ({time_ms[start]:.2f}–{time_ms[end]:.2f} ms)")
    else:
        print("Skipping peak separation (processing full dataset).")
        peak_regions = [(0, stokes_i.shape[1])]

    label_word = "Peak" if args.separate_peaks else "Segment"
    fig_ext = args.ext.strip().lower().lstrip(".") or "png"

    set_pub_style(use_latex=False)

    # ---- Main loop ----
    all_results = []
    grid_n_points = 50 if args.fast else 100

    for i, peak_region in enumerate(peak_regions):
        print("\n" + "=" * 70)
        print(f"Analyzing {label_word} {i + 1}")
        print("=" * 70)

        results = optimiser.compare_methods(
            dm_range, peak_region,
            n_points=grid_n_points,
            dm_step=args.dm_step,
            segment_tag=f"{label_word.lower()}{i + 1}",
            label=args.label,
            selected_methods=selected_method_keys,
        )
        all_results.append(results)

        print(f"\nResults for {label_word} {i + 1}:")
        for method_name, result in results.items():
            print(f"  {result['method']}:")
            print(
                "    Optimal DM: "
                + format_uncertainty(
                    result["dm"],
                    result.get("uncertainty_minus"),
                    result.get("uncertainty_plus"),
                )
                + " pc cm⁻³"
            )
            print(f"    Metric value: {result['metric']:.6f}")
            if result.get("uncertainty_method"):
                print(f"    Uncertainty method: {result['uncertainty_method']}")

        print(f"\nGenerating comparison plot for {label_word} {i + 1}...")
        optimiser.plot_comparison(
            results, dm_range, peak_region,
            save_path=f"dm_comparison_{label_word.lower()}{i + 1}.{fig_ext}",
        )

    # ---- Multi-component diagnostics ----
    if len(all_results) > 1:
        # Determine component peak times for physical ordering
        component_peak_times_ms = np.zeros(len(peak_regions))
        for i, (start_idx, end_idx) in enumerate(peak_regions):
            s = min(max(start_idx, 0), stokes_i.shape[1] - 1)
            e = min(max(end_idx, s + 1), stokes_i.shape[1])
            window = np.mean(stokes_i[:, s:e], axis=0)
            peak_global = s + int(np.argmax(window))
            component_peak_times_ms[i] = float(time_ms[peak_global])

        sort_idx = np.argsort(component_peak_times_ms)
        sorted_results = [all_results[int(i)] for i in sort_idx]
        sorted_times = component_peak_times_ms[sort_idx]
        sorted_ids = (sort_idx + 1).astype(int)

        print(f"\nGenerating multi-{label_word.lower()} DM diagnostics plot...")
        optimiser.plot_component_dm_diagnostics(
            sorted_results, component_ids=sorted_ids,
            save_path=f"dm_component_dm_diagnostics.{fig_ext}",
        )

        dne_diag = optimiser.calculate_dn_e_between_components(
            sorted_results, component_times_ms=sorted_times, comparison="adjacent",
        )

        # Re-label pairs using original sorted component IDs
        pair_labels_time = [
            f"comp{int(sorted_ids[a])}->comp{int(sorted_ids[b])}"
            for a, b in dne_diag["pair_indices"]
        ]
        dne_diag["pair_labels"] = pair_labels_time

        print("\nComponent-to-component dn_e diagnostics (adjacent pairs, L~cΔt):")
        for i, pair_label in enumerate(dne_diag["pair_labels"]):
            sep_pc = float(dne_diag["pair_separations_pc"][i])
            print(f"  {pair_label}: L = {sep_pc:.6e} pc")
            for method_name, method_vals in dne_diag["methods"].items():
                ddm = float(method_vals["delta_dm"][i])
                ddm_lo = float(method_vals["delta_dm_low"][i])
                ddm_hi = float(method_vals["delta_dm_high"][i])
                dne = float(method_vals["dn_e"][i])
                dne_lo = float(method_vals["dn_e_low"][i])
                dne_hi = float(method_vals["dn_e_high"][i])
                print(
                    f"    {method_name}: "
                    f"ΔDM={ddm:.6f} [{ddm_lo:.6f}, {ddm_hi:.6f}] pc cm⁻³, "
                    f"dn_e={dne:.6e} [{dne_lo:.6e}, {dne_hi:.6e}] cm⁻³"
                )

        dne_plot_path = f"dm_component_dne_diagnostics_{label_word.lower()}.{fig_ext}"
        optimiser.plot_component_dne_diagnostics(dne_diag, save_path=dne_plot_path)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - dm_comparison_{label_word.lower()}*.{fig_ext}")
    if len(all_results) > 1:
        print(f"  - dm_component_dm_diagnostics.{fig_ext}")
        print(f"  - dm_component_dne_diagnostics_{label_word.lower()}.{fig_ext}")


if __name__ == "__main__":
    main()
