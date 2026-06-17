"""
CLI for RM fitting using the split rmop modules.
"""

import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from frbop.utils.peaks import (parse_peak_index_pairs,
                               select_frequency_bands_manual,
                               split_frequency_bands_equal,
                               split_frequency_bands_equal_snr)
from frbop.utils.plotting import set_pub_style

from .data_io import find_onpulse_window, load_stokes_data, select_peaks_manual
from .diagnostics import time_series_sigma_rm_diagnostic
from .fitter import RMFitter, fit_rm_time_series
from .plotting import (plot_burns_law_fits, plot_poincare_projections,
                       plot_poincare_projections_frequency,
                       plot_poincare_sphere, plot_poincare_sphere_frequency,
                       plot_poincare_sphere_subbands,
                       plot_polarisation_fraction_acf_ccf, plot_rm_results,
                       plot_rm_time_series)

warnings.filterwarnings("ignore")


def _resolve_freq_band_idx(
    args: argparse.Namespace,
    freq_hz: np.ndarray,
    spec_for_band: np.ndarray,
    dspec: Optional[np.ndarray] = None,
) -> Optional[slice]:
    """Return a slice for frequency band cropping, or None if no band selected."""
    if args.freq_band_mhz is not None:
        fmin, fmax = args.freq_band_mhz
        fmin_hz = fmin * 1e6
        fmax_hz = fmax * 1e6
        idx = np.where((freq_hz >= fmin_hz) & (freq_hz <= fmax_hz))[0]
        if len(idx) == 0:
            print(f"  WARNING: no channels in frequency band [{fmin}, {fmax}] MHz")
            return None
        return slice(idx[0], idx[-1] + 1)
    elif args.freq_band_indices is not None:
        start, stop = args.freq_band_indices
        n_freq = len(freq_hz)
        if start >= stop or start < 0 or stop > n_freq:
            print(f"  WARNING: invalid freq-band-indices [{start}, {stop}) for {n_freq} channels")
            return None
        return slice(start, stop)
    elif args.manual_freq_bands:
        bands = select_frequency_bands_manual(freq_hz / 1e6, spec_for_band, dspec=dspec)
        if bands is None or len(bands) == 0:
            return None
        start, stop = bands[0]
        return slice(start, stop)
    return None


def main() -> None:
    """Entry point for the RM fitting CLI."""
    parser = argparse.ArgumentParser(
        description="RM fitting for Stokes IQUV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fit single spectrum from text files\n"
            "  python -m frbop.rmop.cli -i stokes_i.txt -q stokes_q.txt -u stokes_u.txt --freq freq.txt\n\n"
            "  # Fit from .npy files with time averaging\n"
            "  python -m frbop.rmop.cli -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-avg\n\n"
            "  # Fit with custom RM range\n"
            "  python -m frbop.rmop.cli -i stokes_i.txt -q stokes_q.txt -u stokes_u.txt --rm-range -500 500\n\n"
            "  # Process 2D data as time series\n"
            "  python -m frbop.rmop.cli -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series\n\n"
            "  # Manually click to select on-pulse window\n"
            "  python -m frbop.rmop.cli -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series \\ \n"
            "      --onpulse-only --manual-peaks\n\n"
            "  # Provide peak start/end indices directly\n"
            "  python -m frbop.rmop.cli -i stokes_i.npy -q stokes_q.npy -u stokes_u.npy --time-series \\ \n"
            "      --onpulse-only --peak-indices 100 120 200 220\n"
        ),
    )

    # Input data
    parser.add_argument("-i", "--stokes-i", required=False, help="Stokes I file path")
    parser.add_argument("-q", "--stokes-q", required=False, help="Stokes Q file path")
    parser.add_argument("-u", "--stokes-u", required=False, help="Stokes U file path")
    parser.add_argument("-v", "--stokes-v", default=None, help="Stokes V file path (optional)")
    parser.add_argument(
        "--stokes-cube",
        default=None,
        help="Path to Stokes cube with components ordered I,Q,U,(V)",
    )
    parser.add_argument("--freq", default=None, help="Frequency file (.npy or .txt)")
    parser.add_argument("--time", default=None, help="Time file (.npy or .txt)")

    # Data layout and processing
    parser.add_argument("--time-series", action="store_true", help="Process as time series data")
    parser.add_argument("--time-avg", action="store_true", help="Average over time axis for 2D data")

    # RM fitting
    parser.add_argument(
        "--method",
        choices=["simple", "rm_synthesis", "qu_fitting", "rmnest"],
        default="rm_synthesis",
        help="RM fitting method (default: rm_synthesis)",
    )
    parser.add_argument(
        "--rm-range",
        nargs=2,
        type=float,
        default=[-1000, 1000],
        metavar=("MIN", "MAX"),
        help="RM search range in rad/m^2 (default: -1000 1000)",
    )
    parser.add_argument(
        "--n-rm",
        type=int,
        default=2000,
        help="Number of RM trial values (default: 2000)",
    )

    # On-pulse selection
    parser.add_argument(
        "--onpulse-only",
        action="store_true",
        help="Fit RM only within on-pulse window",
    )
    parser.add_argument(
        "--onpulse-fraction",
        type=float,
        default=0.95,
        help="Fraction of flux to include in on-pulse window (default: 0.95)",
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

    # Output and plotting
    parser.add_argument(
        "-o",
        "--output",
        default="rm_fitting_results",
        help="Output file prefix (default: rm_fitting_results)",
    )
    parser.add_argument("--ext", default="png", help="Output figure extension (default: png)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--hide-rm-frac-panel",
        action="store_true",
        help="Hide the third (L/I and V/I) panel in plot_rm_results",
    )

    # Poincare plots
    parser.add_argument(
        "--poincare",
        action="store_true",
        help="Generate Poincare sphere plot (time-dependent only; requires --time-series)",
    )
    parser.add_argument(
        "--poincare-interactive",
        action="store_true",
        help="Display Poincare plot interactively before saving",
    )
    parser.add_argument(
        "--poincare-surface",
        action="store_true",
        help="Force all Poincare points onto unit sphere surface",
    )
    parser.add_argument(
        "--poincare-subbands",
        type=int,
        default=1,
        help="Split the full band into this many subbands and plot one Poincare sphere per band",
    )
    parser.add_argument(
        "--poincare-projections",
        nargs="?",
        const="all",
        default=None,
        choices=["all", "gnom", "stere", "aeqd", "ortho", "equirect", "robin"],
        help=(
            "Generate Poincare projections. Use 'all' (default when flag is present) "
            "for a panel, or a single projection type: gnom, stere, aeqd, ortho, equirect, robin. "
            "Requires --poincare."
        ),
    )
    parser.add_argument(
        "--poincare-proj-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help=(
            "Projection center as a Stokes (Q,U,V) unit vector. "
            "Defaults to the mean polarisation vector of the data."
        ),
    )
    parser.add_argument(
        "--poincare-circle-fit",
        nargs="?",
        const="auto",
        default=None,
        choices=["auto", "great", "small"],
        help="Fit circles to Poincare segments: auto (default), great, or small.",
    )
    parser.add_argument(
        "--poincare-circle-segments",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Point-index segment pairs for circle fitting: s1 e1 s2 e2 ... "
            "(indices refer to plotted Poincare points after masking/binning)."
        ),
    )
    parser.add_argument(
        "--poincare-freq-bands",
        type=int,
        default=None,
        help=(
            "Split the frequency band into N subbands for Poincare frequency plots. "
            "Use with --poincare and --time-avg."
        ),
    )
    parser.add_argument(
        "--poincare-freq-bands-snr",
        type=int,
        default=None,
        help="Split Poincare frequency subbands by equal SNR weight into N bands.",
    )
    parser.add_argument(
        "--poincare-freq-bands-min-channels",
        type=int,
        default=4,
        help="Minimum channels per Poincare frequency subband (default: 4)",
    )
    parser.add_argument(
        "--poincare-freq-bands-manual",
        action="store_true",
        help="Manually select Poincare frequency subbands (interactive).",
    )
    # Peak separation
    parser.add_argument(
        "--separate-peaks",
        action="store_true",
        help="Create separate side-by-side plots for each detected peak region",
    )
    parser.add_argument(
        "--min-gap-bins",
        type=int,
        default=3,
        help="Minimum number of low-signal bins to separate peaks (default: 3)",
    )
    parser.add_argument(
        "--min-peak-bins",
        type=int,
        default=10,
        help="Minimum number of consecutive significant bins required for a valid peak (default: 10)",
    )
    parser.add_argument(
        "--max-merge-gap",
        type=int,
        default=0,
        help=(
            "Maximum gap size to merge nearby peaks. Peaks separated by fewer bins "
            "will be merged (default: 0, no merging)"
        ),
    )

    # RMNest
    parser.add_argument(
        "--rmnest-gfr",
        action="store_true",
        help="Use RMNest generalized Faraday rotation model",
    )
    parser.add_argument(
        "--rmnest-free-alpha",
        action="store_true",
        help="Allow alpha to vary for RMNest GFR model",
    )
    parser.add_argument(
        "--rmnest-outdir",
        default=None,
        help="Output directory for RMNest results (default: <output>_rmnest)",
    )
    parser.add_argument(
        "--rmnest-label",
        default=None,
        help="Label for RMNest run (default: <output>)",
    )
    parser.add_argument(
        "--rmnest-sampler",
        default="dynesty",
        help="Sampler for RMNest/Bilby (default: dynesty)",
    )

    # Binning and noise
    parser.add_argument(
        "--time-bins",
        type=int,
        default=None,
        help="Number of time bins to fit in time-series mode (default: no binning)",
    )
    parser.add_argument(
        "--pa-bins",
        type=int,
        default=50,
        help="Number of PA/EA/pol fraction bins in lower panel of time-series plot (default: 50)",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        default=None,
        help="Number of frequency bins after --time-avg (default: no binning)",
    )
    parser.add_argument(
        "--exclude-edge-bins",
        type=int,
        default=0,
        help=(
            "Exclude N bins from each end of the spectrum for frequency-domain fitting "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--freq-band-mhz", "--freq-band",
        type=float, nargs=2, default=None, metavar=("FMIN", "FMAX"),
        help="Crop to frequency band [FMIN, FMAX] in MHz",
    )
    parser.add_argument(
        "--freq-band-indices", "--freq-indices",
        type=int, nargs=2, default=None, metavar=("START", "STOP"),
        help="Crop to frequency band indices [START, STOP)",
    )
    parser.add_argument(
        "--manual-freq-bands", "--manual-freqs",
        action="store_true",
        help="Interactively select frequency band(s) from spectrum",
    )
    parser.add_argument(
        "--offpulse",
        type=float,
        default=0.1,
        help="Fraction of Stokes I samples used for offpulse noise estimation (default: 0.10)",
    )

    # Physics helpers
    parser.add_argument(
        "--turbulent-radius-pc",
        type=float,
        default=21.0,
        help="Radius R of turbulent environment in pc for delta(n_e, B_parallel)",
    )
    parser.add_argument(
        "--screen-scale-cm",
        type=float,
        default=1e15,
        help="Plasma-screen scale l_screen in cm for delta(n_e, B_parallel)",
    )

    args = parser.parse_args()

    stokes_axis = 0
    time_axis = 1
    freq_axis = 0
    freq_unit = "MHz"
    time_unit = "ms"

    using_cube = args.stokes_cube is not None
    using_separate = any(v is not None for v in (args.stokes_i, args.stokes_q, args.stokes_u, args.stokes_v))
    if using_cube and using_separate:
        parser.error("Use either --stokes-cube or separate --stokes-i/--stokes-q/--stokes-u inputs, not both")
    if (not using_cube) and (args.stokes_i is None or args.stokes_q is None or args.stokes_u is None):
        parser.error("Provide --stokes-cube or all of --stokes-i, --stokes-q, and --stokes-u")
    if args.exclude_edge_bins < 0:
        parser.error("--exclude-edge-bins must be >= 0")

    circle_segments: Optional[List[Tuple[int, int]]] = None
    if args.poincare_circle_segments is not None:
        if len(args.poincare_circle_segments) == 0:
            circle_segments = []
        elif len(args.poincare_circle_segments) % 2 != 0:
            parser.error("--poincare-circle-segments must contain an even number of integers")
        else:
            circle_segments = list(
                zip(
                    args.poincare_circle_segments[0::2],
                    args.poincare_circle_segments[1::2],
                )
            )

    set_pub_style(use_latex=False)

    print("=" * 60)
    print("RM FITTING FOR STOKES IQUV DATA")
    print("Using RM-Tools library for RM synthesis")
    print("=" * 60)

    print("\nLoading Stokes data...")
    freq_hz, stokes_i, stokes_q, stokes_u, stokes_v, time_array = load_stokes_data(
        i_file=args.stokes_i,
        q_file=args.stokes_q,
        u_file=args.stokes_u,
        v_file=args.stokes_v,
        cube_file=args.stokes_cube,
        stokes_axis=stokes_axis,
        freq_file=args.freq,
        time_file=args.time,
        time_axis=time_axis,
        freq_axis=freq_axis,
        freq_unit=freq_unit,
        time_unit=time_unit,
    )

    burn_pol_frac_err = None
    burn_valid_mask = None
    burn_circ_frac_err = None
    burn_circ_valid_mask = None
    time_avg_extra_regions: List[Tuple[int, int]] = []
    sigma_i_chan_base = None
    sigma_q_chan_base = None
    sigma_u_chan_base = None
    sigma_v_chan_base = None
    freq_hz_unbinned = None
    onpulse_mask = None
    off_std = None

    if stokes_i.ndim == 2:
        print(f"\n  Detected 2D data with shape: {stokes_i.shape}")
        print(f"  Time axis: {time_axis}, Frequency axis: {freq_axis}")

        n_time_noise = stokes_i.shape[1]
        n_frac_noise = max(1, int(n_time_noise * args.offpulse))
        i_off = stokes_i[:, :n_frac_noise]
        q_off = stokes_q[:, :n_frac_noise]
        u_off = stokes_u[:, :n_frac_noise]
        v_off = stokes_v[:, :n_frac_noise]

        sigma_i_chan = np.nanstd(i_off, axis=1)
        sigma_q_chan = np.nanstd(q_off, axis=1)
        sigma_u_chan = np.nanstd(u_off, axis=1)
        sigma_v_chan = np.nanstd(v_off, axis=1)

        off_std = np.array([sigma_i_chan, sigma_q_chan, sigma_u_chan, sigma_v_chan])

        # Save full 2D arrays for later use by extra peaks
        stokes_i_full_noise = stokes_i
        stokes_q_full_noise = stokes_q
        stokes_u_full_noise = stokes_u
        stokes_v_full_noise = stokes_v

        sigma_i_chan = np.where(
            np.isfinite(sigma_i_chan) & (sigma_i_chan > 0), sigma_i_chan, 1e-10
        )
        sigma_q_chan = np.where(
            np.isfinite(sigma_q_chan) & (sigma_q_chan > 0), sigma_q_chan, 1e-10
        )
        sigma_u_chan = np.where(
            np.isfinite(sigma_u_chan) & (sigma_u_chan > 0), sigma_u_chan, 1e-10
        )
        if sigma_v_chan is not None:
            sigma_v_chan = np.where(
                np.isfinite(sigma_v_chan) & (sigma_v_chan > 0), sigma_v_chan, 1e-10
            )
        sigma_i_chan_base = sigma_i_chan.copy()
        sigma_q_chan_base = sigma_q_chan.copy()
        sigma_u_chan_base = sigma_u_chan.copy()
        sigma_v_chan_base = sigma_v_chan.copy() if sigma_v_chan is not None else None

        n_freq_data = stokes_i.shape[freq_axis]
        if len(freq_hz) != n_freq_data:
            print(
                f"\n  WARNING: Frequency array length ({len(freq_hz)}) does not match "
                f"frequency axis dimension ({n_freq_data})"
            )
            print("  Attempting to auto-correct: swapping time and frequency arrays...")
            freq_hz, time_array = time_array, freq_hz
            print(f"  New frequency array length: {len(freq_hz)}")
            if time_array is not None:
                print(f"  New time array length: {len(time_array)}")
        freq_hz_unbinned = np.asarray(freq_hz, dtype=float).copy()

        onpulse_regions = None

        if args.manual_peaks or (args.peak_indices is not None and len(args.peak_indices) > 0):
            if args.manual_peaks:
                print("\nInteractive peak selection requested...")
                if time_array is None:
                    print("  ERROR: manual peak selection requires a time array (--time)")
                else:
                    peaks = select_peaks_manual(time_array, stokes_i)
                    print(f"  Manual peaks: {peaks}")
                    onpulse_regions = peaks
                    start_idx = min(p[0] for p in peaks)
                    end_idx = max(p[1] for p in peaks)
                    print(
                        f"  Using on-pulse window covering manual peaks: {start_idx} to {end_idx}"
                    )
                    onpulse_mask = (start_idx, end_idx)
            elif args.peak_indices is not None:
                pairs = parse_peak_index_pairs(args.peak_indices, stokes_i.shape[1])
                if len(pairs) == 0:
                    print("  Warning: --peak-indices provided but no valid pairs found")
                else:
                    print(f"\nUser-specified peak index pairs: {pairs}")
                    onpulse_regions = pairs
                    start_idx = min(p[0] for p in pairs)
                    end_idx = max(p[1] for p in pairs)
                    print(
                        f"  Using on-pulse window covering provided indices: {start_idx} to {end_idx}"
                    )
                    onpulse_mask = (start_idx, end_idx)
        else:
            if args.onpulse_only:
                print(f"\nDetecting on-pulse window ({args.onpulse_fraction * 100:.1f}% flux)...")

                time_profile = np.sum(stokes_i, axis=freq_axis)
                start_idx, end_idx = find_onpulse_window(time_profile, args.onpulse_fraction)

                if time_array is not None:
                    time_start = time_array[start_idx] * 1e3
                    time_end = time_array[end_idx] * 1e3
                    print(f"  On-pulse window: time bins {start_idx} to {end_idx}")
                    print(f"  Time range: {time_start:.3f} to {time_end:.3f} ms")
                    print(f"  Window width: {end_idx - start_idx + 1} bins")
                else:
                    print(f"  On-pulse window: bins {start_idx} to {end_idx}")

                onpulse_mask = (start_idx, end_idx)

        if onpulse_regions is None and onpulse_mask is not None:
            onpulse_regions = [onpulse_mask]
        if onpulse_regions is not None and len(onpulse_regions) > 0:
            def _print_component_fractions(
                label: str,
                i_seg: np.ndarray,
                q_seg: np.ndarray,
                u_seg: np.ndarray,
                v_seg: Optional[np.ndarray],
            ) -> None:
                i_mean = float(np.nanmean(i_seg))
                q_mean = float(np.nanmean(q_seg))
                u_mean = float(np.nanmean(u_seg))
                l_mean = float(np.sqrt(q_mean ** 2 + u_mean ** 2))
                if v_seg is not None:
                    v_mean = float(np.nanmean(v_seg))
                    p_mean = float(np.sqrt(q_mean ** 2 + u_mean ** 2 + v_mean ** 2))
                else:
                    v_mean = float("nan")
                    p_mean = float(np.sqrt(q_mean ** 2 + u_mean ** 2))
                denom = i_mean if np.isfinite(i_mean) and i_mean != 0.0 else float("nan")
                p_frac = p_mean / denom if np.isfinite(denom) else float("nan")
                l_frac = l_mean / denom if np.isfinite(denom) else float("nan")
                v_frac = v_mean / denom if np.isfinite(denom) else float("nan")
                print(
                    f"  {label}: P/I={p_frac:.4f}, L/I={l_frac:.4f}, V/I={v_frac:.4f}"
                )
            print("\nSelected component fractions:")
            for idx, (start_idx, end_idx) in enumerate(onpulse_regions, start=1):
                if time_axis == 0:
                    i_seg = stokes_i[start_idx : end_idx + 1, :]
                    q_seg = stokes_q[start_idx : end_idx + 1, :]
                    u_seg = stokes_u[start_idx : end_idx + 1, :]
                    v_seg = (
                        stokes_v[start_idx : end_idx + 1, :]
                        if stokes_v is not None
                        else None
                    )
                else:
                    i_seg = stokes_i[:, start_idx : end_idx + 1]
                    q_seg = stokes_q[:, start_idx : end_idx + 1]
                    u_seg = stokes_u[:, start_idx : end_idx + 1]
                    v_seg = (
                        stokes_v[:, start_idx : end_idx + 1]
                        if stokes_v is not None
                        else None
                    )
                _print_component_fractions(
                    f"component {idx} (bins {start_idx}-{end_idx})",
                    i_seg,
                    q_seg,
                    u_seg,
                    v_seg,
                )
        else:
            print("\nSelected component fractions: no on-pulse regions available")

        if args.time_avg:
            print("  Averaging over time axis...")
            n_time_avg_used = n_time_noise

            if onpulse_regions is not None and len(onpulse_regions) > 0:
                first_start, first_end = onpulse_regions[0]
                time_avg_extra_regions = onpulse_regions[1:]
                n_time_avg_used = max(1, first_end - first_start + 1)
                print(f"  Using peak 1 on-pulse region: bins {first_start} to {first_end}")
                if len(time_avg_extra_regions) > 0:
                    print(f"  Additional selected peaks to process separately: {len(time_avg_extra_regions)}")

                if time_axis == 0:
                    stokes_i = np.mean(stokes_i[first_start : first_end + 1, :], axis=0)
                    stokes_q = np.mean(stokes_q[first_start : first_end + 1, :], axis=0)
                    stokes_u = np.mean(stokes_u[first_start : first_end + 1, :], axis=0)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[first_start : first_end + 1, :], axis=0)
                else:
                    stokes_i = np.mean(stokes_i[:, first_start : first_end + 1], axis=1)
                    stokes_q = np.mean(stokes_q[:, first_start : first_end + 1], axis=1)
                    stokes_u = np.mean(stokes_u[:, first_start : first_end + 1], axis=1)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[:, first_start : first_end + 1], axis=1)
            elif onpulse_mask is not None:
                start_idx, end_idx = onpulse_mask
                n_time_avg_used = max(1, end_idx - start_idx + 1)
                print(f"  Using only on-pulse region (bins {start_idx} to {end_idx})...")

                if time_axis == 0:
                    stokes_i = np.mean(stokes_i[start_idx : end_idx + 1, :], axis=time_axis)
                    stokes_q = np.mean(stokes_q[start_idx : end_idx + 1, :], axis=time_axis)
                    stokes_u = np.mean(stokes_u[start_idx : end_idx + 1, :], axis=time_axis)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[start_idx : end_idx + 1, :], axis=time_axis)
                else:
                    stokes_i = np.mean(stokes_i[:, start_idx : end_idx + 1], axis=time_axis)
                    stokes_q = np.mean(stokes_q[:, start_idx : end_idx + 1], axis=time_axis)
                    stokes_u = np.mean(stokes_u[:, start_idx : end_idx + 1], axis=time_axis)
                    if stokes_v is not None:
                        stokes_v = np.mean(stokes_v[:, start_idx : end_idx + 1], axis=time_axis)
            else:
                stokes_i = np.mean(stokes_i, axis=time_axis)
                stokes_q = np.mean(stokes_q, axis=time_axis)
                stokes_u = np.mean(stokes_u, axis=time_axis)
                if stokes_v is not None:
                    stokes_v = np.mean(stokes_v, axis=time_axis)

            print(f"  Averaged data shape: {stokes_i.shape}")

            noise_scale_timeavg = np.sqrt(max(1, n_time_avg_used))
            sigma_i_chan = sigma_i_chan / noise_scale_timeavg
            sigma_q_chan = sigma_q_chan / noise_scale_timeavg
            sigma_u_chan = sigma_u_chan / noise_scale_timeavg
            if sigma_v_chan is not None:
                sigma_v_chan = sigma_v_chan / noise_scale_timeavg

            n_freq = len(freq_hz)
            if args.freq_bins is None or args.freq_bins <= 0 or args.freq_bins >= n_freq:
                if args.freq_bins is not None and args.freq_bins >= n_freq:
                    print(
                        f"  Requested --freq-bins={args.freq_bins} >= number of channels "
                        f"({n_freq}); keeping full resolution."
                    )
            else:
                n_freq_bins_actual = min(args.freq_bins, n_freq)
                freq_bin_size = int(np.ceil(n_freq / n_freq_bins_actual))
                n_freq_bins_actual = (n_freq + freq_bin_size - 1) // freq_bin_size

                freq_hz_binned = np.zeros(n_freq_bins_actual)
                stokes_i_binned = np.zeros(n_freq_bins_actual)
                stokes_q_binned = np.zeros(n_freq_bins_actual)
                stokes_u_binned = np.zeros(n_freq_bins_actual)
                stokes_v_binned = np.zeros(n_freq_bins_actual) if stokes_v is not None else None
                sigma_i_binned = np.zeros(n_freq_bins_actual)
                sigma_q_binned = np.zeros(n_freq_bins_actual)
                sigma_u_binned = np.zeros(n_freq_bins_actual)
                sigma_v_binned = np.zeros(n_freq_bins_actual) if sigma_v_chan is not None else None

                for i_bin in range(n_freq_bins_actual):
                    bin_start = i_bin * freq_bin_size
                    bin_end = min((i_bin + 1) * freq_bin_size, n_freq)
                    if bin_end <= bin_start:
                        continue

                    freq_hz_binned[i_bin] = np.mean(freq_hz[bin_start:bin_end])
                    stokes_i_binned[i_bin] = np.mean(stokes_i[bin_start:bin_end])
                    stokes_q_binned[i_bin] = np.mean(stokes_q[bin_start:bin_end])
                    stokes_u_binned[i_bin] = np.mean(stokes_u[bin_start:bin_end])
                    if stokes_v is not None:
                        stokes_v_binned[i_bin] = np.mean(stokes_v[bin_start:bin_end])
                    n_chan_bin = max(1, bin_end - bin_start)
                    sigma_i_binned[i_bin] = (
                        np.sqrt(np.sum(sigma_i_chan[bin_start:bin_end] ** 2)) / n_chan_bin
                    )
                    sigma_q_binned[i_bin] = (
                        np.sqrt(np.sum(sigma_q_chan[bin_start:bin_end] ** 2)) / n_chan_bin
                    )
                    sigma_u_binned[i_bin] = (
                        np.sqrt(np.sum(sigma_u_chan[bin_start:bin_end] ** 2)) / n_chan_bin
                    )
                    if sigma_v_chan is not None:
                        sigma_v_binned[i_bin] = (
                            np.sqrt(np.sum(sigma_v_chan[bin_start:bin_end] ** 2)) / n_chan_bin
                        )

                freq_hz = freq_hz_binned
                stokes_i = stokes_i_binned
                stokes_q = stokes_q_binned
                stokes_u = stokes_u_binned
                if stokes_v is not None:
                    stokes_v = stokes_v_binned
                sigma_i_chan = sigma_i_binned
                sigma_q_chan = sigma_q_binned
                sigma_u_chan = sigma_u_binned
                sigma_v_chan = sigma_v_binned

                print(
                    f"  Frequency-binned data: {n_freq} -> {len(freq_hz)} channels "
                    f"(--freq-bins={args.freq_bins})"
                )

            q_val = np.asarray(stokes_q, dtype=float)
            u_val = np.asarray(stokes_u, dtype=float)
            i_val = np.asarray(stokes_i, dtype=float)
            l_val = np.sqrt(q_val ** 2 + u_val ** 2)
            sigma_l = np.sqrt(
                (q_val ** 2 * sigma_q_chan ** 2 + u_val ** 2 * sigma_u_chan ** 2)
                / (l_val ** 2 + 1e-20)
            )
            burn_pol_frac_err = np.sqrt(
                (sigma_l / (i_val + 1e-10)) ** 2
                + ((l_val * sigma_i_chan) / ((i_val + 1e-10) ** 2)) ** 2
            )

            i_snr_chan = i_val / (sigma_i_chan + 1e-10)
            burn_valid_mask = i_snr_chan >= 2.0
            burn_pol_frac_err[~burn_valid_mask] = np.nan
            burn_circ_valid_mask = burn_valid_mask.copy()

            if stokes_v is not None and sigma_v_chan is not None:
                v_val = np.asarray(stokes_v, dtype=float)
                burn_circ_frac_err = np.sqrt(
                    (sigma_v_chan / (i_val + 1e-10)) ** 2
                    + ((np.abs(v_val) * sigma_i_chan) / ((i_val + 1e-10) ** 2)) ** 2
                )
                burn_circ_frac_err[~burn_circ_valid_mask] = np.nan
                burn_circ_frac_err = np.where(
                    np.isfinite(burn_circ_frac_err) & (burn_circ_frac_err > 0),
                    burn_circ_frac_err,
                    np.nan,
                )

            burn_pol_frac_err = np.where(
                np.isfinite(burn_pol_frac_err) & (burn_pol_frac_err > 0),
                burn_pol_frac_err,
                np.nan,
            )
        elif not args.time_series:
            print(
                "  Note: Data is 2D. Use --time-avg to average over time, or --time-series to "
                "process each time bin."
            )
            print("  Proceeding with first time sample...")
            idx = [slice(None)] * stokes_i.ndim
            idx[time_axis] = 0
            stokes_i = stokes_i[tuple(idx)]
            stokes_q = stokes_q[tuple(idx)]
            stokes_u = stokes_u[tuple(idx)]
            if stokes_v is not None:
                stokes_v = stokes_v[tuple(idx)]
            print(f"  Using data shape: {stokes_i.shape}")

    if args.exclude_edge_bins > 0:
        n_freq_current = len(freq_hz)
        if (2 * args.exclude_edge_bins) >= n_freq_current:
            parser.error(
                f"--exclude-edge-bins={args.exclude_edge_bins} removes all channels "
                f"(n_freq={n_freq_current})."
            )

        trim_slice = slice(args.exclude_edge_bins, n_freq_current - args.exclude_edge_bins)
        freq_hz = np.asarray(freq_hz, dtype=float)[trim_slice]
        stokes_i = np.asarray(stokes_i, dtype=float)[trim_slice]
        stokes_q = np.asarray(stokes_q, dtype=float)[trim_slice]
        stokes_u = np.asarray(stokes_u, dtype=float)[trim_slice]
        if stokes_v is not None:
            stokes_v = np.asarray(stokes_v, dtype=float)[trim_slice]
        if burn_pol_frac_err is not None:
            burn_pol_frac_err = np.asarray(burn_pol_frac_err, dtype=float)[trim_slice]
        if burn_valid_mask is not None:
            burn_valid_mask = np.asarray(burn_valid_mask, dtype=bool)[trim_slice]
        if burn_circ_frac_err is not None:
            burn_circ_frac_err = np.asarray(burn_circ_frac_err, dtype=float)[trim_slice]
        if burn_circ_valid_mask is not None:
            burn_circ_valid_mask = np.asarray(burn_circ_valid_mask, dtype=bool)[trim_slice]
        if off_std is not None:
            off_std = off_std[:, trim_slice]

        print(
            f"  Excluded {args.exclude_edge_bins} edge bins per side: "
            f"{n_freq_current} -> {len(freq_hz)} channels"
        )

    if args.freq_band_mhz is not None or args.freq_band_indices is not None or args.manual_freq_bands:
        spec_for_band = np.nanmean(stokes_i, axis=time_axis) if stokes_i.ndim == 2 else stokes_i
        freq_band = _resolve_freq_band_idx(args, freq_hz, spec_for_band, dspec=stokes_i if stokes_i.ndim == 2 else None)
        if freq_band is not None:
            freq_hz = freq_hz[freq_band]
            stokes_i = stokes_i[freq_band]
            stokes_q = stokes_q[freq_band]
            stokes_u = stokes_u[freq_band]
            if stokes_v is not None:
                stokes_v = stokes_v[freq_band]
            if burn_pol_frac_err is not None:
                burn_pol_frac_err = burn_pol_frac_err[freq_band]
            if burn_valid_mask is not None:
                burn_valid_mask = burn_valid_mask[freq_band]
            if burn_circ_frac_err is not None:
                burn_circ_frac_err = burn_circ_frac_err[freq_band]
            if burn_circ_valid_mask is not None:
                burn_circ_valid_mask = burn_circ_valid_mask[freq_band]
            if off_std is not None:
                off_std = off_std[:, freq_band]
            print(
                f"  Cropped to frequency band: {freq_hz.min() / 1e6:.2f} - "
                f"{freq_hz.max() / 1e6:.2f} MHz ({len(freq_hz)} channels)"
            )

    print(f"  Frequency range: {freq_hz.min() / 1e6:.2f} - {freq_hz.max() / 1e6:.2f} MHz")
    print(f"  Number of channels: {len(freq_hz)}")

    if not args.time_series:
        print(f"\nPerforming RM fitting using method: {args.method}")

        fitter = RMFitter(freq_hz, stokes_i, stokes_q, stokes_u, stokes_v)

        if args.method == "simple":
            result = fitter._fit_rm_with_rmtools(rm_range=tuple(args.rm_range), n_rm=args.n_rm)
            rm_peak_print = result.get("rm_clean_peak", result.get("rm_peak", np.nan))
            rm_err = result.get("rm_clean_err", result.get("noise_estimate", 0) * 2)
            print("\nResults (RM Synthesis - Simple Mode):")
            print(f"  Peak RM = {rm_peak_print:.4f} +/- {rm_err:.4f} rad/m^2")
            print(f"  SNR = {result.get('rm_peak_snr', np.nan):.2f}")
            print(f"  Noise level = {result.get('noise_estimate', 0):.6f}")

            if not args.no_plot:
                plot_rm_results(
                    fitter,
                    result,
                    f"{args.output}_rm_results.{args.ext}",
                    pol_frac_err=burn_pol_frac_err,
                    valid_mask=burn_valid_mask,
                    circ_frac_err=burn_circ_frac_err,
                    circ_valid_mask=burn_circ_valid_mask,
                    show_frac_panel=not args.hide_rm_frac_panel,
                )

        elif args.method == "rm_synthesis":
            result = fitter._fit_rm_with_rmtools(rm_range=tuple(args.rm_range), n_rm=args.n_rm)
            rm_peak_print = result.get("rm_clean_peak", result.get("rm_peak", np.nan))
            rm_err = result.get("rm_clean_err", result.get("noise_estimate", 0) * 2)
            print("\nResults (RM Synthesis):")
            print(f"  Peak RM = {rm_peak_print:.4f} +/- {rm_err:.4f} rad/m^2")
            print(f"  SNR = {result.get('rm_peak_snr', np.nan):.2f}")
            print(f"  Noise level = {result.get('noise_estimate', 0):.6f}")

            if not args.no_plot:
                plot_rm_results(
                    fitter,
                    result,
                    f"{args.output}_rm_results.{args.ext}",
                    pol_frac_err=burn_pol_frac_err,
                    valid_mask=burn_valid_mask,
                    circ_frac_err=burn_circ_frac_err,
                    circ_valid_mask=burn_circ_valid_mask,
                    show_frac_panel=not args.hide_rm_frac_panel,
                )

        elif args.method == "qu_fitting":
            result = fitter.fit_rm_qufitting()
            if result.get("success"):
                print("\nResults (QU Fitting):")
                print(f"  RM = {result['rm']:.4f} +/- {result['rm_err']:.4f} rad/m^2")
                print(f"  Q0 = {result['q0']:.6f} +/- {result['q0_err']:.6f}")
                print(f"  U0 = {result['u0']:.6f} +/- {result['u0_err']:.6f}")
            else:
                print("QU fitting failed!")

        elif args.method == "rmnest":
            rmnest_outdir = args.rmnest_outdir or f"{args.output}_rmnest"
            rmnest_label = args.rmnest_label or args.output
            try:
                result = fitter.fit_rm_rmnest(
                    gfr=args.rmnest_gfr,
                    free_alpha=args.rmnest_free_alpha,
                    outdir=rmnest_outdir,
                    label=rmnest_label,
                    sampler=args.rmnest_sampler,
                )
            except ImportError as exc:
                print(f"RMNest unavailable: {exc}")
                return

            param_name = result["param_name"].upper()
            median = result["median"]
            low = result["low"]
            high = result["high"]
            print("\nResults (RMNest):")
            print(f"  {param_name} = {median:.4f} +{high - median:.4f}/-{median - low:.4f}")
            print(f"  Output directory: {result['rmnest_outdir']}")
            print(f"  Bilby result: {result['rmnest_post_json']}")

        if args.time_avg and not args.no_plot:
            burn_out = f"{args.output}_burns_law.{args.ext}"
            plot_burns_law_fits(
                fitter,
                burn_out,
                pol_frac_err=burn_pol_frac_err,
                valid_mask=burn_valid_mask,
                circ_frac_err=burn_circ_frac_err,
                circ_valid_mask=burn_circ_valid_mask,
                turbulent_radius_pc=args.turbulent_radius_pc,
                screen_scale_cm=args.screen_scale_cm,
            )

            plot_polarisation_fraction_acf_ccf(
                fitter,
                f"{args.output}_frac_correlation.{args.ext}",
            )

            if args.poincare:
                freq_band_ranges: Optional[List[Tuple[int, int]]] = None
                if args.poincare_freq_bands_manual:
                    freq_band_ranges = select_frequency_bands_manual(
                        freq_hz / 1e6,
                        np.nanmean(stokes_i, axis=-1) if stokes_i.ndim > 1 else stokes_i,
                        dspec=stokes_i if stokes_i.ndim == 2 else None,
                    )
                elif args.poincare_freq_bands is not None or args.poincare_freq_bands_snr is not None:
                    if args.poincare_freq_bands_snr is not None:
                        n_bands = args.poincare_freq_bands_snr
                        n_common = len(freq_hz)
                        if sigma_i_chan is not None:
                            n_common = min(n_common, len(sigma_i_chan))
                        n_common = min(n_common, len(stokes_i))
                        freq_mhz_common = (freq_hz[:n_common] / 1e6) if n_common > 0 else freq_hz / 1e6
                        stokes_i_common = stokes_i[:n_common] if n_common > 0 else stokes_i
                        if sigma_i_chan is not None:
                            sigma_i_common = sigma_i_chan[:n_common] if n_common > 0 else sigma_i_chan
                            snr_weights = np.where(
                                np.isfinite(sigma_i_common),
                                np.abs(stokes_i_common) / (sigma_i_common + 1e-10),
                                0.0,
                            )
                        else:
                            snr_weights = np.abs(stokes_i_common)
                        freq_band_ranges = split_frequency_bands_equal_snr(
                            freq_mhz_common,
                            snr_weights,
                            n_bands,
                            min_channels=args.poincare_freq_bands_min_channels,
                        )
                    else:
                        freq_band_ranges = split_frequency_bands_equal(
                            freq_hz / 1e6,
                            args.poincare_freq_bands,
                        )
                    if freq_band_ranges is not None and len(freq_band_ranges) == 0:
                        freq_band_ranges = None

                plot_poincare_sphere_frequency(
                    freq_hz,
                    stokes_i,
                    stokes_q,
                    stokes_u,
                    stokes_v,
                    f"{args.output}_poincare_frequency.{args.ext}",
                    sigma_i=sigma_i_chan,
                    sigma_q=sigma_q_chan,
                    sigma_u=sigma_u_chan,
                    sigma_v=sigma_v_chan,
                    snr_threshold=2.0,
                    exclude_edge_bins=args.exclude_edge_bins,
                    interactive=args.poincare_interactive,
                    force_surface=args.poincare_surface,
                    circle_fit_mode=args.poincare_circle_fit,
                    circle_fit_segments=circle_segments,
                )

                if args.poincare_projections:
                    proj_tag = str(args.poincare_projections).lower()
                    plot_poincare_projections_frequency(
                        freq_hz,
                        stokes_i,
                        stokes_q,
                        stokes_u,
                        stokes_v,
                        f"{args.output}_poincare_projections_{proj_tag}.{args.ext}",
                        projection_type=args.poincare_projections,
                        sigma_i=sigma_i_chan,
                        sigma_q=sigma_q_chan,
                        sigma_u=sigma_u_chan,
                        sigma_v=sigma_v_chan,
                        snr_threshold=2.0,
                        exclude_edge_bins=args.exclude_edge_bins,
                        force_surface=args.poincare_surface,
                        circle_fit_mode=args.poincare_circle_fit,
                        circle_fit_segments=circle_segments,
                        center=tuple(args.poincare_proj_center)
                        if args.poincare_proj_center is not None
                        else None,
                    )

                if freq_band_ranges is not None:
                    for band_idx, (start_idx, stop_idx) in enumerate(freq_band_ranges, start=1):
                        band_slice = slice(start_idx, stop_idx)
                        band_tag = f"band{band_idx:02d}"
                        plot_poincare_sphere_frequency(
                            freq_hz[band_slice],
                            stokes_i[band_slice],
                            stokes_q[band_slice],
                            stokes_u[band_slice],
                            stokes_v[band_slice] if stokes_v is not None else None,
                            f"{args.output}_poincare_frequency_{band_tag}.{args.ext}",
                            sigma_i=sigma_i_chan[band_slice] if sigma_i_chan is not None else None,
                            sigma_q=sigma_q_chan[band_slice] if sigma_q_chan is not None else None,
                            sigma_u=sigma_u_chan[band_slice] if sigma_u_chan is not None else None,
                            sigma_v=sigma_v_chan[band_slice] if sigma_v_chan is not None else None,
                            snr_threshold=2.0,
                            exclude_edge_bins=args.exclude_edge_bins,
                            interactive=args.poincare_interactive,
                            force_surface=args.poincare_surface,
                            circle_fit_mode=args.poincare_circle_fit,
                            circle_fit_segments=circle_segments,
                        )

                        if args.poincare_projections:
                            proj_tag = str(args.poincare_projections).lower()
                            plot_poincare_projections_frequency(
                                freq_hz[band_slice],
                                stokes_i[band_slice],
                                stokes_q[band_slice],
                                stokes_u[band_slice],
                                stokes_v[band_slice] if stokes_v is not None else None,
                                f"{args.output}_poincare_projections_{proj_tag}_{band_tag}.{args.ext}",
                                projection_type=args.poincare_projections,
                                sigma_i=sigma_i_chan[band_slice] if sigma_i_chan is not None else None,
                                sigma_q=sigma_q_chan[band_slice] if sigma_q_chan is not None else None,
                                sigma_u=sigma_u_chan[band_slice] if sigma_u_chan is not None else None,
                                sigma_v=sigma_v_chan[band_slice] if sigma_v_chan is not None else None,
                                snr_threshold=2.0,
                                exclude_edge_bins=args.exclude_edge_bins,
                                force_surface=args.poincare_surface,
                                circle_fit_mode=args.poincare_circle_fit,
                                circle_fit_segments=circle_segments,
                                center=tuple(args.poincare_proj_center)
                                if args.poincare_proj_center is not None
                                else None,
                            )

        if args.time_avg and len(time_avg_extra_regions) > 0 and stokes_i is not None:
            for i_extra, (pk_start, pk_end) in enumerate(time_avg_extra_regions, start=2):
                print(f"\nProcessing additional selected peak {i_extra}: bins {pk_start} to {pk_end}")
                n_time_pk = max(1, pk_end - pk_start + 1)

                stokes_i_pk = np.mean(stokes_i_full_noise[:, pk_start : pk_end + 1], axis=1)
                stokes_q_pk = np.mean(stokes_q_full_noise[:, pk_start : pk_end + 1], axis=1)
                stokes_u_pk = np.mean(stokes_u_full_noise[:, pk_start : pk_end + 1], axis=1)
                stokes_v_pk = np.mean(stokes_v_full_noise[:, pk_start : pk_end + 1], axis=1) if stokes_v is not None else None

                freq_pk = (
                    freq_hz_unbinned.copy()
                    if freq_hz_unbinned is not None
                    else np.asarray(freq_hz, dtype=float).copy()
                )
                sigma_i_pk = sigma_i_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_q_pk = sigma_q_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_u_pk = sigma_u_chan_base.copy() / np.sqrt(n_time_pk)
                sigma_v_pk = (
                    sigma_v_chan_base.copy() / np.sqrt(n_time_pk)
                    if sigma_v_chan_base is not None
                    else None
                )

                n_freq_pk = len(freq_pk)
                if args.freq_bins is not None and args.freq_bins > 0 and args.freq_bins < n_freq_pk:
                    n_freq_bins_actual = min(args.freq_bins, n_freq_pk)
                    freq_bin_size = int(np.ceil(n_freq_pk / n_freq_bins_actual))
                    n_freq_bins_actual = (n_freq_pk + freq_bin_size - 1) // freq_bin_size

                    freq_b = np.zeros(n_freq_bins_actual)
                    i_b = np.zeros(n_freq_bins_actual)
                    q_b = np.zeros(n_freq_bins_actual)
                    u_b = np.zeros(n_freq_bins_actual)
                    v_b = np.zeros(n_freq_bins_actual) if stokes_v_pk is not None else None
                    si_b = np.zeros(n_freq_bins_actual)
                    sq_b = np.zeros(n_freq_bins_actual)
                    su_b = np.zeros(n_freq_bins_actual)
                    sv_b = np.zeros(n_freq_bins_actual) if sigma_v_pk is not None else None

                    for i_bin in range(n_freq_bins_actual):
                        bin_start = i_bin * freq_bin_size
                        bin_end = min((i_bin + 1) * freq_bin_size, n_freq_pk)
                        if bin_end <= bin_start:
                            continue
                        n_chan_bin = max(1, bin_end - bin_start)
                        freq_b[i_bin] = np.mean(freq_pk[bin_start:bin_end])
                        i_b[i_bin] = np.mean(stokes_i_pk[bin_start:bin_end])
                        q_b[i_bin] = np.mean(stokes_q_pk[bin_start:bin_end])
                        u_b[i_bin] = np.mean(stokes_u_pk[bin_start:bin_end])
                        if stokes_v_pk is not None:
                            v_b[i_bin] = np.mean(stokes_v_pk[bin_start:bin_end])
                        si_b[i_bin] = np.sqrt(np.sum(sigma_i_pk[bin_start:bin_end] ** 2)) / n_chan_bin
                        sq_b[i_bin] = np.sqrt(np.sum(sigma_q_pk[bin_start:bin_end] ** 2)) / n_chan_bin
                        su_b[i_bin] = np.sqrt(np.sum(sigma_u_pk[bin_start:bin_end] ** 2)) / n_chan_bin
                        if sigma_v_pk is not None:
                            sv_b[i_bin] = (
                                np.sqrt(np.sum(sigma_v_pk[bin_start:bin_end] ** 2)) / n_chan_bin
                            )

                    freq_pk = freq_b
                    stokes_i_pk = i_b
                    stokes_q_pk = q_b
                    stokes_u_pk = u_b
                    if stokes_v_pk is not None:
                        stokes_v_pk = v_b
                    sigma_i_pk = si_b
                    sigma_q_pk = sq_b
                    sigma_u_pk = su_b
                    sigma_v_pk = sv_b

                if args.exclude_edge_bins > 0:
                    n_freq_pk_cur = len(freq_pk)
                    if (2 * args.exclude_edge_bins) >= n_freq_pk_cur:
                        print(
                            f"  Skipping peak {i_extra}: --exclude-edge-bins={args.exclude_edge_bins} "
                            f"would remove all channels (n_freq={n_freq_pk_cur})."
                        )
                        continue
                    trim_slice_pk = slice(args.exclude_edge_bins, n_freq_pk_cur - args.exclude_edge_bins)
                    freq_pk = np.asarray(freq_pk, dtype=float)[trim_slice_pk]
                    stokes_i_pk = np.asarray(stokes_i_pk, dtype=float)[trim_slice_pk]
                    stokes_q_pk = np.asarray(stokes_q_pk, dtype=float)[trim_slice_pk]
                    stokes_u_pk = np.asarray(stokes_u_pk, dtype=float)[trim_slice_pk]
                    if stokes_v_pk is not None:
                        stokes_v_pk = np.asarray(stokes_v_pk, dtype=float)[trim_slice_pk]
                    sigma_i_pk = np.asarray(sigma_i_pk, dtype=float)[trim_slice_pk]
                    sigma_q_pk = np.asarray(sigma_q_pk, dtype=float)[trim_slice_pk]
                    sigma_u_pk = np.asarray(sigma_u_pk, dtype=float)[trim_slice_pk]
                    if sigma_v_pk is not None:
                        sigma_v_pk = np.asarray(sigma_v_pk, dtype=float)[trim_slice_pk]

                if args.freq_band_mhz is not None or args.freq_band_indices is not None or args.manual_freq_bands:
                    freq_band_pk = _resolve_freq_band_idx(args, freq_pk, stokes_i_pk, dspec=stokes_i_pk if stokes_i_pk.ndim == 2 else None)
                    if freq_band_pk is not None:
                        freq_pk = freq_pk[freq_band_pk]
                        stokes_i_pk = stokes_i_pk[freq_band_pk]
                        stokes_q_pk = stokes_q_pk[freq_band_pk]
                        stokes_u_pk = stokes_u_pk[freq_band_pk]
                        if stokes_v_pk is not None:
                            stokes_v_pk = stokes_v_pk[freq_band_pk]
                        sigma_i_pk = sigma_i_pk[freq_band_pk]
                        sigma_q_pk = sigma_q_pk[freq_band_pk]
                        sigma_u_pk = sigma_u_pk[freq_band_pk]
                        if sigma_v_pk is not None:
                            sigma_v_pk = sigma_v_pk[freq_band_pk]
                        print(
                            f"    Cropped peak {i_extra} to frequency band: "
                            f"{freq_pk.min() / 1e6:.2f} - {freq_pk.max() / 1e6:.2f} MHz "
                            f"({len(freq_pk)} channels)"
                        )

                l_pk = np.sqrt(stokes_q_pk ** 2 + stokes_u_pk ** 2)
                sigma_l_pk = np.sqrt(
                    (stokes_q_pk ** 2 * sigma_q_pk ** 2 + stokes_u_pk ** 2 * sigma_u_pk ** 2)
                    / (l_pk ** 2 + 1e-20)
                )
                burn_err_pk = np.sqrt(
                    (sigma_l_pk / (stokes_i_pk + 1e-10)) ** 2
                    + ((l_pk * sigma_i_pk) / ((stokes_i_pk + 1e-10) ** 2)) ** 2
                )
                burn_mask_pk = (stokes_i_pk / (sigma_i_pk + 1e-10)) >= 2.0
                burn_err_pk[~burn_mask_pk] = np.nan
                burn_err_pk = np.where(
                    np.isfinite(burn_err_pk) & (burn_err_pk > 0), burn_err_pk, np.nan
                )
                burn_circ_err_pk = None
                if stokes_v_pk is not None and sigma_v_pk is not None:
                    burn_circ_err_pk = np.sqrt(
                        (sigma_v_pk / (stokes_i_pk + 1e-10)) ** 2
                        + ((np.abs(stokes_v_pk) * sigma_i_pk) / ((stokes_i_pk + 1e-10) ** 2)) ** 2
                    )
                    burn_circ_err_pk[~burn_mask_pk] = np.nan
                    burn_circ_err_pk = np.where(
                        np.isfinite(burn_circ_err_pk) & (burn_circ_err_pk > 0),
                        burn_circ_err_pk,
                        np.nan,
                    )

                fitter_pk = RMFitter(freq_pk, stokes_i_pk, stokes_q_pk, stokes_u_pk, stokes_v_pk)
                output_prefix_pk = f"{args.output}_peak{i_extra}"

                if args.method in ["simple", "rm_synthesis"]:
                    result_pk = fitter_pk._fit_rm_with_rmtools(
                        rm_range=tuple(args.rm_range), n_rm=args.n_rm
                    )
                    rm_peak_pk = result_pk.get("rm_clean_peak", result_pk.get("rm_peak", np.nan))
                    rm_err_pk = result_pk.get("rm_clean_err", result_pk.get("noise_estimate", 0) * 2)
                    print(f"  Peak RM = {rm_peak_pk:.4f} +/- {rm_err_pk:.4f} rad/m^2")
                    print(f"  SNR = {result_pk.get('rm_peak_snr', np.nan):.2f}")

                    if not args.no_plot:
                        plot_rm_results(
                            fitter_pk,
                            result_pk,
                            f"{output_prefix_pk}_rm_results.{args.ext}",
                            pol_frac_err=burn_err_pk,
                            valid_mask=burn_mask_pk,
                            circ_frac_err=burn_circ_err_pk,
                            circ_valid_mask=burn_mask_pk,
                            show_frac_panel=not args.hide_rm_frac_panel,
                        )
                        plot_burns_law_fits(
                            fitter_pk,
                            f"{output_prefix_pk}_burns_law.{args.ext}",
                            pol_frac_err=burn_err_pk,
                            valid_mask=burn_mask_pk,
                            circ_frac_err=burn_circ_err_pk,
                            circ_valid_mask=burn_mask_pk,
                            turbulent_radius_pc=args.turbulent_radius_pc,
                            screen_scale_cm=args.screen_scale_cm,
                        )
                elif args.method == "qu_fitting":
                    result_pk = fitter_pk.fit_rm_qufitting()
                    if result_pk.get("success"):
                        print(f"  RM = {result_pk['rm']:.4f} +/- {result_pk['rm_err']:.4f} rad/m^2")
                        if not args.no_plot:
                            plot_burns_law_fits(
                                fitter_pk,
                                f"{output_prefix_pk}_burns_law.{args.ext}",
                                pol_frac_err=burn_err_pk,
                                valid_mask=burn_mask_pk,
                                circ_frac_err=burn_circ_err_pk,
                                circ_valid_mask=burn_mask_pk,
                                turbulent_radius_pc=args.turbulent_radius_pc,
                                screen_scale_cm=args.screen_scale_cm,
                            )
                elif args.method == "rmnest":
                    outdir_pk = (args.rmnest_outdir or f"{args.output}_rmnest") + f"_peak{i_extra}"
                    label_pk = (args.rmnest_label or args.output) + f"_peak{i_extra}"
                    try:
                        result_pk = fitter_pk.fit_rm_rmnest(
                            gfr=args.rmnest_gfr,
                            free_alpha=args.rmnest_free_alpha,
                            outdir=outdir_pk,
                            label=label_pk,
                            sampler=args.rmnest_sampler,
                        )
                        print(f"  RMNest output directory: {result_pk['rmnest_outdir']}")
                    except ImportError as exc:
                        print(f"  RMNest unavailable for peak {i_extra}: {exc}")
# TIME SERIES MODE
    else:
        if stokes_i.ndim != 2:
            print("\nError: Time series mode requires 2D data arrays.")
            return

        if args.method == "rmnest":
            print("\nNote: RMNest time-series fitting can be slow. Outputs will be written per time bin.")

        print(f"\nProcessing time series data: {stokes_i.shape[time_axis]} time samples")
        print(f"Using method: {args.method}")
        print("This may take a while...")

        full_time_series_data = {
            "time": time_array if time_array is not None else np.arange(stokes_i.shape[time_axis]),
            "I": stokes_i,
            "Q": stokes_q,
            "U": stokes_u,
        }
        if stokes_v is not None:
            full_time_series_data["V"] = stokes_v

        ts_regions: List[Tuple[int, int]] = []
        if onpulse_regions is not None and len(onpulse_regions) > 0:
            ts_regions = onpulse_regions
        elif onpulse_mask is not None:
            ts_regions = [onpulse_mask]
        else:
            n_time = stokes_i.shape[time_axis]
            ts_regions = [(0, n_time - 1)]

        if onpulse_mask is not None:
            m_start, m_end = onpulse_mask
            if time_axis == 0:
                plot_ts_data = {
                    "time": (time_array if time_array is not None else np.arange(stokes_i.shape[time_axis]))[m_start:m_end + 1],
                    "I": stokes_i[m_start:m_end + 1, :],
                    "Q": stokes_q[m_start:m_end + 1, :],
                    "U": stokes_u[m_start:m_end + 1, :],
                }
                if stokes_v is not None:
                    plot_ts_data["V"] = stokes_v[m_start:m_end + 1, :]
            else:
                plot_ts_data = {
                    "time": (time_array if time_array is not None else np.arange(stokes_i.shape[time_axis]))[m_start:m_end + 1],
                    "I": stokes_i[:, m_start:m_end + 1],
                    "Q": stokes_q[:, m_start:m_end + 1],
                    "U": stokes_u[:, m_start:m_end + 1],
                }
                if stokes_v is not None:
                    plot_ts_data["V"] = stokes_v[:, m_start:m_end + 1]
        else:
            plot_ts_data = full_time_series_data

        all_rm_results: List[Dict] = []

        for region_idx, (r_start, r_end) in enumerate(ts_regions):
            print(f"\n  --- Fitting region {region_idx + 1}: time bins {r_start} to {r_end} ---")

            if time_axis == 0:
                region_data = {
                    "time": (time_array if time_array is not None else np.arange(stokes_i.shape[time_axis]))[r_start:r_end + 1],
                    "I": stokes_i[r_start:r_end + 1, :],
                    "Q": stokes_q[r_start:r_end + 1, :],
                    "U": stokes_u[r_start:r_end + 1, :],
                }
                if stokes_v is not None:
                    region_data["V"] = stokes_v[r_start:r_end + 1, :]
            else:
                region_data = {
                    "time": (time_array if time_array is not None else np.arange(stokes_i.shape[time_axis]))[r_start:r_end + 1],
                    "I": stokes_i[:, r_start:r_end + 1],
                    "Q": stokes_q[:, r_start:r_end + 1],
                    "U": stokes_u[:, r_start:r_end + 1],
                }
                if stokes_v is not None:
                    region_data["V"] = stokes_v[:, r_start:r_end + 1]

            rm_result = fit_rm_time_series(
                freq_hz,
                region_data,
                method=args.method,
                rm_range=tuple(args.rm_range),
                n_rm=args.n_rm,
                rmnest_gfr=args.rmnest_gfr,
                rmnest_free_alpha=args.rmnest_free_alpha,
                rmnest_outdir=args.rmnest_outdir or f"{args.output}_rmnest_ts",
                rmnest_label=args.rmnest_label or args.output,
                rmnest_sampler=args.rmnest_sampler,
                n_time_bins=args.time_bins,
                exclude_edge_bins=0,
                noise_fraction=args.offpulse,
                offpulse_std=off_std
            )

            l_weights_region = None
            if "L_frac_bin" in rm_result:
                l_weights_region = np.asarray(rm_result["L_frac_bin"], dtype=float) ** 2
            rm_diag_region = time_series_sigma_rm_diagnostic(rm_result["rm"], weights=l_weights_region)

            print(f"    RM bins used = {rm_diag_region['n_valid']}/{rm_diag_region['n_total']}")
            print(f"    Mean RM = {rm_diag_region['rm_mean']:.4f} rad/m^2")
            print(f"    std_RM(time) = {rm_diag_region['std_rm_time']:.4f} rad/m^2")
            if np.isfinite(rm_diag_region["weighted_std_rm_time"]):
                print(f"    Weighted Mean RM (L^2) = {rm_diag_region['weighted_rm_mean']:.4f} rad/m^2")
                print(f"    Weighted std_RM(time) (L^2) = {rm_diag_region['weighted_std_rm_time']:.4f} rad/m^2")
            print(f"    Min RM = {rm_diag_region['rm_min']:.4f} rad/m^2")
            print(f"    Max RM = {rm_diag_region['rm_max']:.4f} rad/m^2")

            if "rm_err" in rm_result and np.any(np.isfinite(rm_result["rm_err"])):
                print(f"    Mean RM err = {np.nanmean(rm_result['rm_err']):.4f} rad/m^2")

            if "snr" in rm_result and np.any(rm_result["snr"] > 0):
                print(f"    Mean SNR = {np.nanmean(rm_result['snr']):.2f}")

            if "pa_deg" in rm_result:
                print(f"    Mean PA = {np.nanmean(rm_result['pa_deg']):.2f} deg")
                print(f"    Mean EA = {np.nanmean(rm_result['ea_deg']):.2f} deg")

            all_rm_results.append(rm_result)

        rm_results = {}
        _concat_keys = ['time', 'rm', 'rm_err', 'snr', 'pa_deg', 'ea_deg',
                        'pa_err', 'ea_err', 'i_snr', 'pol_angle_0',
                        'pol_angle_ref', 'P_frac_bin', 'L_frac_bin',
                        'V_frac_bin', 'q_bin', 'u_bin', 'v_bin',
                        'time_bin_start', 'time_bin_end']
        for key in _concat_keys:
            parts = [r[key] for r in all_rm_results if key in r]
            if parts:
                rm_results[key] = np.concatenate(parts, axis=0)
            else:
                rm_results[key] = np.array([])
        rm_results['is_binned'] = any(
            r.get('is_binned', False) for r in all_rm_results
        )

        l_weights = None
        if "L_frac_bin" in rm_results and len(rm_results["L_frac_bin"]) > 0:
            l_weights = np.asarray(rm_results["L_frac_bin"], dtype=float) ** 2
        rm_diag = time_series_sigma_rm_diagnostic(rm_results["rm"], weights=l_weights)

        print("\nCombined Time Series Results:")
        print(f"  RM bins used = {rm_diag['n_valid']}/{rm_diag['n_total']}")
        print(f"  Mean RM = {rm_diag['rm_mean']:.4f} rad/m^2")
        print(f"  std_RM(time) = {rm_diag['std_rm_time']:.4f} rad/m^2")
        if np.isfinite(rm_diag["weighted_std_rm_time"]):
            print(
                f"  Weighted Mean RM (L^2) = {rm_diag['weighted_rm_mean']:.4f} rad/m^2"
            )
            print(
                f"  Weighted std_RM(time) (L^2) = {rm_diag['weighted_std_rm_time']:.4f} rad/m^2"
            )
        print(f"  Min RM = {rm_diag['rm_min']:.4f} rad/m^2")
        print(f"  Max RM = {rm_diag['rm_max']:.4f} rad/m^2")

        if "rm_err" in rm_results and np.any(np.isfinite(rm_results["rm_err"])):
            print(f"  Mean RM err = {np.nanmean(rm_results['rm_err']):.4f} rad/m^2")

        if "snr" in rm_results and np.any(rm_results["snr"] > 0):
            print(f"  Mean SNR = {np.nanmean(rm_results['snr']):.2f}")

        if "pa_deg" in rm_results:
            print(f"  Mean PA = {np.nanmean(rm_results['pa_deg']):.2f} deg")
            print(f"  Mean EA = {np.nanmean(rm_results['ea_deg']):.2f} deg")

        if not args.no_plot:
            if args.separate_peaks:
                if time_axis == 0:
                    if onpulse_mask is not None:
                        start_idx, end_idx = onpulse_mask
                        full_time_profile = np.zeros(end_idx - start_idx + 1)
                        if plot_ts_data["I"].ndim == 2:
                            full_time_profile = np.sum(plot_ts_data["I"], axis=1)
                    else:
                        full_time_profile = (
                            np.sum(plot_ts_data["I"], axis=1)
                            if plot_ts_data["I"].ndim == 2
                            else None
                        )
                else:
                    if onpulse_mask is not None:
                        start_idx, end_idx = onpulse_mask
                        full_time_profile = np.zeros(end_idx - start_idx + 1)
                        if plot_ts_data["I"].ndim == 2:
                            full_time_profile = np.sum(plot_ts_data["I"], axis=0)
                    else:
                        full_time_profile = (
                            np.sum(plot_ts_data["I"], axis=0)
                            if plot_ts_data["I"].ndim == 2
                            else None
                        )
            else:
                full_time_profile = None

            plot_rm_time_series(
                rm_results["time"],
                rm_results,
                f"{args.output}_time_series.{args.ext}",
                time_profile=full_time_profile,
                separate_peaks=args.separate_peaks,
                min_gap_bins=args.min_gap_bins,
                min_peak_bins=args.min_peak_bins,
                max_merge_gap=args.max_merge_gap,
                time_series_data=full_time_series_data,
                freq_hz=freq_hz,
                n_rm_bins=args.time_bins if args.time_bins and args.time_bins > 0 else None,
                n_pa_bins=args.pa_bins,
                noise_fraction=args.offpulse,
                offpulse_std=off_std,
                full_time_series=full_time_series_data.get('time') if 'time' in full_time_series_data else None,
                peak_mask_bounds=onpulse_regions
            )

            if args.poincare:
                pt_bins = args.time_bins if args.time_bins and args.time_bins > 0 else None
                if args.poincare_subbands and args.poincare_subbands > 1:
                    plot_poincare_sphere_subbands(
                        plot_ts_data,
                        freq_hz,
                        f"{args.output}_poincare.{args.ext}",
                        n_subbands=args.poincare_subbands,
                        n_time_bins=pt_bins,
                        noise_fraction=args.offpulse,
                        time_unit=time_unit,
                        interactive=args.poincare_interactive,
                        force_surface=args.poincare_surface,
                        offpulse_std=full_time_series_data,
                        rm_results=rm_results,
                        circle_fit_mode=args.poincare_circle_fit,
                        circle_fit_segments=circle_segments,
                    )
                else:
                    plot_poincare_sphere(
                        plot_ts_data,
                        f"{args.output}_poincare.{args.ext}",
                        n_time_bins=pt_bins,
                        noise_fraction=args.offpulse,
                        time_unit=time_unit,
                        interactive=args.poincare_interactive,
                        force_surface=args.poincare_surface,
                        rm_results=rm_results,
                        offpulse_std=full_time_series_data,
                        circle_fit_mode=args.poincare_circle_fit,
                        circle_fit_segments=circle_segments,
                    )
                if args.poincare_projections:
                    proj_tag = str(args.poincare_projections).lower()
                    plot_poincare_projections(
                        plot_ts_data,
                        f"{args.output}_poincare_projections_{proj_tag}.{args.ext}",
                        projection_type=args.poincare_projections,
                        n_time_bins=pt_bins,
                        noise_fraction=args.offpulse,
                        time_unit=time_unit,
                        force_surface=args.poincare_surface,
                        rm_results=rm_results,
                        offpulse_std=full_time_series_data,
                        circle_fit_mode=args.poincare_circle_fit,
                        circle_fit_segments=circle_segments,
                        center=tuple(args.poincare_proj_center)
                        if args.poincare_proj_center is not None
                        else None,
                    )

    print("\n" + "=" * 60)
    print("RM fitting completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
