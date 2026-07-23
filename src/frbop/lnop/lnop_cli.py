import argparse

import numpy as np

from frbop.utils.plotting import set_pub_col, set_pub_style

from .core import SearchConfig, run_search
from .plotting import (plot_bin_summary, plot_epsilon,
                       plot_polarization_scatter, plot_time_lag_correlation)


def main():
    parser = argparse.ArgumentParser(
        description="Coherent gravitational lensing search from voltage timestreams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipeline (Kader, Leung et al. 2022, arXiv:2204.06014):\n"
            "  1. Build a matched filter from the burst's intensity profile.\n"
            "  2. Auto-correlate the voltage stream in the time-lag domain.\n"
            "  3. Calibrate noise using off-pulse (burst-free) data.\n"
            "  4. Bin the time-lag spectrum logarithmically; compute chi^2 per bin.\n"
            "  5. Apply delay/significance/polarization vetoes to flag candidates.\n"
            "\n"
            "Examples:\n"
            "  frbop ln Vx.npy Vy.npy --dt 1.25e-9\n"
            "  frbop ln Vx.npy Vy.npy --n-log-bins 10 --output results\n"
        ),
    )

    parser.add_argument("Vx", help="X-polarisation voltage timestream (.npy)")
    parser.add_argument("Vy", help="Y-polarisation voltage timestream (.npy)")

    parser.add_argument("--dt", type=float, default=2.97619048e-9,
                        help="Voltage sample spacing [s] (default: 2.976e-09)")
    parser.add_argument("--min-lag", type=float, default=2.56e-6,
                        help="Minimum absolute lag for first logarithmic bin [s] (default: 2.560e-06)")
    parser.add_argument("--n-log-bins", type=int, default=8,
                        help="Number of logarithmic lag bins (default: 8)")
    parser.add_argument("--n-off-pulse", type=int, default=5,
                        help="Number of off-pulse noise realisations (default: 5)")

    adv = parser.add_argument_group("advanced search-config options")
    adv.add_argument("--frame", type=float, default=None,
                     help="PFB frame duration for delay veto [s] (default: None, veto disabled)")
    adv.add_argument("--delay-veto-tol", type=float, default=0.625e-9,
                     help="Delay veto tolerance [s] (default: 6.250e-10)")
    adv.add_argument("--n-gauss-threshold", type=float, default=1e-2,
                     help="N_gauss significance threshold (default: 1.00e-02)")
    adv.add_argument("--polarization-percentile", type=float, default=99.0,
                     help="Polarisation consistency percentile (default: 99.0)")
    adv.add_argument("--max-excursions", type=int, default=2048,
                     help="Max top chi^2 values kept per bin (default: 2048)")
    adv.add_argument("--off-pulse-gap-widths", type=int, default=5,
                     help="Off-pulse filter shift in burst widths (default: 5)")

    parser.add_argument("-o", "--output", default="lnop_results",
                        help="Output prefix (default: lnop_results)")
    parser.add_argument("--ext", default="png",
                        help="Figure extension (default: png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable all plotting")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-bin diagnostic output")
    parser.add_argument("--pub-col", type=float, default=2,
                        help="Publication figure column count (1, 2, 3, ...). Default: 2")

    args = parser.parse_args()
    set_pub_col(args.pub_col)
    set_pub_style(use_latex=False)

    cfg = SearchConfig(
        dt=args.dt,
        min_lag=args.min_lag,
        frame=args.frame,
        delay_veto_tol=args.delay_veto_tol,
        n_gauss_threshold=args.n_gauss_threshold,
        polarization_percentile=args.polarization_percentile,
        max_excursions_per_bin=args.max_excursions,
    )

    print("=" * 60)
    print("GRAVITATIONAL LENSING SEARCH (Kader, Leung et al. 2022)")
    print("=" * 60)

    print(f"\nLoading voltages ...")
    Vx = np.load(args.Vx)
    Vy = np.load(args.Vy)
    Vx = np.squeeze(Vx)
    Vy = np.squeeze(Vy)
    print(f"  Vx: {Vx.shape}, dtype={Vx.dtype}")
    print(f"  Vy: {Vy.shape}, dtype={Vy.dtype}")

    if Vx.ndim != 1 or Vy.ndim != 1:
        parser.error("Voltage arrays must be 1-D timestreams")

    N = len(Vx)
    duration_s = N * cfg.dt
    print(f"  Samples: {N}, duration: {duration_s:.4e} s")
    print(f"  dt: {cfg.dt:.3e} s, min_lag: {cfg.min_lag:.3e} s"
          + (f", frame: {cfg.frame:.3e} s" if cfg.frame is not None else ""))
    print(f"  Log bins: {args.n_log_bins}, off-pulse realisations: {args.n_off_pulse}")
    print()

    result = run_search(Vx, Vy, cfg,
                        n_off_pulse=args.n_off_pulse,
                        off_pulse_gap_widths=args.off_pulse_gap_widths,
                        n_log_bins=args.n_log_bins,
                        verbose=not args.quiet)

    candidates = result["candidates"]
    print(f"\n{len(candidates)} candidate(s) survived all veto conditions.")

    if candidates:
        print(f"\n{'=' * 60}")
        print("SURVIVING CANDIDATES")
        print(f"{'=' * 60}")
        for i, cand in enumerate(candidates):
            print(f"  Candidate {i + 1}:")
            print(f"    Bin index     : {cand['bin_index']}")
            print(f"    Lag range     : [{cand['lag_range_s'][0]:.3e}, "
                  f"{cand['lag_range_s'][1]:.3e}] s")
            print(f"    τ             : {cand['tau_seconds']:.3e} s")
            print(f"    χ² max        : {cand['chi2_max']:.2f}")
            print(f"    N_gauss       : {cand['ngauss']:.3e}")
            print(f"    ε_X           : {cand['eps_x']:.6f}")
            print(f"    ε_Y           : {cand['eps_y']:.6f}")
            print(f"    Delay veto    : {'PASS' if cand['delay_ok'] else 'FAIL'}")
            print(f"    Significance  : {'PASS' if cand['significance_ok'] else 'FAIL'}")
            print(f"    Polarisation  : {'PASS' if cand['polarization_ok'] else 'FAIL'}")

    np.savez(f"{args.output}_candidates.npz",
             candidates=np.array(candidates) if candidates else np.array([]),
             dt=cfg.dt, min_lag=cfg.min_lag, frame=cfg.frame,
             Vx_file=args.Vx, Vy_file=args.Vy)
    print(f"\nSaved: {args.output}_candidates.npz")

    if not args.no_plot:
        lags_s = result["lags_seconds"]
        Cx = result["Cx"]
        Cy = result["Cy"]
        eps_x = result["eps_x"]
        eps_y = result["eps_y"]
        edges = result["edges"]

        plot_time_lag_correlation(lags_s, Cx, Cy,
                                  candidates=candidates,
                                  output=args.output, ext=args.ext)
        plot_epsilon(lags_s, eps_x, eps_y,
                     candidates=candidates,
                     output=args.output, ext=args.ext)
        plot_bin_summary(candidates, edges,
                         output=args.output, ext=args.ext)
        plot_polarization_scatter(eps_x, eps_y,
                                   candidates=candidates,
                                   output=args.output, ext=args.ext)


if __name__ == "__main__":
    main()
