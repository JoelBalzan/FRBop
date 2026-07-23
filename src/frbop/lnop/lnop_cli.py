import argparse
import os

import numpy as np

from frbop.utils.plotting import set_pub_col, set_pub_style

from .core import (SearchConfig, detect_leakage_period, make_matched_filter,
                    off_pulse_stats, run_search, time_lag_correlation)
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
            "  4. Bin the time-lag spectrum logarithmically; compute chi^2 per\n"
            "     bin against an OFF-PULSE-derived null model.\n"
            "  5. Apply delay/significance/polarization vetoes to flag candidates.\n"
            "\n"
            "--frame / --min-lag / --delay-veto-tol default to None, in which\n"
            "case they're derived from --dt (see SearchConfig.resolve()) rather\n"
            "than assuming CHIME-specific constants. If you don't know your\n"
            "instrument's PFB-inversion leakage period, pass\n"
            "--detect-leakage-period to estimate it empirically from your own\n"
            "off-pulse data before the search runs.\n"
            "\n"
            "Examples:\n"
            "  frbop ln Vx.npy Vy.npy --dt 2.976e-9\n"
            "  frbop ln Vx.npy Vy.npy --detect-leakage-period --n-log-bins 10 "
            "--outdir /scratch/user/frb123 --label frb123\n"
        ),
    )

    parser.add_argument("Vx", help="X-polarisation voltage timestream (.npy)")
    parser.add_argument("Vy", help="Y-polarisation voltage timestream (.npy)")

    parser.add_argument("--dt", type=float, default=2.97619048e-9,
                        help="Voltage sample spacing [s] (default: 2.976e-09, "
                             "i.e. 1/336 MHz -- appropriate for ASKAP/CELEBI "
                             "voltages already PFB-inverted to full band "
                             "resolution; check this matches your data)")
    parser.add_argument("--n-log-bins", type=int, default=8,
                        help="Number of logarithmic lag bins (default: 8)")
    parser.add_argument("--n-off-pulse", type=int, default=5,
                        help="Number of off-pulse noise realisations (default: 5)")

    adv = parser.add_argument_group("advanced search-config options")
    adv.add_argument("--frame", type=float, default=None,
                     help="PFB-inversion leakage period for the delay veto [s]. "
                          "Default: None (veto disabled). Use "
                          "--detect-leakage-period to estimate this from your "
                          "own data, or set explicitly if known.")
    adv.add_argument("--min-lag", type=float, default=None,
                     help="Minimum |lag| for the first logarithmic bin [s]. "
                          "Default: None -> derived as 4*frame if --frame is "
                          "set, else falls back to CHIME's 2.56e-6 s (verify "
                          "this is meaningful for your instrument).")
    adv.add_argument("--delay-veto-tol", type=float, default=None,
                     help="Delay veto tolerance [s]. Default: None -> dt/2 "
                          "(half a native sample), not CHIME's fixed 0.625 ns.")
    adv.add_argument("--n-gauss-threshold", type=float, default=1e-2,
                     help="N_gauss significance threshold (default: 1.00e-02)")
    adv.add_argument("--polarization-percentile", type=float, default=99.0,
                     help="Polarisation consistency percentile (default: 99.0)")
    adv.add_argument("--max-excursions", type=int, default=2048,
                     help="Max top chi^2 values kept per bin (default: 2048)")
    adv.add_argument("--off-pulse-gap-widths", type=int, default=5,
                     help="Off-pulse filter shift in burst widths (default: 5)")
    adv.add_argument("--detect-leakage-period", action="store_true",
                     help="Empirically scan the off-pulse correlation for a "
                          "periodic leakage comb before the search, and use "
                          "the best match for --frame if --frame wasn't set "
                          "explicitly. Always inspect the printed candidates "
                          "yourself -- this is a starting point, not a "
                          "substitute for knowing your pipeline's PFB "
                          "inversion parameters.")

    parser.add_argument("-o", "--outdir", default=".",
                        help="Output directory (default: .)")
    parser.add_argument("--label", default="lnop",
                        help="Run label / output prefix (default: lnop)")
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
    os.makedirs(args.outdir, exist_ok=True)
    output_prefix = os.path.join(args.outdir, args.label)

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
    print(f"  dt: {cfg.dt:.3e} s")

    # ---- Optional: empirically estimate the PFB-inversion leakage period ----
    if args.detect_leakage_period:
        print("\nRunning preliminary PFB leakage-period detection "
              "(off-pulse data only) ...")
        Wx2_prelim, Wy2_prelim, mask_prelim = make_matched_filter(Vx, Vy)
        burst_width = max(int(mask_prelim.sum()), 1)
        off_shift = args.off_pulse_gap_widths * burst_width
        lags_x_prelim, _ = time_lag_correlation(Vx, Wx2_prelim)
        lags_seconds_prelim = lags_x_prelim * cfg.dt
        _, Cx_off_stack_prelim = off_pulse_stats(Vx, Wx2_prelim, off_shift,
                                                  args.n_off_pulse)
        detected_period, _ = detect_leakage_period(
            lags_seconds_prelim, Cx_off_stack_prelim, cfg.dt)

        if detected_period is None:
            print("  No clear periodic leakage detected in the searched "
                  "range -- leaving --frame as given.")
        else:
            print(f"  Best candidate leakage period: {detected_period:.4e} s")
            if cfg.frame is None:
                cfg.frame = detected_period
                print(f"  --frame not set explicitly; using this detected "
                      f"period. VERIFY this against the printed candidates "
                      f"above before trusting the delay veto on real "
                      f"candidates.")
            else:
                print(f"  --frame was set explicitly to {cfg.frame:.4e} s; "
                      f"leaving it unchanged (detection is informational).")

    # ---- Print resolved (instrument-derived, not hardcoded) config ----
    resolved_tol, resolved_min_lag = cfg.resolve()
    print(f"\n  frame (delay veto period): "
          f"{cfg.frame if cfg.frame is not None else 'disabled (None)'}")
    print(f"  delay_veto_tol (resolved): {resolved_tol:.3e} s"
          + ("" if args.delay_veto_tol is not None else "  [derived as dt/2]"))
    print(f"  min_lag (resolved)       : {resolved_min_lag:.3e} s"
          + ("" if args.min_lag is not None else
             ("  [derived as 4*frame]" if cfg.frame is not None
              else "  [CHIME fallback -- verify this applies to your data]")))
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

    np.savez(f"{output_prefix}_candidates.npz",
             candidates=np.array(candidates, dtype=object) if candidates else np.array([]),
             bin_diagnostics=np.array(result["bin_diagnostics"], dtype=object),
             dt=cfg.dt, min_lag=result["resolved_min_lag"],
             frame=cfg.frame if cfg.frame is not None else np.nan,
             delay_veto_tol=result["resolved_delay_veto_tol"],
             Vx_file=args.Vx, Vy_file=args.Vy)
    print(f"\nSaved: {output_prefix}_candidates.npz")

    if not args.no_plot:
        lags_s = result["lags_seconds"]
        Cx = result["Cx"]
        Cy = result["Cy"]
        eps_x = result["eps_x"]
        eps_y = result["eps_y"]
        edges = result["edges"]

        plot_time_lag_correlation(lags_s, Cx, Cy,
                                  candidates=candidates,
                                  output=output_prefix, ext=args.ext)
        plot_epsilon(lags_s, eps_x, eps_y,
                     candidates=candidates,
                     output=output_prefix, ext=args.ext)
        plot_bin_summary(candidates, edges,
                         output=output_prefix, ext=args.ext)
        plot_polarization_scatter(eps_x, eps_y,
                                   candidates=candidates,
                                   output=output_prefix, ext=args.ext)


if __name__ == "__main__":
    main()