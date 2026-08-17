"""
CLI for RVM fitting of Stokes Q/U dynamic spectra.

Fits the Rotating Vector Model (Everett & Weisberg 2001) to the
polarisation angle as a function of rotational phase within an FRB burst.

When no rotation period is known (typical for FRBs), the phase sweep rate
*k* (rad/ms) is treated as a free parameter — the 5-parameter fit
(α, ζ, φ₀, ψ₀, k) avoids needing a period.

Noise is estimated from the first 10% of time bins (off-pulse baseline)
when 2D dynamic spectra are provided, or from the median absolute deviation
when 1D data is given.

Examples::

  # Basic fit (k-mode — no period needed)
  frbop pa -i I.npy -q Q.npy -u U.npy --time time.npy

  # Larger grid for better global minimum search
  frbop pa -i I.npy -q Q.npy -u U.npy --time time.npy \\
      --n-alpha 60 --n-zeta 60 --n-phi 101

  # With Stokes cube
  frbop pa -c stokes_cube.npy --time time.npy
"""

import argparse
import os
import warnings

import numpy as np

from frbop.utils.linear_pol import debiased_linear_from_qu
from frbop.utils.noise import noise_from_first_fraction
from frbop.utils.scrunch import tscrunch_array
from frbop.utils.significance import build_pa_mask

from .rvm_fitter import fit_rvm
from .rvm_plotting import plot_grid_chi2, plot_rvm_corner, plot_rvm_fit

warnings.filterwarnings("ignore")


def _load(path: str) -> np.ndarray:
    """Load .npy or .txt, squeeze singleton dims."""
    d = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    return np.squeeze(d)


def _estimate_noise_from_offpulse(q_2d: np.ndarray, u_2d: np.ndarray,
                                   time_axis: int, frac: float = 0.1
                                   ) -> tuple:
    """Estimate per-bin Q/U noise from the first *frac* of time bins."""
    n_time = q_2d.shape[time_axis]
    n_off = max(1, int(n_time * frac))

    if time_axis == 0:
        q_off = q_2d[:, :n_off]
        u_off = u_2d[:, :n_off]
    else:
        q_off = q_2d[:n_off, :]
        u_off = u_2d[:n_off, :]

    q_std_chan = np.nanstd(q_off, axis=1 if time_axis == 0 else 0)
    u_std_chan = np.nanstd(u_off, axis=1 if time_axis == 0 else 0)

    sigma_q = float(np.nanmedian(q_std_chan[q_std_chan > 0])
                    if np.any(q_std_chan > 0) else 1.0)
    sigma_u = float(np.nanmedian(u_std_chan[u_std_chan > 0])
                    if np.any(u_std_chan > 0) else 1.0)
    return sigma_q, sigma_u


def _build_pa_mask(stokes_q: np.ndarray, stokes_u: np.ndarray,
                   stokes_i: np.ndarray | None = None,
                   li_i_sigma_cut: float = 2.0,
                   pa_fit_post_peak_only: bool = False,
                   pa_min_run: int = 3) -> np.ndarray:
    """Build the same PA significance mask used by the time-series pipeline."""
    q = np.asarray(stokes_q, dtype=float).ravel()
    u = np.asarray(stokes_u, dtype=float).ravel()
    q_rms = float(noise_from_first_fraction(q, frac=0.05))
    u_rms = float(noise_from_first_fraction(u, frac=0.05))

    l_debias, sigma_l, _ = debiased_linear_from_qu(q, u, q_rms, u_rms)

    i_ts = None
    sigma_i = med_i = None
    if stokes_i is not None:
        i = np.asarray(stokes_i, dtype=float).ravel()
        if i.shape != l_debias.shape:
            raise ValueError("Stokes I, Q, and U must have matching lengths for PA masking")
        i_ts = i
        sigma_i = float(noise_from_first_fraction(i, frac=0.05))
        med_i = float(np.nanmedian(i))

    return build_pa_mask(
        l_debias, sigma_l,
        i_ts=i_ts, sigma_i=sigma_i, med_i=med_i,
        li_i_sigma_cut=li_i_sigma_cut,
        post_peak_only=pa_fit_post_peak_only,
        min_run=pa_min_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RVM fitting for Stokes Q/U dynamic spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  frbop pa -q Q.npy -u U.npy --time time.npy\n"
            "  frbop pa -c stokes_cube.npy --time time.npy\n"
        ),
    )

    # Input data
    parser.add_argument("-i", "--stokes-i", help="Stokes I file (.npy / .txt)")
    parser.add_argument("-q", "--stokes-q", default=None,
                        help="Stokes Q file (.npy / .txt)")
    parser.add_argument("-u", "--stokes-u", default=None,
                        help="Stokes U file (.npy / .txt)")
    parser.add_argument("-c", "--stokes-cube", default=None,
                        help="Stokes cube (I, Q, U, [V]) — axis 0 is Stokes")
    parser.add_argument("-t", "--time", help="Time file (ms)")

    # Averaging
    parser.add_argument("--time-avg", action="store_true",
                        help="Average over time axis (2D → 1D)")
    parser.add_argument("--freq-avg", action="store_true",
                        help="Average over frequency axis (2D → 1D)")
    parser.add_argument("-tscr", "--tscrunch", type=int, default=1,
                        help="Time scrunch factor applied before PA masking and fitting (average every N time bins)")

    # PA masking
    parser.add_argument("--pa-min-run", type=int, default=3,
                        help="Minimum consecutive PA samples to keep after masking (default: 3)")
    parser.add_argument("--pa-fit-post-peak-only", action="store_true",
                        help="Restrict PA masking to samples at or after the Stokes-I peak")
    parser.add_argument("--li-sigma-cut", type=float, default=2.0,
                        help="Stokes-I significance cutoff in sigma for PA masking (default: 2.0)")

    # Fit configuration
    parser.add_argument("--n-alpha", type=int, default=40,
                        help="Grid points in α (default: 40)")
    parser.add_argument("--n-zeta", type=int, default=40,
                        help="Grid points in ζ (default: 40)")
    parser.add_argument("--n-phi", type=int, default=101,
                        help="Grid points in φ₀ per (α,ζ) cell (default: 101)")
    parser.add_argument("--n-k", type=int, default=20,
                        help="Grid points in k scan (default: 20)")
    parser.add_argument("--no-lm", action="store_true",
                        help="Skip Nelder-Mead refinement step")
    parser.add_argument("--no-mcmc", action="store_true",
                        help="Skip MCMC posterior sampling")

    # MCMC settings
    parser.add_argument("--mcmc-walkers", type=int, default=32)
    parser.add_argument("--mcmc-steps", type=int, default=3000)
    parser.add_argument("--mcmc-burn", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=5)
    parser.add_argument("--mcmc-silent", action="store_true",
                        help="Suppress MCMC progress bar")

    # Output
    parser.add_argument("-o", "--output", default="paop_results",
                        help="Output prefix (default: paop_results)")
    parser.add_argument("--ext", default="png",
                        help="Figure extension (default: png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable all plotting")

    args = parser.parse_args()

    if not args.stokes_cube and (args.stokes_q is None or args.stokes_u is None):
        parser.error("Provide either --stokes-cube or both --stokes-q and --stokes-u")

    # ── Data loading ────────────────────────────────────────────────
    print("=" * 60)
    print("RVM FITTING FOR STOKES Q/U DATA")
    print("=" * 60)

    if args.stokes_cube:
        print("\nLoading Stokes cube ...")
        cube = np.load(args.stokes_cube)
        if cube.ndim < 2:
            parser.error("Stokes cube must have at least 2 dims")
        stokes_i = cube[0]
        stokes_q = cube[1]
        stokes_u = cube[2]
    else:
        stokes_i = _load(args.stokes_i) if args.stokes_i else None
        stokes_q = _load(args.stokes_q)
        stokes_u = _load(args.stokes_u)

    data_ndim = stokes_q.ndim
    n_freq, n_time = (stokes_q.shape if data_ndim == 2 else (1, stokes_q.size))
    print(f"\n  Data dimensions: {data_ndim}D  "
          f"({'n_freq=' + str(n_freq) + ', ' if data_ndim == 2 else ''}"
          f"n_time={n_time})")

    # ── Noise estimation (before averaging) ─────────────────────────
    if data_ndim == 2:
        sigma_q, sigma_u = _estimate_noise_from_offpulse(
            stokes_q, stokes_u, time_axis=-1, frac=0.1
        )
        print(f"  Off-pulse noise: σ_Q = {sigma_q:.4g}, σ_U = {sigma_u:.4g}")
    else:
        sigma_q = float(np.median(np.abs(stokes_q - np.median(stokes_q)))
                        / 0.6745)
        sigma_u = float(np.median(np.abs(stokes_u - np.median(stokes_u)))
                        / 0.6745)
        sigma_q = max(sigma_q, 1e-15)
        sigma_u = max(sigma_u, 1e-15)
        print(f"  MAD-based noise: σ_Q = {sigma_q:.4g}, σ_U = {sigma_u:.4g}")

    # ── Averaging (default: freq-average 2D → 1D time series) ──────
    if data_ndim == 2:
        if args.freq_avg and args.time_avg:
            stokes_q = np.atleast_1d(np.nanmean(stokes_q))
            stokes_u = np.atleast_1d(np.nanmean(stokes_u))
            if stokes_i is not None:
                stokes_i = np.atleast_1d(np.nanmean(stokes_i))
            n_scale = np.sqrt(n_freq * n_time)
            sigma_q /= n_scale
            sigma_u /= n_scale
            print(f"  Full average: σ_Q = {sigma_q:.4g}, σ_U = {sigma_u:.4g}")
        elif args.freq_avg or args.time_avg:
            axis = 0 if args.time_avg else -1
            label = "Time average (→ 1D spectrum)" if args.time_avg else "Frequency average (→ 1D time series)"
            scale = np.sqrt(n_time if args.time_avg else n_freq)
            sigma_q /= scale
            sigma_u /= scale
            stokes_q = np.nanmean(stokes_q, axis=axis)
            stokes_u = np.nanmean(stokes_u, axis=axis)
            if stokes_i is not None:
                stokes_i = np.nanmean(stokes_i, axis=axis)
            print(f"  {label}: σ_Q = {sigma_q:.4g}, σ_U = {sigma_u:.4g}")
        else:
            sigma_q /= np.sqrt(n_freq)
            sigma_u /= np.sqrt(n_freq)
            stokes_q = np.nanmean(stokes_q, axis=0)
            stokes_u = np.nanmean(stokes_u, axis=0)
            if stokes_i is not None:
                stokes_i = np.nanmean(stokes_i, axis=0)
            print(f"  Auto frequency-average (→ 1D time series): "
                  f"σ_Q = {sigma_q:.4g}, σ_U = {sigma_u:.4g}")

    stokes_q = np.asarray(stokes_q, dtype=float).ravel()
    stokes_u = np.asarray(stokes_u, dtype=float).ravel()
    if stokes_i is not None:
        stokes_i = np.asarray(stokes_i, dtype=float).ravel()

    n_pts = len(stokes_q)
    print(f"\n  Data points: {n_pts}")
    print(f"  Stokes Q range: [{stokes_q.min():.4g}, {stokes_q.max():.4g}]")
    print(f"  Stokes U range: [{stokes_u.min():.4g}, {stokes_u.max():.4g}]")

    # ── Time / phase array ──────────────────────────────────────────
    if args.time:
        time_vals = _load(args.time)
        if len(time_vals) != n_pts:
            parser.error(f"Time array length {len(time_vals)} != "
                         f"{n_pts} data points")
        time_vals = np.asarray(time_vals, dtype=float).ravel()
        burst_dur = np.ptp(time_vals)
        print(f"  Time range: [{time_vals.min():.4f}, {time_vals.max():.4f}] ms"
              f"  (duration {burst_dur:.4f} ms)")
    else:
        time_vals = np.linspace(0, 1.0, n_pts)
        burst_dur = 1.0
        print("  No time file — using dummy 0–1 ms")

    if args.tscrunch < 1:
        parser.error(f"--tscrunch must be >= 1, got {args.tscrunch}")
    if args.tscrunch > 1:
        print(f"  Time scrunch factor: {args.tscrunch}")
        n_pts_orig = n_pts
        stokes_q = tscrunch_array(stokes_q, args.tscrunch)
        stokes_u = tscrunch_array(stokes_u, args.tscrunch)
        time_vals = tscrunch_array(time_vals, args.tscrunch)
        if stokes_i is not None:
            stokes_i = tscrunch_array(stokes_i, args.tscrunch)
        n_pts = len(stokes_q)
        if n_pts < 4:
            parser.error(
                f"Time scrunching reduced the data from {n_pts_orig} to {n_pts} samples; need at least 4 for RVM fitting"
            )
        print(f"  Time samples: {n_pts_orig} -> {n_pts}")

    # ── PA masking ─────────────────────────────────────────────────
    pa_mask = _build_pa_mask(
        stokes_q,
        stokes_u,
        stokes_i=stokes_i,
        li_i_sigma_cut=args.li_sigma_cut,
        pa_fit_post_peak_only=args.pa_fit_post_peak_only,
        pa_min_run=args.pa_min_run,
    )
    n_masked = int(np.sum(pa_mask))
    if n_masked < 4:
        parser.error(
            f"PA masking left {n_masked} samples; need at least 4 for RVM fitting"
        )
    print(
        f"  PA mask: keeping {n_masked}/{n_pts} samples "
        f"(L significance, I threshold, min run = {args.pa_min_run})"
    )

    stokes_q = np.asarray(stokes_q, dtype=float)[pa_mask]
    stokes_u = np.asarray(stokes_u, dtype=float)[pa_mask]
    time_vals = np.asarray(time_vals, dtype=float)[pa_mask]
    if stokes_i is not None:
        stokes_i = np.asarray(stokes_i, dtype=float)[pa_mask]
    n_pts = len(stokes_q)

    # ── Fit ─────────────────────────────────────────────────────────
    print("\nRunning RVM fit...")
    print(f"  Grid: α = {args.n_alpha} × ζ = {args.n_zeta} "
          f"(φ₀ = {args.n_phi} per cell, k = {args.n_k})")
    print(f"  Nelder-Mead refine: {'no' if args.no_lm else 'yes'}")
    print(f"  MCMC: {'no' if args.no_mcmc else 'yes'}")

    result = fit_rvm(
        q=stokes_q, u=stokes_u,
        time=time_vals,
        sigma_q=sigma_q, sigma_u=sigma_u,
        n_alpha=args.n_alpha, n_zeta=args.n_zeta, n_phi=args.n_phi,
        n_k=args.n_k,
        do_lm=not args.no_lm,
        do_mcmc=not args.no_mcmc,
        mcmc_walkers=args.mcmc_walkers,
        mcmc_steps=args.mcmc_steps,
        mcmc_burn=args.mcmc_burn,
        mcmc_thin=args.mcmc_thin,
        mcmc_progress=not args.mcmc_silent,
    )

    # ── Results ─────────────────────────────────────────────────────
    alpha = result["best_alpha"]
    zeta = result["best_zeta"]
    beta = result["best_beta"]
    best_k = result.get("best_k", None)
    delta_phi = best_k * burst_dur if best_k is not None else None

    print("\n" + "─" * 40)
    print("BEST-FIT RVM PARAMETERS")
    print("─" * 40)
    print(f"  α = {np.degrees(alpha):.2f}°   "
          f"ζ = {np.degrees(zeta):.2f}°   "
          f"β = {np.degrees(beta):.2f}°")
    print(f"  φ₀ = {result['best_phi0']:.4f} rad  "
          f"({np.degrees(result['best_phi0']):.2f}°)")
    print(f"  ψ₀ = {result['best_psi0']:.4f} rad  "
          f"({np.degrees(result['best_psi0']):.2f}°)")
    if best_k is not None and delta_phi is not None:
        print(f"  k  = {best_k:.4f} rad/ms  "
              f"(Δφ = {delta_phi:.4f} rad = {delta_phi / (2*np.pi):.4f} cycles)")
    print(f"  χ² = {result['best_chi2']:.2f}  (N_data = {n_pts})")
    alt_a = np.pi - alpha
    alt_z = np.pi - zeta
    alt_b = alt_z - alt_a
    print(f"\n  Complementary: α' = {np.degrees(alt_a):.2f}°  "
          f"ζ' = {np.degrees(alt_z):.2f}°  β' = {np.degrees(alt_b):.2f}°")

    if result["mcmc"] is not None:
        mc = result["mcmc"]
        flat = mc["flatchain"]
        p16, p50, p84 = np.percentile(flat, [16, 50, 84], axis=0)
        ndim = mc["ndim"]
        labels = [r"φ₀", r"ψ₀", r"α", r"ζ"]
        if ndim == 5:
            labels.append(r"k")
        print("\nMCMC 16th–50th–84th percentiles:")
        for j, lab in enumerate(labels):
            print(f"  {lab} = "
                  f"{np.degrees(p50[j]):.2f}° "
                  f"+{np.degrees(p84[j]) - np.degrees(p50[j]):.2f}°"
                  f"/{np.degrees(p50[j]) - np.degrees(p16[j]):.2f}°"
                  f"  ({p50[j]:.4f} rad)"
                  if lab != "k" else
                  f"  {lab} = {p50[j]:.4f} "
                  f"+{p84[j] - p50[j]:.4f}"
                  f"/{p50[j] - p16[j]:.4f}  rad/ms")
        print(f"  Acceptance fraction: {mc['acceptance_fraction']:.3f}")

    # ── Plots ───────────────────────────────────────────────────────
    if not args.no_plot:
        print("")
        for label, fn in [
            ("PA fit", f"{args.output}_rvm_fit.{args.ext}"),
            ("χ² grid", f"{args.output}_rvm_grid.{args.ext}"),
            ("corner", f"{args.output}_rvm_corner.{args.ext}"),
        ]:
            out = os.path.join(os.getcwd(), fn)
            print(f"  {label}: {out}")

        plot_rvm_fit(time_vals, stokes_q, stokes_u, result,
                     save_path=f"{args.output}_rvm_fit.{args.ext}")
        plot_grid_chi2(result,
                       save_path=f"{args.output}_rvm_grid.{args.ext}")
        if result["mcmc"] is not None:
            plot_rvm_corner(result,
                            save_path=f"{args.output}_rvm_corner.{args.ext}")

    # ── Save results ────────────────────────────────────────────────
    out_npz = f"{args.output}_rvm_result.npz"
    np.savez(out_npz,
             time=time_vals,
             q=stokes_q,
             u=stokes_u,
             sigma_q=sigma_q,
             sigma_u=sigma_u,
             best_phi0=result["best_phi0"],
             best_psi0=result["best_psi0"],
             best_alpha=result["best_alpha"],
             best_zeta=result["best_zeta"],
             best_beta=result["best_beta"],
             best_k=result.get("best_k", np.nan),
             best_chi2=result["best_chi2"],
             best_pa=result["best_pa"],
             best_L=result.get("best_L", np.array([])))
    print(f"\n  Saved: {out_npz}")
    print("Done.")


if __name__ == "__main__":
    main()
