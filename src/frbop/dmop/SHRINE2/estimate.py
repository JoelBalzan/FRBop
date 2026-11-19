"""The structure-maximising DM and its 1-sigma uncertainty from one set of profiles I(d_j, t).

Paper: Section 3.3, steps in the order of Sections 3.3.1-3.3.4. Each step below names the
equation or section it evaluates.
"""
import numpy as np

from .covariance import noise_covariance
from .fit import fit_parabola, fit_window
from .noise import filtered_noise_ratio, noise_variance, sigma_S
from .structure import coefficients, structure_parameter


def dm_uncertainty(I, d, kc, dt_nu, share=None, cov_n=None, verbose=False):
    """The structure-maximising DM and its 1-sigma uncertainty, sigma_DM, from the profiles
    I(d_j, t): the whole of the paper's Section 3.3.

    In order: Sp at every trial DM; sigma_t^2 and sigma_S at the peak; the fit window; the
    parabola and its vertex, d_fit; the noise covariance V between the trial DMs in the window;
    and from these, sigma_DM.

    By default the noise is taken as white, with sigma_t^2 estimated from the profile at d_max
    (Sec. 3.3.2). For noise correlated between samples, pass its covariance as cov_n, measured
    away from the burst (e.g. Toeplitz, from the autocovariance of off-pulse profiles, for
    stationary noise): it then replaces diag(sigma_t^2) in sigma_S (Eq. sigSgen), in V (Eq. cov)
    and in e_ratio, and sigma_t^2 is not estimated (Sec. 5.3).

    The covariance does not need uniformly spaced trial DMs, but check_trial_dms and the verbose
    output take the step as d[1] - d[0].

    Args:
        I: I(d_j, t), shape (n_d, N), one row per trial DM.
        d: the trial DMs, shape (n_d,), pc cm^-3.
        kc: filter cutoff k_c.
        dt_nu: each channel's dispersion delay per unit DM in samples, shape (n_chan,)
            (profiles.channel_delays divided by dt).
        share: each channel's share of the noise variance, shape (n_chan,); None for equal shares.
        cov_n: Cov(n) of the frequency-summed profile noise, shape (N, N) (or its diagonal, shape
            (N,)), at d_max; None to estimate sigma_t^2 from the profile (white noise).
        verbose: print each key result of Section 3.3 as it is found, with its equation.
    Returns:
        dict with
            ok, why: whether the window and fit succeeded, and why not;
            d, Sp, kc: the inputs and Sp(d_j);
            j_max, d_max, Sp_max: the trial DM with the largest Sp;
            sig_S, sig2, e_ratio: sigma_S, sigma_t^2 (the diagonal of cov_n, if given) and
                sqrt(<||e||^2>)/Sp at d_max;
            and, when ok:
            win, n_win, d_lo, d_hi: the fit window, its size and its first and last trial DM;
            d_fit, c, B: the vertex, (c0, c1, c2) about it, and B;
            V: the covariance between the trial DMs in the window;
            sig_DM, c2_err: sigma_DM and std[c2]/|c2|.
    """
    d = np.asarray(d, float)

    # Sp(d_j) at every trial DM (Eq. sp), and d_max, the trial DM with the largest Sp.
    Sp = structure_parameter(I, kc)
    j_max = int(np.argmax(Sp))
    if verbose:
        print(f"  Sp(d_j) (Eq. sp): largest at d_max = {d[j_max]:.6g} "
              f"(trial DM {j_max} of {d.size}), Sp_max = {Sp[j_max]:.4g}")

    # Cov(n): diag(sigma_t^2), with sigma_t^2 from the profile at d_max (Sec. 3.3.2), unless
    # given (Sec. 5.3).
    if cov_n is None:
        sig2 = noise_variance(I[j_max], kc, verbose=verbose)
        cov = sig2
    else:
        cov = np.asarray(cov_n, float)
        if cov.shape not in [(I.shape[1],), (I.shape[1], I.shape[1])]:
            raise ValueError(f"cov_n has shape {cov.shape}; expected ({I.shape[1]}, {I.shape[1]}) "
                             f"or ({I.shape[1]},)")
        sig2 = np.diag(cov).copy() if cov.ndim == 2 else cov
        if verbose:
            print(f"  Cov(n) given (Sec. 5.3), {'full' if cov.ndim == 2 else 'diagonal'}: variance "
                  f"per sample {np.mean(sig2):.4g} on average; sigma_t^2 not estimated")

    # w at d_max (Eq. ahat), and sigma_S (Eq. sigSgen; Eq. sigS for white noise).
    w_max = coefficients(I[j_max], kc)
    sig_S = sigma_S(w_max, cov)
    if verbose:
        print(f"  sigma_S (Eq. sigS) = {sig_S:.4g}, i.e. {sig_S / Sp[j_max]:.3g} of Sp_max")

    # The results so far; the fit below adds to them.
    out = dict(
        ok=False,
        why="",
        d=d,
        Sp=Sp,
        kc=kc,
        j_max=j_max,
        d_max=d[j_max],
        Sp_max=Sp[j_max],
        sig_S=sig_S,
        sig2=sig2,
        e_ratio=filtered_noise_ratio(cov, kc, Sp[j_max]),
    )
    if verbose:
        print(f"  sqrt(<||e||^2>)/Sp (Eq. small, Sec. 3.4) = {out['e_ratio']:.3g}"
              "  (first order holds for <~ 0.3)")

    # The fit window: the contiguous trial DMs with Sp(d_j) >= Sp_max - sigma_S (Sec. 3.3.3).
    win = fit_window(Sp, sig_S, j_max)
    if win is None:
        out["why"] = "the fit window reaches the end of the DM range: widen the DM range"
        if verbose:
            print(f"  no fit: {out['why']}")
        return out

    n_win = int(win.sum())
    out["n_win"] = n_win
    if verbose:
        print(f"  fit window (Sec. 3.3.3): Sp >= Sp_max - sigma_S = {Sp[j_max] - sig_S:.4g} "
              f"over {n_win} trial DMs, {d[win][0]:.6g} to {d[win][-1]:.6g}")
    if n_win < 3:
        out["why"] = f"only {n_win} trial DMs in the fit window: use a finer DM step"
        if verbose:
            print(f"  no fit: {out['why']}")
        return out

    # The parabola about its vertex: d_fit, (c0, c1, c2) and B (Eqs. parab, lsq).
    fit = fit_parabola(d[win], Sp[win])
    if fit is None:
        out["why"] = "the parabola over the fit window has no maximum"
        if verbose:
            print(f"  no fit: {out['why']}")
        return out
    d_fit, c, B = fit
    if verbose:
        print(f"  parabola about its vertex (Eqs. parab, lsq): d_fit = {d_fit:.6g}, "
              f"c0 = {c[0]:.4g}, c1 = {c[1]:.2g} (~0 about the vertex), c2 = {c[2]:.4g}")

    # w(t; d_j) at every trial DM in the window (Eq. ahat), and V_jl between them (App. 2,
    # Eqs. covprof, cov), with Cov(n) attached to the dynamic spectrum at d_max, where it was
    # measured: dedispersing at d_j shifts channel nu by a further delta_jnu = (d_j - d_max) dt_nu.
    W = coefficients(I[win], kc)
    delay = np.outer(d[win] - d[j_max], dt_nu)
    V = noise_covariance(W, cov, delay, share)
    if verbose:
        # How strongly the noise is correlated between trial DMs, which is why V is needed.
        sd = np.sqrt(np.diag(V))
        rho_neighbours = np.mean(np.diag(V, 1) / (sd[:-1] * sd[1:]))
        rho_ends = V[0, -1] / (sd[0] * sd[-1])
        print(f"  V_jl (App. 2, Eqs. covprof, cov): {n_win} x {n_win}; correlation {rho_neighbours:.4f} "
              f"between neighbouring trial DMs, {rho_ends:.3g} between the window's ends")

    # std[c1]^2 = sum_jl B_1j B_1l V_jl (Eq. varc1), and the same for c2.
    var_c1 = B[1] @ V @ B[1]
    var_c2 = B[2] @ V @ B[2]

    # Both must be positive. Otherwise V is not a valid covariance (or the noise is zero), and
    # sigma_DM would be meaningless.
    if var_c1 <= 0 or var_c2 <= 0:
        out["why"] = (f"the propagated variances are not positive (std[c1]^2 = {var_c1:.3g}, "
                      f"std[c2]^2 = {var_c2:.3g}): sigma_DM cannot be computed")
        if verbose:
            print(f"  no fit: {out['why']}")
        return out

    # sigma_DM = std[c1] / |2 c2| (Eq. final).
    sig_DM = np.sqrt(var_c1) / abs(2 * c[2])

    # The relative curvature error std[c2] / |c2| (Eq. c2err).
    c2_err = np.sqrt(var_c2) / abs(c[2])

    if verbose:
        step = d[1] - d[0]
        print(f"  std[c1] (Eq. varc1) = {np.sqrt(var_c1):.4g}")
        print(f"  sigma_DM = std[c1] / |2 c2| (Eq. final) = {sig_DM:.4g} pc cm^-3 "
              f"({sig_DM / step:.3g} trial-DM steps)")
        print(f"  std[c2]/|c2| (Eq. c2err) = {c2_err:.3g}"
              "  (sigma_DM is unreliable as it approaches 1)")

    out.update(
        ok=True,
        win=win,
        n_win=n_win,
        d_lo=d[win][0],
        d_hi=d[win][-1],
        d_fit=d_fit,
        c=c,
        B=B,
        V=V,
        sig_DM=sig_DM,
        c2_err=c2_err,
    )
    return out


def check_trial_dms(res):
    """Check whether the trial DMs of a result of dm_uncertainty were fine and wide enough, and
    suggest a better grid if not.

    The paper's simulations use a step of sigma_DM/6 (35-163 trial DMs in the fit window) and trial
    DMs out to about 15 sigma_DM either side. If the fit window holds fewer than three trial DMs,
    the suggestion is a step ten times finer about d_max. Returns (warnings, grid), with
    grid = (d_low, d_high, step), or None if the trial DMs are fine enough.

    Args:
        res: the dict returned by dm_uncertainty.
    Returns:
        warnings: list of str.
        grid: (d_low, d_high, step) in pc cm^-3 for generate_profiles.py, or None.
    """
    step = res["d"][1] - res["d"][0]

    # No fit: pass on the reason. If the window held too few trial DMs, suggest a step ten times
    # finer about d_max. (n_win is missing when the window reached the end of the DM range.)
    if not res["ok"]:
        grid = None
        if res.get("n_win", 3) < 3:
            grid = (res["d_max"] - 10 * step, res["d_max"] + 10 * step, step / 10)
        return [res["why"]], grid

    messages = []
    grid = None
    sig_DM = res["sig_DM"]

    # A step coarser than sigma_DM/3: suggest sigma_DM/6, out to 15 sigma_DM or 1.5 window widths
    # either side of d_fit, whichever is wider.
    if step > sig_DM / 3:
        messages.append(f"DM step {step:.3g} is coarser than sigma_DM/3 = {sig_DM / 3:.3g}")
        half_width = max(15 * sig_DM, 1.5 * (res["d_hi"] - res["d_lo"]))
        grid = (res["d_fit"] - half_width, res["d_fit"] + half_width, sig_DM / 6)

    # A poorly determined curvature: sigma_DM is unreliable.
    if res["c2_err"] > 0.5:
        messages.append(f"relative curvature error std[c2]/|c2| = {res['c2_err']:.2f} > 0.5: "
                        "sigma_DM is unreliable (the peak may be flat or split; Sec. 3.3.4)")

    # Filtered noise not small compared with the structure: the first-order expansion may fail.
    if res["e_ratio"] > 0.3:
        messages.append(f"sqrt(<||e||^2>)/Sp = {res['e_ratio']:.2f} > 0.3: "
                        "the first-order expansion may fail (Secs. 3.4, 5.2)")
    return messages, grid
