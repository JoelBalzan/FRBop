"""Diagnostic plots (written by maximise_structure.py with -s/--save_all, and by maximise_sn.py)."""
# FRBop: SHRINE-II's matplotlib.use("Agg") is left out here, so that importing these plots does not
# change FRBop's backend. The plots are only written to files, so they work with any backend.
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

from .structure import kc_search, smooth

DM_LABEL = r"$\Delta$DM (pc cm$^{-3}$)"


def plot_structure_parameter(res, path):
    """Plot Sp against trial DM with +-sigma_S, the fit window, the parabola and d_fit +- sigma_DM
    (as Figure 6 of the paper).

    Args:
        res: the dict returned by estimate.dm_uncertainty.
        path: output file name.
    """
    d = res["d"]
    Sp = res["Sp"]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Sp(d) with a band of +-sigma_S.
    ax.fill_between(d, Sp - res["sig_S"], Sp + res["sig_S"], color="C1", alpha=0.2, lw=0)
    ax.plot(d, Sp, color="C1", label=r"$\mathcal{S}\pm\sigma_\mathcal{S}$")

    if res["ok"]:
        # The fit window.
        ax.axvspan(res["d_lo"], res["d_hi"], color="C4", alpha=0.1, label="Fit window")

        # The fitted parabola over the window, about its vertex d_fit.
        x = np.linspace(res["d_lo"], res["d_hi"], 200) - res["d_fit"]
        c0, c1, c2 = res["c"]
        ax.plot(x + res["d_fit"], c0 + c1 * x + c2 * x**2, "k--", label="Parabola fit")

        # d_fit +- sigma_DM at the parabola's peak.
        ax.errorbar(res["d_fit"], c0, xerr=res["sig_DM"], fmt="o", color="C4", capsize=3,
                    label=r"$d_{\rm fit}\pm\sigma_{\rm DM}$")

        # Show three window widths about d_fit.
        span = 3 * (res["d_hi"] - res["d_lo"])
        ax.set_xlim(res["d_fit"] - span / 2, res["d_fit"] + span / 2)

    ax.set_xlabel(DM_LABEL)
    ax.set_ylabel(r"Structure parameter $\mathcal{S}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dct_spectrum(I, kc, path):
    """Plot |DCT coefficient| of every profile against mode number, with the k_c search marked
    (S23's diagnostic plot, with the paper's Appendix 1 added).

    Shows the running mean Mbar of M(k) (the largest |coefficient| of each mode over the trial
    DMs), the noise floor M_n, the extended search's level M_n + 4 sig_Mbar, S23's k_c and, if the
    extended search moved it, the updated k_c. A k_c given with -kc is marked too. Everything is
    plotted against mode number k = index + 1, and each k_c at its value, which is the cutoff mode
    in f(k); so Mbar reaches M_n one mode to the right of S23's k_c (see structure.choose_kc).

    Args:
        I: I(d_j, t), shape (n_d, N).
        kc: the filter cutoff k_c used.
        path: output file name.
    """
    modes = np.arange(1, I.shape[1] + 1)
    abs_coefficients = np.abs(dct(I, norm="ortho", axis=-1))
    search = kc_search(I)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(modes, abs_coefficients.T, ".", ms=1, color="0.5")

    # The running mean over complete six-mode windows (index 5 onwards), and the two levels.
    ax.plot(modes[5:], search["Mbar"][5:], color="k", lw=1,
            label=r"$\bar{M}(k)$, 6-mode running mean of $M(k)$")
    ax.axhline(search["Mn"], color="k", ls=":", lw=1, label=r"noise floor $M_{\rm n}$")
    ax.axhline(search["rise_level"], color="0.4", ls="--", lw=1,
               label=r"$M_{\rm n} + 4\sigma_{\bar{M}}$")

    # S23's k_c, the updated k_c if the extended search moved it, and a k_c forced with -kc; the
    # one that was used is marked in the legend.
    kc_s23 = search["kc_s23"]
    kc_ext = search["kc_ext"]
    if kc_s23 is not None:
        label = f"$k_c$ = {kc_s23} (S23)"
        if kc_s23 == kc:
            label = f"$k_c$ = {kc_s23} (S23, used)"
        ax.axvline(kc_s23, color="C0", label=label)
    if kc_ext is not None and kc_ext != kc_s23:
        label = f"$k_c$ = {kc_ext} (extended search)"
        if kc_ext == kc:
            label = f"$k_c$ = {kc_ext} (extended search, used)"
        ax.axvline(kc_ext, color="C3", ls="--", label=label)
    if kc not in (kc_s23, kc_ext):
        ax.axvline(kc, color="C2", ls="-.", label=f"$k_c$ = {kc} (-kc, used)")

    ax.set_xlabel("Mode $k$")
    ax.set_ylabel("|DCT coefficient|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_profile_and_smoothed(x, kc, dt, path):
    """Plot one profile x and its smoothed profile S x against time.

    Args:
        x: one profile, shape (N,).
        kc: filter cutoff k_c.
        dt: time resolution, us.
        path: output file name.
    """
    t_ms = np.arange(x.size) * dt / 1000

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_ms, x, color="0.6", lw=0.6, label=r"$\mathbf{x}$")
    ax.plot(t_ms, smooth(x, kc), color="C1", label=r"$\mathbf{Sx}$")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("I")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_I_DM_t(I, d, dt, path):
    """Plot an array over trial DM and time as an image: I(d, t), or S x for
    {label}_{dt}us_I_DM_t_smoothed.png and I - S x for {label}_{dt}us_detrended_noise.png.

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), uniformly spaced, pc cm^-3.
        dt: time resolution, us.
        path: output file name.
    """
    duration_ms = I.shape[1] * dt / 1000

    # Each row is centred on its trial DM, so the image extends half a step beyond the first and
    # last trial DMs.
    if d.size > 1:
        half_step = (d[1] - d[0]) / 2
    else:
        half_step = 0.5
    d_bottom = d[0] - half_step
    d_top = d[-1] + half_step

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(I, aspect="auto", origin="lower", extent=(0, duration_ms, d_bottom, d_top))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(DM_LABEL)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sn_vs_dm(d, SN, path):
    """Plot the largest boxcar S/N against trial DM (SHRINE's {label}_SN.png).

    Args:
        d: the trial DMs, shape (n_d,), pc cm^-3.
        SN: best boxcar S/N at each, shape (n_d,).
        path: output file name.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, SN)
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel(DM_LABEL)
    ax.set_ylabel("Signal to Noise Ratio")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sn_window(Sx, start, length, dt, path):
    """Plot the smoothed profile at the S/N-maximising DM with its boxcar marked (SHRINE's
    {label}_Max_SN_window.png).

    Args:
        Sx: the smoothed profile at the S/N-maximising DM, shape (N,).
        start, length: the boxcar, in samples.
        dt: time resolution, us.
        path: output file name.
    """
    t_ms = np.arange(Sx.size) * dt / 1000
    boxcar_start_ms = start * dt / 1000
    boxcar_end_ms = (start + length) * dt / 1000

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_ms, Sx)
    ax.axvspan(boxcar_start_ms, boxcar_end_ms, color="red", alpha=0.3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("I")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noise_at_d_max(noise_only, sig2, k_min, d_max, dt, path):
    """Plot the noise in the profile at d_max, rebuilt from its DCT modes k > k_min, with +-sigma_t
    (Sec. 3.3.2; {label}_{dt}us_noise_max.png).

    The reconstruction holds the fraction 1 - k_min/N of the noise variance, so it is divided by
    sqrt(1 - k_min/N) here: sigma_t is then its local rms.

    Args:
        noise_only: the reconstruction from noise.high_mode_noise, shape (N,).
        sig2: sigma_t^2, shape (N,).
        k_min: the number of low modes removed.
        d_max: the trial DM of the profile, pc cm^-3.
        dt: time resolution, us.
        path: output file name.
    """
    t_ms = np.arange(noise_only.size) * dt / 1000
    noise = noise_only / np.sqrt(1 - k_min / noise_only.size)
    sig_t = np.sqrt(sig2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_ms, noise, color="0.6", lw=0.6, label=rf"DCT modes $k > k_{{\rm min}} = {k_min}$")
    ax.plot(t_ms, sig_t, color="C3", label=r"$\pm\sigma_t$")
    ax.plot(t_ms, -sig_t, color="C3")
    ax.set_title(rf"Noise at $d_{{\rm max}}$ = {d_max:.5g} pc cm$^{{-3}}$")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("I")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
