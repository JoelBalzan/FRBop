"""RVM fitting visualisation: PA-fit panel, corner plot, grid χ² map."""

from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .rvm_model import iau_pa_from_qu, intensity_from_qu, rvm_pa

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


def plot_rvm_fit(phi: np.ndarray, q: np.ndarray, u: np.ndarray,
                 result: dict,
                 save_path: Optional[str] = None,
                 title: Optional[str] = None,
                 show: bool = False,
                 phase_unit: str = "rad") -> plt.Figure:
    """
    Three-panel summary of an RVM fit: PA, Stokes Q/U, and linear
    amplitude, with the best-fit RVM overlaid.

    The x-axis uses ``result['phi_fit']`` if available (the best-fit
    rotational phase), otherwise *phi*.

    Parameters
    ----------
    phi : (N,) array
        Input phase or time array (used as fallback x-axis).
    q, u : (N,) array
        Stokes Q, U data.
    result : dict
        Output from ``fit_rvm()``.
    save_path : str, optional
        If given, save figure to this path.
    title : str, optional
        Figure title.
    show : bool
        Call plt.show().
    phase_unit : str
        Label for the x-axis unit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pa = iau_pa_from_qu(q, u)
    l = intensity_from_qu(q, u)
    pa_fit = result.get("best_pa", None)
    L_fit = result.get("best_L", None)
    xx = result.get("phi_fit", phi)  # best-fit rotational phase

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, hspace=0.35, height_ratios=[1, 1, 1])

    # ── Panel 1: PA vs phase ──
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(xx, np.degrees(pa), fmt=".", color="C0", alpha=0.6,
                 label="data")
    if pa_fit is not None:
        # Wrap model PA to [-π/2, π/2] for visual comparison with data
        pa_wrap = (pa_fit + np.pi / 2) % np.pi - np.pi / 2
        ax1.plot(xx, np.degrees(pa_wrap), "-", color="C3", lw=2,
                 label="RVM fit")
    ax1.set_ylabel("PA [deg]")
    ax1.legend(loc="best", fontsize=9)
    ax1.axhline(0, color="grey", ls="--", lw=0.5)

    # ── Panel 2: Stokes Q / U ──
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(xx, q, ".", color="C1", alpha=0.6, label="Q")
    ax2.plot(xx, u, ".", color="C2", alpha=0.6, label="U")
    if pa_fit is not None and L_fit is not None:
        ax2.plot(xx, L_fit * np.cos(2 * pa_fit), "-", color="C1", lw=1.5)
        ax2.plot(xx, L_fit * np.sin(2 * pa_fit), "-", color="C2", lw=1.5)
    ax2.set_ylabel("Stokes")
    ax2.legend(loc="best", fontsize=9)

    # ── Panel 3: Linear amplitude ──
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(xx, l, ".", color="C4", alpha=0.6, label="L (data)")
    if L_fit is not None:
        ax3.plot(xx, L_fit, "-", color="C3", lw=2, label="L (fit)")
    ax3.set_xlabel(f"Rotational phase [{phase_unit}]")
    ax3.set_ylabel("L")
    ax3.legend(loc="best", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.align_ylabels()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_rvm_corner(result: dict,
                    save_path: Optional[str] = None,
                    show: bool = False,
                    **corner_kw) -> Optional[plt.Figure]:
    """
    Corner plot of the MCMC chain for the 4 RVM parameters.

    Parameters
    ----------
    result : dict
        Output from ``fit_rvm()`` (must contain an ``mcmc`` key).
    save_path : str, optional
    show : bool
    **corner_kw
        Passed to ``corner.corner()``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None (if corner not available or no
          MCMC chain).
    """
    if not HAS_CORNER:
        print("corner.py not installed — skipping corner plot")
        return None

    mcmc = result.get("mcmc")
    if mcmc is None or mcmc.get("flatchain") is None:
        print("No MCMC chain found in result")
        return None

    flat = mcmc["flatchain"]
    labels = [r"$\phi_0$", r"$\psi_0$", r"$\alpha$", r"$\zeta$"]

    defaults = dict(
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        truths=(result.get("best_phi0"), result.get("best_psi0"),
                result.get("best_alpha"), result.get("best_zeta")),
        truth_color="C3",
        range=[0.999] * 4,
    )
    defaults.update(corner_kw)

    fig = corner.corner(flat, labels=labels, **defaults)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_grid_chi2(result: dict,
                   save_path: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """
    2D χ² map from the grid search over (α, ζ).

    Parameters
    ----------
    result : dict
        Output from ``fit_rvm()`` (must contain a ``grid`` key).
    save_path : str, optional
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    grid = result.get("grid", {})
    alpha_rad = grid.get("grid_alpha")
    zeta_rad = grid.get("grid_zeta")
    chi2 = grid.get("grid_chi2")
    if chi2 is None:
        print("No grid χ² data found in result")
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    alpha_deg = np.degrees(alpha_rad)
    zeta_deg = np.degrees(zeta_rad)

    # Clip for visualisation
    vmin = np.nanmin(chi2)
    vmax = vmin + 10 * (np.nanpercentile(chi2, 90) - vmin + 1e-10)
    im = ax.pcolormesh(alpha_deg, zeta_deg, chi2.T,
                       shading="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, label=r"$\chi^2$")

    # Mark best point
    best_a = np.degrees(result.get("best_alpha", 0))
    best_z = np.degrees(result.get("best_zeta", 0))
    ax.plot(best_a, best_z, "*", color="C3", ms=12, mew=1,
            mec="white", label="best")
    ax.legend()

    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\zeta$ [deg]")
    ax.set_title(r"RVM $\chi^2$ grid: $\alpha$ vs $\zeta$")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
