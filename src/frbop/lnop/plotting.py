import matplotlib.pyplot as plt
import numpy as np

from frbop.utils.plotting import (IBM_PALETTE, pub_figsize, savefig_rasterized,
                                   set_pub_col)
from frbop.utils.peaks import parse_peak_index_pairs


def plot_time_lag_correlation(lags_seconds, Cx, Cy, lags_seconds_y=None,
                               candidates=None, output=None, ext="png"):
    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.75))
    ax.plot(lags_seconds * 1e9, Cx, color=IBM_PALETTE[0], label="X pol", lw=0.8)
    ax.plot(lags_seconds * 1e9, Cy, color=IBM_PALETTE[1], label="Y pol", lw=0.8)
    if candidates:
        for cand in candidates:
            ax.axvline(cand["tau_seconds"] * 1e9, color=IBM_PALETTE[3],
                       ls="--", lw=0.6, alpha=0.7)
    ax.set_xlabel("Lag τ [ns]")
    ax.set_ylabel("C(τ)")
    ax.legend(fontsize=8)
    if output:
        savefig_rasterized(f"{output}_time_lag_correlation.{ext}")
    plt.close(fig)


def plot_epsilon(lags_seconds, eps_x, eps_y, candidates=None,
                  output=None, ext="png"):
    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.75))
    ax.plot(lags_seconds * 1e9, eps_x, color=IBM_PALETTE[0], label="ε_X", lw=0.8)
    ax.plot(lags_seconds * 1e9, eps_y, color=IBM_PALETTE[1], label="ε_Y", lw=0.8)
    if candidates:
        for cand in candidates:
            ax.axvline(cand["tau_seconds"] * 1e9, color=IBM_PALETTE[3],
                       ls="--", lw=0.6, alpha=0.7)
    ax.set_xlabel("Lag τ [ns]")
    ax.set_ylabel("ε(τ)")
    ax.legend(fontsize=8)
    if output:
        savefig_rasterized(f"{output}_epsilon.{ext}")
    plt.close(fig)


def plot_bin_summary(candidates, edges, output=None, ext="png"):
    if not candidates:
        return
    fig, axes = plt.subplots(2, 1, figsize=pub_figsize(height_ratio=1.2),
                             sharex=True)
    bins = [c["bin_index"] for c in candidates]
    chi2_vals = [c["chi2_max"] for c in candidates]
    ngauss_vals = [c["ngauss"] for c in candidates]

    axes[0].bar(bins, chi2_vals, color=IBM_PALETTE[0], width=0.6)
    axes[0].set_ylabel("χ² max")
    axes[1].bar(bins, ngauss_vals, color=IBM_PALETTE[1], width=0.6)
    axes[1].set_ylabel("N_gauss")
    axes[1].set_xlabel("Logarithmic lag bin index")

    for i, c in enumerate(candidates):
        axes[0].annotate(f"τ={c['tau_seconds']*1e9:.0f}ns",
                         (c["bin_index"], c["chi2_max"]),
                         fontsize=7, ha="center", va="bottom")

    if output:
        savefig_rasterized(f"{output}_bin_summary.{ext}")
    plt.close(fig)


def plot_polarization_scatter(eps_x, eps_y, candidates=None,
                               output=None, ext="png"):
    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.75))
    ax.scatter(eps_x, eps_y, s=1, c=IBM_PALETTE[0], alpha=0.3, label="All lags")
    if candidates:
        for cand in candidates:
            ax.scatter(cand["eps_x"], cand["eps_y"], s=40,
                       c=IBM_PALETTE[3], marker="x", zorder=5)
    lim = max(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("ε_X")
    ax.set_ylabel("ε_Y")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    if output:
        savefig_rasterized(f"{output}_polarization_scatter.{ext}")
    plt.close(fig)


def plot_chi2_vs_lag(lags_seconds, chi2, edges, bin_idx,
                      candidates=None, output=None, ext="png"):
    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.75))
    for i in range(len(edges) - 1):
        sel = bin_idx == i
        if sel.sum() < 10:
            continue
        ax.scatter(lags_seconds[sel] * 1e9, chi2[sel], s=1, alpha=0.3,
                   label=f"bin {i}" if i < 5 else "")
    if candidates:
        for cand in candidates:
            ax.axvline(cand["tau_seconds"] * 1e9, color=IBM_PALETTE[3],
                       ls="--", lw=0.6)
    ax.set_xlabel("Lag τ [ns]")
    ax.set_ylabel("χ²")
    ax.legend(fontsize=7, ncol=2)
    if output:
        savefig_rasterized(f"{output}_chi2_vs_lag.{ext}")
    plt.close(fig)
