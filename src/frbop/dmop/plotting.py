"""
Visualisation for DM optimisation results.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from frbop.utils.plotting import savefig_rasterized

from .noise import robust_vmin_vmax
from .uncertainty import format_dm, format_uncertainty


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCAN_COLORS = {
    "structure": "tab:blue",
    "snr": "tab:red",
    "pa_slope": "tab:green",
    "pa_slope_shrine": "tab:cyan",
    "l_i_mean": "tab:purple",
}
_SCAN_LABELS = {
    "structure": "Structure",
    "snr": "S/N",
    "pa_slope": "PA",
    "pa_slope_shrine": "PA (SHRINE)",
    "l_i_mean": "L/I mean",
}


# ---------------------------------------------------------------------------
# Main comparison figure
# ---------------------------------------------------------------------------

def plot_comparison(
    results: Dict,
    stokes_i: np.ndarray,
    freq_mhz: np.ndarray,
    time_ms: np.ndarray,
    input_dm: float,
    dm_range: Tuple[float, float],
    dedisp_mode: str,
    get_delay_samples_fn,
    pa_series_fn,
    pa_smoothed_and_fit_fn,
    pa_shrine_smoothed_and_fit_fn,
    stokes_q: Optional[np.ndarray] = None,
    stokes_u: Optional[np.ndarray] = None,
    peak_region: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Multi-panel comparison figure: one row per method plus an original-data row.

    Callable arguments (*_fn) are thin closures supplied by DMOptimiser so
    this module stays decoupled from the optimiser internals.
    """
    n_methods = len(results)
    has_qu = stokes_q is not None and stokes_u is not None

    fig_width = 15.5
    row_height = 2.7
    fig_height = max(11.0, row_height * (n_methods + 1))
    fig, axes = plt.subplots(
        n_methods + 1,
        5,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [0.85, 0.11, 0.85, 0.32, 0.85]},
    )
    if n_methods == 0:
        axes = np.atleast_2d(axes)

    for spacer_ax in axes[:, 1]:
        fig.delaxes(spacer_ax)
    for spacer_ax in axes[:, 3]:
        fig.delaxes(spacer_ax)
    axes = np.stack((axes[:, 0], axes[:, 2], axes[:, 4]), axis=1)

    fs_title, fs_label, fs_tick = 16, 14, 12
    fs_legend, fs_overlay, fs_lpad = 11, 12, 2

    # ---- original data region ----
    if peak_region is not None:
        orig_data = stokes_i[:, peak_region[0]:peak_region[1]]
        time_range = time_ms[peak_region[0]:peak_region[1]]
        q_region = None if not has_qu else stokes_q[:, peak_region[0]:peak_region[1]]
        u_region = None if not has_qu else stokes_u[:, peak_region[0]:peak_region[1]]
    else:
        orig_data = stokes_i
        time_range = time_ms
        q_region = stokes_q
        u_region = stokes_u

    # Pre-compute shared PA y-limits
    pa_limits = None
    if has_qu:
        pa_series_all = [pa_series_fn(q_region, u_region, orig_data)]
        for result in results.values():
            n_t = result["dedispersed"].shape[1]
            dq = result.get("dedispersed_q")
            du = result.get("dedispersed_u")
            if dq is None or du is None:
                continue
            pa_series_all.append(pa_series_fn(dq, du, result["dedispersed"]))
        pa_all = np.concatenate([p[np.isfinite(p)] for p in pa_series_all if p is not None])
        if pa_all.size > 0:
            pa_limits = (float(np.nanmin(pa_all)) - 10.0, float(np.nanmax(pa_all)) + 10.0)

    def _label_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=fs_title)
        ax.set_xlabel(xlabel, fontsize=fs_label, labelpad=fs_lpad)
        ax.set_ylabel(ylabel, fontsize=fs_label, labelpad=fs_lpad)
        ax.tick_params(axis="both", labelsize=fs_tick)

    # ---- Row 0: original data ----
    vmin0, vmax0 = robust_vmin_vmax(orig_data)
    axes[0, 0].imshow(
        orig_data, aspect="auto",
        extent=[time_range[0], time_range[-1], freq_mhz[0], freq_mhz[-1]],
        cmap="viridis", origin="lower", vmin=vmin0, vmax=vmax0,
    )
    _label_ax(
        axes[0, 0],
        f"Original Data (SHRINE structure-maximised)\n"+rf"Input DM = {format_dm(input_dm, 3)} pc cm$^{{-3}}$",
        "Time (ms)", "Frequency (MHz)",
    )

    ts_orig = np.mean(orig_data, axis=0)
    axes[0, 1].plot(time_range, ts_orig, "k-", linewidth=1, label="I")
    ax0r = None
    if has_qu:
        ax0r = axes[0, 1].twinx()
        L0 = np.mean(np.sqrt(q_region ** 2 + u_region ** 2), axis=0)
        n_edge = max(1, int(0.05 * len(L0)))
        L0_plot = L0 - float(np.median(L0[:n_edge]))
        axes[0, 1].plot(time_range, L0_plot, "r", linewidth=1, label="L")
        pa_deg0 = pa_series_fn(q_region, u_region, orig_data)
        pa_sm0, fit0 = pa_smoothed_and_fit_fn(q_region, u_region, orig_data, time_range)
        ax0r.plot(time_range, pa_deg0, color="silver", linewidth=1, alpha=0.9)
        ax0r.plot(time_range, pa_sm0, color="tab:purple", linewidth=2, alpha=0.8, label="PA")
        ax0r.plot(time_range, fit0, color="tab:orange", linewidth=1.5, linestyle="--", alpha=0.7, label="PA fit")
        h1, l1 = axes[0, 1].get_legend_handles_labels()
        h2, l2 = ax0r.get_legend_handles_labels()
        axes[0, 1].legend(h1 + h2, l1 + l2, loc="best", fontsize=fs_legend)
        ax0r.set_ylabel("PA (deg)", fontsize=fs_label, labelpad=fs_lpad)
        ax0r.tick_params(axis="y", labelsize=fs_tick)
        if pa_limits:
            ax0r.set_ylim(pa_limits)
    else:
        axes[0, 1].legend(loc="best", fontsize=fs_legend)
    _label_ax(axes[0, 1], "Original Time Series", "Time (ms)", "Flux")
    axes[0, 1].grid(True, alpha=0.3)

    # Top-right: DM summary error-bar chart
    all_scan_ax = axes[0, 2]
    if results:
        method_names = list(results.keys())
        y_pos = np.arange(len(method_names), dtype=float)
        for j, mname in enumerate(method_names):
            r = results[mname]
            minus = r.get("uncertainty_minus")
            plus = r.get("uncertainty_plus")
            xerr = np.array([
                [0.0 if minus is None else float(minus)],
                [0.0 if plus is None else float(plus)],
            ])
            all_scan_ax.errorbar(
                x=[r["dm"]], y=[y_pos[j]], xerr=xerr,
                fmt="o", markersize=5, capsize=3, elinewidth=1.8,
                color=_SCAN_COLORS.get(mname, "black"),
            )
        all_scan_ax.axvline(input_dm, color="gray", linestyle=":", linewidth=1.5, alpha=0.9)
        all_scan_ax.set_xlim(dm_range)
        all_scan_ax.set_yticks(y_pos)
        all_scan_ax.set_yticklabels([_SCAN_LABELS.get(n, n) for n in method_names])
        all_scan_ax.invert_yaxis()
        all_scan_ax.grid(True, axis="x", alpha=0.3)
        all_scan_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _label_ax(all_scan_ax, rf"Best DM Summary", "DM (pc cm$^{{-3}}$)", "")
    else:
        all_scan_ax.text(0.5, 0.5, "No method results", ha="center", va="center",
                         transform=all_scan_ax.transAxes)
        all_scan_ax.set_axis_off()

    # ---- Per-method rows ----
    show_scan_legend = True
    for idx, (mname, result) in enumerate(results.items(), start=1):
        n_t = result["dedispersed"].shape[1]
        dt = float(np.median(np.diff(time_range))) if len(time_range) > 1 else 1.0
        delay_samples = get_delay_samples_fn(result["dm"])
        if dedisp_mode == "crop":
            start_shift = int(np.max(delay_samples))
        else:
            start_shift = int(np.min(delay_samples))
        tr_d = time_range[0] + start_shift * dt + np.arange(n_t) * dt

        vmin, vmax = robust_vmin_vmax(result["dedispersed"])
        axes[idx, 0].imshow(
            result["dedispersed"], aspect="auto",
            extent=[tr_d[0], tr_d[-1], freq_mhz[0], freq_mhz[-1]],
            cmap="viridis", origin="lower", vmin=vmin, vmax=vmax,
        )
        _label_ax(axes[idx, 0], result["method"], "Time (ms)", "Frequency (MHz)")
        axes[idx, 0].text(
            0.98, 0.98,
            "DM=" + format_uncertainty(result["dm"], result.get("uncertainty_minus"),
                                       result.get("uncertainty_plus"), precision=3) + rf" pc cm$^{{-3}}$",
            transform=axes[idx, 0].transAxes, ha="right", va="top",
            color="white", fontsize=fs_overlay,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=2.0),
        )

        ts = np.mean(result["dedispersed"], axis=0)
        axes[idx, 1].plot(tr_d, ts, "k-", linewidth=1, label="I")
        axr = None
        if has_qu:
            axr = axes[idx, 1].twinx()
            dq = result.get("dedispersed_q")
            du = result.get("dedispersed_u")
            if dq is None or du is None:
                dq = result.get("_dq_fallback")
                du = result.get("_du_fallback")
            L_s = np.mean(np.sqrt(dq ** 2 + du ** 2), axis=0)
            n_edge = max(1, int(0.05 * len(L_s)))
            axes[idx, 1].plot(tr_d, L_s - float(np.median(L_s[:n_edge])), "r", linewidth=1, label="L")

            if "pa_plot_series" in result:
                pa_deg = np.asarray(result["pa_plot_series"])
            else:
                pa_deg = np.asarray(pa_series_fn(dq, du, result["dedispersed"], use_data_rms=False))

            if mname == "pa_slope_shrine" and "pa_plot_fit" not in result:
                pa_sm, fit = pa_shrine_smoothed_and_fit_fn(
                    dq, du, result["dedispersed"], tr_d, force_kc=result.get("kc")
                )
            elif mname == "pa_slope" and "pa_plot_fit" not in result:
                pa_sm, fit = pa_smoothed_and_fit_fn(dq, du, result["dedispersed"], tr_d)
            else:
                pa_sm = np.asarray(result.get("pa_plot_smooth", pa_deg))
                fit = np.asarray(result.get("pa_plot_fit", np.full_like(pa_deg, np.nan)))

            axr.plot(tr_d, pa_deg, color="silver", linewidth=1, alpha=0.9, label="PA")
            # Only plot smoothed PA and fit for PA-specific methods
            if mname in ("pa_slope", "pa_slope_shrine"):
                suffix = "(S)" if mname == "pa_slope_shrine" else ""
                axr.plot(tr_d, pa_sm, color="tab:purple", linewidth=2, alpha=0.8, label=f"PA sm{suffix}")
                axr.plot(tr_d, fit, color="tab:orange", linewidth=1.5, linestyle="--", alpha=0.7, label=f"PA fit{suffix}")
            axr.set_ylabel("PA (deg)", fontsize=fs_label, labelpad=fs_lpad)
            axr.tick_params(axis="y", labelsize=fs_tick)
            if pa_limits:
                axr.set_ylim(pa_limits)

        _label_ax(axes[idx, 1], f"Metric = {result['metric']:.6f}", "Time (ms)", "Flux")
        axes[idx, 1].grid(True, alpha=0.3)

        # DM scan curve
        scan_ax = axes[idx, 2]
        dm_vals = result.get("dm_values")
        metric_vals = result.get("metric_values")
        if dm_vals is not None and metric_vals is not None:
            low_dm = result.get("uncertainty_low_dm")
            high_dm = result.get("uncertainty_high_dm")
            shade_low = float(dm_range[0]) if low_dm is None else float(low_dm)
            shade_high = float(dm_range[1]) if high_dm is None else float(high_dm)
            if shade_low <= shade_high:
                scan_ax.axvspan(shade_low, shade_high, color="tab:orange", alpha=0.18,
                                label="DM uncertainty" if show_scan_legend else None)
            scan_ax.plot(dm_vals, metric_vals, linewidth=2.0, color=_SCAN_COLORS.get(mname, "black"))
            scan_ax.axvline(input_dm, color="gray", linestyle=":", linewidth=1.4, alpha=0.9,
                            label="Input DM" if show_scan_legend else None)
            scan_ax.axvline(result["dm"], color="red", linestyle="--", linewidth=1.4, alpha=0.9,
                            label="Best DM" if show_scan_legend else None)
            scan_ax.set_xlim(dm_range)
            scan_ax.grid(True, alpha=0.3)
            scan_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            _label_ax(scan_ax, "", r"DM (pc cm$^{{-3}}$)", "Metric")
            if show_scan_legend:
                scan_ax.legend(loc="upper left", fontsize=fs_legend)
                show_scan_legend = False
        else:
            scan_ax.text(0.5, 0.5, "No scan data", ha="center", va="center",
                         transform=scan_ax.transAxes)
            scan_ax.set_axis_off()

    plt.tight_layout(rect=[0.02, 0.02, 0.995, 0.995])
    fig.subplots_adjust(wspace=0.04, hspace=0.5)

    if save_path:
        savefig_rasterized(save_path, dpi=600, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# DM scan overview
# ---------------------------------------------------------------------------

def plot_dm_scan(
    dm_values: np.ndarray,
    metrics: Dict,
    input_dm: float,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-method metric curves across a DM grid."""
    labels = {
        "structure": "Structure Metric (SHRINE)",
        "snr": "S/N",
        "pa_slope": "Weighted PA Slope magnitude",
        "pa_slope_shrine": "Weighted PA Slope magnitude (SHRINE-smoothed PA)",
        "l_i_mean": "L/I (mean)",
    }
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (mname, mvals) in zip(axes, metrics.items()):
        ax.plot(dm_values, mvals, color=_SCAN_COLORS.get(mname, "black"), linewidth=2)
        ax.set_xlabel(r"DM (pc cm$^{{-3}}$)")
        ax.set_ylabel("Metric Value")
        ax.set_title(labels.get(mname, mname))
        ax.grid(True, alpha=0.3)
        ax.axvline(input_dm, color="gray", linestyle=":", linewidth=2, alpha=1,
                   label=f"Input DM={format_dm(input_dm, 3)}")
        max_idx = int(np.argmax(mvals))
        ax.axvline(dm_values[max_idx], color="red", linestyle="--", alpha=1,
                   label=f"Max at DM={format_dm(dm_values[max_idx], 3)}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        savefig_rasterized(save_path, dpi=150, bbox_inches="tight")
        print(f"DM scan plot saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Component DM diagnostics
# ---------------------------------------------------------------------------

def plot_component_dm_diagnostics(
    all_results: List[Dict],
    component_ids: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Best-DM per component for each method with asymmetric error bars."""
    n_components = len(all_results)
    if n_components < 2:
        print("Component DM diagnostics skipped (need ≥ 2 components).")
        return

    if component_ids is None:
        component_ids = np.arange(1, n_components + 1, dtype=int)
    else:
        component_ids = np.asarray(component_ids, dtype=int)

    preferred = ["structure", "snr", "pa_slope", "pa_slope_shrine", "l_i_mean"]
    first = list(all_results[0].keys())
    common = [m for m in first if all(m in c for c in all_results)]
    if not common:
        print("Component DM diagnostics skipped (no common methods).")
        return

    ordered = [m for m in preferred if m in common] + [m for m in common if m not in preferred]
    comp_idx = np.arange(1, n_components + 1)
    dm_mat = np.zeros((len(ordered), n_components))
    dm_minus = np.zeros_like(dm_mat)
    dm_plus = np.zeros_like(dm_mat)

    for i, mname in enumerate(ordered):
        for j, comp in enumerate(all_results):
            r = comp[mname]
            dm_mat[i, j] = float(r["dm"])
            dm_minus[i, j] = max(0.0, float(r.get("uncertainty_minus") or 0.0))
            dm_plus[i, j] = max(0.0, float(r.get("uncertainty_plus") or 0.0))

    draw_order = list(np.argsort(-(dm_minus + dm_plus).mean(axis=1)))
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))
    for rank, i in enumerate(draw_order):
        mname = ordered[i]
        ax.errorbar(
            comp_idx, dm_mat[i],
            yerr=np.vstack((dm_minus[i], dm_plus[i])),
            fmt="o-", linewidth=1.8, capsize=2.5, elinewidth=1.1,
            label=_SCAN_LABELS.get(mname, mname),
            color=_SCAN_COLORS.get(mname),
            zorder=2 + rank,
        )

    ax.set_ylabel("Best DM (pc cm⁻³)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(fontsize=8)
    ax.set_xticks(comp_idx)

    component_names = []
    for cid in component_ids:
        cid_int = int(cid)
        if cid_int == 1:
            component_names.append("Main component")
        elif cid_int == 2:
            component_names.append("Precursor")
        else:
            component_names.append(f"Precursor {cid_int - 1}")
    ax.set_xticklabels(component_names, fontsize=7)
    x_pad = 0.2
    ax.set_xlim(1 - x_pad, n_components + x_pad)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    if save_path:
        savefig_rasterized(save_path, dpi=600, bbox_inches="tight")
        print(f"Component DM diagnostics saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Component dn_e diagnostics
# ---------------------------------------------------------------------------

def plot_component_dne_diagnostics(
    dne_diag: Dict,
    save_path: Optional[str] = None,
) -> None:
    """dn_e between component pairs for all methods."""
    pair_labels = dne_diag.get("pair_labels", [])
    methods = dne_diag.get("methods", {})
    if not pair_labels or not methods:
        print("dn_e diagnostics plot skipped (no data).")
        return

    preferred = ["structure", "snr", "pa_slope", "pa_slope_shrine", "l_i_mean"]
    method_names = [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]
    n_methods = len(method_names)

    x = np.arange(len(pair_labels), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))

    offset_span = 0.18
    offsets = np.array([0.0]) if n_methods == 1 else np.linspace(-offset_span, offset_span, n_methods)
    all_abs = []

    for i, mname in enumerate(method_names):
        vals = methods[mname]
        y = np.asarray(vals.get("dn_e", np.zeros_like(x)), dtype=float)
        y_low = np.asarray(vals.get("dn_e_low", y), dtype=float)
        y_high = np.asarray(vals.get("dn_e_high", y), dtype=float)
        yerr = np.vstack((np.abs(y - y_low), np.abs(y_high - y)))

        all_abs.extend(np.abs(y[np.isfinite(y)]).tolist())
        all_abs.extend(np.abs(y_low[np.isfinite(y_low)]).tolist())
        all_abs.extend(np.abs(y_high[np.isfinite(y_high)]).tolist())

        ax.errorbar(
            x + offsets[i], y, yerr=yerr, fmt="o", capsize=3, elinewidth=1.3, markersize=5,
            label=_SCAN_LABELS.get(mname, mname), color=_SCAN_COLORS.get(mname),
        )

    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylabel(r"$\Delta n_e$ (cm$^{-3}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    finite_abs = np.asarray([v for v in all_abs if np.isfinite(v) and v > 0])
    if finite_abs.size > 1 and float(np.max(finite_abs) / np.min(finite_abs)) > 100.0:
        ax.set_yscale("symlog", linthresh=max(1.0, float(np.min(finite_abs))))

    plt.tight_layout()
    if save_path:
        savefig_rasterized(save_path, dpi=600, bbox_inches="tight")
        print(f"Component dn_e diagnostics saved to: {save_path}")
    else:
        plt.show()
