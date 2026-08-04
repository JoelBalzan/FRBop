"""Plotting helpers for scintillation workflows."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from frbop.scop.fit_utils import _decode_lorentzian_components
from frbop.scop.models import (lorentzian, lorentzian_2c, lorentzian_3c,
                               scattered_gaussian)
from frbop.scop.power import correct_spectrum_powerlaw
from frbop.utils.plotting import (IBM_PALETTE, pub_figsize, savefig_rasterized,
                                  set_pub_col, set_pub_style)


def _apply_publication_style() -> dict:
    set_pub_style(use_latex=True)
    return {
        "title": float(plt.rcParams.get("axes.titlesize", 11)),
        "label": float(plt.rcParams.get("axes.labelsize", 10)),
        "tick": float(plt.rcParams.get("xtick.labelsize", 10)),
        "legend": float(plt.rcParams.get("legend.fontsize", 10)),
        "line": float(plt.rcParams.get("lines.linewidth", 1.4)),
    }

def plot_spectrum_powerlaw_fit(
    freq_mhz,
    spectrum,
    mean_model,
    spectral_index: float | None = None,
    output=None,
):
    styles = _apply_publication_style()
    fig, axs = plt.subplots(2, 1, figsize=pub_figsize(height_ratio=0.8))
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mean_model = np.asarray(mean_model, dtype=float)

    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum)
    if np.any(finite):
        axs[0].plot(freq_mhz[finite], spectrum[finite], color='0.25', lw=0.8, label='Spectrum')

    model_mask = np.isfinite(freq_mhz) & np.isfinite(mean_model)
    if np.any(model_mask):
        idx_str = "" if spectral_index is None else f" ($\\alpha_s$={float(spectral_index):.2f})"
        axs[0].plot(
            freq_mhz[model_mask],
            mean_model[model_mask],
            color='tab:orange',
            lw=1,
            ls='--',
            label=f"Power-law fit{idx_str}",
        )

    #axs[0].set_title('Spectrum power-law fit', fontsize=styles['title'])
    axs[0].set_xlabel('Frequency [MHz]', fontsize=styles['label'])
    axs[0].set_ylabel(r'S [arb.]', fontsize=styles['label'])
    axs[0].tick_params(labelsize=styles['tick'])
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=styles['legend'])

    corrected = np.full_like(spectrum, np.nan, dtype=float)
    corr_mask = np.isfinite(mean_model) & (mean_model > 0) & np.isfinite(spectrum)
    if np.any(corr_mask):
        corrected[corr_mask] = (spectrum[corr_mask] - mean_model[corr_mask]) / mean_model[corr_mask]
        axs[1].plot(freq_mhz[corr_mask], corrected[corr_mask], color='k', lw=0.8)
    else:
        axs[1].text(0.5, 0.5, "No valid corrected spectrum", ha='center', va='center', transform=axs[1].transAxes)

    #axs[1].set_title('Corrected spectrum', fontsize=styles['title'])
    axs[1].set_xlabel('Frequency [MHz]', fontsize=styles['label'])
    axs[1].set_ylabel('Corrected '+r'S [arb.]', fontsize=styles['label'])
    axs[1].tick_params(labelsize=styles['tick'])
    axs[1].grid(alpha=0.25)
    plt.tight_layout()

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_spectrum_powerlaw' + (ext if ext else '.png')
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Saved spectrum power-law plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_lorentzian_diagnostics(
    lags_plot_sym,
    acf_plot_sym,
    lags_lorentz_fit,
    acf_lorentz_fit,
    fit_models,
    lag_zoom,
    output=None,
):
    styles = _apply_publication_style()
    fig, axs = plt.subplots(2, 3, figsize=pub_figsize(1, height_ratio=1.05))
    xabs = np.abs(lags_plot_sym)

    panel_cfg = [
        ("1-Component Lorentzian", fit_models[0], lorentzian, 'tab:blue', axs[0, 0]),
        ("2-Component Lorentzian", fit_models[1], lorentzian_2c, 'tab:red', axs[0, 1]),
        ("3-Component Lorentzian", fit_models[2], lorentzian_3c, 'tab:orange', axs[0, 2]),
    ]

    comp_colors = [
        ['tab:cyan', 'tab:purple', 'tab:brown'],
        ['tab:red', 'tab:pink', 'tab:brown'],
        ['tab:orange', 'tab:pink', 'tab:brown'],
    ]

    for i, (title, (name, result, _), model_fn, sum_color, ax) in enumerate(panel_cfg):
        ax.plot(lags_plot_sym, acf_plot_sym, color='k', lw=1.3, label='ACF data')
        if "popt" in result:
            n_comp = i + 1
            components, A, C = _decode_lorentzian_components(n_comp, result["popt"])
            sum_curve = model_fn(xabs, *result["popt"])
            ax.plot(lags_plot_sym, sum_curve, lw=1.6, color=sum_color, label=f'{n_comp}c sum')
            for j, (w, d) in enumerate(components):
                comp = A * w / (1.0 + (xabs / d) ** 2)
                ax.plot(
                    lags_plot_sym,
                    comp,
                    ls='--',
                    lw=1.1,
                    alpha=0.9,
                    color=comp_colors[i][j],
                    label=f'comp {j+1} (d={d:.3f} MHz)',
                )
            ax.plot(
                lags_plot_sym,
                np.full_like(lags_plot_sym, C),
                ls=':',
                lw=1.0,
                alpha=0.8,
                color='tab:gray',
                label='offset',
            )
        ax.set_xlim(-lag_zoom, lag_zoom)
        ax.set_title(title, fontsize=styles['title'])
        ax.set_xlabel("Delta nu [MHz]", fontsize=styles['label'])
        ax.set_ylabel("ACF", fontsize=styles['label'])
        ax.tick_params(labelsize=styles['tick'])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=styles['legend'])

    # Residuals
    ax1 = axs[1, 0]
    ax1.axhline(0.0, color='0.5', lw=1)
    for name, result, _ in fit_models:
        if "ymod" in result:
            ax1.plot(lags_lorentz_fit, acf_lorentz_fit - result["ymod"], lw=1.3, label=name)
    ax1.set_title("Residuals (positive lags)", fontsize=styles['title'])
    ax1.set_xlabel("Delta nu [MHz]", fontsize=styles['label'])
    ax1.set_ylabel("ACF residual", fontsize=styles['label'])
    ax1.tick_params(labelsize=styles['tick'])
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=styles['legend'])

    # Delta AIC bar
    valid_names = [n for n, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
    valid_aic = [r["aic"] for _, r, _ in fit_models if "aic" in r and np.isfinite(r["aic"])]
    valid_bic = [r["bic"] for _, r, _ in fit_models if "bic" in r and np.isfinite(r["bic"])]

    ax2 = axs[1, 1]
    if valid_names:
        daic = [a - min(valid_aic) for a in valid_aic]
        x = np.arange(len(valid_names))
        ax2.bar(x, daic, color='tab:blue', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(valid_names, rotation=15)
        ax2.set_ylabel("Delta AIC")
        ax2.grid(axis='y', alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "No valid AIC values", ha='center', va='center', transform=ax2.transAxes)

    ax3 = axs[1, 2]
    if valid_names:
        dbic = [b - min(valid_bic) for b in valid_bic]
        x = np.arange(len(valid_names))
        ax3.bar(x, dbic, color='tab:green', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(valid_names, rotation=15)
        ax3.set_ylabel("Delta BIC")
        ax3.grid(axis='y', alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "No valid BIC values", ha='center', va='center', transform=ax3.transAxes)

    plt.tight_layout()
    if output:
        base, ext = os.path.splitext(output)
        out = base + '_lorentzian_diagnostics' + (ext if ext else '.pdf')
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Saved Lorentzian diagnostics plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_scintillation_band_power_law(
    band_results,
    power_law_fit,
    *,
    output=None,
    fit_max_lag_mhz: float | None = None,
):
    styles = _apply_publication_style()
    # Top row: power-law plot + residual ratio
    # Bottom rows: per-band spectra + ACF Lorentzian fits
    n_bands = len(band_results)
    base_width, _ = pub_figsize(1, 0.5)
    fig_height = max(6.0, 2.5 + 1.8 * n_bands)
    fig = plt.figure(figsize=(base_width, fig_height), constrained_layout=False)
    # Top gridspec: power-law + residual ratio
    gs_top = fig.add_gridspec(1, 2, top=0.93, bottom=0.62, wspace=0.3)
    # Bottom gridspec: per-band spectra + ACF fits
    gs_bot = fig.add_gridspec(2, n_bands, top=0.56, bottom=0.07, hspace=0.4, wspace=0.35)

    ax0 = fig.add_subplot(gs_top[0, 0])
    ax1 = fig.add_subplot(gs_top[0, 1])

    centers = np.array([r["center_mhz"] for r in band_results], dtype=float)
    dnus = np.array([r["dnu_mhz"] for r in band_results], dtype=float)
    errs = np.array([r.get("dnu_err_mhz", np.nan) for r in band_results], dtype=float)
    finite = np.isfinite(centers) & np.isfinite(dnus) & (centers > 0) & (dnus > 0)
    centers, dnus, errs = centers[finite], dnus[finite], errs[finite]

    if centers.size > 0:
        order = np.argsort(centers)
        centers, dnus, errs = centers[order], dnus[order], errs[order]
        has_err = np.isfinite(errs)
        if np.any(has_err):
            ax0.errorbar(
                centers[has_err],
                dnus[has_err],
                yerr=errs[has_err],
                fmt='o',
                color='tab:blue',
                capsize=3,
                label='Measured bands',
            )
            if np.any(~has_err):
                ax0.loglog(centers[~has_err], dnus[~has_err], 'o', color='tab:blue')
        else:
            ax0.loglog(centers, dnus, 'o', color='tab:blue', label='Measured bands')
        ax0.set_xscale('log')
        ax0.set_yscale('log')

    if power_law_fit is not None and centers.size > 0:
        x_fit = np.asarray(power_law_fit["fit_freq_mhz"], dtype=float)
        grid = np.logspace(
            np.log10(float(np.nanmin(x_fit)) * 0.9),
            np.log10(float(np.nanmax(x_fit)) * 1.1),
            256,
        )
        ax0.loglog(
            grid,
            np.exp(power_law_fit["log_norm_fit"] + power_law_fit["alpha_fit"] * np.log(grid)),
            color='tab:orange',
            lw=1.8,
            label=f"Fit: alpha={power_law_fit['alpha_fit']:.2f}",
        )
        ax0.loglog(
            grid,
            np.exp(power_law_fit["log_norm_44"] + power_law_fit["comparison_alpha"] * np.log(grid)),
            color='tab:green',
            lw=1.5,
            ls='--',
            label='Kolmogorov alpha=4.4',
        )
        ax0.axvline(
            power_law_fit["reference_freq_mhz"],
            color='0.45',
            lw=1.0,
            ls=':',
            label=f"nu_c={power_law_fit['reference_freq_mhz']:.1f} MHz",
        )

        fit_at = np.exp(power_law_fit["log_norm_fit"] + power_law_fit["alpha_fit"] * np.log(centers))
        fit_44 = np.exp(
            power_law_fit["log_norm_44"] + power_law_fit["comparison_alpha"] * np.log(centers)
        )
        ax1.axhline(1.0, color='0.5', lw=1.0, ls=':')
        ax1.plot(centers, dnus / fit_at, 'o-', color='tab:orange', lw=1.5, label='Data / fit')
        ax1.plot(centers, dnus / fit_44, 's--', color='tab:green', lw=1.2, label='Data / Kolmogorov')
        ax1.axvline(power_law_fit["reference_freq_mhz"], color='0.45', lw=1.0, ls=':')
        ax1.set_ylim(0.2, 2.5)

    ax0.set_xlabel('Frequency [MHz]', fontsize=styles['label'])
    ax0.set_ylabel(r'$\Delta\nu_d$ [MHz]', fontsize=styles['label'])
    ax0.set_title('Scintillation bandwidth vs frequency', fontsize=styles['title'])
    ax0.tick_params(labelsize=styles['tick'])
    ax0.grid(alpha=0.25, which='both')
    ax0.legend(fontsize=styles['legend'])
    ax1.set_xlabel('Frequency [MHz]', fontsize=styles['label'])
    ax1.set_ylabel('Observed / model', fontsize=styles['label'])
    ax1.set_title('Residual ratio diagnostic', fontsize=styles['title'])
    ax1.tick_params(labelsize=styles['tick'])
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=styles['legend'])

    # Per-band spectrum + ACF panels
    for k, row in enumerate(band_results):
        ax_spec = fig.add_subplot(gs_bot[0, k])
        ax = fig.add_subplot(gs_bot[1, k])
        spec_freq = row.get("_spec_freq")
        spec_flux = row.get("_spec_flux")

        if spec_freq is not None and spec_flux is not None:
            spec_freq = np.asarray(spec_freq, dtype=float)
            spec_flux = np.asarray(spec_flux, dtype=float)
            mask = np.isfinite(spec_freq) & np.isfinite(spec_flux)
            if np.any(mask):
                ax_spec.plot(spec_freq[mask], spec_flux[mask], color='0.25', lw=1.2)
            else:
                ax_spec.text(0.5, 0.5, "No spectrum", ha='center', va='center', transform=ax_spec.transAxes)
        else:
            ax_spec.text(0.5, 0.5, "No spectrum", ha='center', va='center', transform=ax_spec.transAxes)

        ax_spec.set_title(f"Band {row['band_idx']}  ({row['center_mhz']:.0f} MHz)", fontsize=8)
        ax_spec.set_xlabel('Frequency [MHz]', fontsize=7)
        ax_spec.set_ylabel('Flux', fontsize=7)
        ax_spec.tick_params(labelsize=7)
        ax_spec.grid(alpha=0.25)

        lags = row.get("_lags")
        acf = row.get("_acf")
        lf = row.get("_lags_fit")
        ym = row.get("_ymod")

        # x-axis limit: the tighter of fit_max_lag_mhz and half the band width
        half_bw = row.get("band_width_mhz", np.nan) / 2.0
        if fit_max_lag_mhz is not None and np.isfinite(half_bw):
            x_max = min(float(fit_max_lag_mhz), half_bw)
        elif fit_max_lag_mhz is not None:
            x_max = float(fit_max_lag_mhz)
        elif np.isfinite(half_bw):
            x_max = half_bw
        else:
            x_max = None

        if lags is not None and acf is not None:
            lags_sym = np.concatenate((-lags[1:][::-1], lags))
            acf_sym = np.concatenate((acf[1:][::-1], acf))
            ax.plot(lags_sym, acf_sym, color='0.3', lw=1.2, label='ACF')
        if lf is not None and ym is not None:
            lf_sym = np.concatenate((-lf[::-1], lf))
            ym_sym = np.concatenate((ym[::-1], ym))
            ax.plot(
                lf_sym,
                ym_sym,
                color='tab:orange',
                lw=1.6,
                ls='--',
                label=f"Lorentzian\nDelta nu_d={row['dnu_mhz']:.3f} MHz",
            )
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(0, color='0.6', lw=0.8)
        if x_max is not None:
            ax.set_xlim(-x_max, x_max)
        ax.set_xlabel('Delta nu [MHz]', fontsize=7)
        ax.set_ylabel('ACF', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_scint_bw_powerlaw' + (ext if ext else '.png')
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Saved scintillation bandwidth power-law plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_subband_diagnostic(
    fit_details,
    t_burst,
    fig_width,
    output,
    scattering_index,
    fitted_index_err,
    tau_at_ref,
    ref_freq,
):
    """Two-column subband diagnostic: per-band profiles (left) and τ vs frequency (right)."""
    n_bands = len(fit_details["freq"])
    fig = plt.figure(figsize=(fig_width * 2, max(4, n_bands * 1.5)))
    gs = plt.GridSpec(n_bands, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.35)

    ax_left = None
    for i in range(n_bands):
        if ax_left is None:
            ax = fig.add_subplot(gs[i, 0])
            ax_left = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=ax_left)
        profile = fit_details["profile"][i]
        popt = fit_details["popt"][i]
        tau_val = fit_details["tau"][i]
        tau_err = fit_details["tau_err"][i]
        freq_val = fit_details["freq"][i]
        fit_curve = scattered_gaussian(t_burst, *popt)

        tau_label = (
            f"$\\tau={tau_val:.3f}\\pm{tau_err:.3f}$ ms"
            if np.isfinite(tau_err) and tau_err > 0
            else f"$\\tau={tau_val:.3f}$ ms"
        )
        ax.plot(t_burst, profile, "k-", linewidth=1.0)
        ax.plot(t_burst, fit_curve, color=IBM_PALETTE[2], linewidth=1.5, label=tau_label)
        ax.set_ylabel(r"S [arb.]")
        ax.set_yticklabels([])
        ax.set_ylim(bottom=np.nanmin(profile) * 1.1, top=np.nanmax(profile) * 1.3)
        ax.text(0.02, 0.95, f"{freq_val:.1f} MHz", transform=ax.transAxes, va="top", fontsize=8)
        if i == n_bands - 1:
            ax.set_xlabel("Time [ms]")
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    ax_tau = fig.add_subplot(gs[:, 1])
    band_freqs = np.array(fit_details["freq"])
    band_taus = np.array(fit_details["tau"])
    ax_tau.plot(band_freqs, band_taus, "ko", markersize=4, label="Measured $\\tau$")
    freq_grid = np.linspace(band_freqs.min(), band_freqs.max(), 200)
    tau_grid = tau_at_ref * (freq_grid / ref_freq) ** scattering_index
    ax_tau.plot(
        freq_grid,
        tau_grid,
        "b-",
        linewidth=1.5,
        label=f"α={scattering_index:.2f}±{fitted_index_err:.2f}",
    )
    ax_tau.set_xlabel("Frequency [MHz]")
    ax_tau.set_ylabel("τ [ms]")
    ax_tau.legend()
    ax_tau.grid(True, alpha=0.3)

    if output:
        base, ext = os.path.splitext(output)
        out = base + "_subbands" + (ext if ext else ".png")
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Subband diagnostic plot saved to {out}")
    plt.close(fig)


def plot_subband_pa(
    sorted_bands,
    burst_ds,
    burst_ds_q,
    burst_ds_u,
    t_burst,
    fit_details,
    freq,
    fig_width,
    ds,
    ntime,
    output,
):
    """Single-column subband PA overplot. Returns pa_band_info for summary use."""
    n_bands = len(sorted_bands)
    fig = plt.figure(figsize=(fig_width, max(4, n_bands * 1.5)), constrained_layout=False)
    gs = plt.GridSpec(n_bands, 1, hspace=0)
    ax_share = None
    twin_axes = []
    pa_smoothed = []
    pa_band_info = []

    for i, (freq_val, lo, hi) in enumerate(sorted_bands):
        prof_i = np.nanmean(burst_ds[lo:hi, :], axis=0)
        q_prof = np.nanmean(burst_ds_q[lo:hi, :], axis=0)
        u_prof = np.nanmean(burst_ds_u[lo:hi, :], axis=0)
        pa = 0.5 * np.degrees(np.arctan2(u_prof, q_prof))
        pa_smooth = gaussian_filter1d(pa, sigma=2.0, mode="nearest")

        off_n = max(1, ntime // 10)
        I_off = ds[lo:hi, :off_n]
        I_mean_off = np.nanmean(I_off)
        sigma_I = np.nanstd(I_off)
        pa_masked = pa_smooth.copy()
        pa_masked[prof_i < I_mean_off + 0.5 * sigma_I] = np.nan
        pa_smoothed.append(pa_masked)
        pa_band_info.append((freq_val, pa_masked))

        ax = fig.add_subplot(gs[i, 0], sharex=ax_share)
        if ax_share is None:
            ax_share = ax
        ax.plot(t_burst, prof_i, "k-", linewidth=1.0)

        for j, fv in enumerate(fit_details["freq"]):
            if abs(fv - freq_val) < 0.01:
                popt_j = fit_details["popt"][j]
                tau_j = fit_details["tau"][j]
                tau_label = (
                    f"$\\tau={tau_j:.3f}\\pm{fit_details['tau_err'][j]:.3f}$ ms"
                    if np.isfinite(fit_details["tau_err"][j]) and fit_details["tau_err"][j] > 0
                    else f"$\\tau={tau_j:.3f}$ ms"
                )
                fit_curve = scattered_gaussian(t_burst, *popt_j)
                ax.plot(t_burst, fit_curve, color=IBM_PALETTE[2], linewidth=1.5, label=tau_label)
                break

        ax_twin = ax.twinx()
        ax_twin.scatter(t_burst, pa_masked, color=IBM_PALETTE[0], s=2)
        ax_twin.set_ylabel("PA [deg.]")
        twin_axes.append(ax_twin)

        ax.set_ylabel(r"S [arb.]")
        ax.tick_params(labelleft=False)
        i0 = np.nanmin(prof_i)
        i1 = np.nanmax(prof_i)
        dy = i1 - i0
        if dy > 0:
            ax.set_ylim(i0 - 0.1 * dy, i1 * 1.3)
        ax.text(0.02, 0.95, f"{freq_val:.1f} MHz",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(loc="upper right")
        if i == n_bands - 1:
            ax.set_xlabel("Time [ms]")
        else:
            ax.tick_params(labelbottom=False)

    if pa_smoothed:
        all_pa = np.concatenate(pa_smoothed)
        pa_min, pa_max = np.nanmin(all_pa), np.nanmax(all_pa)
        pa_range = pa_max - pa_min
        if pa_range > 0:
            pa_pad = 0.1 * pa_range
            for ax_t in twin_axes:
                ax_t.set_ylim(pa_min - pa_pad, pa_max + pa_pad)

    if output:
        base, ext = os.path.splitext(output)
        out = base + "_subbands_pa" + (ext if ext else ".png")
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Subband PA plot saved to {out}")
    plt.close(fig)
    return pa_band_info


def plot_pa_summary(pa_band_info, t_burst, fig_width, output):
    """Single-panel PA summary: all subbands overplotted."""
    if not pa_band_info:
        return
    pa_band_info_sorted = sorted(pa_band_info, key=lambda x: -x[0])
    fig = plt.figure(figsize=(fig_width, fig_width * 0.6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap("plasma", len(pa_band_info_sorted))
    for k, (fv, pa_vals) in enumerate(pa_band_info_sorted):
        ax.plot(t_burst, pa_vals, color=cmap(k), linewidth=1.0, label=f"{fv:.0f} MHz")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("PA [deg.]")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    if output:
        base, ext = os.path.splitext(output)
        out = base + "_pa_summary" + (ext if ext else ".png")
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"PA summary plot saved to {out}")
    plt.close(fig)


def plot_cn2_profile(s, cn2, ldeg, bdeg, lg_peak, lg_eff_kpc, output=None):
    """Single-panel C_n^2 profile with peak and effective-distance markers."""
    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.65))
    ax.plot(s, cn2, color="tab:blue", lw=1.2, label=r"$C_n^2$")
    ax.set_xlabel("Distance from observer (kpc)")
    ax.set_ylabel(r"$C_n^2$ (m$^{-20/3}$)")
    ax.set_title(f"NE2025  (l={ldeg:.2f} deg, b={bdeg:.2f} deg)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.axvline(
        lg_peak,
        color="tab:green",
        lw=1.0,
        ls="--",
        label=rf"$L_g$ peak = {lg_peak:.3f} kpc",
    )
    if lg_eff_kpc is not None:
        ax.axvline(
            lg_eff_kpc,
            color="tab:orange",
            lw=1.0,
            ls="-.",
            label=rf"$L_g$ (weighted) = {lg_eff_kpc:.3f} kpc",
        )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    if output:
        base, ext = os.path.splitext(output)
        out = base + "_Cn2" + (ext if ext else ".pdf")
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Saved Cn2 profile plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_acf_fit(
    lags_sym,
    acf_sym,
    best_fit,
    best_n_comp,
    component_noise_errs,
    lag_zoom,
    delta_nu_d,
    dnu_err,
    output,
    figsize=None,
):
    """Single-panel ACF fit with best Lorentzian model and component overlays."""
    if figsize is None:
        figsize = pub_figsize(height_ratio=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    xabs = np.abs(lags_sym)
    comp_colors = IBM_PALETTE[::-1]
    labels = ["Lorentzian", "Double Lorentzian", "Triple Lorentzian"]

    ax.plot(lags_sym, acf_sym, label="ACF", color="k", lw=2)
    if delta_nu_d is not None and best_fit and "popt" in best_fit:
        model_fn = [lorentzian, lorentzian_2c, lorentzian_3c][best_n_comp - 1]
        label = (
            f"{labels[best_n_comp - 1]}\n"
            + rf"$\Delta \nu_{{\rm d}} = {delta_nu_d:.2f} \pm {dnu_err:.2f}$ MHz"
            if best_n_comp == 1
            else f"{labels[best_n_comp - 1]} fit"
        )
        ax.plot(
            lags_sym,
            model_fn(xabs, *best_fit["popt"]),
            "-",
            label=label,
            lw=1.5,
            color=comp_colors[0],
        )
        if best_n_comp > 1:
            components, A_fit, C_fit = _decode_lorentzian_components(best_n_comp, best_fit["popt"])
            for i, (w, d) in enumerate(components, start=1):
                errs = component_noise_errs[i - 1] if (i - 1) < len(component_noise_errs) else {}
                dnu_err_i = errs.get("dnu_err", np.nan)
                comp = A_fit * w / (1.0 + (xabs / d) ** 2)
                ax.plot(
                    lags_sym,
                    comp,
                    ls="--",
                    lw=1.5,
                    alpha=0.9,
                    label=rf"$\Delta \nu_{{\rm d}} = {d:.2f} \pm {dnu_err_i:.2f}$ MHz",
                    color=comp_colors[i],
                )
    ax.set_xlim(-lag_zoom, lag_zoom)
    ax.set_xlabel("Frequency lag [MHz]")
    ax.set_ylabel("ACF power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if output:
        savefig_rasterized(output, dpi=300, fig=fig)
        print(f"\nSaved spectrum+ACF plot to {output}")
    else:
        plt.show()
    plt.close(fig)


def _acf_1d(x):
    """Unbiased autocovariance of 1D array x (zero-lag and positive lags)."""
    n = len(x)
    xc = x - np.nanmean(x)
    xc[~np.isfinite(xc)] = 0.0
    result = np.correlate(xc, xc, mode="full")[n - 1 :]
    counts = np.arange(n, 0, -1)
    return result / counts


def _weighted_linear_fit(x, y, yerr):
    """Weighted least-squares line y = a + b*x with weights w = 1/yerr**2.

    Returns (a, a_err, b, b_err) or None when the fit is not possible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if np.count_nonzero(finite) < 2:
        return None
    w = 1.0 / yerr[finite] ** 2
    xf, yf = x[finite], y[finite]
    X = np.column_stack([np.ones_like(xf), xf])
    WX = X * w[:, None]
    A = X.T @ WX
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    a, b = cov @ (WX.T @ yf)
    return a, float(np.sqrt(cov[0, 0])), b, float(np.sqrt(cov[1, 1]))


def compute_modulation_index(ds_onpulse, off_pulse, freq, i_sigma=3.0, nbins=None,
                             min_snr=1.0, raw_acf=False, freq_mask=None):
    """Compute time-resolved modulation index from ACF Lorentzian fits per time bin.

    Each time bin applies the same processing as the full-band measurement: the mean
    spectrum is power-law corrected to the fractional residual (S - S̄)/S̄ via
    correct_spectrum_powerlaw, its ACF computed, and a 1-component Lorentzian
    A(Δν) = C + A/(1 + (Δν/d)²) fitted. The modulation index for that bin is then
    m = √A (dimensionless, since the spectrum is already normalised). With
    raw_acf=True the raw spectrum is used and m = √A/⟨I⟩.

    Parameters
    ----------
    ds_onpulse : ndarray, shape (nfreq, ntime)
        Baseline-subtracted on-pulse dynamic spectrum.
    off_pulse : ndarray, shape (nfreq, n_off)
        Baseline-subtracted off-pulse data for noise estimation.
    freq : ndarray, shape (nfreq,)
        Channel frequencies in MHz.
    i_sigma : float
        Number of off-pulse standard deviations for the I threshold.
    nbins : int or None
        If set, average the time axis into nbins before computing.
    min_snr : float
        Minimum per-channel SNR for the power-law correction (matches --threshold-sigma).
    raw_acf : bool
        Skip the power-law residual correction and use the raw spectrum (matches --raw-acf).
    freq_mask : ndarray of bool or None
        Channel mask applied to the ACF fit, matching --fmin/--fmax (default: all channels).

    Returns
    -------
    dict with keys: i_profile, mod_index, mod_err, weighted_mean,
                    weighted_mean_err, mask, i_cut, t_centers
    """
    nfreq = ds_onpulse.shape[0]
    if freq_mask is None:
        freq_mask = np.ones(nfreq, dtype=bool)
    freq_fit = np.asarray(freq, dtype=float)[freq_mask]
    df = float(np.abs(freq_fit[1] - freq_fit[0])) if freq_fit.size > 1 else 1.0
    total_bw = float(np.abs(freq_fit[-1] - freq_fit[0]))
    max_lag_mhz = min(20.0, 0.15 * total_bw)

    # I threshold from off-pulse noise
    i_off_std = np.nanstd(off_pulse) if off_pulse is not None and off_pulse.size > 0 else 1.0
    i_cut = i_sigma * i_off_std / np.sqrt(nfreq)
    off_pulse_rms = np.nanstd(off_pulse, axis=1) \
        if off_pulse is not None and off_pulse.size > 0 else None

    # Time binning
    ntime = ds_onpulse.shape[1]
    t_centers = None
    if nbins is not None and nbins > 1:
        bin_edges = np.linspace(0, ntime, nbins + 1).astype(int)
    else:
        nbins = ntime
        bin_edges = np.arange(ntime + 1)

    i_prof = np.zeros(nbins)
    mod_idx = np.full(nbins, np.nan)
    mod_err = np.full(nbins, np.nan)
    dnu_d = np.full(nbins, np.nan)
    dnu_d_err = np.full(nbins, np.nan)
    t_centers = np.zeros(nbins)

    for k in range(nbins):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        t_centers[k] = 0.5 * (lo + hi - 1)
        spectrum = np.nanmean(ds_onpulse[:, lo:hi], axis=1)
        i_prof[k] = np.nanmean(spectrum)

        # Skip bins below the I threshold
        if i_prof[k] < i_cut or not np.isfinite(i_prof[k]):
            continue

        # Identical processing to the full-band path, applied per time bin:
        # power-law residual correction (unless raw_acf) before the ACF fit
        try:
            if raw_acf:
                fit_spectrum = spectrum[freq_mask]
            else:
                corrected, _, _, _, _ = correct_spectrum_powerlaw(
                    freq, spectrum, off_pulse_rms, min_snr=min_snr)
                fit_spectrum = corrected[freq_mask]

            # ACF and Lorentzian fit
            acf = _acf_1d(fit_spectrum)
            lags = np.arange(len(acf)) * df
            fit_ok = (lags > 0) & (lags < max_lag_mhz) & np.isfinite(acf)
            lag_fit = lags[fit_ok]
            acf_fit = acf[fit_ok]
            if lag_fit.size < 5:
                continue

            d_guess = max_lag_mhz / 4.0
            A_guess = float(acf[0]) if np.isfinite(acf[0]) else 0.1
            C_guess = float(np.nanmedian(acf_fit[-min(5, len(acf_fit)):]))
            popt, pcov = curve_fit(
                lorentzian, lag_fit, acf_fit, p0=[d_guess, A_guess, C_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=2000,
            )
            A_val = float(popt[1])
            d_val = float(popt[0])
            if A_val <= 0 or d_val <= 0:
                continue
            m_val = np.sqrt(A_val)
            if raw_acf:
                m_val /= i_prof[k]
            mod_idx[k] = m_val
            dnu_d[k] = d_val
            if pcov is not None and np.isfinite(pcov[1, 1]):
                mod_err[k] = np.sqrt(max(0.0, pcov[1, 1])) / (2.0 * mod_idx[k])
            if pcov is not None and np.isfinite(pcov[0, 0]):
                dnu_d_err[k] = np.sqrt(max(0.0, pcov[0, 0]))
        except Exception:
            continue

    mask = i_prof >= i_cut

    # Weighted mean over masked bins
    good = mask & np.isfinite(mod_idx) & (mod_err > 0)
    wmean = np.nan
    wmean_err = np.nan
    if np.any(good):
        w = 1.0 / mod_err[good] ** 2
        wsum = np.nansum(w)
        if wsum > 0:
            wmean = np.nansum(mod_idx[good] * w) / wsum
            wmean_err = 1.0 / np.sqrt(wsum)

    # Weighted linear fit over the same good bins (t_centers in sample units)
    fit = _weighted_linear_fit(t_centers[good], mod_idx[good], mod_err[good]) if np.any(good) else None

    return {
        "i_profile": i_prof,
        "mod_index": mod_idx,
        "mod_err": mod_err,
        "dnu_d": dnu_d,
        "dnu_d_err": dnu_d_err,
        "weighted_mean": wmean,
        "weighted_mean_err": wmean_err,
        "fit_ok": fit is not None,
        "fit_intercept": fit[0] if fit is not None else np.nan,
        "fit_intercept_err": fit[1] if fit is not None else np.nan,
        "fit_slope": fit[2] if fit is not None else np.nan,
        "fit_slope_err": fit[3] if fit is not None else np.nan,
        "mask": mask,
        "i_cut": i_cut,
        "t_centers": t_centers,
    }


def plot_modulation_index(t_mod, t_profile, mod_index, mod_err, i_profile,
                          weighted_mean, weighted_mean_err=0.0,
                          dnu_d=None, dnu_d_err=None,
                          i_cut=None, output=None, fig_width=None,
                          max_err_frac=0.5, ncol=None):
    """Three-panel: modulation index (top), scintillation bandwidth (middle),
    and pulse profile (bottom)."""
    if fig_width is None:
        fig_width, _ = pub_figsize(ncol=ncol)
    fig = plt.figure(figsize=(fig_width, fig_width * 0.7), constrained_layout=False)
    gs = plt.GridSpec(2, 1, hspace=0)

    # Mask unreliable points
    with np.errstate(divide='ignore', invalid='ignore'):
        good = np.isfinite(mod_index) & (mod_index > 0) & np.isfinite(mod_err) & (mod_err > 0) & (mod_err / mod_index < max_err_frac)
    if dnu_d is not None:
        good &= np.isfinite(dnu_d) & (dnu_d > 0)
        if dnu_d_err is not None:
            good &= np.isfinite(dnu_d_err) & (dnu_d_err > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                good &= (dnu_d_err / dnu_d < max_err_frac)

    # Top panel: modulation index
    ax_m = fig.add_subplot(gs[0, 0])
    if np.any(good):
        ax_m.errorbar(t_mod[good], mod_index[good], yerr=mod_err[good],
                      fmt='o', color=IBM_PALETTE[2],
                      markersize=2, capsize=2, capthick=0.5, linewidth=0.5)

        peak_idx = np.argmax(mod_index[good])
        m_peak = mod_index[good][peak_idx]
        m_peak_err = mod_err[good][peak_idx]

    ax_m.axhline(weighted_mean, color=IBM_PALETTE[2], alpha=0.7, linewidth=1.5, linestyle='--',
                 label=rf'$\langle m_g \rangle = {weighted_mean:.4f} \pm {weighted_mean_err:.4f}$')
    #if np.any(good):
    #    mean_unw = float(np.nanmean(mod_index[good]))
    #    n_good   = int(np.count_nonzero(good))
    #    mean_unw_err = float(np.nanstd(mod_index[good], ddof=1) / np.sqrt(n_good)) if n_good > 1 else float('nan')
    #    ax_m.axhline(mean_unw, color='k', alpha=0.6, linewidth=1.0, linestyle='-.',
    #                 label=rf'$\overline{{m_g}} = {mean_unw:.4f} \pm {mean_unw_err:.4f}$')
    fit = _weighted_linear_fit(t_mod[good], mod_index[good], mod_err[good]) if np.any(good) else None
    fit_line = None
    if fit is not None:
        a, a_err, b, b_err = fit
        x_line = np.array([float(np.min(t_mod[good])), float(np.max(t_mod[good]))])
        fit_line, = ax_m.plot(x_line, a + b * x_line, color=IBM_PALETTE[3], alpha=0.8,
                              linewidth=1.5, linestyle=':',
                              #label=rf'$m_g(t) = {a:.3f} + ({b:.3f}\pm{b_err:.3f})\,t$'
                              )
        print(f"  m_g(t) weighted linear fit: intercept = {a:.4f} ± {a_err:.4f}, "
              f"slope = {b:.4f} ± {b_err:.4f} per ms")
    if np.any(good):
        ax_m.plot([], [], ' ', label=rf'$m_g^{{\rm peak}} = {m_peak:.4f} \pm {m_peak_err:.4f}$')
    ax_m.set_ylabel(r'$m_g$')
    ax_m.tick_params(labelbottom=False)
    ax_m.legend(loc='upper right', fontsize=8)
    ax_m.grid(True, alpha=0.3)

    ## Middle panel: scintillation bandwidth
    #ax_d = fig.add_subplot(gs[1, 0], sharex=ax_m)
    #if dnu_d is not None and np.any(good):
    #    y = dnu_d[good]
    #    yerr = dnu_d_err[good] if dnu_d_err is not None else None
    #    ax_d.errorbar(t_mod[good], y, yerr=yerr,
    #                  fmt='o', color=IBM_PALETTE[1],
    #                  markersize=2, capsize=2, capthick=0.5, linewidth=0.5)
    #    if np.any(np.isfinite(y)):
    #        dnu_mean = np.nanmean(y)
    #        dnu_sem = np.nanstd(y, ddof=1) / np.sqrt(np.sum(np.isfinite(y)))
    #        ax_d.axhline(dnu_mean, color=IBM_PALETTE[1], alpha=0.7,
    #                     linewidth=1.5, linestyle='--',
    #                     label=rf'$\langle \delta\nu_d \rangle = {dnu_mean:.3f} \pm {dnu_sem:.3f}$ MHz')
    #        ax_d.legend(loc='upper right', fontsize=8)
#
    #ax_d.set_ylabel(r'$\delta\nu_d$ [MHz]')
    #ax_d.tick_params(labelbottom=False)
    #ax_d.grid(True, alpha=0.3)

    # Bottom panel: pulse profile (full resolution)
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_p.plot(t_profile, i_profile, 'k-', linewidth=1.0)
    ax_p.set_xlabel('Time [ms]')
    ax_p.set_yticklabels([])
    ax_p.set_ylabel(r'S [arb.]')
    ax_p.grid(True, alpha=0.3)

    # Extend the fitted line across the full width of the (shared) axes
    if fit_line is not None:
        x_bounds = np.asarray(ax_m.get_xlim())
        fit_line.set_xdata(x_bounds)
        fit_line.set_ydata(a + b * x_bounds)

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_modulation' + (ext if ext else '.png')
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Modulation index plot saved to {out}")
    else:
        plt.show()
    plt.close(fig)
