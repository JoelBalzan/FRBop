"""Plotting helpers for scintillation workflows."""

import os

import matplotlib.pyplot as plt
import numpy as np

from frbop.scop.fit_utils import _decode_lorentzian_components
from frbop.scop.models import lorentzian, lorentzian_2c, lorentzian_3c
from frbop.utils.plotting import savefig_rasterized, set_pub_style


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
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.0))
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mean_model = np.asarray(mean_model, dtype=float)

    finite = np.isfinite(freq_mhz) & np.isfinite(spectrum)
    if np.any(finite):
        axs[0].plot(freq_mhz[finite], spectrum[finite], color='0.25', lw=1.4, label='Spectrum')

    model_mask = np.isfinite(freq_mhz) & np.isfinite(mean_model)
    if np.any(model_mask):
        idx_str = "" if spectral_index is None else f" (index={float(spectral_index):.2f})"
        axs[0].plot(
            freq_mhz[model_mask],
            mean_model[model_mask],
            color='tab:orange',
            lw=1.4,
            ls='--',
            label=f"Power-law fit{idx_str}",
        )

    axs[0].set_title('Spectrum power-law fit', fontsize=styles['title'])
    axs[0].set_xlabel('Frequency (MHz)', fontsize=styles['label'])
    axs[0].set_ylabel('Flux / intensity', fontsize=styles['label'])
    axs[0].tick_params(labelsize=styles['tick'])
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=styles['legend'])

    corrected = np.full_like(spectrum, np.nan, dtype=float)
    corr_mask = np.isfinite(mean_model) & (mean_model > 0) & np.isfinite(spectrum)
    if np.any(corr_mask):
        corrected[corr_mask] = (spectrum[corr_mask] - mean_model[corr_mask]) / mean_model[corr_mask]
        axs[1].plot(freq_mhz[corr_mask], corrected[corr_mask], color='k', lw=1.4)
    else:
        axs[1].text(0.5, 0.5, "No valid corrected spectrum", ha='center', va='center', transform=axs[1].transAxes)

    axs[1].set_title('Corrected spectrum', fontsize=styles['title'])
    axs[1].set_xlabel('Frequency (MHz)', fontsize=styles['label'])
    axs[1].set_ylabel('Corrected flux', fontsize=styles['label'])
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
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
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
        ax.set_xlabel("Delta nu (MHz)", fontsize=styles['label'])
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
    ax1.set_xlabel("Delta nu (MHz)", fontsize=styles['label'])
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
    fig = plt.figure(figsize=(max(13.0, 4.5 * n_bands), 12.5))
    gs_top = fig.add_gridspec(1, 2, top=0.95, bottom=0.62, hspace=0.35, wspace=0.3)
    gs_bot = fig.add_gridspec(2, n_bands, top=0.57, bottom=0.07, hspace=0.45, wspace=0.35)

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

    ax0.set_xlabel('Frequency (MHz)', fontsize=styles['label'])
    ax0.set_ylabel(r'$\Delta\nu_d$ (MHz)', fontsize=styles['label'])
    ax0.set_title('Scintillation bandwidth vs frequency', fontsize=styles['title'])
    ax0.tick_params(labelsize=styles['tick'])
    ax0.grid(alpha=0.25, which='both')
    ax0.legend(fontsize=styles['legend'])
    ax1.set_xlabel('Frequency (MHz)', fontsize=styles['label'])
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
        ax_spec.set_xlabel('Frequency (MHz)', fontsize=7)
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
        ax.set_xlabel('Delta nu (MHz)', fontsize=7)
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
