"""Plotting helpers for scintillation workflows."""

import os

import matplotlib.pyplot as plt
import numpy as np

from frbop.scop.fit_utils import _decode_lorentzian_components
from frbop.scop.macquart import _powerlaw_mean_spectrum
from frbop.scop.models import lorentzian, lorentzian_2c, lorentzian_3c

# Publication-friendly sizing (in inches) adapted from RM plotting helpers
TWO_COLUMN_WIDTH_IN = 7.1
SINGLE_COLUMN_WIDTH_IN = 4.8


def _pub_figsize(single_column: bool = True, height_ratio: float = 0.62, min_height: float = 3.0):
    """Return a figure size (width, height) in inches suitable for LaTeX figures.

    By default returns a single-column width for a two-column layout. Set
    single_column=False to get a full two-column width.
    """
    width = SINGLE_COLUMN_WIDTH_IN if single_column else TWO_COLUMN_WIDTH_IN
    height = max(min_height, width * height_ratio)
    return (width, height)


def _plot_style():
    """Return a small dict of plotting sizes used for publication-style figures."""
    return {
        'title': 11,
        'label': 10,
        'tick': 8,
        'legend': 8,
        'annotation': 7,
        'line': 1.2,
    }


def plot_macquart_diagnostics(
    freq_mhz,
    raw_spectrum,
    raw_result,
    corrected_result,
    output=None,
    fit_max_lag_mhz=None,
):
    m2_raw, dnu_raw, lags_raw, acov_raw = raw_result
    m2_corr, dnu_corr, lags_corr, acov_corr = corrected_result

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    raw_spectrum = np.asarray(raw_spectrum, dtype=float)
    finite = np.isfinite(freq_mhz) & np.isfinite(raw_spectrum)

    ax0 = axs[0]
    if np.any(finite):
        ax0.plot(freq_mhz[finite], raw_spectrum[finite], color='0.25', lw=1.4, label='Raw spectrum')
        try:
            raw_mean = float(np.nanmean(raw_spectrum[finite]))
            if np.isfinite(raw_mean) and raw_mean > 0:
                ax0.axhline(raw_mean, color='tab:blue', lw=1.1, ls='--', label='Raw mean')
            cm = _powerlaw_mean_spectrum(freq_mhz[finite], raw_spectrum[finite])
            if np.all(np.isfinite(cm)):
                ax0.plot(
                    freq_mhz[finite],
                    cm,
                    color='tab:orange',
                    lw=1.2,
                    ls='--',
                    label=r'$\nu^{-1.5}$ mean model',
                )
        except Exception:
            pass
    ax0.set_title('Macquart spectrum reference')
    ax0.set_xlabel('Frequency (MHz)')
    ax0.set_ylabel('Flux / intensity')
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=8)

    ax1 = axs[1]
    plotted = False
    for lags, acov, label, color, dnu in (
        (lags_raw, acov_raw, 'Raw mean normalisation', 'tab:blue', dnu_raw),
        (lags_corr, acov_corr, r'$\nu^{-1.5}$ corrected', 'tab:orange', dnu_corr),
    ):
        if lags.size == 0 or acov.size == 0:
            continue
        plotted = True
        mask = np.isfinite(lags) & np.isfinite(acov)
        if fit_max_lag_mhz is not None:
            mask &= lags <= float(fit_max_lag_mhz)
        if not np.any(mask):
            continue
        half = 0.5 * float(acov[0]) if np.isfinite(acov[0]) else np.nan
        ax1.plot(lags[mask], acov[mask], color=color, lw=1.6, label=f'{label} ACF')
        if np.isfinite(half):
            ax1.axhline(half, color=color, ls=':', lw=1.0, alpha=0.85)
        if dnu is not None and np.isfinite(dnu):
            ax1.axvline(dnu, color=color, ls='--', lw=1.1, alpha=0.85)
            ax1.text(
                dnu,
                half if np.isfinite(half) else 0.05,
                f'  nu_dc~{dnu:.3f} MHz',
                color=color,
                fontsize=8,
                va='bottom',
            )
    ax1.axhline(0.0, color='0.5', lw=1.0)
    ax1.set_title('Macquart autocovariance')
    ax1.set_xlabel('Frequency lag Delta nu (MHz)')
    ax1.set_ylabel('Mean-normalised autocovariance')
    ax1.grid(alpha=0.25)
    if plotted:
        ax1.legend(fontsize=8)

    fig.suptitle('Macquart modulation-index diagnostics')
    plt.tight_layout()

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_macquart_diagnostics' + (ext if ext else '.png')
        plt.savefig(out, dpi=220)
        print(f"Saved Macquart diagnostics plot to {out}")
    else:
        plt.show()
    plt.close(fig)


def plot_lorentzian_diagnostics(
    lags_plot_sym,
    acf_plot_sym,
    lags_lorentz_fit,
    acf_lorentz_fit,
    fit_models,
    output=None,
):
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
        ax.set_title(title)
        ax.set_xlabel("Delta nu (MHz)")
        ax.set_ylabel("ACF")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    # Residuals
    ax1 = axs[1, 0]
    ax1.axhline(0.0, color='0.5', lw=1)
    for name, result, _ in fit_models:
        if "ymod" in result:
            ax1.plot(lags_lorentz_fit, acf_lorentz_fit - result["ymod"], lw=1.3, label=name)
    ax1.set_title("Residuals (positive lags)")
    ax1.set_xlabel("Delta nu (MHz)")
    ax1.set_ylabel("ACF residual")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

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
        plt.savefig(out, dpi=220)
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
    # Top row: power-law plot + residual ratio
    # Bottom row: per-band ACF + Lorentzian fits
    n_bands = len(band_results)
    fig = plt.figure(figsize=(max(13.0, 4.5 * n_bands), 10.0))
    gs_top = fig.add_gridspec(1, 2, top=0.95, bottom=0.55, hspace=0.35, wspace=0.3)
    gs_bot = fig.add_gridspec(1, n_bands, top=0.45, bottom=0.07, hspace=0.35, wspace=0.35)

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

    ax0.set_xlabel('Frequency (MHz)')
    ax0.set_ylabel(r'$\Delta\nu_d$ (MHz)')
    ax0.set_title('Scintillation bandwidth vs frequency')
    ax0.grid(alpha=0.25, which='both')
    ax0.legend(fontsize=8)
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Observed / model')
    ax1.set_title('Residual ratio diagnostic')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    # Per-band ACF panels
    for k, row in enumerate(band_results):
        ax = fig.add_subplot(gs_bot[0, k])
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
        ax.set_title(f"Band {row['band_idx']}  ({row['center_mhz']:.0f} MHz)", fontsize=8)
        ax.set_xlabel('Delta nu (MHz)', fontsize=7)
        ax.set_ylabel('ACF', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)

    if output:
        base, ext = os.path.splitext(output)
        out = base + '_scint_bw_powerlaw' + (ext if ext else '.png')
        plt.savefig(out, dpi=220)
        print(f"Saved scintillation bandwidth power-law plot to {out}")
    else:
        plt.show()
    plt.close(fig)
