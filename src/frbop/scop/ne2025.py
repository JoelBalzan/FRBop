"""NE2025 Cn2 profile and scattering predictions."""

import os

import matplotlib.pyplot as plt
import numpy as np

from frbop.utils.plotting import pub_figsize, savefig_rasterized, set_pub_style


def get_cn2_profile(l_deg, b_deg, da_kpc, ndir=-1):
    import mwprop.nemod.NE2025 as _ne2025_mod
    ne2025 = _ne2025_mod.ne2025
    outdir = os.path.join(os.getcwd(), 'output_ne2025p')
    os.makedirs(outdir, exist_ok=True)
    ne2025(
        l_deg,
        b_deg,
        da_kpc,
        ndir,
        classic=False,
        dmd_only=False,
        do_analysis=True,
        plotting=False,
        verbose=False,
    )
    prefix = "d2dm" if ndir < 0 else "dm2d"
    f25 = os.path.join(outdir, f'f25_{prefix}_ne_dsm_vs_s.txt')
    if not os.path.exists(f25):
        raise FileNotFoundError(f"NE2025 LoS profile not found at {f25}")
    data = np.loadtxt(f25, skiprows=3)
    s, ne, cn2 = data[:, 0], data[:, 4], data[:, 5]
    nonzero = np.where(ne != 0)[0]
    if nonzero.size > 0:
        indkeep = min(int(1.1 * nonzero[-1]), s.size)
        s, cn2 = s[:indkeep], cn2[:indkeep]
    return s, cn2


def estimate_lg_kpc_from_ne2025(ldeg, bdeg, da_kpc, max_dist_kpc=50.0, output=None):
    """Return (lg_peak_kpc, cn2_peak, lg_eff_kpc) from an NE2025 Cn2 profile."""
    s, cn2 = get_cn2_profile(ldeg, bdeg, da_kpc=max_dist_kpc)

    set_pub_style(use_latex=False)

    fig, ax = plt.subplots(figsize=pub_figsize(height_ratio=0.65))
    ax.plot(s, cn2, color='tab:blue', lw=1.2, label=r'$C_n^2$')
    ax.set_xlabel("Distance from observer (kpc)")
    ax.set_ylabel(r"$C_n^2$ (m$^{-20/3}$)")
    ax.set_title(f"NE2025  (l={ldeg:.2f} deg, b={bdeg:.2f} deg)")
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    if np.nansum(cn2) == 0.0:
        print("Warning: Cn2 profile is all zeros - check coordinates and NE2025 model.")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return float(s[0]), 0.0

    lg_peak = float(s[np.argmax(cn2)])
    cn2_peak = float(np.max(cn2))
    lg_eff_kpc = None
    if da_kpc is not None and np.isfinite(da_kpc) and da_kpc > 0:
        geom_weight = s * (1.0 - s / da_kpc)
        numer = np.trapezoid(cn2 * geom_weight, s)
        denom = np.trapezoid(cn2, s)
        if denom > 0 and np.isfinite(numer):
            lg_eff_kpc = float(numer / denom)
    ax.axvline(
        lg_peak,
        color='tab:green',
        lw=1.0,
        ls='--',
        label=rf'$L_g$ peak = {lg_peak:.3f} kpc',
    )
    if lg_eff_kpc is not None:
        ax.axvline(
            lg_eff_kpc,
            color='tab:orange',
            lw=1.0,
            ls='-.',
            label=rf'$L_g$ (weighted) = {lg_eff_kpc:.3f} kpc',
        )
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    if output:
        base, ext = os.path.splitext(output)
        out = base + '_Cn2' + (ext if ext else '.pdf')
        savefig_rasterized(out, dpi=300, fig=fig)
        print(f"Saved Cn2 profile plot to {out}")
    else:
        plt.show()
    plt.close(fig)
    return lg_peak, cn2_peak, lg_eff_kpc


# ---------------------------------------------------------------------------
# NE2025 scattering / scintillation predictions
# ---------------------------------------------------------------------------


def ne2025_scattering_prediction(
    s_kpc: np.ndarray,
    cn2: np.ndarray,
    lg_kpc: float,
    ds_kpc: float,
    nu_ref_mhz: float,
    v_iss_km_s: float = 100.0,
) -> dict:
    """Predict tau_scatt, Delta nu_d, and t_scint from an NE2025 Cn2 profile."""
    kpc_to_m = 3.085677581e19  # 1 kpc in metres
    c_m_s = 2.99792458e8  # speed of light m/s

    # Numerical SM = integral Cn2(s) ds  [kpc * m^{-20/3}]
    s_arr = np.asarray(s_kpc, dtype=float)
    cn2_arr = np.asarray(cn2, dtype=float)
    finite = np.isfinite(s_arr) & np.isfinite(cn2_arr) & (cn2_arr >= 0)
    if not np.any(finite):
        raise ValueError("Cn2 profile has no finite non-negative values")
    SM_kpc = float(np.trapezoid(cn2_arr[finite], s_arr[finite]))  # kpc * m^{-20/3}

    # Convert to SI: m^{-17/3}
    SM_si = SM_kpc * kpc_to_m

    # Weighted effective geometric distance for an extended medium
    if ds_kpc > 0 and np.isfinite(ds_kpc):
        geom_weight = s_arr[finite] * (1.0 - s_arr[finite] / ds_kpc)
        numer = np.trapezoid(cn2_arr[finite] * geom_weight, s_arr[finite])
        denom = np.trapezoid(cn2_arr[finite], s_arr[finite])
        if denom > 0 and np.isfinite(numer):
            lg_eff_kpc = float(numer / denom)
        else:
            lg_eff_kpc = lg_kpc
    else:
        lg_eff_kpc = lg_kpc
    D_eff_m = lg_eff_kpc * kpc_to_m

    nu_hz = nu_ref_mhz * 1e6
    lam_m = c_m_s / nu_hz

    # Pulse broadening: Cordes & Lazio (2003) eq.(4) numeric constant
    r_e = 2.8179403e-15
    tau_scatt_s = (r_e ** 2 * lam_m ** 4 * SM_si * D_eff_m) / (2.0 * np.pi * c_m_s)
    tau_scatt_ms = tau_scatt_s * 1e3

    # Decorrelation bandwidth Delta nu_d = 1 / (2 pi tau_scatt)
    if tau_scatt_s > 0:
        delta_nu_d_hz = 1.0 / (2.0 * np.pi * tau_scatt_s)
        delta_nu_d_mhz = delta_nu_d_hz / 1e6
    else:
        delta_nu_d_mhz = np.nan

    # Diffractive scintillation timescale
    V_iss_m_s = v_iss_km_s * 1e3
    if np.isfinite(delta_nu_d_mhz) and delta_nu_d_mhz > 0:
        delta_nu_d_hz_val = delta_nu_d_mhz * 1e6
        r_diff_m = np.sqrt(c_m_s * D_eff_m * delta_nu_d_hz_val) / nu_hz
        t_scint_s = r_diff_m / V_iss_m_s
    else:
        r_diff_m = np.nan
        t_scint_s = np.nan

    return dict(
        SM_kpc=SM_kpc,
        SM_si=SM_si,
        lg_eff_kpc=lg_eff_kpc,
        tau_scatt_ms=tau_scatt_ms,
        delta_nu_d_mhz=delta_nu_d_mhz,
        t_scint_s=t_scint_s,
        r_diff_m=r_diff_m,
        nu_ref_mhz=nu_ref_mhz,
        v_iss_km_s=v_iss_km_s,
    )


def print_ne2025_scattering_prediction(pred: dict, lg_peak_kpc: float | None, ds_kpc: float) -> None:
    print("\n  NE2025 predicted scattering (Galactic screen):")
    print(f"    Reference frequency    = {pred['nu_ref_mhz']:.3f} MHz")
    print(f"    SM (integral)          = {pred['SM_kpc']:.4e} kpc m^{{-20/3}}")
    print(f"    SM (SI)                = {pred['SM_si']:.4e} m^{{-17/3}}")
    if lg_peak_kpc is not None:
        print(f"    L_g (peak)             = {lg_peak_kpc:.4f} kpc")
    print(f"    L_g (weighted)         = {pred['lg_eff_kpc']:.4f} kpc")
    print(f"    D_s                    = {ds_kpc:.4e} kpc")
    print(f"    tau_scatt (predicted)  = {pred['tau_scatt_ms']:.4e} ms")
    print(f"    Delta nu_d (predicted) = {pred['delta_nu_d_mhz']:.4e} MHz")
    if np.isfinite(pred['t_scint_s']):
        print(
            f"    t_scint (predicted)    = {pred['t_scint_s']:.4e} s  "
            f"(V_ISS = {pred['v_iss_km_s']:.0f} km/s assumed)"
        )
    else:
        print("    t_scint (predicted)    = N/A")
