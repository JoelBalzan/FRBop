"""Two-screen distance calculations."""

import numpy as np


def two_screen_estimate(
    delta_nu_d_mhz: float,
    tau_ms: float,
    nu_obs_mhz: float,
    redshift: float,
    ds_kpc: float,
    mg: float | None,
    lg_kpc: float | None,
    label: str = "",
) -> dict:
    """Compute LxLg (and optionally Lx) from the two-screen scattering model.

    Based on Sammons et al. (2023) Eqs. (2) and (4):
        LxLg <= Ds^2 * nu_dc / (2 pi nu^2 (1+z) tau_scatt)    [Eq. 2, mg=1 limit]
        LxLg ~= LxLg_upper / mg^2                            [Eq. 4, mg<1]

    Returns a dict with keys: label, dnu, lxlg_upper, lxlg_partial, lx_upper, lx_partial.
    """
    nu_hz = nu_obs_mhz * 1e6
    nu_dc_hz = delta_nu_d_mhz * 1e6
    t_s = tau_ms * 1e-3

    geom = (ds_kpc ** 2) / (2.0 * np.pi * (nu_hz ** 2) * (1.0 + redshift))
    lxlg_upper = geom * (nu_dc_hz / t_s)
    lxlg_partial = lxlg_upper / (mg ** 2) if (mg is not None and mg > 0) else None

    lx_upper = lxlg_upper / lg_kpc if (lg_kpc is not None and lg_kpc > 0) else None
    lx_partial = (
        lxlg_partial / lg_kpc if (lxlg_partial is not None and lg_kpc is not None and lg_kpc > 0) else None
    )

    return dict(
        label=label,
        dnu_mhz=delta_nu_d_mhz,
        c_val=2.0 * np.pi * nu_dc_hz * t_s,
        lxlg_upper=lxlg_upper,
        lxlg_partial=lxlg_partial,
        lx_upper=lx_upper,
        lx_partial=lx_partial,
    )


def print_two_screen_results(
    results: list[dict],
    tau_ms,
    nu_obs_mhz,
    redshift,
    mg,
    lg_kpc,
    delta_nu_d_for_calc_source,
    lg_source,
):
    print("\n" + "=" * 60)
    print("TWO-SCREEN DISTANCE ESTIMATES")
    print("=" * 60)
    print(f"  tau_scatt        = {tau_ms:.4f} ms")
    print(f"  nu_obs           = {nu_obs_mhz:.3f} MHz")
    print(f"  z                = {redshift:.6f}")
    print(f"  Delta nu_d source = {delta_nu_d_for_calc_source}")
    if mg is not None:
        print(f"  m_g              = {mg:.6f}")
    if lg_kpc is not None:
        print(f"  L_g              = {lg_kpc:.4f} kpc  ({lg_source})")

    for r in results:
        print(f"\n  --- {r['label']} ---")
        print(f"    Delta nu_d              = {r['dnu_mhz']:.4f} MHz")
        print(f"    C = 2pi nu_dc tau       = {r['c_val']:.3e}")
        print(f"    Eq.(2) L_x L_g <=        {r['lxlg_upper']:.4e} kpc^2  (m_g=1 limit)")
        if r['lxlg_partial'] is not None:
            if mg is not None and mg > 1.0:
                print(f"    Eq.(4) skipped: m_g={mg:.3f} > 1 is unphysical")
            else:
                print(f"    Eq.(4) L_x L_g ~=        {r['lxlg_partial']:.4e} kpc^2  (m_g={mg:.4f})")
        if r['lx_upper'] is not None:
            print(f"    Eq.(2) L_x <=            {r['lx_upper']:.4e} kpc")
        if r['lx_partial'] is not None:
            print(f"    Eq.(4) L_x ~=            {r['lx_partial']:.4e} kpc")
