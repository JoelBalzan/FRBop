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

    Also computes the corrected upper limit from Pradeep et al. (2025, A&A 700, A99)
    Eq. (7.6), "this work":
        LxLg <= (1+z) Ds^2 / (8 pi nu^2) * nu_dc / (mg * tau_scatt)

    Relative to the Sammons et al. (2023) form, Pradeep et al. (2025) note two
    corrections (see their Sec. 2 footnote 1 and Sec. 7.2):
      (1) the (1+z) factor moves from the denominator to the numerator, following
          their correction to Macquart & Koay (2013);
      (2) an additional factor of 4 appears in the denominator (2 pi -> 8 pi);
      (3) m_g enters linearly (as an upper-limit/Narayan 1992 factor) rather than
          squared, since it is derived from Eq. (7.4)/(7.5) rather than assuming
          m_g^2 directly cancels the broadening as in Sammons et al. (2023). The
          paper notes the Sammons et al. (2023) mg^2 form should be interpreted
          as an upper limit rather than an estimate, since it assumes a different
          (incompatible) scaling of nu_s and m with resolution power.

    Returns a dict with keys: label, dnu, lxlg_upper, lxlg_partial, lx_upper,
    lx_partial, lxlg_thiswork, lx_thiswork.
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

    # Pradeep et al. (2025) Eq. (7.6) "this work" upper limit
    geom_thiswork = (1.0 + redshift) * (ds_kpc ** 2) / (8.0 * np.pi * (nu_hz ** 2))
    lxlg_thiswork = (
        geom_thiswork * (nu_dc_hz / (mg * t_s)) if (mg is not None and mg > 0) else None
    )
    lx_thiswork = (
        lxlg_thiswork / lg_kpc
        if (lxlg_thiswork is not None and lg_kpc is not None and lg_kpc > 0)
        else None
    )

    return dict(
        label=label,
        dnu_mhz=delta_nu_d_mhz,
        c_val=2.0 * np.pi * nu_dc_hz * t_s,
        lxlg_upper=lxlg_upper,
        lxlg_partial=lxlg_partial,
        lx_upper=lx_upper,
        lx_partial=lx_partial,
        lxlg_thiswork=lxlg_thiswork,
        lx_thiswork=lx_thiswork,
    )


def print_two_screen_results(
    results: list[dict],
    tau_ms,
    nu_obs_mhz,
    redshift,
    mg,
    lg_kpc,
    lg_peak_kpc,
    lg_eff_kpc,
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
        print(f"  L_g (used)       = {lg_kpc:.4f} kpc  ({lg_source})")
    if lg_peak_kpc is not None:
        print(f"  L_g (peak)       = {lg_peak_kpc:.4f} kpc")
    if lg_eff_kpc is not None and (lg_kpc is None or abs(lg_eff_kpc - lg_kpc) > 1e-6):
        print(f"  L_g (weighted) = {lg_eff_kpc:.4f} kpc")

    for r in results:
        print(f"\n  --- {r['label']} ---")
        print(f"    Delta nu_d              = {r['dnu_mhz']:.4f} MHz")
        print(f"    C = 2pi nu_dc tau       = {r['c_val']:.3e}")
        print(f"    Eq.(2) L_x L_g <=        {r['lxlg_upper']:.4e} kpc^2  (m_g=1 limit)")
        if r['lxlg_partial'] is not None:
            if mg is not None and mg > 1.0:
                print(f"    Eq.(4) skipped: m_g={mg:.3f} > 1 is unphysical")
            else:
                print(f"    Eq.(4) L_x L_g ~=        {r['lxlg_partial']:.4e} kpc^2  (m_g={mg:.4f})  [Sammons et al. 2023]")
        if r['lx_upper'] is not None:
            print(f"    Eq.(2) L_x <=            {r['lx_upper']:.4e} kpc")
        if r['lx_partial'] is not None:
            print(f"    Eq.(4) L_x ~=            {r['lx_partial']:.4e} kpc")
        if r.get('lxlg_thiswork') is not None:
            if mg is not None and mg > 1.0:
                print(f"    Eq.(7.6) skipped: m_g={mg:.3f} > 1 is unphysical")
            else:
                print(f"    Eq.(7.6) L_x L_g <=      {r['lxlg_thiswork']:.4e} kpc^2  (m_g={mg:.4f})  [Pradeep et al. 2025]")
        if r.get('lx_thiswork') is not None:
            print(f"    Eq.(7.6) L_x <=          {r['lx_thiswork']:.4e} kpc")