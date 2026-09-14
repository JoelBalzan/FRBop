"""NE2025 Cn2 profile and scattering predictions."""

import os

import numpy as np

from frbop.scop.plotting import plot_cn2_profile
from frbop.utils.plotting import set_pub_style


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

    if np.nansum(cn2) == 0.0:
        print("Warning: Cn2 profile is all zeros - check coordinates and NE2025 model.")
        return float(s[0]), 0.0, None

    lg_peak = float(s[np.argmax(cn2)])
    cn2_peak = float(np.max(cn2))
    lg_eff_kpc = None
    if da_kpc is not None and np.isfinite(da_kpc) and da_kpc > 0:
        geom_weight = s * (1.0 - s / da_kpc)
        numer = np.trapezoid(cn2 * geom_weight, s)
        denom = np.trapezoid(cn2, s)
        if denom > 0 and np.isfinite(numer):
            lg_eff_kpc = float(numer / denom)

    plot_cn2_profile(s, cn2, ldeg, bdeg, lg_peak, lg_eff_kpc, output=output)

    return lg_peak, cn2_peak, lg_eff_kpc


# ---------------------------------------------------------------------------
# NE2025 scattering / scintillation predictions
# ---------------------------------------------------------------------------


def ne2025_scattering_prediction(
    l_deg: float,
    b_deg: float,
    ds_kpc: float,
    nu_ref_mhz: float = 1000.0,
    v_iss_km_s: float = 100.0,
    lg_eff_kpc: float | None = None,
    lg_max_dist_kpc: float = 50.0,
) -> dict:
    """Predict DM, SM, tau_scatt, Delta nu_d, and t_scint from NE2025.

    Calls mwprop's dmdsm_d2dm directly (no file output, no Cn2-profile
    re-integration) to get SM, SMtau, and model DM for the line of sight.

    Only the Milky Way (NE2025) screen is integrated: the model distance is
    capped at min(D_s, lg_max_dist_kpc) -- mwprop's own integration maximum is
    dmax_ne2001p_integrate = 50 kpc -- so for extragalactic sources this
    predicts the Galactic-screen contribution rather than integrating the
    Galactic model out to the source.

    Then the calibrated Cordes & Lazio relations
      tau_scatt = 1000 * (SMtau / 292)^1.2 * D_s * nu^-4.4        ms
      Delta nu_d = 1e-3 * 1.16 / (2 pi tau_scatt)                 MHz
      t_scint = 3.3 * nu^1.2 * SMtau^-0.6 * (100 / V_ISS)         s
    matching mwprop's own scattering_functions2020.py outputs.
    """
    from astropy import units as u

    if ds_kpc is None or not np.isfinite(ds_kpc) or ds_kpc <= 0:
        raise ValueError("ne2025_scattering_prediction requires ds_kpc > 0 (kpc)")

    import mwprop.nemod.dmdsm as _dmdsm
    from mwprop.nemod.config_nemod import ds_coarse as _ds_coarse
    from mwprop.nemod.config_nemod import ds_fine as _ds_fine
    from mwprop.scattering_functions import scattering_functions2020 as _sf

    if lg_max_dist_kpc is None or not np.isfinite(lg_max_dist_kpc) or lg_max_dist_kpc <= 0:
        lg_max_dist_kpc = 50.0
    model_dist_kpc = float(min(ds_kpc, float(lg_max_dist_kpc)))

    # d -> DM/SM/SMtau directly from mwprop, MW screen only
    # (do_analysis=False => no files written)
    out = _dmdsm.dmdsm_d2dm(
        np.deg2rad(float(l_deg)), np.deg2rad(float(b_deg)), model_dist_kpc,
        ds_coarse=_ds_coarse, ds_fine=_ds_fine, Nsmin=10,
        d2dm_only=False, do_analysis=False, plotting=False,
    )
    limit, dhat_kpc, dm_pc_cm3, sm_kpc, smtau_kpc, smtheta_kpc, smiso_kpc = out

    nu_ghz = nu_ref_mhz * 1e-3
    tau_scatt_ms = float(_sf.tauiss(dhat_kpc, smtau_kpc, nu_ghz))
    delta_nu_d_mhz = float(_sf.scintbw(dhat_kpc, smtau_kpc, nu_ghz))
    t_scint_s = float(_sf.scintime(smtau_kpc, nu_ghz, v_iss_km_s))

    theta_broadening_mas = float(_sf.theta_xgal(smtheta_kpc, nu_ghz))  # FWHM, mas
    fwhm2sigma = 2.0 / np.sqrt(8.0 * np.log(2.0))  # 2*sigma = FWHM * this factor
    theta_2sigma_mas = theta_broadening_mas * fwhm2sigma

    DMW_kpc = float(dhat_kpc)
    if lg_eff_kpc is not None and np.isfinite(lg_eff_kpc) and lg_eff_kpc > 0:
        DMW_kpc = float(lg_eff_kpc)
    LMW_kpc = 4.0 * theta_2sigma_mas * _sf.mas * DMW_kpc
    LMW_au = float((LMW_kpc * u.kpc).to_value(u.AU))

    def _sm_si(sm_kpc_val):
        if sm_kpc_val is None or not np.isfinite(sm_kpc_val):
            return None
        return float((sm_kpc_val * u.kpc / u.m ** (20.0 / 3.0)).to_value(u.m ** (-17.0 / 3.0)))

    return dict(
        SM_kpc=float(sm_kpc),
        SM_si=_sm_si(sm_kpc),
        smtau_kpc=float(smtau_kpc),
        smtau_si=_sm_si(smtau_kpc),
        smtheta_kpc=float(smtheta_kpc),
        smtheta_si=_sm_si(smtheta_kpc),
        smiso_kpc=float(smiso_kpc),
        smiso_si=_sm_si(smiso_kpc),
        dm_pc_cm3=float(dm_pc_cm3),
        dhat_kpc=float(dhat_kpc),
        model_dist_kpc=model_dist_kpc,
        lg_eff_kpc=lg_eff_kpc,
        tau_scatt_ms=tau_scatt_ms,
        delta_nu_d_mhz=delta_nu_d_mhz,
        t_scint_s=t_scint_s,
        theta_broadening_mas=theta_broadening_mas,
        theta_2sigma_mas=theta_2sigma_mas,
        LMW_kpc=LMW_kpc,
        LMW_au=LMW_au,
        DMW_kpc=DMW_kpc,
        r_diff_m=t_scint_s * v_iss_km_s * 1e3,
        r_diff_au=float((t_scint_s * u.s * v_iss_km_s * u.km / u.s).to_value(u.AU)),
        nu_ref_mhz=float(nu_ref_mhz),
        v_iss_km_s=float(v_iss_km_s),
    )


def print_ne2025_scattering_prediction(pred: dict, lg_peak_kpc: float | None, ds_kpc: float) -> None:
    print("\n  NE2025 predicted scattering (Galactic screen):")
    print(f"    Reference frequency    = {pred['nu_ref_mhz']:.3f} MHz")
    print(f"    DM (model)             = {pred['dm_pc_cm3']:.4f} pc cm^{{-3}}")
    print(f"    SM (integral)          = {pred['SM_kpc']:.4e} kpc m^{{-20/3}}")
    print(f"    SM (SI)                = {pred['SM_si']:.4e} m^{{-17/3}}")
    if pred.get('smtau_kpc') is not None and np.isfinite(pred['smtau_kpc']):
        print(f"    SMtau (pulse-broad.)  = {pred['smtau_kpc']:.4e} kpc m^{{-20/3}}")
    if pred.get('smtau_si') is not None and np.isfinite(pred['smtau_si']):
        print(f"    SMtau (SI)            = {pred['smtau_si']:.4e} m^{{-17/3}}")
    if pred.get('smtheta_kpc') is not None and np.isfinite(pred['smtheta_kpc']):
        print(f"    SMtheta (angular)     = {pred['smtheta_kpc']:.4e} kpc m^{{-20/3}}")
    if pred.get('smtheta_si') is not None and np.isfinite(pred['smtheta_si']):
        print(f"    SMtheta (SI)          = {pred['smtheta_si']:.4e} m^{{-17/3}}")
    if pred.get('smiso_kpc') is not None and np.isfinite(pred['smiso_kpc']):
        print(f"    SMiso (isoplanatic)   = {pred['smiso_kpc']:.4e} kpc m^{{-20/3}}")
    if pred.get('smiso_si') is not None and np.isfinite(pred['smiso_si']):
        print(f"    SMiso (SI)            = {pred['smiso_si']:.4e} m^{{-17/3}}")
    if pred.get('dhat_kpc') is not None and np.isfinite(pred['dhat_kpc']):
        print(f"    dhat (integ. dist)    = {pred['dhat_kpc']:.4e} kpc")
    if lg_peak_kpc is not None:
        print(f"    L_g (peak)             = {lg_peak_kpc:.4f} kpc")
    lg_eff = pred.get('lg_eff_kpc')
    if lg_eff is not None and np.isfinite(lg_eff):
        print(f"    L_g (weighted)         = {lg_eff:.4f} kpc")
    print(f"    D_s                    = {ds_kpc:.4e} kpc")
    d_model = pred.get('model_dist_kpc')
    if d_model is not None and d_model < ds_kpc - 1e-12:
        print(f"    Screen depth (model)   = {d_model:.4e} kpc (capped at MW model max)")
    print(f"    tau_scatt (predicted)  = {pred['tau_scatt_ms']:.4e} ms")
    print(f"    Delta nu_d (predicted) = {pred['delta_nu_d_mhz']:.4e} MHz")
    if pred.get('theta_broadening_mas') is not None and np.isfinite(pred['theta_broadening_mas']):
        print(f"    theta_broadening (FWHM) = {pred['theta_broadening_mas']:.4e} mas")
        print(f"    theta_broadening (2sig) = {pred['theta_2sigma_mas']:.4e} mas")
    if pred.get('LMW_kpc') is not None and np.isfinite(pred['LMW_kpc']):
        print(f"    L_MW (screen)          = {pred['LMW_kpc']:.4e} kpc  ({pred['LMW_au']:.4e} AU)  (D_MW = {pred['DMW_kpc']:.4f} kpc)")
    if np.isfinite(pred['t_scint_s']):
        print(
            f"    t_scint (predicted)    = {pred['t_scint_s']:.4e} s  "
            f"(V_ISS = {pred['v_iss_km_s']:.0f} km/s assumed)"
        )
        print(f"    r_diff (predicted)     = {pred['r_diff_m']:.4e} m  ({pred['r_diff_au']:.4e} AU)")
    else:
        print("    t_scint (predicted)    = N/A")
