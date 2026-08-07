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
    kpc_to_m = 3.085677581e19  # 1 kpc in metres

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
    _limit, _dhat, dm_pc_cm3, sm_kpc, smtau_kpc, _smtheta_kpc, _smiso_kpc = out

    nu_ghz = nu_ref_mhz * 1e-3
    tau_scatt_ms = float(_sf.tauiss(_dhat, smtau_kpc, nu_ghz))
    delta_nu_d_mhz = float(_sf.scintbw(_dhat, smtau_kpc, nu_ghz))
    t_scint_s = float(_sf.scintime(smtau_kpc, nu_ghz, v_iss_km_s))

    return dict(
        SM_kpc=float(sm_kpc),
        SM_si=float(sm_kpc) * kpc_to_m,
        smtau_kpc=float(smtau_kpc),
        dm_pc_cm3=float(dm_pc_cm3),
        lg_eff_kpc=lg_eff_kpc,
        model_dist_kpc=model_dist_kpc,
        tau_scatt_ms=tau_scatt_ms,
        delta_nu_d_mhz=delta_nu_d_mhz,
        t_scint_s=t_scint_s,
        r_diff_m=t_scint_s * v_iss_km_s * 1e3,
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
    if np.isfinite(pred['t_scint_s']):
        print(
            f"    t_scint (predicted)    = {pred['t_scint_s']:.4e} s  "
            f"(V_ISS = {pred['v_iss_km_s']:.0f} km/s assumed)"
        )
    else:
        print("    t_scint (predicted)    = N/A")
