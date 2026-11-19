"""SHRINE-II: the structure-maximising DM of S23 (Sutinjo et al. 2023) with the 1-sigma
uncertainties of Balzan et al. (in prep.), and SHRINE's S/N-maximising DM.

Modules, in the order the paper uses them:
    structure   Sp = ||D1 S x||, the coefficients w, k_c          (Sections 2.2, 3.1, 3.3.1; App. 1)
    noise       sigma_t^2 and sigma_S                             (Sections 3.3.2, 3.1)
    fit         the fit window and the parabola                   (Sections 3.3.3, 3.3.4)
    covariance  V_jl between trial DMs                            (Appendix 2)
    estimate    all of Section 3.3 in one call: dm_uncertainty()
    profiles    I(d_j, t) from CRAFT voltages, as SHRINE's generate_profiles.py, and the channel
                delays
    snr         S/N maximisation, as SHRINE's maximise_sn.py: the standalone alternative for very
                low S/N bursts
    summary     the summary files written by the scripts
    plots       the diagnostic plots
"""
from .estimate import check_trial_dms, dm_uncertainty
from .structure import choose_kc, coefficients, structure_parameter

__all__ = [
    "dm_uncertainty",
    "check_trial_dms",
    "choose_kc",
    "structure_parameter",
    "coefficients",
]
