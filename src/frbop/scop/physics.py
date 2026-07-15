"""Astrophysical helpers for scintillation calculations."""

from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.cosmology import WMAP5, Planck18


def scale_scintillation_bandwidth(delta_nu_d_mhz, nu_from_mhz, nu_to_mhz, alpha=4.0):
    """Scale Delta nu_d proportional to nu^alpha (default alpha=4)."""
    if nu_from_mhz <= 0 or nu_to_mhz <= 0:
        raise ValueError("Frequencies must be > 0 for Delta nu_d scaling")
    return float(delta_nu_d_mhz) * (float(nu_to_mhz) / float(nu_from_mhz)) ** float(alpha)


def estimate_ds_kpc_from_redshift(z):
    return Planck18.angular_diameter_distance(z).to(u.kpc).value


def radec_to_galactic_deg(ra_hms, dec_dms):
    c = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    print(f"\n  RA={ra_hms}, Dec={dec_dms} -> l={c.galactic.l.deg:.4f} deg, b={c.galactic.b.deg:.4f} deg")
    return float(c.galactic.l.deg), float(c.galactic.b.deg)
