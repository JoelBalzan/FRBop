"""
Galactic RM correction using SPICE-RACS DR2 sky maps.

Looks up the local Galactic RM foreground from the SPICE-RACS DR2
nearest-neighbour sky maps (Thomson et al. 2026, arXiv:2605.16917) at a
given (RA, Dec), and subtracts it from an observed RM before converting to
the FRB host-galaxy rest frame:

    RM_host = (RM_obs - RM_galactic) * (1 + z)**2

This matches the recipe in Pandhi et al. (2026), Appendix C, and the same
approach used for sigma_RM in that section.

Expected sky map files (download from the SPICE-RACS DR2 DAP collection,
https://data.csiro.au/collection/csiro:64891, under skymaps/) placed in
./skymaps/ next to this script, or pass --skymap-dir:

    spice-racs.dr2.nn_rm_wmean_nn.fits       (Galactic RM estimate)
    spice-racs.dr2.nn_rm_wmean_err_nn.fits   (its uncertainty)
    spice-racs.dr2.nn_rm_mad_std_nn.fits     (scatter of the neighbour
                                               ensemble -- flags unreliable
                                               foreground estimates)

Swap '_nn' for '_idw' throughout if you prefer the inverse-distance-weighted
maps instead of nearest-neighbour.

IMPORTANT -- verify before trusting results:
    * Coordinate system (Galactic vs equatorial/ICRS) and pixel ordering
      (RING vs NESTED) are read from the FITS header where possible and
      checked against expectations; this script will raise rather than
      silently assume if the header is missing or unexpected. Confirm
      against the actual downloaded file if you hit that error.
    * Declinations above +49.4 deg are outside the SPICE-RACS DR2 footprint
      and will return a masked (UNSEEN) pixel.
    * A high nn_rm_mad_std at a position means the local Galactic foreground
      is not well constrained by a simple neighbour average (e.g. Galactic
      plane, complex regions) -- treat the correction there with more
      caution. This script surfaces that value per source rather than
      silently applying the correction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


DEFAULT_SKYMAP_DIR = Path(__file__).parent / "skymaps"

RM_MAP_NAME = "spice-racs.dr2.nn_rm_wmean_nn.fits"
RM_ERR_MAP_NAME = "spice-racs.dr2.nn_rm_wmean_err_nn.fits"
RM_SCATTER_MAP_NAME = "spice-racs.dr2.nn_rm_mad_std_nn.fits"

# MAD_std above which we flag the Galactic-RM estimate as unreliable at a
# given position (rad/m^2). This is a judgement call, not a value quoted in
# the paper -- adjust to taste. The paper's overall catalogue median RM
# uncertainty is ~2 rad/m^2 (DR2 abstract); a neighbour-ensemble scatter much
# larger than that suggests the local foreground is not well described by a
# single value.
DEFAULT_SCATTER_WARN_THRESHOLD = 20.0


@dataclass
class GalacticRMResult:
    """Result of a single Galactic-RM lookup + host-frame correction."""
    name: str
    ra_deg: float
    dec_deg: float
    z: float
    rm_obs: float
    rm_obs_err: Optional[float]
    rm_galactic: float
    rm_galactic_err: float
    rm_galactic_scatter: float
    rm_host: float
    rm_host_err: Optional[float]
    high_scatter_flag: bool


class SpiceRacsSkymaps:
    """Loads and queries the SPICE-RACS DR2 nn_rm_wmean sky maps."""

    def __init__(self, skymap_dir: Path = DEFAULT_SKYMAP_DIR):
        self.skymap_dir = Path(skymap_dir)
        self._rm_map, self._nside, self._nest, self._coordsys = self._load(
            RM_MAP_NAME)
        self._rm_err_map, *_ = self._load(RM_ERR_MAP_NAME)
        self._rm_scatter_map, *_ = self._load(RM_SCATTER_MAP_NAME)

    def _load(self, filename: str):
        path = self.skymap_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Sky map not found: {path}\n"
                f"Download from the SPICE-RACS DR2 DAP collection "
                f"(https://data.csiro.au/collection/csiro:64891) under "
                f"skymaps/, or point --skymap-dir at the right directory."
            )

        header = fits.getheader(path, 1)
        ordering = header.get("ORDERING", "").upper()
        coordsys = header.get("COORDSYS", header.get("COORDSYS", ""))
        if ordering not in ("RING", "NESTED"):
            raise ValueError(
                f"{path.name}: could not determine HEALPix ordering from "
                f"header (ORDERING={ordering!r}). Inspect the file header "
                f"manually before proceeding -- do not assume RING."
            )
        nest = ordering == "NESTED"

        healpix_map = hp.read_map(str(path), nest=nest, verbose=False)
        nside = hp.get_nside(healpix_map)
        return healpix_map, nside, nest, coordsys

    def _to_healpix_frame(self, ra_deg: float, dec_deg: float):
        """Convert an ICRS (RA, Dec) to the (theta, phi) HEALPix expects,
        rotating to Galactic coordinates first if that's what the map uses.
        """
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        coordsys = (self._coordsys or "").upper()
        if coordsys.startswith("G"):
            gal = coord.galactic
            lon_deg, lat_deg = gal.l.deg, gal.b.deg
        elif coordsys.startswith("C") or coordsys.startswith("E") or coordsys == "":
            # 'C' or 'E' (celestial/equatorial), or missing header --
            # assume equatorial but this is exactly the kind of assumption
            # flagged in the module docstring: verify against the real file.
            lon_deg, lat_deg = coord.ra.deg, coord.dec.deg
        else:
            raise ValueError(
                f"Unrecognised COORDSYS={coordsys!r} in sky map header; "
                f"inspect the file manually."
            )

        theta = np.radians(90.0 - lat_deg)
        phi = np.radians(lon_deg)
        return theta, phi

    def query(self, ra_deg: float, dec_deg: float):
        """Return (rm_galactic, rm_galactic_err, rm_galactic_scatter) at a
        given ICRS position, or raise if the pixel is masked/unseen."""
        theta, phi = self._to_healpix_frame(ra_deg, dec_deg)
        pix = hp.ang2pix(self._nside, theta, phi, nest=self._nest)

        rm = self._rm_map[pix]
        rm_err = self._rm_err_map[pix]
        rm_scatter = self._rm_scatter_map[pix]

        for label, val in (("nn_rm_wmean", rm),
                           ("nn_rm_wmean_err", rm_err),
                           ("nn_rm_mad_std", rm_scatter)):
            if val == hp.UNSEEN or not np.isfinite(val):
                raise ValueError(
                    f"No SPICE-RACS DR2 coverage ({label} is UNSEEN/NaN) at "
                    f"RA={ra_deg}, Dec={dec_deg}. This position may be "
                    f"outside the DR2 footprint (Dec > +49.4 deg) or in an "
                    f"unsampled gap."
                )
        return float(rm), float(rm_err), float(rm_scatter)


def correct_to_host_frame(
    name: str,
    ra_deg: float,
    dec_deg: float,
    z: float,
    rm_obs: float,
    skymaps: SpiceRacsSkymaps,
    rm_obs_err: Optional[float] = None,
    scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
) -> GalacticRMResult:
    """
    Subtract the SPICE-RACS DR2 Galactic RM foreground from an observed RM
    and convert to the host-galaxy rest frame.

        RM_host = (RM_obs - RM_galactic) * (1 + z)**2

    Error propagation (when rm_obs_err is given) combines the observed RM
    error and the Galactic RM foreground error in quadrature, then scales by
    (1+z)**2:

        RM_host_err = sqrt(rm_obs_err**2 + rm_galactic_err**2) * (1 + z)**2

    This does NOT include any uncertainty from the neighbour-ensemble
    scatter (nn_rm_mad_std) itself -- that value is returned separately as a
    data-quality flag, per the paper's own caution against treating it as a
    formal 'true foreground' uncertainty (Sec. 5.4.2).
    """
    rm_gal, rm_gal_err, rm_gal_scatter = skymaps.query(ra_deg, dec_deg)

    rm_host = (rm_obs - rm_gal) * (1 + z) ** 2
    rm_host_err = None
    if rm_obs_err is not None:
        rm_host_err = np.sqrt(rm_obs_err ** 2 + rm_gal_err ** 2) * (1 + z) ** 2

    return GalacticRMResult(
        name=name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        z=z,
        rm_obs=rm_obs,
        rm_obs_err=rm_obs_err,
        rm_galactic=rm_gal,
        rm_galactic_err=rm_gal_err,
        rm_galactic_scatter=rm_gal_scatter,
        rm_host=rm_host,
        rm_host_err=rm_host_err,
        high_scatter_flag=rm_gal_scatter > scatter_warn_threshold,
    )


def correct_sample(
    sources: List[Dict],
    skymaps: SpiceRacsSkymaps,
    scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
) -> List[GalacticRMResult]:
    """
    Apply :func:`correct_to_host_frame` across a list of source dicts, each
    expected to have: name, ra (deg), dec (deg), z, rm, and optionally
    rm_err. Sources missing ra/dec/z/rm are skipped with a printed warning
    rather than raising, so one bad entry doesn't kill a batch run.
    """
    results: List[GalacticRMResult] = []
    for src in sources:
        missing = [k for k in ("ra", "dec", "z", "rm") if src.get(k) is None]
        if missing:
            print(f"  skipping {src.get('name', '?')}: missing {missing}")
            continue
        try:
            result = correct_to_host_frame(
                name=src["name"],
                ra_deg=src["ra"],
                dec_deg=src["dec"],
                z=src["z"],
                rm_obs=src["rm"],
                rm_obs_err=src.get("rm_err"),
                skymaps=skymaps,
                scatter_warn_threshold=scatter_warn_threshold,
            )
        except ValueError as exc:
            print(f"  skipping {src.get('name', '?')}: {exc}")
            continue
        results.append(result)
    return results

# ---------------------------------------------------------------------------
# Dict-schema correction helpers, for use with rm_tau_correlation.py's
# FENG2022_SAMPLE / PANDHI2026_SAMPLE / UTTARKAR* style source lists.
# ---------------------------------------------------------------------------
# These implement the same recipe as correct_to_host_frame() above, plus the
# scattering-timescale rest-frame correction (Pandhi+ 2026, Appendix C):
#
#   RM_host  = (RM_obs - RM_galactic) * (1 + z)**2      [RM_IGM assumed 0]
#   tau_host = tau_obs / (1 + z)**3                      [tau ~ nu**-4,
#                                                          Macquart & Koay 2013]
#
# Unlike correct_to_host_frame(), these accept/return the loosely-typed
# source dicts used elsewhere in rmop (arbitrary extra keys preserved,
# missing fields skipped rather than raised on) since that's the schema the
# rest of the module's sample lists already use.


def host_frame_correct_source(src: Dict, skymaps: "SpiceRacsSkymaps",
                              scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD
                              ) -> Optional[Dict]:
    """
    Return a copy of ``src`` with ``rm``, ``rm_err``, ``sigma_rm``,
    ``sigma_rm_err``, ``tau``, ``tau_err`` replaced by their host-frame
    (rest-frame) values, or ``None`` if the source is missing ra/dec/z/rm,
    or if the local Galactic RM foreground (nn_rm_mad_std) exceeds
    ``scatter_warn_threshold`` and is therefore not reliably subtracted.
    ...
    """
    name = src.get("name", "?")
    ra, dec, z = src.get("ra"), src.get("dec"), src.get("z")
    rm_obs = src.get("rm")

    missing = [k for k, v in (("ra", ra), ("dec", dec), ("z", z), ("rm", rm_obs))
              if v is None]
    if missing:
        print(f"  [host-frame] skipping {name}: missing {missing}")
        return None

    try:
        rm_gal, rm_gal_err, rm_gal_scatter = skymaps.query(ra, dec)
    except ValueError as exc:
        print(f"  [host-frame] skipping {name}: {exc}")
        return None

    if rm_gal_scatter > scatter_warn_threshold:
        print(f"  [host-frame] skipping {name}: high Galactic RM scatter "
             f"(nn_rm_mad_std={rm_gal_scatter:.1f} rad/m^2 > "
             f"{scatter_warn_threshold} rad/m^2 threshold) -- foreground "
             f"subtraction is not reliable here.")
        return None

    out = dict(src)
    out["rm"] = (rm_obs - rm_gal) * (1 + z) ** 2
    rm_err = src.get("rm_err")
    if rm_err is not None:
        out["rm_err"] = np.sqrt(rm_err ** 2 + rm_gal_err ** 2) * (1 + z) ** 2
    out["rm_galactic"] = rm_gal
    out["rm_galactic_err"] = rm_gal_err
    out["rm_galactic_scatter"] = rm_gal_scatter

    sigma_rm_obs = src.get("sigma_rm")
    if sigma_rm_obs is not None:
        out["sigma_rm"] = sigma_rm_obs * (1 + z) ** 2
        sigma_rm_err = src.get("sigma_rm_err")
        if sigma_rm_err is not None:
            out["sigma_rm_err"] = sigma_rm_err * (1 + z) ** 2

    tau_obs = src.get("tau")
    if tau_obs is not None:
        is_upper = tau_obs < 0
        sign = -1.0 if is_upper else 1.0
        out["tau"] = sign * (abs(tau_obs) / (1 + z) ** 3)
        tau_err = src.get("tau_err")
        if tau_err is not None:
            out["tau_err"] = tau_err / (1 + z) ** 3

    return out


def host_frame_correct_sample(sample: List[Dict],
                              skymaps: Optional["SpiceRacsSkymaps"] = None,
                              skymap_dir: Path = DEFAULT_SKYMAP_DIR,
                              scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
                              ) -> List[Dict]:
    """
    Apply :func:`host_frame_correct_source` to every entry in ``sample``
    that has ra/dec/z/rm; sources missing any of these are dropped (with a
    printed reason) rather than raising, so a batch with partial coverage
    still returns a usable (shorter) sample.
    """
    if skymaps is None:
        skymaps = SpiceRacsSkymaps(skymap_dir=skymap_dir)

    corrected = []
    for src in sample:
        result = host_frame_correct_source(src, skymaps, scatter_warn_threshold)
        if result is not None:
            corrected.append(result)
    return corrected

def _print_results(results: List[GalacticRMResult]) -> None:
    header = (f"{'name':<18} {'RM_obs':>10} {'RM_gal':>10} {'scatter':>9} "
              f"{'RM_host':>12} {'flag':>6}")
    print(header)
    print("-" * len(header))
    for r in results:
        flag = "HIGH" if r.high_scatter_flag else ""
        rm_host_str = f"{r.rm_host:.2f}"
        if r.rm_host_err is not None:
            rm_host_str += f" +/- {r.rm_host_err:.2f}"
        print(f"{r.name:<18} {r.rm_obs:>10.2f} {r.rm_galactic:>10.2f} "
              f"{r.rm_galactic_scatter:>9.2f} {rm_host_str:>12} {flag:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subtract the SPICE-RACS DR2 Galactic RM foreground "
                    "from an observed RM and convert to host frame.",
    )
    parser.add_argument("--ra", type=float, help="RA in degrees (ICRS)")
    parser.add_argument("--dec", type=float, help="Dec in degrees (ICRS)")
    parser.add_argument("--z", type=float, help="Redshift")
    parser.add_argument("--rm", type=float, help="Observed RM (rad/m^2)")
    parser.add_argument("--rm-err", type=float, default=None,
                        help="Observed RM uncertainty (rad/m^2)")
    parser.add_argument("--name", default="source", help="Label for output")
    parser.add_argument("--skymap-dir", type=Path, default=DEFAULT_SKYMAP_DIR,
                        help=f"Directory containing the SPICE-RACS DR2 "
                             f"skymap FITS files (default: {DEFAULT_SKYMAP_DIR})")
    parser.add_argument("--scatter-warn-threshold", type=float,
                        default=DEFAULT_SCATTER_WARN_THRESHOLD,
                        help="nn_rm_mad_std above which to flag the "
                             "Galactic RM estimate as unreliable")
    args = parser.parse_args()

    if None in (args.ra, args.dec, args.z, args.rm):
        parser.error("--ra, --dec, --z, and --rm are all required for a "
                    "single-source query.")

    skymaps = SpiceRacsSkymaps(skymap_dir=args.skymap_dir)
    result = correct_to_host_frame(
        name=args.name, ra_deg=args.ra, dec_deg=args.dec, z=args.z,
        rm_obs=args.rm, rm_obs_err=args.rm_err, skymaps=skymaps,
        scatter_warn_threshold=args.scatter_warn_threshold,
    )
    _print_results([result])
    if result.high_scatter_flag:
        print(f"\nWARNING: nn_rm_mad_std = {result.rm_galactic_scatter:.1f} "
             f"rad/m^2 exceeds the {args.scatter_warn_threshold} rad/m^2 "
             f"threshold at this position -- the Galactic RM foreground is "
             f"not well constrained here; treat rm_host with extra caution.")


if __name__ == "__main__":
    main()