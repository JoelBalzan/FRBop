"""
Galactic RM correction using SPICE-RACS DR2 sky maps.

Looks up the local Galactic RM foreground from the SPICE-RACS DR2
nearest-neighbour sky maps and subtracts it from an observed RM before
converting to the FRB host-galaxy rest frame:

	RM_host = (RM_obs - RM_galactic) * (1 + z)**2

The same (1+z)**2 scaling is applied to sigma_RM, while the scattering
timescale is converted as:

	tau_host = tau_obs / (1 + z)**3

A high SPICE-RACS neighbour scatter does NOT cause a source to be discarded.
Instead, the Galactic RM correction falls back to the Hutschenreuter et al.
(2022) all-sky reconstruction. The original SPICE-RACS neighbour scatter is
retained as metadata so the quality of the foreground estimate remains
visible in the final table.

Expected sky map files:

	spice-racs.dr2.nn_rm_wmean_nn.fits
	spice-racs.dr2.nn_rm_wmean_err_nn.fits
	spice-racs.dr2.nn_rm_mad_std_nn.fits

placed in ./skymaps/ next to this script, or supplied through --skymap-dir.

Swap '_nn' for '_idw' throughout if the inverse-distance-weighted maps are
preferred.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits


DEFAULT_SKYMAP_DIR = Path(__file__).parent / "skymaps"

RM_MAP_NAME = "spice-racs.dr2.nn_rm_wmean_nn.fits"
RM_ERR_MAP_NAME = "spice-racs.dr2.nn_rm_wmean_err_nn.fits"
RM_SCATTER_MAP_NAME = "spice-racs.dr2.nn_rm_mad_std_nn.fits"

DEFAULT_SCATTER_WARN_THRESHOLD = 20.0


@dataclass
class GalacticRMResult:
	"""Result of a single Galactic-RM lookup and host correction."""

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
	rm_galactic_source: str


class SpiceRacsSkymaps:
	"""Loads and queries the SPICE-RACS DR2 nearest-neighbour maps."""

	def __init__(
		self,
		skymap_dir: Path = DEFAULT_SKYMAP_DIR,
	):
		self.skymap_dir = Path(skymap_dir)

		(
			self._rm_map,
			self._nside,
			self._nest,
			self._coordsys,
		) = self._load(RM_MAP_NAME)

		self._rm_err_map, _, _, _ = self._load(
			RM_ERR_MAP_NAME
		)

		self._rm_scatter_map, _, _, _ = self._load(
			RM_SCATTER_MAP_NAME
		)

		# All maps must describe the same HEALPix grid.
		for filename, data in (
			(RM_ERR_MAP_NAME, self._rm_err_map),
			(RM_SCATTER_MAP_NAME, self._rm_scatter_map),
		):
			nside = hp.get_nside(data)
			if nside != self._nside:
				raise ValueError(
					f"{filename}: NSIDE={nside} does not match "
					f"{RM_MAP_NAME}: NSIDE={self._nside}."
				)

	def _load(self, filename: str):
		path = self.skymap_dir / filename

		if not path.exists():
			raise FileNotFoundError(
				f"Sky map not found: {path}\n"
				"Download the SPICE-RACS DR2 skymaps and place them "
				"in the supplied skymap directory."
			)

		header = fits.getheader(path, 1)

		ordering = str(
			header.get("ORDERING", "")
		).upper()

		coordsys = str(
			header.get("COORDSYS", "")
		).upper()

		if ordering not in ("RING", "NESTED"):
			raise ValueError(
				f"{path.name}: could not determine HEALPix ordering "
				f"from header (ORDERING={ordering!r}). Inspect the "
				"file header manually rather than assuming RING."
			)

		nest = ordering == "NESTED"

		healpix_map = hp.read_map(
			str(path),
			nest=nest,
			verbose=False,
		)

		nside = hp.get_nside(healpix_map)

		return (
			healpix_map,
			nside,
			nest,
			coordsys,
		)

	def _to_healpix_frame(
		self,
		ra_deg: float,
		dec_deg: float,
	):
		"""
		Convert ICRS coordinates into the coordinate system used by the map.
		"""
		coord = SkyCoord(
			ra=ra_deg * u.deg,
			dec=dec_deg * u.deg,
			frame="icrs",
		)

		coordsys = (self._coordsys or "").upper()

		if coordsys.startswith("G"):
			gal = coord.galactic
			lon_deg = gal.l.deg
			lat_deg = gal.b.deg

		elif coordsys.startswith("C") or coordsys.startswith("E"):
			lon_deg = coord.ra.deg
			lat_deg = coord.dec.deg

		elif coordsys == "":
			raise ValueError(
				"SPICE-RACS sky map has no COORDSYS header. "
				"Inspect the downloaded file and explicitly verify "
				"its coordinate system before using it."
			)

		else:
			raise ValueError(
				f"Unrecognised COORDSYS={coordsys!r} in sky map header; "
				"inspect the file manually."
			)

		theta = np.radians(90.0 - lat_deg)
		phi = np.radians(lon_deg)

		return theta, phi

	def query(
		self,
		ra_deg: float,
		dec_deg: float,
	):
		"""
		Return:

			(rm_galactic, rm_galactic_err, rm_galactic_scatter)

		at the supplied ICRS position.

		Raises ValueError for masked/unseen pixels.
		"""
		theta, phi = self._to_healpix_frame(
			ra_deg,
			dec_deg,
		)

		pix = hp.ang2pix(
			self._nside,
			theta,
			phi,
			nest=self._nest,
		)

		rm = self._rm_map[pix]
		rm_err = self._rm_err_map[pix]
		rm_scatter = self._rm_scatter_map[pix]

		for label, val in (
			("nn_rm_wmean", rm),
			("nn_rm_wmean_err", rm_err),
			("nn_rm_mad_std", rm_scatter),
		):
			if (
				val == hp.UNSEEN
				or not np.isfinite(val)
			):
				raise ValueError(
					f"No SPICE-RACS DR2 coverage ({label} is "
					f"UNSEEN/NaN) at RA={ra_deg}, Dec={dec_deg}."
				)

		return (
			float(rm),
			float(rm_err),
			float(rm_scatter),
		)

	def query_galactic_rm(
		self,
		ra_deg: float,
		dec_deg: float,
		scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
	):
		"""
		Query the Galactic RM foreground.

		SPICE-RACS is used when its pixel is usable and the local neighbour
		scatter is below the supplied threshold.

		If there is no usable SPICE-RACS coverage, or the local scatter is
		above the threshold, Hutschenreuter et al. (2022) is used instead.

		Returns:
			rm_galactic,
			rm_galactic_err,
			rm_galactic_scatter,
			source
		"""
		try:
			rm, rm_err, rm_scatter = self.query(
				ra_deg,
				dec_deg,
			)
		except ValueError as exc:
			print(
				f"SPICE-RACS DR2 has no usable coverage at "
				f"RA={ra_deg:.6f}, Dec={dec_deg:.6f}; "
				"falling back to Hutschenreuter et al. (2022)."
			)

			try:
				rm, rm_err = query_hutschenreuter2022(
					ra_deg,
					dec_deg,
				)
			except Exception as fallback_exc:
				raise RuntimeError(
					"Both SPICE-RACS and the Hutschenreuter et al. "
					"fallback failed. "
					f"SPICE-RACS error: {exc}; "
					f"Hutschenreuter error: {fallback_exc}"
				) from fallback_exc

			if rm is None or rm_err is None:
				raise RuntimeError(
					"Hutschenreuter et al. (2022) returned no usable "
					"Galactic RM for this position."
				)

			return (
				rm,
				rm_err,
				np.nan,
				"Hutschenreuter 2022",
			)

		if (
			np.isfinite(rm_scatter)
			and rm_scatter > scatter_warn_threshold
		):
			print(
				f"SPICE-RACS DR2 has high Galactic RM scatter "
				f"(nn_rm_mad_std={rm_scatter:.1f} rad/m^2 > "
				f"{scatter_warn_threshold:.1f} rad/m^2 threshold) "
				"-- foreground subtraction is not reliable here; "
				"falling back to Hutschenreuter et al. (2022)."
			)

			try:
				fallback_rm, fallback_err = (
					query_hutschenreuter2022(
						ra_deg,
						dec_deg,
					)
				)
			except Exception as exc:
				raise RuntimeError(
					"Hutschenreuter et al. (2022) fallback failed "
					"after a high SPICE-RACS neighbour-scatter flag."
				) from exc

			if (
				fallback_rm is None
				or fallback_err is None
			):
				raise RuntimeError(
					"Hutschenreuter et al. (2022) returned no usable "
					"Galactic RM after the high-scatter SPICE-RACS flag."
				)

			return (
				fallback_rm,
				fallback_err,
				rm_scatter,
				"Hutschenreuter 2022",
			)

		return (
			rm,
			rm_err,
			rm_scatter,
			"SPICE-RACS DR2",
		)


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
	Subtract the Galactic RM foreground and convert to the host frame.

	RM_host = (RM_obs - RM_galactic) * (1 + z)^2

	The observed RM uncertainty and Galactic foreground uncertainty are
	combined in quadrature before applying the same redshift scaling.

	The neighbour scatter is deliberately NOT included in the formal
	uncertainty. It is returned separately as a quality indicator.
	"""
	(
		rm_gal,
		rm_gal_err,
		rm_gal_scatter,
		rm_gal_source,
	) = skymaps.query_galactic_rm(
		ra_deg,
		dec_deg,
		scatter_warn_threshold,
	)

	zp1 = 1.0 + z

	rm_host = (
		rm_obs - rm_gal
	) * zp1**2

	rm_host_err = None

	if rm_obs_err is not None:
		rm_host_err = (
			np.sqrt(
				rm_obs_err**2
				+ rm_gal_err**2
			)
			* zp1**2
		)

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
		high_scatter_flag=(
			np.isfinite(rm_gal_scatter)
			and rm_gal_scatter > scatter_warn_threshold
		),
		rm_galactic_source=rm_gal_source,
	)


def correct_sample(
	sources: List[Dict],
	skymaps: SpiceRacsSkymaps,
	scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
) -> List[GalacticRMResult]:
	"""
	Apply correct_to_host_frame across a list of source dictionaries.
	"""
	results: List[GalacticRMResult] = []

	for src in sources:
		missing = [
			k
			for k in ("ra", "dec", "z", "rm")
			if src.get(k) is None
		]

		if missing:
			print(
				f"  skipping {src.get('name', '?')}: "
				f"missing {missing}"
			)
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
		except Exception as exc:
			print(
				f"  skipping {src.get('name', '?')}: {exc}"
			)
			continue

		results.append(result)

	return results


# ---------------------------------------------------------------------------
# Dict-schema correction helpers for rm_tau_correlation.py
# ---------------------------------------------------------------------------


def host_frame_correct_source(
	src: Dict,
	skymaps: "SpiceRacsSkymaps",
	scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
) -> Optional[Dict]:
	"""
	Return a copy of src with RM, sigma_RM and tau converted to the host frame.

	Observer-frame quantities are preserved as *_obs.

	Corrections:

		RM_host = (RM_obs - RM_galactic) * (1 + z)^2

		sigma_RM_host = sigma_RM_obs * (1 + z)^2

		sigma_RM_upper,host = sigma_RM_upper,obs * (1 + z)^2

		tau_host = tau_obs / (1 + z)^3

	A high SPICE-RACS neighbour scatter does not reject the source.
	query_galactic_rm() falls back to Hutschenreuter et al. (2022) and
	preserves the original SPICE-RACS scatter as metadata.
	"""
	name = src.get("name", "?")

	ra = src.get("ra")
	dec = src.get("dec")
	z = src.get("z")
	rm_obs = src.get("rm")

	missing = [
		k
		for k, value in (
			("ra", ra),
			("dec", dec),
			("z", z),
			("rm", rm_obs),
		)
		if value is None
	]

	if missing:
		print(
			f"  [host-frame] skipping {name}: "
			f"missing {missing}"
		)
		return None

	try:
		(
			rm_gal,
			rm_gal_err,
			rm_gal_scatter,
			rm_gal_source,
		) = skymaps.query_galactic_rm(
			ra,
			dec,
			scatter_warn_threshold,
		)
	except Exception as exc:
		print(
			f"  [host-frame] skipping {name}: {exc}"
		)
		return None

	out = dict(src)

	# Preserve every observer-frame quantity before replacement.
	for key in (
		"rm",
		"rm_err",
		"sigma_rm",
		"sigma_rm_err",
		"sigma_rm_upper",
		"tau",
		"tau_err",
	):
		if key in src:
			out[f"{key}_obs"] = src[key]

	zp1 = 1.0 + z

	# ------------------------------------------------------------------
	# RM
	# ------------------------------------------------------------------

	out["rm"] = (
		rm_obs - rm_gal
	) * zp1**2

	rm_err = src.get("rm_err")

	if rm_err is not None:
		out["rm_err"] = (
			np.sqrt(
				rm_err**2
				+ rm_gal_err**2
			)
			* zp1**2
		)
	else:
		out["rm_err"] = None

	# Galactic foreground metadata.
	out["rm_galactic"] = rm_gal
	out["rm_galactic_err"] = rm_gal_err
	out["rm_galactic_source"] = rm_gal_source
	out["rm_galactic_scatter"] = rm_gal_scatter

	out["high_scatter_flag"] = (
		np.isfinite(rm_gal_scatter)
		and rm_gal_scatter > scatter_warn_threshold
	)

	# ------------------------------------------------------------------
	# sigma_RM
	# ------------------------------------------------------------------

	sigma_rm_obs = src.get("sigma_rm")

	if sigma_rm_obs is not None:
		out["sigma_rm"] = (
			sigma_rm_obs * zp1**2
		)

		sigma_rm_err = src.get("sigma_rm_err")

		if sigma_rm_err is not None:
			out["sigma_rm_err"] = (
				sigma_rm_err * zp1**2
			)
		else:
			out["sigma_rm_err"] = None

	else:
		# Preserve None for a non-detection.
		out["sigma_rm"] = None
		out["sigma_rm_err"] = None

	# IMPORTANT BUG FIX:
	# sigma_RM upper limits also need the (1+z)^2 scaling when the source
	# has no sigma_RM detection. The previous implementation only scaled
	# sigma_rm_upper inside the sigma_rm_obs != None branch.
	sigma_rm_upper = src.get("sigma_rm_upper")

	if sigma_rm_upper is not None:
		out["sigma_rm_upper"] = (
			sigma_rm_upper * zp1**2
		)
	else:
		out["sigma_rm_upper"] = None

	# ------------------------------------------------------------------
	# Scattering timescale
	# ------------------------------------------------------------------

	tau_obs = src.get("tau")

	if tau_obs is not None:
		is_upper = tau_obs < 0
		sign = -1.0 if is_upper else 1.0

		out["tau"] = (
			sign
			* abs(tau_obs)
			/ zp1**3
		)

		tau_err = src.get("tau_err")

		if tau_err is not None:
			out["tau_err"] = (
				tau_err / zp1**3
			)
		else:
			out["tau_err"] = None

	else:
		out["tau"] = None
		out["tau_err"] = None

	return out


def host_frame_correct_sample(
	sample: List[Dict],
	skymaps: Optional["SpiceRacsSkymaps"] = None,
	skymap_dir: Path = DEFAULT_SKYMAP_DIR,
	scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
) -> List[Dict]:
	"""
	Apply host_frame_correct_source to every source with the required data.

	Sources without RA, Dec, redshift or RM are dropped with a printed
	explanation rather than raising.
	"""
	if skymaps is None:
		skymaps = SpiceRacsSkymaps(
			skymap_dir=skymap_dir
		)

	corrected = []

	for src in sample:
		result = host_frame_correct_source(
			src,
			skymaps,
			scatter_warn_threshold,
		)

		if result is not None:
			corrected.append(result)

	return corrected


def query_hutschenreuter2022(
	ra_deg: float,
	dec_deg: float,
):
	"""
	Query the Hutschenreuter et al. (2022) Galactic Faraday-rotation map.

	Returns
	-------
	rm_galactic : float
		Posterior-mean Galactic RM [rad m^-2].

	rm_galactic_err : float
		Posterior standard deviation [rad m^-2].
	"""
	try:
		from frb import rm
	except ImportError as exc:
		raise ImportError(
			"The 'frb' package is required for the "
			"Hutschenreuter et al. (2022) Galactic-RM fallback."
		) from exc

	coord = SkyCoord(
		ra=ra_deg * u.deg,
		dec=dec_deg * u.deg,
		frame="icrs",
	)

	rm_gal, rm_gal_err = rm.galactic_rm(
		coord,
		use_map=2020,
	)

	return (
		float(
			rm_gal.to_value(
				u.rad / u.m**2
			)
		),
		float(
			rm_gal_err.to_value(
				u.rad / u.m**2
			)
		),
	)


def _print_results(
	results: List[GalacticRMResult],
) -> None:
	"""Print compact single-source host-frame correction results."""
	header = (
		f"{'name':<18} "
		f"{'RM_obs':>10} "
		f"{'RM_gal':>10} "
		f"{'scatter':>9} "
		f"{'RM_host':>16} "
		f"{'source':>22} "
		f"{'flag':>6}"
	)

	print(header)
	print("-" * len(header))

	for r in results:
		flag = "HIGH" if r.high_scatter_flag else ""

		rm_host_str = f"{r.rm_host:.2f}"

		if r.rm_host_err is not None:
			rm_host_str += f" +/- {r.rm_host_err:.2f}"

		print(
			f"{r.name:<18} "
			f"{r.rm_obs:>10.2f} "
			f"{r.rm_galactic:>10.2f} "
			f"{r.rm_galactic_scatter:>9.2f} "
			f"{rm_host_str:>16} "
			f"{r.rm_galactic_source:>22} "
			f"{flag:>6}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Subtract the SPICE-RACS DR2 Galactic RM foreground "
			"from an observed RM and convert to host frame."
		)
	)

	parser.add_argument(
		"--ra",
		type=float,
		help="RA in degrees (ICRS)",
	)

	parser.add_argument(
		"--dec",
		type=float,
		help="Dec in degrees (ICRS)",
	)

	parser.add_argument(
		"--z",
		type=float,
		help="Redshift",
	)

	parser.add_argument(
		"--rm",
		type=float,
		help="Observed RM (rad/m^2)",
	)

	parser.add_argument(
		"--rm-err",
		type=float,
		default=None,
		help="Observed RM uncertainty (rad/m^2)",
	)

	parser.add_argument(
		"--name",
		default="source",
		help="Label for output",
	)

	parser.add_argument(
		"--skymap-dir",
		type=Path,
		default=DEFAULT_SKYMAP_DIR,
		help=(
			"Directory containing the SPICE-RACS DR2 skymap FITS "
			f"files (default: {DEFAULT_SKYMAP_DIR})"
		),
	)

	parser.add_argument(
		"--scatter-warn-threshold",
		type=float,
		default=DEFAULT_SCATTER_WARN_THRESHOLD,
		help=(
			"nn_rm_mad_std above which the SPICE-RACS Galactic RM "
			"estimate triggers the Hutschenreuter fallback"
		),
	)

	args = parser.parse_args()

	if None in (
		args.ra,
		args.dec,
		args.z,
		args.rm,
	):
		parser.error(
			"--ra, --dec, --z, and --rm are all required "
			"for a single-source query."
		)

	skymaps = SpiceRacsSkymaps(
		skymap_dir=args.skymap_dir
	)

	result = correct_to_host_frame(
		name=args.name,
		ra_deg=args.ra,
		dec_deg=args.dec,
		z=args.z,
		rm_obs=args.rm,
		rm_obs_err=args.rm_err,
		skymaps=skymaps,
		scatter_warn_threshold=args.scatter_warn_threshold,
	)

	_print_results([result])

	if result.high_scatter_flag:
		print(
			f"\nWARNING: nn_rm_mad_std = "
			f"{result.rm_galactic_scatter:.1f} rad/m^2 exceeds "
			f"the {args.scatter_warn_threshold} rad/m^2 threshold "
			"at this position."
		)
		print(
			"The Galactic RM value used for the correction was obtained "
			"from the Hutschenreuter et al. (2022) fallback."
		)


if __name__ == "__main__":
	main()