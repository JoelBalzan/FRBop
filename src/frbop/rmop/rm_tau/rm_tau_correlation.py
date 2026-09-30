"""
sigma_RM / |RM| vs tau correlation plots.

Implements the Feng+ (2022) comparison sample of FRBs with measured
rotation-measure dispersion (sigma_RM) and scattering timescale (tau), adds an
optional burst of your own, and over-plots two linear (power-law) fits:

	* one to the Feng+ (2022) sample only
	* one to the combined Feng + your data set

Fits are performed in log-log space. For the sigma_RM panel the fit follows
the convention of Feng+ (2022), who regress tau on sigma_RM and report
tau ~ sigma_RM^0.81: we fit ln(tau) = a + b*ln(sigma_RM). The FRB 121102
upper limit is plotted but excluded from the regression. The |RM| panel uses
a direct regression of |RM| on sigma_RM. Slopes / intercepts (with
uncertainties) are reported and annotated on the figure.

Reference samples combined into the fit/plot: Feng+ (2022), Uttarkar+
(2024/2026), and Pandhi+ (2026) [FRB 20220529A].

The plotting follows the FRBop publication conventions: pub_figsize,
set_pub_style, IBM_PALETTE and the _savefig wrapper used across the rmop
module.

The terminal output includes a complete observer-frame table showing the
input values and frequency-rescaled tau values. The host-frame wrapper
additionally prints a complete table containing the Galactic foreground
correction and all rest-frame quantities.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit

from frbop.utils.plotting import IBM_PALETTE, pub_figsize

from ..constants import plot_style
from ..plotting import _savefig
from .galactic_rm import (
	DEFAULT_SCATTER_WARN_THRESHOLD,
	DEFAULT_SKYMAP_DIR,
	SpiceRacsSkymaps,
	host_frame_correct_sample,
	host_frame_correct_source,
)


def _radec(
	ra_str: str,
	dec_str: Optional[str] = None,
) -> Dict[str, float]:
	"""
	Parse coordinates into {'ra': deg, 'dec': deg}.

	Accepts either:
	  - a single combined string, e.g. "01h16m24.99s +20d37m56.55s"
	  - separate RA/Dec strings in any of astropy's sexagesimal formats.
	"""
	if dec_str is None:
		c = SkyCoord(
			ra_str,
			frame="icrs",
			unit=(u.hourangle, u.deg),
		)
	else:
		c = SkyCoord(
			ra=ra_str,
			dec=dec_str,
			frame="icrs",
			unit=(u.hourangle, u.deg),
		)

	return {"ra": c.ra.deg, "dec": c.dec.deg}


# ---------------------------------------------------------------------------
# Reference samples
# ---------------------------------------------------------------------------

# Pandhi+ (2026) — FRB 20220529A
PANDHI2026_SAMPLE = [
	{
		"name": "FRB 20220529A",
		"z": 0.1839,
		"nu_centre": 600,
		"rm": -4.4,
		"rm_err": 0.4,
		"sigma_rm": 2.50,
		"sigma_rm_err": 0.02,
		"tau": 0.43,
		"tau_err": 0.01,
		**_radec("01h16m24.99s +20d37m56.55s"),
	},
]


# name           RM      sigma_RM   sigma_RM_err  tau      tau_err
# FRB 20121102A  81542   30.9       0.4          <0.43     -
# FRB 20180301A    546    6.3       0.4          -         -
# FRB 20180916B   -115    0.12      0.01          0.009     -
# FRB 20190303A   -411    3.6       0.1           0.19      0.04
# FRB 20190417A   4681    6.1       0.5           0.21      0.06
# FRB 20190520B   2759  218.9      10.2           9.8       2.0
# FRB 20201124A   -684    2.5       0.1           0.59      -
#
# A negative tau marks an upper limit. None means no tau measurement.

FENG2022_SAMPLE = [
	{
		"name": "FRB 20121102A",
		"z": 0.19273,
		"nu_centre": 1300,
		"rm": 81542.0,
		"rm_err": None,
		"sigma_rm": 30.9,
		"sigma_rm_err": 0.4,
		"tau": -0.43,
		"tau_err": None,
		**_radec("05h32m09s +33d05m13s"),
	},
	{
		"name": "FRB 20180301A",
		"z": 0.3304,
		"nu_centre": 1300,
		"rm": 546.0,
		"rm_err": None,
		"sigma_rm": 6.3,
		"sigma_rm_err": 0.4,
		"tau": None,
		"tau_err": None,
		**_radec("06h12m43.4s +04d33m44.8s"),
	},
	{
		"name": "FRB 20180916B",
		"z": 0.0337,
		"nu_centre": 1300,
		"rm": -115.0,
		"rm_err": None,
		"sigma_rm": 0.12,
		"sigma_rm_err": 0.01,
		"tau": 0.009,
		"tau_err": None,
		**_radec("01h56m40.8s +65d44m24s"),
	},
	{
		"name": "FRB 20190303A",
		"z": 0.0640,
		"nu_centre": 1300,
		"rm": -411.0,
		"rm_err": None,
		"sigma_rm": 3.6,
		"sigma_rm_err": 0.1,
		"tau": 0.19,
		"tau_err": 0.04,
		**_radec("13:52:19.200 +48:15:00.00"),
	},
	{
		"name": "FRB 20190417A",
		"z": 0.12817,
		"nu_centre": 1300,
		"rm": 4681.0,
		"rm_err": None,
		"sigma_rm": 6.1,
		"sigma_rm_err": 0.5,
		"tau": 0.21,
		"tau_err": 0.06,
		**_radec("19:38:52.800 +59:24:36.00"),
	},
	{
		"name": "FRB 20190520B",
		"z": 0.241,
		"nu_centre": 1300,
		"rm": 2759.0,
		"rm_err": None,
		"sigma_rm": 218.9,
		"sigma_rm_err": 10.2,
		"tau": 9.8,
		"tau_err": 2.0,
		**_radec("16:02:04.270 -11:17:17.32"),
	},
	{
		"name": "FRB 20201124A",
		"z": 0.098,
		"nu_centre": 1300,
		"rm": -684.0,
		"rm_err": None,
		"sigma_rm": 2.5,
		"sigma_rm_err": 0.1,
		"tau": 0.59,
		"tau_err": None,
		**_radec("05:07:57.600 +26:11:24.00"),
	},
]


UTTARKAR2024_SAMPLE = [
	{
		"name": "FRB 20180924B",
		"nu_centre": 1271,
		"rm": 22,
		"rm_err": 2,
		"z": 0.3214,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.17,
		"sigma_rm_prime_upper": 4.01,
		"tau": 0.91,
		"tau_err_low": 0.07,
		"tau_err_high": 0.06,
		**_radec("21:44:25.255 -40:54:00.10"),
	},
	{
		"name": "FRB 20190102C",
		"nu_centre": 1271,
		"rm": -105,
		"rm_err": 1,
		"z": 0.29,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.52,
		"sigma_rm_prime_upper": 4.39,
		"tau": 0.84,
		"tau_err_low": 0.03,
		"tau_err_high": 0.05,
		**_radec("21:29:39.760 -79:28:32.50"),
	},
	{
		"name": "FRB 20190608B",
		"nu_centre": 1271,
		"rm": 353,
		"rm_err": 2,
		"z": 0.1178,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.42,
		"sigma_rm_prime_upper": 4.81,
		"tau": 0.93,
		"tau_err_low": 0.06,
		"tau_err_high": 0.05,
		**_radec("22:16:04.770 -07:53:53.70"),
	},
	{
		"name": "FRB 20190611B",
		"nu_centre": 1271,
		"rm": 20,
		"rm_err": 4,
		"z": 0.378,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 6.41,
		"sigma_rm_prime_upper": 5.48,
		"tau": 0.85,
		"tau_err_low": 0.04,
		"tau_err_high": 0.06,
		**_radec("21:22:58.940 -79:23:51.30"),
	},
	{
		"name": "FRB 20190711A",
		"nu_centre": 1271,
		"rm": 9,
		"rm_err": 2,
		"z": 0.522,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 8.64,
		"sigma_rm_prime_upper": 8.93,
		"tau": 0.85,
		"tau_err_low": 0.20,
		"tau_err_high": 0.10,
		**_radec("21:57:40.62 -80:21:28.80"),
	},
	{
		"name": "FRB 20191001A",
		"nu_centre": 1271,
		"rm": 55.5,
		"rm_err": 9,
		"z": 0.23,
		"sigma_rm": 4.1,
		"sigma_rm_err_low": 0.10,
		"sigma_rm_err_high": 0.09,
		"sigma_rm_upper": None,
		"sigma_rm_prime_upper": 2.8,
		"tau": 0.6,
		"tau_err_low": 0.03,
		"tau_err_high": 0.06,
		**_radec("21:33:24.313 -54:44:51.86"),
	},
]


UTTARKAR2026_SAMPLE = [
	{
		"name": "FRB 20191228A",
		**_radec("22:57:43.327", "-29:35:38.82"),
		"nu_centre": 1272,
		"z": 0.2432,
		"rm": 11.3,
		"rm_err": 0.8,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.17,
		"sigma_rm_prime_upper": 3.93,
		"tau": 5.69,
		"tau_err_low": 0.14,
		"tau_err_high": 0.15,
	},
	{
		"name": "FRB 20200430A",
		**_radec("15:18:49.549", "+12:22:34.784"),
		"nu_centre": 864,
		"z": 0.161,
		"rm": 195.3,
		"rm_err": 0.6,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 4.41,
		"sigma_rm_prime_upper": 3.51,
		"tau": 7.27,
		"tau_err_low": 0.21,
		"tau_err_high": 0.20,
	},
	{
		"name": "FRB 20210117A",
		**_radec("22:39:55.007", "-16:09:05.203"),
		"nu_centre": 1272,
		"z": 0.214,
		"rm": -45.4,
		"rm_err": 0.7,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 3.05,
		"sigma_rm_prime_upper": 2.25,
		"tau": 0.09,
		"tau_err_low": 0.06,
		"tau_err_high": 0.07,
	},
	{
		"name": "FRB 20210320C",
		**_radec("13:37:50.104", "-16:07:21.611"),
		"nu_centre": 864,
		"z": 0.28,
		"rm": 288.8,
		"rm_err": 0.2,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": None,
		"sigma_rm_prime_upper": None,
		"tau": 0.38,
		"tau_err_low": 0.18,
		"tau_err_high": 0.20,
	},
	{
		"name": "FRB 20210407E",
		**_radec("05:14:36.230", "+27:03:29.73"),
		"nu_centre": 1272,
		"z": None,
		"rm": -8.9,
		"rm_err": 0.5,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 2.53,
		"sigma_rm_prime_upper": 1.98,
		"tau": 0.27,
		"tau_err_low": 0.15,
		"tau_err_high": 0.04,
	},
	{
		"name": "FRB 20210912A",
		**_radec("23:23:10.441", "-30:24:20.089"),
		"nu_centre": 1272,
		"z": None,
		"rm": 5.7,
		"rm_err": 0.4,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.06,
		"sigma_rm_prime_upper": 4.30,
		"tau": 0.45,
		"tau_err_low": 0.02,
		"tau_err_high": 0.02,
	},
	{
		"name": "FRB 20220501C",
		**_radec("23:29:30.987", "-32:29:26.63"),
		"nu_centre": 864,
		"z": 0.381,
		"rm": 35.2,
		"rm_err": 0.4,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 2.07,
		"sigma_rm_prime_upper": 1.38,
		"tau": None,
		"tau_err_low": None,
		"tau_err_high": None,
	},
	{
		"name": "FRB 20220610A",
		**_radec("23:24:17.583", "-33:30:49.859"),
		"nu_centre": 1272,
		"z": 1.015,
		"rm": 217.0,
		"rm_err": 2.0,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 1.70,
		"sigma_rm_prime_upper": 1.19,
		"tau": 0.54,
		"tau_err_low": 0.05,
		"tau_err_high": 0.03,
	},
	{
		"name": "FRB 20220725A",
		**_radec("23:33:15.649", "-35:59:24.92"),
		"nu_centre": 920,
		"z": 0.1926,
		"rm": -26.0,
		"rm_err": 2.0,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 4.13,
		"sigma_rm_prime_upper": 3.22,
		"tau": 2.40,
		"tau_err_low": 0.12,
		"tau_err_high": 0.12,
	},
	{
		"name": "FRB 20221106A",
		**_radec("03:46:49.154", "-25:34:11.316"),
		"nu_centre": 1632,
		"z": 0.204,
		"rm": -445.0,
		"rm_err": 1.0,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 10.04,
		"sigma_rm_prime_upper": 8.38,
		"tau": None,
		"tau_err_low": None,
		"tau_err_high": None,
	},
	{
		"name": "FRB 20230526A",
		**_radec("01:28:55.828", "-52:43:02.39"),
		"nu_centre": 1272,
		"z": 0.157,
		"rm": 613.0,
		"rm_err": 2.0,
		"sigma_rm": 12.62,
		"sigma_rm_err_low": 0.23,
		"sigma_rm_err_high": 0.25,
		"sigma_rm_upper": None,
		"sigma_rm_prime": 11.23,
		"sigma_rm_prime_err_low": 1.04,
		"sigma_rm_prime_err_high": 0.83,
		"tau": 1.39,
		"tau_err_low": 0.04,
		"tau_err_high": 0.04,
	},
	{
		"name": "FRB 20230718A",
		**_radec("08:32:38.863", "-40:27:06.955"),
		"nu_centre": 1272,
		"z": 0.035,
		"rm": 243.0,
		"rm_err": 1.0,
		"sigma_rm": None,
		"sigma_rm_err": None,
		"sigma_rm_upper": 5.91,
		"sigma_rm_prime_upper": 5.38,
		"tau": None,
		"tau_err_low": None,
		"tau_err_high": None,
	},
]


# Balzan et al. (2026) — FRB 20250607A
BALZAN2026_SAMPLE = [
	{
		"name": "FRB 20250607A",
		**_radec("02:32:43.44488639", "-39:20:27.9387337"),
		"z": 0.2106,
		"nu_centre": 919.5,
		"rm": -428.2,
		"rm_err": 0.14,
		"sigma_rm": 3.556142,
		"sigma_rm_err": 0.033343,
		"tau": 1.25,
		"tau_err": 0.02,
	},
]


def _find_in_sample(name: str, *samples: List[Dict]) -> Optional[Dict]:
	"""Return the first entry matching name across one or more samples."""
	for sample in samples:
		for src in sample:
			if src["name"] == name:
				return src
	return None


def apply_named_defaults(
	args: argparse.Namespace,
	parser: argparse.ArgumentParser,
) -> None:
	"""
	Fill CLI defaults from a known source when --tau-name is used.
	Explicit CLI flags always win.
	"""
	src = _find_in_sample(
		args.tau_name,
		BALZAN2026_SAMPLE,
		PANDHI2026_SAMPLE,
	)

	if src is None:
		return

	field_map = {
		"ra": "ra",
		"dec": "dec",
		"z": "z",
		"rm": "rm",
		"tau": "tau",
		"tau_err": "tau_err",
	}

	for arg_field, src_field in field_map.items():
		if getattr(args, arg_field, None) == parser.get_default(arg_field):
			val = src.get(src_field)
			if val is not None:
				setattr(args, arg_field, val)

	if (
		src.get("nu_centre") is not None
		and args.tau_freq == parser.get_default("tau_freq")
	):
		args.tau_freq = src["nu_centre"]


def _asym(lo, hi):
	"""Return a symmetric error from asymmetric low/high errors."""
	if lo is not None and hi is not None:
		return (float(lo) + float(hi)) / 2.0
	if lo is not None:
		return float(lo)
	if hi is not None:
		return float(hi)
	return None


def _append_simple_group(
	out: List[Dict],
	group: str,
	rlist: List[Dict],
) -> None:
	"""Append a sample using symmetric errors."""
	for r in rlist:
		out.append(
			{
				"name": r["name"],
				"group": group,
				"nu_centre": r.get("nu_centre", 1300.0),
				"sigma_rm": r.get("sigma_rm"),
				"sigma_rm_err": r.get("sigma_rm_err"),
				"sigma_rm_upper": None,
				"rm": r.get("rm"),
				"rm_err": r.get("rm_err"),
				"tau": r.get("tau"),
				"tau_err": r.get("tau_err"),
				"ra": r.get("ra"),
				"dec": r.get("dec"),
				"z": r.get("z"),
			}
		)


def _unified_sources() -> List[Dict]:
	"""Flatten all reference samples into a common schema."""
	out: List[Dict] = []

	_append_simple_group(out, "feng", FENG2022_SAMPLE)
	_append_simple_group(out, "pandhi2026", PANDHI2026_SAMPLE)

	for grp, rlist in (
		("uttarkar2024", UTTARKAR2024_SAMPLE),
		("uttarkar2026", UTTARKAR2026_SAMPLE),
	):
		for r in rlist:
			sig = r.get("sigma_rm")

			sig_err = (
				_asym(
					r.get("sigma_rm_err_low"),
					r.get("sigma_rm_err_high"),
				)
				if sig is not None
				else None
			)

			tau = r.get("tau")

			out.append(
				{
					"name": r["name"],
					"group": grp,
					"nu_centre": r.get("nu_centre", 1300.0),
					"sigma_rm": sig,
					"sigma_rm_err": sig_err,
					"sigma_rm_upper": (
						r.get("sigma_rm_upper")
						if sig is None
						else None
					),
					"rm": r.get("rm"),
					"rm_err": r.get("rm_err"),
					"tau": tau,
					"tau_err": (
						_asym(
							r.get("tau_err_low"),
							r.get("tau_err_high"),
						)
						if tau is not None
						else None
					),
					"ra": r.get("ra"),
					"dec": r.get("dec"),
					"z": r.get("z"),
				}
			)

	return out


def feng2022_sample() -> List[Dict]:
	"""Return the unified reference sample."""
	return _unified_sources()


def _extract(sample: List[Dict]) -> Tuple[np.ndarray, ...]:
	"""Split a sample into parallel arrays of raw values."""
	names = [r["name"] for r in sample]
	group = np.array([r.get("group", "feng") for r in sample])

	rm = np.array(
		[r["rm"] if r["rm"] is not None else np.nan for r in sample],
		dtype=float,
	)

	rm_err = np.array(
		[r["rm_err"] if r.get("rm_err") is not None else np.nan for r in sample],
		dtype=float,
	)

	sigma_rm = np.array(
		[
			r["sigma_rm"] if r["sigma_rm"] is not None else np.nan
			for r in sample
		],
		dtype=float,
	)

	sigma_rm_err = np.array(
		[
			r["sigma_rm_err"]
			if r["sigma_rm_err"] is not None
			else np.nan
			for r in sample
		],
		dtype=float,
	)

	sigma_rm_upper = np.array(
		[
			r["sigma_rm_upper"]
			if r["sigma_rm_upper"] is not None
			else np.nan
			for r in sample
		],
		dtype=float,
	)

	has_rm = np.array(
		[r["rm"] is not None for r in sample],
		dtype=bool,
	)

	tau = np.array(
		[r["tau"] if r["tau"] is not None else np.nan for r in sample],
		dtype=float,
	)

	tau_err = np.array(
		[
			r["tau_err"] if r["tau_err"] is not None else np.nan
			for r in sample
		],
		dtype=float,
	)

	nu_centre = np.array(
		[r.get("nu_centre", 1300.0) for r in sample],
		dtype=float,
	)

	return (
		names,
		group,
		rm,
		rm_err,
		sigma_rm,
		sigma_rm_err,
		sigma_rm_upper,
		has_rm,
		tau,
		tau_err,
		nu_centre,
	)


def _tau_arrays(
	tau: np.ndarray,
	tau_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Resolve negative tau values into positive upper limits."""
	upper = tau < 0
	t = np.where(upper, -tau, tau)
	t_err = np.where(upper, np.nan, tau_err)
	return t, t_err, upper


def _linear(
	logx: np.ndarray,
	a: float,
	b: float,
) -> np.ndarray:
	"""Linear model in log space: ln(y) = a + b*ln(x)."""
	return a + b * logx


def _loglog_fit(
	x: np.ndarray,
	y: np.ndarray,
	yerr: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
	"""
	Fit ln(y) = a + b*ln(x).

	When y errors are available and positive, they are propagated into
	fractional log-space errors and used as weights.
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)

	finite = (
		np.isfinite(x)
		& (x > 0)
		& np.isfinite(y)
		& (y > 0)
	)

	if np.nansum(finite) < 2:
		return None

	logx = np.log(x[finite])
	logy = np.log(y[finite])

	sigma = None

	if yerr is not None:
		yerr = np.asarray(yerr, dtype=float)[finite]
		frac = yerr / y[finite]
		ok = np.isfinite(frac) & (frac > 0)

		if np.any(ok):
			sigma = np.ones_like(logy)
			sigma[ok] = frac[ok]
			sigma[~ok] = 10.0

	try:
		if sigma is not None:
			popt, pcov = curve_fit(
				_linear,
				logx,
				logy,
				p0=[0.0, 1.0],
				sigma=sigma,
				absolute_sigma=False,
				maxfev=20000,
			)
		else:
			popt, pcov = curve_fit(
				_linear,
				logx,
				logy,
				p0=[0.0, 1.0],
				maxfev=20000,
			)
	except Exception:
		return None

	perr = np.sqrt(np.diag(pcov))

	return {
		"a": float(popt[0]),
		"b": float(popt[1]),
		"a_err": float(perr[0]),
		"b_err": float(perr[1]),
	}


def _logrange(
	vals: np.ndarray,
	pad: float = 0.3,
) -> Tuple[float, float]:
	"""Decade-padded log limits for positive values."""
	vals = np.asarray(vals, dtype=float)
	finite = vals[np.isfinite(vals) & (vals > 0)]

	if finite.size == 0:
		return 1e-3, 1e3

	return (
		float(np.nanmin(finite) * 10 ** (-pad)),
		float(np.nanmax(finite) * 10 ** pad),
	)


def _fit_summary(
	tag: str,
	fit: Optional[Dict[str, float]],
	lines: List[str],
	tau_first: bool = False,
	absrm_first: bool = False,
) -> None:
	if fit is None:
		lines.append(f"  {tag}: fit failed / insufficient points")
		return

	if np.isfinite(fit["b_err"]):
		if tau_first:
			lines.append(
				f"  {tag}: tau ~ sigma_RM^{fit['b']:.3f} "
				f"± {fit['b_err']:.3f} "
				f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})"
			)
		elif absrm_first:
			lines.append(
				f"  {tag}: |RM| ~ sigma_RM^{fit['b']:.3f} "
				f"± {fit['b_err']:.3f} "
				f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})"
			)
		else:
			lines.append(
				f"  {tag}: b = {fit['b']:.3f} ± {fit['b_err']:.3f} "
				f"(a = {fit['a']:.3f} ± {fit['a_err']:.3f})"
			)
	else:
		if tau_first:
			lines.append(
				f"  {tag}: tau ~ sigma_RM^{fit['b']:.3f} "
				f"(a = {fit['a']:.3f})"
			)
		elif absrm_first:
			lines.append(
				f"  {tag}: |RM| ~ sigma_RM^{fit['b']:.3f} "
				f"(a = {fit['a']:.3f})"
			)
		else:
			lines.append(
				f"  {tag}: b = {fit['b']:.3f} "
				f"(a = {fit['a']:.3f})"
			)


def _slope_text(
	fit: Optional[Dict[str, float]],
	tag: str,
	tau_first: bool,
	absrm_first: bool = False,
) -> Optional[str]:
	"""Format a slope annotation."""
	if fit is None:
		return None

	prefix = tag

	if tau_first:
		prefix += r"$\tau \propto \sigma_{\mathrm{RM}}^{"
	elif absrm_first:
		prefix += r"$|\mathrm{RM}| \propto \sigma_{\mathrm{RM}}^{"
	else:
		prefix += "$b = "

	if np.isfinite(fit["b_err"]):
		if tau_first or absrm_first:
			return (
				prefix
				+ f"{fit['b']:.2f} \\pm {fit['b_err']:.2f}"
				+ "}$"
			)
		return prefix + f"{fit['b']:.2f} \\pm {fit['b_err']:.2f}$"

	if tau_first or absrm_first:
		return prefix + f"{fit['b']:.2f}" + "}$"

	return prefix + f"{fit['b']:.2f}$"


def _annotate_slope(
	ax,
	text: Optional[str],
	color: str,
	yfrac: float,
	style: Dict,
) -> None:
	"""Add an annotation box to an axis."""
	if text is None:
		return

	ax.text(
		0.04,
		yfrac,
		text,
		transform=ax.transAxes,
		fontsize=9,
		color=color,
		va="top",
		bbox=dict(
			boxstyle="round,pad=0.25",
			fc="white",
			ec="none",
			alpha=0.75,
		),
	)


_GROUP_STYLE = (
	("feng", IBM_PALETTE[0], "Feng+ (2022)"),
	("uttarkar2024", IBM_PALETTE[1], "Uttarkar+ (2024)"),
	("uttarkar2026", IBM_PALETTE[3], "Uttarkar+ (2026)"),
	("pandhi2026", "#009E73", "Pandhi+ (2026)"),
)


def _fmt(
	x,
	digits: int = 2,
) -> str:
	"""Compact formatting for terminal tables."""
	if x is None:
		return "—"

	try:
		if not np.isfinite(x):
			return "—"
	except TypeError:
		return str(x)

	return f"{x:.{digits}f}"


def _fmt_signed_with_err(
	value,
	err=None,
	digits: int = 2,
) -> str:
	if value is None:
		return "—"

	if err is None or not np.isfinite(err):
		return _fmt(value, digits)

	return f"{value:.{digits}f} ± {err:.{digits}f}"


def _fmt_limit_or_value(
	value,
	err=None,
	digits: int = 2,
) -> str:
	"""Format a detection or upper limit."""
	if value is None:
		return "—"

	if value < 0:
		return f"<{abs(value):.{digits}f}"

	return _fmt_signed_with_err(value, err, digits)


def _fmt_tau(
	value,
	err=None,
) -> str:
	"""Format tau, preserving negative-value upper-limit convention."""
	if value is None:
		return "—"

	prefix = "<" if value < 0 else ""
	value = abs(value)

	if err is None or not np.isfinite(err):
		return f"{prefix}{value:.3f}"

	return f"{prefix}{value:.3f} ± {err:.3f}"


def _print_observer_frame_table(
	rows: List[Dict],
) -> None:
	"""Print the complete observer-frame sample and tau rescaling table."""
	columns = [
		("Source", 18),
		("Group", 14),
		("nu_c [MHz]", 11),
		("RM [rad m^-2]", 17),
		("sigma_RM", 17),
		("tau obs [ms]", 15),
		("tau used [ms]", 15),
	]

	header = "  ".join(
		f"{name:>{width}}" for name, width in columns
	)
	separator = "-" * len(header)

	print()
	print("=" * len(header))
	print("OBSERVER-FRAME SAMPLE / TAU FREQUENCY RESCALING")
	print("=" * len(header))
	print(header)
	print(separator)

	for src in rows:
		rm = _fmt_signed_with_err(
			src.get("rm_obs", src.get("rm")),
			src.get("rm_err_obs", src.get("rm_err")),
			2,
		)

		sigma = _fmt_limit_or_value(
			src.get("sigma_rm_obs", src.get("sigma_rm")),
			src.get("sigma_rm_err_obs", src.get("sigma_rm_err")),
			2,
		)

		tau_obs = _fmt_tau(
			src.get("tau_obs", src.get("tau")),
			src.get("tau_err_obs", src.get("tau_err")),
		)

		tau_used = _fmt_tau(
			src.get("tau_used"),
			src.get("tau_used_err"),
		)

		row = [
			f"{src.get('name', '?'):<18}",
			f"{src.get('group', 'user'):>14}",
			f"{_fmt(src.get('nu_centre'), 1):>11}",
			f"{rm:>17}",
			f"{sigma:>17}",
			f"{tau_obs:>15}",
			f"{tau_used:>15}",
		]

		print("  ".join(row))

	print(separator)
	print(
		"Reference tau values: tau_used = tau_obs * "
		"(1300 / nu_centre)^(-4)."
	)
	print(
		"Negative tau values denote upper limits; the displayed limit "
		"is the absolute value."
	)
	print()


def _print_host_frame_table(sample: List[Dict]) -> None:
	"""Print observer-frame and host-frame products for the corrected sample."""
	columns = [
		("Source", 18),
		("z", 6),
		("RM obs", 19),
		("RM Gal", 20),
		("RM host", 19),
		("sigmaRM obs", 17),
		("sigmaRM host", 17),
		("tau obs [ms]", 17),
		("tau host [ms]", 17),
		("Galactic RM source", 20),
		("scatter", 9),
	]

	header = "  ".join(
		f"{name:>{width}}" for name, width in columns
	)
	separator = "-" * len(header)

	print()
	print("=" * len(header))
	print("HOST-FRAME CORRECTED SAMPLE")
	print("=" * len(header))
	print(header)
	print(separator)

	for src in sample:
		rm_obs = _fmt_signed_with_err(
			src.get("rm_obs"),
			src.get("rm_err_obs"),
			2,
		)

		rm_gal = _fmt_signed_with_err(
			src.get("rm_galactic"),
			src.get("rm_galactic_err"),
			2,
		)

		rm_host = _fmt_signed_with_err(
			src.get("rm"),
			src.get("rm_err"),
			2,
		)

		sigma_obs = _fmt_limit_or_value(
			src.get("sigma_rm_obs"),
			src.get("sigma_rm_err_obs"),
			2,
		)

		sigma_host = _fmt_limit_or_value(
			src.get("sigma_rm"),
			src.get("sigma_rm_err"),
			2,
		)

		tau_obs = _fmt_tau(
			src.get("tau_obs"),
			src.get("tau_err_obs"),
		)

		tau_host = _fmt_tau(
			src.get("tau"),
			src.get("tau_err"),
		)

		row = [
			f"{src.get('name', '?'):<18}",
			f"{_fmt(src.get('z'), 3):>6}",
			f"{rm_obs:>19}",
			f"{rm_gal:>20}",
			f"{rm_host:>19}",
			f"{sigma_obs:>17}",
			f"{sigma_host:>17}",
			f"{tau_obs:>17}",
			f"{tau_host:>17}",
			f"{src.get('rm_galactic_source', '—'):>20}",
			f"{_fmt(src.get('rm_galactic_scatter'), 2):>9}",
		]

		print("  ".join(row))

	print(separator)
	print(
		"RM_host = (RM_obs − RM_Gal) × (1+z)^2    "
		"sigmaRM_host = sigmaRM_obs × (1+z)^2    "
		"tau_host = tau_obs / (1+z)^3"
	)
	print(
		"Negative tau values and sigma_RM values denote upper limits; "
		"displayed limits are absolute values."
	)
	print()


def plot_rm_tau_correlation(
	sigma_rm: float,
	sigma_rm_err: float,
	rm: float,
	tau: float,
	tau_err: Optional[float] = None,
    rm_err: Optional[float] = None,
	sample: Optional[List[Dict]] = None,
	output_file: str = "rm_tau_correlation.png",
	name: str = "This work",
	freq_mhz: Optional[float] = None,
	ref_freq_mhz: float = 1300.0,
	scattering_index: Optional[float] = None,
	scattering_index_err: Optional[float] = None,
	label_suffix: str = "",
	exclude_upper_limits: bool = False,
	print_table: bool = True,
) -> Dict:
	"""
	Plot tau vs sigma_RM (top) and |RM| vs sigma_RM (bottom).

	Reference tau values are rescaled from their central observing frequency
	to ref_freq_mhz using alpha = -4.

	The user's tau is rescaled from freq_mhz using scattering_index, defaulting
	to -4 when no index is supplied.
	"""
	if sample is None:
		sample = feng2022_sample()

	if tau <= 0:
		raise ValueError("tau must be > 0 (log-scale axes)")

	if sigma_rm <= 0:
		raise ValueError("sigma_rm must be > 0 (log-scale axes)")

	if rm == 0:
		raise ValueError("rm must be non-zero (|RM| axis is log-scale)")

	(
		names,
		group,
		rm_arr,
		rm_err_arr,
		sigma_rm_arr,
		sigma_rm_err_arr,
		sigma_rm_upper,
		has_rm,
		tau_raw,
		tau_err_raw,
		nu_centre,
	) = _extract(sample)

	# Make table rows before modifying the plotting arrays.
	table_rows = [dict(src) for src in sample]

	for row in table_rows:
		row["rm_obs"] = row.get("rm")
		row["rm_err_obs"] = row.get("rm_err")
		row["sigma_rm_obs"] = row.get("sigma_rm")
		row["sigma_rm_err_obs"] = row.get("sigma_rm_err")
		row["tau_obs"] = row.get("tau")
		row["tau_err_obs"] = row.get("tau_err")

	# Reference-sample frequency rescaling.
	ref_alpha = -4.0
	ref_scale = (ref_freq_mhz / nu_centre) ** ref_alpha

	tau_raw = np.where(
		np.isfinite(tau_raw),
		tau_raw * ref_scale,
		tau_raw,
	)
	tau_err_raw = tau_err_raw * ref_scale

	taus, taus_err, upper = _tau_arrays(
		tau_raw,
		tau_err_raw,
	)

	for row, used_tau, used_err in zip(
		table_rows,
		taus,
		taus_err,
	):
		row["tau_used"] = (
			float(used_tau) if np.isfinite(used_tau) else None
		)
		row["tau_used_err"] = (
			float(used_err) if np.isfinite(used_err) else None
		)

	# Rescale the user burst tau.
	user_tau_input = tau
	user_tau_err_input = tau_err

	if freq_mhz is not None and freq_mhz != ref_freq_mhz:
		if freq_mhz <= 0 or ref_freq_mhz <= 0:
			raise ValueError("frequencies must be > 0 (MHz)")

		alpha = (
			-4.0
			if scattering_index is None
			else scattering_index
		)

		scale = (ref_freq_mhz / freq_mhz) ** alpha

		tau = tau * scale

		if tau_err is not None and tau_err > 0:
			frac_tau = tau_err / user_tau_input
			frac_alpha = np.log(ref_freq_mhz / freq_mhz) * (
				scattering_index_err
				if scattering_index_err is not None
				else 0.0
			)

			tau_err = tau * np.sqrt(
				frac_tau**2 + frac_alpha**2
			)

	user_row = {
		"name": name,
		"group": "user",
		"nu_centre": ref_freq_mhz,
		"rm_obs": rm,
		"rm_err_obs": rm_err_arr[-1]
		if rm_err_arr.size
		else None,
		"rm": rm,
		"rm_err": None,
		"sigma_rm_obs": sigma_rm,
		"sigma_rm_err_obs": sigma_rm_err,
		"sigma_rm": sigma_rm,
		"sigma_rm_err": sigma_rm_err,
		"sigma_rm_upper": None,
		"tau_obs": user_tau_input,
		"tau_err_obs": user_tau_err_input,
		"tau_used": tau,
		"tau_used_err": tau_err,
	}

	# Append the user burst.
	names = names + [name]
	group = np.append(group, "user")
	rm_arr = np.append(rm_arr, rm)
	rm_err_arr = np.append(
		rm_err_arr,
		np.nan if rm_err is None else rm_err,
	)
	sigma_rm_arr = np.append(sigma_rm_arr, sigma_rm)
	sigma_rm_err_arr = np.append(sigma_rm_err_arr, sigma_rm_err)
	sigma_rm_upper = np.append(sigma_rm_upper, np.nan)
	has_rm = np.append(has_rm, True)

	taus = np.append(taus, tau)
	taus_err = np.append(
		taus_err,
		tau_err if tau_err is not None else np.nan,
	)
	upper = np.append(upper, False)

	tau_raw = np.append(tau_raw, tau)
	tau_err_raw = np.append(
		tau_err_raw,
		tau_err if tau_err is not None else np.nan,
	)
	nu_centre = np.append(nu_centre, ref_freq_mhz)

	table_rows.append(user_row)

	# User is always retained by the normal input validation.
	user_mask = np.zeros_like(taus, dtype=bool)
	if len(user_mask) > 0:
		user_mask[-1] = True

	abs_rm = np.abs(rm_arr)

	# ---- Fits -------------------------------------------------------------

	has_tau = np.isfinite(taus) & (taus > 0)
	sig_det = np.isfinite(sigma_rm_arr) & (sigma_rm_arr > 0)

	feng_g = group == "feng"

	# ---- Upper-limit exclusion (per panel) --------------------------------

	# The tau panel needs a sigma_RM detection and a tau detection; the
	# |RM| panel does not use tau at all, so a missing or limiting tau
	# must not remove a source from it. The exclusion is therefore
	# applied when plotting, not by discarding rows up front, so both
	# panels keep every source they can actually show.
	tau_usable = (
		sig_det
		& np.isfinite(tau_raw)
		& (tau_raw > 0)
	)

	drop_mask = {
		0: np.zeros(taus.shape, dtype=bool),
		1: np.zeros(taus.shape, dtype=bool),
	}

	if exclude_upper_limits:
		drop_mask[0] = ~tau_usable
		drop_mask[1] = ~sig_det

	# Sigma_RM / tau:
	# tau upper limits are plotted but excluded from the regression.
	sigma_fit_members = (
		has_tau
		& sig_det
		& ~upper
	)

	feng_sigma = sigma_fit_members & feng_g

	feng_sigma_fit = (
		_loglog_fit(
			sigma_rm_arr[feng_sigma],
			taus[feng_sigma],
		)
		if np.any(feng_sigma)
		else None
	)

	all_sigma_fit = _loglog_fit(
		sigma_rm_arr[sigma_fit_members],
		taus[sigma_fit_members],
	)

	# |RM| / sigma_RM.
	absrm_fit_members = (
		has_rm
		& np.isfinite(abs_rm)
		& (abs_rm > 0)
		& sig_det
	)

	feng_absrm = absrm_fit_members & feng_g

	feng_absrm_fit = (
		_loglog_fit(
			sigma_rm_arr[feng_absrm],
			abs_rm[feng_absrm],
		)
		if np.any(feng_absrm)
		else None
	)

	all_absrm_fit = _loglog_fit(
		sigma_rm_arr[absrm_fit_members],
		abs_rm[absrm_fit_members],
	)

	# ---- Terminal summary -------------------------------------------------

	summary: List[str] = [
		"RM vs tau correlation fit summary:"
	]

	_fit_summary(
		"Feng only sigma_RM ",
		feng_sigma_fit,
		summary,
		tau_first=True,
	)
	_fit_summary(
		"All data sigma_RM  ",
		all_sigma_fit,
		summary,
		tau_first=True,
	)
	_fit_summary(
		"Feng only |RM|     ",
		feng_absrm_fit,
		summary,
		absrm_first=True,
	)
	_fit_summary(
		"All data |RM|      ",
		all_absrm_fit,
		summary,
		absrm_first=True,
	)

	for line in summary:
		print(line)

	# ---- Figure -----------------------------------------------------------

	style = plot_style()

	sig_is_upper = np.isfinite(sigma_rm_upper)

	sig_plot = np.where(
		sig_is_upper,
		sigma_rm_upper,
		sigma_rm_arr,
	)

	sigma_lim = _logrange(sig_plot)

	fig_w, _ = pub_figsize(
		height_ratio=1.0,
		ncol=2,
	)

	fig, axes = plt.subplots(
		2,
		1,
		sharex=True,
		figsize=(fig_w, fig_w * 0.62 * 2),
	)

	fig.subplots_adjust(
		left=0.16,
		right=0.97,
		top=0.96,
		bottom=0.14,
		hspace=0.08,
	)

	_tau_sub = (
		r"\tau"
		if not label_suffix
		else rf"\tau_{{\mathrm{{{label_suffix}}}}}"
	)

	_rm_sub = (
		r"|\mathrm{RM}|"
		if not label_suffix
		else rf"|\mathrm{{RM}}_{{\mathrm{{{label_suffix}}}}}|"
	)

	_sigma_sub = (
		r"\sigma_{\mathrm{RM}}"
		if not label_suffix
		else rf"\sigma_{{\mathrm{{RM}},\mathrm{{{label_suffix}}}}}"
	)

	panels = [
		(
			0,
			taus,
			taus_err,
			_logrange(taus),
			rf"${_tau_sub}$ [ms]",
			feng_sigma_fit,
			all_sigma_fit,
			True,
			False,
		),
		(
			1,
			abs_rm,
			None,
			_logrange(abs_rm),
			rf"${_rm_sub}$ [rad m$^{{-2}}$]",
			feng_absrm_fit,
			all_absrm_fit,
			False,
			True,
		),
	]

	for (
		idx,
		y,
		yerr,
		y_lim,
		ylabel,
		feng_fit,
		all_fit,
		tau_first,
		absrm_first,
	) in panels:

		ax = axes[idx]
		feng = ~user_mask

		need_tau = idx == 0

		if need_tau:
			has_xy = (
				np.isfinite(y)
				& np.isfinite(taus)
			)
		else:
			has_xy = has_rm & np.isfinite(y)

		ref = feng & has_xy & ~drop_mask[idx]

		for grp, grp_color, grp_label in _GROUP_STYLE:
			sig_det_mask = (
				ref
				& (group == grp)
				& ~sig_is_upper
				& np.isfinite(sigma_rm_arr)
				& (sigma_rm_arr > 0)
			)

			sig_up = (
				ref
				& (group == grp)
				& sig_is_upper
				& (sigma_rm_upper > 0)
			)

			tau_up_det = (
				sig_det_mask
				& upper
				& need_tau
			)

			tau_up_sigu = (
				sig_up
				& upper
				& need_tau
			)

			sig_det_plot = sig_det_mask & (
				~upper | (not need_tau)
			)

			sig_up_plot = sig_up & (
				~upper | (not need_tau)
			)

			if np.any(sig_det_plot):
				ax.errorbar(
					sigma_rm_arr[sig_det_plot],
					y[sig_det_plot],
					xerr=(
						sigma_rm_err_arr[sig_det_plot]
						if np.all(
							np.isfinite(
								sigma_rm_err_arr[sig_det_plot]
							)
						)
						else None
					),
					yerr=(
						yerr[sig_det_plot]
						if (
							yerr is not None
							and np.all(
								np.isfinite(
									yerr[sig_det_plot]
								)
							)
						)
						else None
					),
					fmt="o",
					ms=6,
					color=grp_color,
					markeredgecolor="white",
					markeredgewidth=0.8,
					ecolor="0.4",
					elinewidth=1,
					capsize=2.5,
					zorder=3,
					label=grp_label if idx == 0 else None,
				)

			if np.any(sig_up_plot):
				ax.errorbar(
					sigma_rm_upper[sig_up_plot],
					y[sig_up_plot],
					fmt="<",
					ms=6,
					color=grp_color,
					markeredgecolor="white",
					markeredgewidth=0.5,
					ecolor="0.4",
					elinewidth=1,
					capsize=2.5,
					zorder=3,
				)

			if np.any(tau_up_det):
				ax.errorbar(
					sigma_rm_arr[tau_up_det],
					y[tau_up_det],
					fmt="v",
					ms=6,
					color=grp_color,
					markeredgecolor="white",
					markeredgewidth=0.5,
					zorder=3,
				)

			if np.any(tau_up_sigu):
				ax.errorbar(
					sigma_rm_upper[tau_up_sigu],
					y[tau_up_sigu],
					fmt="v",
					ms=6,
					color=grp_color,
					markeredgecolor="white",
					markeredgewidth=0.5,
					zorder=3,
				)

		if np.any(user_mask):
			user_yerr = None

			if yerr is not None:
				user_yerr_values = yerr[user_mask]
				if np.all(np.isfinite(user_yerr_values)):
					user_yerr = user_yerr_values

			ax.errorbar(
				sigma_rm_arr[user_mask],
				y[user_mask],
				xerr=sigma_rm_err_arr[user_mask],
				yerr=user_yerr,
				marker=(10, 1, 0),
				ms=8,
				color=IBM_PALETTE[-1],
				markeredgecolor="white",
				markeredgewidth=0.5,
				ecolor="0.4",
				elinewidth=1,
				capsize=2.5,
				zorder=4,
				label=name,
			)

		grid_x = np.logspace(
			np.log10(sigma_lim[0]),
			np.log10(sigma_lim[1]),
			400,
		)

		if feng_fit is not None:
			line_y = np.exp(
				_linear(
					np.log(grid_x),
					feng_fit["a"],
					feng_fit["b"],
				)
			)

			ax.plot(
				grid_x,
				line_y,
				color="0.35",
				lw=style["line"],
				ls="--",
				zorder=2,
			)

			_annotate_slope(
				ax,
				_slope_text(
					feng_fit,
					"Feng+ (2022): ",
					tau_first,
					absrm_first,
				),
				"0.35",
				0.88,
				style,
			)

		if all_fit is not None:
			line_y = np.exp(
				_linear(
					np.log(grid_x),
					all_fit["a"],
					all_fit["b"],
				)
			)

			lo_y = np.exp(
				_linear(
					np.log(grid_x),
					all_fit["a"] - all_fit["a_err"],
					all_fit["b"] - all_fit["b_err"],
				)
			)

			hi_y = np.exp(
				_linear(
					np.log(grid_x),
					all_fit["a"] + all_fit["a_err"],
					all_fit["b"] + all_fit["b_err"],
				)
			)

			ax.fill_between(
				grid_x,
				lo_y,
				hi_y,
				color=IBM_PALETTE[2],
				alpha=0.15,
				zorder=1,
			)

			ax.plot(
				grid_x,
				line_y,
				color=IBM_PALETTE[2],
				lw=style["line"],
				ls="-",
				zorder=2,
			)

			_annotate_slope(
				ax,
				_slope_text(
					all_fit,
					"",
					tau_first,
					absrm_first,
				),
				IBM_PALETTE[2],
				0.70,
				style,
			)

		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xlim(*sigma_lim)
		ax.set_ylim(*y_lim)

		ax.grid(
			True,
			which="major",
			alpha=0.3,
		)

		ax.tick_params(
			axis="both",
			labelsize=style["tick"],
		)

		ax.set_ylabel(
			ylabel,
			fontsize=style["label"],
		)

		if idx == 0:
			ax.legend(
				fontsize=style["legend"],
				loc="best",
			)

		if idx == 1:
			ax.set_xlabel(
				rf"${_sigma_sub}$ [rad m$^{{-2}}$]",
				fontsize=style["label"],
			)

	_savefig(
		output_file,
		dpi=600,
		bbox_inches="tight",
	)

	print(
		f"RM-tau correlation plot saved to {output_file}"
	)

	plt.close(fig)

	if print_table:
		_print_observer_frame_table(table_rows)

	return {
		"x": taus,
		"xerr": taus_err,
		"sigma_rm": sigma_rm_arr,
		"sigma_rm_err": sigma_rm_err_arr,
		"abs_rm": abs_rm,
		"upper": upper,
		"user_mask": user_mask,
		"names": names,
		"feng_sigma_fit": feng_sigma_fit,
		"all_sigma_fit": all_sigma_fit,
		"feng_absrm_fit": feng_absrm_fit,
		"all_absrm_fit": all_absrm_fit,
	}


# ---------------------------------------------------------------------------
# Host-frame correction
# ---------------------------------------------------------------------------


def plot_rm_tau_correlation_host_frame(
	sigma_rm: float,
	sigma_rm_err: float,
	rm: float,
	tau: float,
	ra: float,
	dec: float,
	z: float,
	tau_err: Optional[float] = None,
	rm_err: Optional[float] = None,
	sample: Optional[List[Dict]] = None,
	skymap_dir: Path = DEFAULT_SKYMAP_DIR,
	output_file: str = "rm_tau_correlation_host_frame.png",
	name: str = "This work",
	scatter_warn_threshold: float = DEFAULT_SCATTER_WARN_THRESHOLD,
	exclude_upper_limits: bool = False,
	print_table: bool = True
) -> Dict:
	"""
	Apply Galactic-RM subtraction and rest-frame corrections before plotting.

	RM_host = (RM_obs - RM_galactic) * (1 + z)^2
	sigma_RM_host = sigma_RM_obs * (1 + z)^2
	tau_host = tau_obs / (1 + z)^3

	Sources without the required coordinates/redshift/RM are skipped.
	High SPICE-RACS neighbour scatter causes a fallback to the
	Hutschenreuter et al. (2022) Galactic RM reconstruction rather than
	rejecting the source.
	"""
	if sample is None:
		sample = feng2022_sample()

	skymaps = SpiceRacsSkymaps(
		skymap_dir=skymap_dir
	)

	print("Applying host-frame correction to reference sample:")

	corrected_sample = host_frame_correct_sample(
		sample,
		skymaps=skymaps,
		scatter_warn_threshold=scatter_warn_threshold,
	)

	print("Applying host-frame correction to your burst:")

	user_src = host_frame_correct_source(
		{
			"name": name,
			"ra": ra,
			"dec": dec,
			"z": z,
			"rm": rm,
			"rm_err": rm_err,
			"sigma_rm": sigma_rm,
			"sigma_rm_err": sigma_rm_err,
			"tau": tau,
			"tau_err": tau_err,
		},
		skymaps,
		scatter_warn_threshold,
	)

	if user_src is None:
		raise ValueError(
			f"Could not apply host-frame correction to your burst "
			f"'{name}' -- check ra/dec/z/rm are all provided and "
			f"valid."
		)

	corrected_sample_with_user = [
		dict(src) for src in corrected_sample
	]
	corrected_sample_with_user.append(user_src)

	# Freeze nu_centre so plot_rm_tau_correlation performs no additional
	# observing-frequency correction. Tau has already been corrected into
	# the host rest frame.
	for src in corrected_sample_with_user:
		src["nu_centre"] = 1300.0

	result = plot_rm_tau_correlation(
		sigma_rm=user_src["sigma_rm"],
		sigma_rm_err=user_src["sigma_rm_err"],
		rm=user_src["rm"],
		rm_err=user_src.get("rm_err"),
		tau=abs(user_src["tau"]),
		tau_err=user_src.get("tau_err"),
		sample=corrected_sample,
		output_file=output_file,
		name=name,
		freq_mhz=None,
		ref_freq_mhz=1300.0,
		label_suffix="host",
		exclude_upper_limits=exclude_upper_limits,
		print_table=print_table,
	)

	# Print the complete host-frame table last, after the plot and fit output.
	_print_host_frame_table(corrected_sample_with_user)

	return result


def main() -> None:
	"""CLI entry point for the RM-tau correlation plot."""
	parser = argparse.ArgumentParser(
		description=(
			"Plot sigma_RM and |RM| vs tau with the Feng+ (2022)/Pandhi+ "
			"(2026)/Uttarkar+ (2024/2026) sample plus your own burst, "
			"with power-law fits."
		)
	)

	parser.add_argument(
		"--sigma-rm",
		type=float,
		required=True,
		help="Your burst's sigma_RM (rad/m^2)",
	)

	parser.add_argument(
		"--sigma-rm-err",
		type=float,
		required=True,
		help="Uncertainty on sigma_RM (rad/m^2)",
	)

	parser.add_argument(
		"--rm",
		type=float,
		required=True,
		help="Your burst's RM (rad/m^2)",
	)

	parser.add_argument(
		"--tau",
		type=float,
		required=True,
		help="Your burst's scattering timescale tau (ms)",
	)

	parser.add_argument(
		"--tau-err",
		type=float,
		default=None,
		help="Uncertainty on tau (ms)",
	)

	parser.add_argument(
		"--freq",
		type=float,
		default=None,
		help=(
			"Observing frequency (MHz) of your tau measurement; "
			"if given, tau is rescaled to --ref-freq"
		),
	)

	parser.add_argument(
		"--ref-freq",
		type=float,
		default=1300.0,
		help=(
			"Reference frequency (MHz) for tau rescaling "
			"(default: 1300)"
		),
	)

	parser.add_argument(
		"--scattering-index",
		type=float,
		default=None,
		help="Scattering index alpha in tau ~ nu**alpha",
	)

	parser.add_argument(
		"--scattering-index-err",
		type=float,
		default=None,
		help="Uncertainty on the scattering index",
	)

	parser.add_argument(
		"--name",
		default="This work",
		help="Legend label for your burst",
	)

	parser.add_argument(
		"-o",
		"--output",
		default="rm_tau_correlation.png",
		help="Output PNG filename",
	)

	parser.add_argument(
		"--pub-col",
		type=float,
		default=2,
		help="Publication figure column count (default: 2)",
	)

	args = parser.parse_args()

	from frbop.utils.plotting import (
		set_pub_col,
		set_pub_style,
	)

	set_pub_col(args.pub_col)
	set_pub_style(use_latex=False)

	plot_rm_tau_correlation(
		sigma_rm=args.sigma_rm,
		sigma_rm_err=args.sigma_rm_err,
		rm=args.rm,
		tau=args.tau,
		tau_err=args.tau_err,
		output_file=args.output,
		name=args.name,
		freq_mhz=args.freq,
		ref_freq_mhz=args.ref_freq,
		scattering_index=args.scattering_index,
		scattering_index_err=args.scattering_index_err,
	)


if __name__ == "__main__":
	main()