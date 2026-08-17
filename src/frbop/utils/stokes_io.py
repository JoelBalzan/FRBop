"""Shared helpers for loading and normalizing Stokes cubes and axes."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def normalise_pol_order(pol_order) -> str:
	"""Convert a polarization order value to a compact uppercase string."""
	if pol_order is None:
		return ""
	value = pol_order
	if isinstance(value, np.ndarray):
		if value.shape == ():
			value = value.item()
		elif value.dtype.kind in {"S", "U", "O"}:
			parts = []
			for item in value.ravel():
				if isinstance(item, (bytes, np.bytes_)):
					parts.append(item.decode())
				else:
					parts.append(str(item))
			value = "".join(parts)
		else:
			value = "".join(str(item) for item in value.ravel())
	elif isinstance(value, (bytes, np.bytes_)):
		value = value.decode()
	else:
		value = str(value)
	return re.sub(r"[^A-Z0-9]", "", value.upper())


def load_stokes_cube_from_npz(npz_path: Path):
	"""Load an archive and return the cube plus non-stokes metadata."""
	with np.load(npz_path, allow_pickle=True) as archive:
		if "stokes" not in archive.files:
			raise ValueError(f"{npz_path} does not contain a 'stokes' array")
		cube = np.asarray(archive["stokes"])
		metadata = {key: archive[key] for key in archive.files if key != "stokes"}
	return cube, metadata


def extract_stokes_components(cube: np.ndarray, pol_order=None, nchan=None, nsamp=None):
	"""Split a Stokes cube into I/Q/U/V arrays using archive metadata when available."""
	cube = np.asarray(cube)
	if cube.ndim != 3:
		raise ValueError(f"Stokes cube must be 3D, got shape {cube.shape}")

	order = normalise_pol_order(pol_order)
	if not order:
		if cube.shape[0] in (3, 4):
			order = "IQUV"[: cube.shape[0]]
		elif cube.shape[-1] in (3, 4):
			order = "IQUV"[: cube.shape[-1]]
		else:
			raise ValueError(
				"Unable to infer polarisation order from the archive; provide a 'pol_order' key"
			)

	if len(order) not in (3, 4):
		raise ValueError(f"Unsupported polarisation order {order!r}; expected 3 or 4 components")

	n_stokes = len(order)
	aligned = None
	if nchan is not None and nsamp is not None:
		nchan = int(np.asarray(nchan).reshape(()))
		nsamp = int(np.asarray(nsamp).reshape(()))
		for stokes_axis in range(3):
			if cube.shape[stokes_axis] != n_stokes:
				continue
			remaining_axes = [axis for axis in range(3) if axis != stokes_axis]
			for freq_axis in remaining_axes:
				for time_axis in remaining_axes:
					if freq_axis == time_axis:
						continue
					if cube.shape[freq_axis] == nchan and cube.shape[time_axis] == nsamp:
						aligned = np.moveaxis(cube, (stokes_axis, freq_axis, time_axis), (0, 1, 2))
						break
				if aligned is not None:
					break
			if aligned is not None:
				break

	if aligned is None:
		if cube.shape[0] == n_stokes:
			aligned = cube
		elif cube.shape[-1] == n_stokes:
			aligned = np.moveaxis(cube, -1, 0)
		else:
			raise ValueError(
				f"Unable to identify the Stokes axis in cube shape {cube.shape}; expected one axis of length {n_stokes}"
			)

	component_map = {label: aligned[idx] for idx, label in enumerate(order)}
	stokes_i = component_map.get("I")
	stokes_q = component_map.get("Q")
	stokes_u = component_map.get("U")
	stokes_v = component_map.get("V")
	if stokes_i is None:
		raise ValueError(f"Archive polarisation order {order!r} does not include Stokes I")
	return stokes_i, stokes_q, stokes_u, stokes_v, aligned, order


def derive_axes_from_metadata(metadata, nchan: int, nsamp: int):
	"""Build frequency/time axes from archive metadata when standalone files are absent."""
	freq_mhz = None
	time_ms = None

	if "freq_mhz" in metadata:
		freq_mhz = np.asarray(metadata["freq_mhz"], dtype=float)
	elif "freq" in metadata:
		freq_mhz = np.asarray(metadata["freq"], dtype=float)
	elif all(key in metadata for key in ("fch1_mhz", "foff_mhz", "nchan")):
		fch1_mhz = float(np.asarray(metadata["fch1_mhz"]).reshape(()))
		foff_mhz = float(np.asarray(metadata["foff_mhz"]).reshape(()))
		nchan_meta = int(np.asarray(metadata["nchan"]).reshape(()))
		freq_mhz = fch1_mhz + np.arange(nchan_meta, dtype=float) * foff_mhz
		if freq_mhz.size != nchan:
			freq_mhz = fch1_mhz + np.arange(nchan, dtype=float) * foff_mhz

	if "time_ms" in metadata:
		time_ms = np.asarray(metadata["time_ms"], dtype=float)
	elif "time" in metadata:
		time_ms = np.asarray(metadata["time"], dtype=float)
	elif all(key in metadata for key in ("tsamp_s", "nsamp")):
		tsamp_s = float(np.asarray(metadata["tsamp_s"]).reshape(()))
		nsamp_meta = int(np.asarray(metadata["nsamp"]).reshape(()))
		time_ms = np.arange(nsamp_meta, dtype=float) * tsamp_s * 1_000.0
		if time_ms.size != nsamp:
			time_ms = np.arange(nsamp, dtype=float) * tsamp_s * 1_000.0

	return freq_mhz, time_ms