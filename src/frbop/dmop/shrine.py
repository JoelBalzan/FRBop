"""SHRINE-II integration: structure and S/N maximisation, k_c and the low-pass filter.

FRBop uses the copy of SHRINE-II in dmop/SHRINE2 (Balzan et al., in prep.: "the paper"). The old SHRINE
(dmop/SHRINE) is no longer used. SHRINE-II is called directly, on profiles made here from FRBop's
dynamic spectra:

	I(d_j, t) = sum_nu x_nu(t + d_j dt_nu)

with x_nu the data in channel nu (already dedispersed at input_dm), d_j = DM_j - input_dm the trial
DM relative to input_dm, and dt_nu the channel's dispersion delay per unit DM in samples. Each shift
is a Fourier phase ramp, so fractional shifts are exact and the shifts are circular in time. This is
the dedispersion that sigma_DM assumes (paper Section 2.2 and Appendix 2), whatever FRBop's own
dedispersion mode, which is used for everything else.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from frbop.utils.plotting import pub_figsize, savefig

from .SHRINE2 import check_trial_dms, choose_kc, dm_uncertainty
from .SHRINE2.noise import high_mode_noise
from .SHRINE2.snr import maximise_sn
from .SHRINE2.structure import smooth, structure_parameter
from .SHRINE2.summary import write_sn_summary, write_structure_summary

# A burst is too faint for structure maximisation when the filtered noise is not small compared
# with the structure, sqrt(<||e||^2>)/Sp > 0.3 (paper Sections 3.4, 5.2), or the curvature of the
# parabola is poorly determined, std[c2]/|c2| > 0.5 (Section 3.3.4). As SHRINE-II.
E_RATIO_MAX = 0.3
C2_ERR_MAX = 0.5


class ShrineMixin:
	def apply_kc_lowpass_2d(self, data_2d: np.ndarray, kc: int) -> np.ndarray:
		"""
		S23's low-pass filter along time, S x = C diag[f(k)] C^T x with f(k) = 1/(1 + (k/k_c)^6)
		(paper Eq. smooth; SHRINE2.structure.smooth), applied to every row of data_2d.
		"""
		if data_2d.ndim != 2:
			raise ValueError("data_2d must be 2D (freq x time)")
		if kc <= 0:
			return data_2d.copy()
		kc_eff = max(1, min(int(kc), data_2d.shape[1]))
		return smooth(np.asarray(data_2d, dtype=float), kc_eff)

	def _dt_us(self) -> float:
		"""The sample interval in us (SHRINE-II's dt)."""
		dt_ms = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
		return 1000.0 * dt_ms

	@staticmethod
	def _dt_label(dt_us: float) -> str:
		"""The sample interval as it appears in SHRINE's file names ({label}_I_{dt}us.npy)."""
		return f"{dt_us:g}"

	def channel_delays_per_dm(self) -> np.ndarray:
		"""
		Each channel's dispersion delay per unit DM in samples, relative to the reference frequency:

			dt_nu = K_DM (1/f_nu^2 - 1/f_ref^2) / dt

		with K_DM = DM_CONSTANT (ms MHz^2 per pc cm^-3), f in MHz and dt in ms. Positive below f_ref,
		so a positive trial DM moves low frequencies earlier (SHRINE-II's sign convention).
		"""
		dt_ms = self._dt_us() / 1000.0
		freq = np.asarray(self.freq_mhz, dtype=float)
		f_ref = float(self.reference_freq)
		return self.DM_CONSTANT * (1.0 / freq**2 - 1.0 / f_ref**2) / dt_ms

	def shrine2_profiles(self, data: np.ndarray, dm_values: np.ndarray):
		"""
		I(d_j, t) for SHRINE-II, from a dynamic spectrum dedispersed at input_dm.

		Each channel kept is shifted by d_j dt_nu samples with a Fourier phase ramp (circular in
		time; exact for fractional shifts) and the channels are summed. Channels that are constant
		in time (zapped channels, which FRBop sets to zero) are left out, so that the channel
		delays describe the channels actually summed. Each kept channel's share of the noise
		variance, s_nu (paper Appendix 2), is estimated from the spread of its first differences:
		for white noise of variance sigma_nu^2, Var(n_{t+1} - n_t) = 2 sigma_nu^2, and the median
		absolute deviation keeps the burst, which changes slowly, from inflating it.

		Parameters
		----------
		data : (n_freq, n_time) dynamic spectrum, dedispersed at input_dm
		dm_values : trial DMs (pc cm^-3)

		Returns
		-------
		I : (n_dm, n_time) profiles
		d : trial DMs relative to input_dm, d_j = DM_j - input_dm
		dt_nu : delay per unit DM (samples) of each kept channel
		share : s_nu of each kept channel (sums to 1)
		zapped : indices of the channels left out
		"""
		data = np.asarray(data, dtype=float)
		n_freq, n_time = data.shape
		d = np.asarray(dm_values, dtype=float) - self.input_dm
		dt_nu_all = self.channel_delays_per_dm()

		# The channels kept: finite and not constant in time.
		kept = []
		zapped = []
		for c in range(n_freq):
			row = data[c]
			if np.all(np.isfinite(row)) and np.ptp(row) > 0:
				kept.append(c)
			else:
				zapped.append(c)
		if len(kept) == 0:
			raise ValueError("no channels left for SHRINE-II: every channel is constant or non-finite")
		x = data[kept]
		dt_nu = dt_nu_all[kept]

		# s_nu from each channel's noise variance, sigma_nu^2 = (1.4826 MAD[n_{t+1} - n_t])^2 / 2.
		var_nu = np.zeros(len(kept))
		for i in range(len(kept)):
			diff = np.diff(x[i])
			mad = np.median(np.abs(diff - np.median(diff)))
			var_nu[i] = (1.4826 * mad) ** 2 / 2.0
			if var_nu[i] <= 0:
				var_nu[i] = np.var(diff) / 2.0
		if np.sum(var_nu) > 0:
			share = var_nu / np.sum(var_nu)
		else:
			share = np.full(len(kept), 1.0 / len(kept))

		# I(d_j, t) = sum_nu x_nu(t + d_j dt_nu): a phase ramp exp(2 pi i f d_j dt_nu) on each
		# channel's Fourier transform (f in cycles per sample), then the sum over channels.
		f = np.fft.rfftfreq(n_time)
		X = np.fft.rfft(x, axis=1)
		I = np.zeros((d.size, n_time))
		for j in range(d.size):
			ramp = np.exp(2j * np.pi * f[np.newaxis, :] * (d[j] * dt_nu)[:, np.newaxis])
			I[j] = np.fft.irfft(np.sum(X * ramp, axis=0), n_time)
		return I, d, dt_nu, share, zapped

	def _shrine2_kc(self, I: np.ndarray, force_kc: Optional[int] = None):
		"""
		k_c for I(d_j, t): force_kc if given, otherwise SHRINE-II's automatic choice (S23's
		procedure with the paper's extended search, Appendix 1). As SHRINE-II's
		maximise_structure.py, a forced k_c below the automatic one is flagged: sigma_t^2 uses the
		DCT modes above k_min = max(N/4, 2 k_c), which must lie above the burst's power.

		Returns
		-------
		kc, note (how k_c was chosen), warnings (list of str)
		"""
		try:
			kc_auto = int(choose_kc(I))
		except RuntimeError:
			if force_kc is None:
				raise
			kc_auto = None

		if force_kc is None:
			return kc_auto, "S23's procedure, with the extended search (paper Appendix 1)", []

		kc = int(force_kc)
		note = f"forced (--shrine-kc); the automatic choice is {kc_auto}"
		warnings = []
		if kc_auto is not None and kc < kc_auto:
			warnings.append(f"forced k_c = {kc} is below the automatic {kc_auto}: sigma_t^2 may "
							"include burst power, which overestimates sigma_S and sigma_DM")
		return kc, note, warnings

	def run_shrine2_structure(self,
							  run_prefix: str,
							  dm_values: np.ndarray,
							  data: np.ndarray,
							  force_kc: Optional[int] = None,
							  save_all: bool = True) -> Dict:
		"""
		Structure maximisation with SHRINE-II (paper Section 3.3).

		Makes I(d_j, t) from data (shrine2_profiles), chooses k_c, and finds the structure-maximising
		DM d_fit and its 1-sigma uncertainty sigma_DM (SHRINE2.dm_uncertainty), then checks the
		trial DMs (SHRINE2.check_trial_dms). Writes, in shrine_logs/{run_prefix}: the trial DMs,
		I(d_j, t), Sp at each trial DM ({run_prefix}_SPs.dat), S x, SHRINE-II's summary file and,
		with save_all, DM.txt and SHRINE-II's plots. Delta DMs in these files are relative to
		input_dm.

		Returns
		-------
		dict with res (from dm_uncertainty, with d relative to input_dm), I, d, kc, kc_note,
		warnings, run_dir, and dm_best: input_dm + d_fit, or input_dm + d_max if there was no fit.
		"""
		dt_us = self._dt_us()
		I, d, dt_nu, share, zapped = self.shrine2_profiles(data, dm_values)
		print(f"    SHRINE-II: {d.size} trial DMs, {I.shape[1]} samples of {dt_us:g} us, "
			  f"{dt_nu.size} channels summed ({len(zapped)} left out)")

		kc, kc_note, kc_warnings = self._shrine2_kc(I, force_kc)
		res = dm_uncertainty(I, d, kc, dt_nu, share=share)

		# The checks of SHRINE-II's maximise_structure.py, with the suggested trial DMs in absolute DM.
		grid_warnings, grid = check_trial_dms(res)
		warnings = list(kc_warnings) + list(grid_warnings)
		if grid is not None:
			d_low, d_high, step = grid
			warnings.append(f"suggested trial DMs: {self.input_dm + d_low:.6g} to "
							f"{self.input_dm + d_high:.6g} pc cm^-3 in steps of {step:.3g}")
		faint = res["e_ratio"] > E_RATIO_MAX
		if res["ok"] and res["c2_err"] > C2_ERR_MAX:
			faint = True
		if faint:
			warnings.append("structure maximisation may be unreliable for this burst: compare with "
							"the S/N-maximising DM (method snr)")

		run_dir = Path("shrine_logs") / run_prefix
		run_dir.mkdir(parents=True, exist_ok=True)
		dt_label = self._dt_label(dt_us)
		np.save(run_dir / f"{run_prefix}_DMs.npy", np.asarray(dm_values, dtype=float))
		np.save(run_dir / f"{run_prefix}_I_{dt_label}us.npy", I)
		np.savetxt(run_dir / f"{run_prefix}_SPs.dat", res["Sp"])
		Sx = smooth(I, res["kc"])
		np.save(run_dir / f"{run_prefix}_I_smooth_{dt_label}us.npy", Sx)
		write_structure_summary(run_dir / f"{run_prefix}_structure_summaryfile.txt", res, run_prefix,
								dt_label, self.input_dm, kc_note, warnings, None, save_all, zapped)
		if save_all:
			self._save_shrine2_structure_plots(run_dir, run_prefix, res, I, Sx, dt_us)

		if res["ok"]:
			dm_best = self.input_dm + res["d_fit"]
			print(f"    DM = {dm_best:.6f} +/- {res['sig_DM']:.3g} (1 sigma), k_c = {res['kc']}")
		else:
			dm_best = self.input_dm + res["d_max"]
			print(f"    No sigma_DM ({res['why']}); DM of the largest structure parameter = "
				  f"{dm_best:.6f}, k_c = {res['kc']}")
		for message in warnings:
			print(f"    Warning: {message}")

		return dict(res=res, I=I, d=d, kc=res["kc"], kc_note=kc_note, warnings=warnings,
					run_dir=run_dir, dm_best=float(dm_best))

	def _save_shrine2_structure_plots(self, run_dir: Path, run_prefix: str, res: Dict,
									  I: np.ndarray, Sx: np.ndarray, dt_us: float) -> None:
		"""DM.txt and SHRINE-II's plots, as its maximise_structure.py writes them with -s."""
		from .SHRINE2 import plots

		tag = run_dir / f"{run_prefix}_{self._dt_label(dt_us)}us"

		# The profile plotted: the trial DM nearest d_fit, or d_max if there was no fit.
		if res["ok"]:
			j = int(np.argmin(np.abs(res["d"] - res["d_fit"])))
		else:
			j = res["j_max"]

		# As SHRINE: DM.txt holds the row of I(DM, t) with the largest structure parameter.
		with open(run_dir / "DM.txt", "w") as out:
			out.write(f"{res['j_max']}")

		plots.plot_structure_parameter(res, f"{tag}_structure_parameter.png")
		plots.plot_dct_spectrum(I, res["kc"], f"{tag}_DCT_v_k_index.png")
		plots.plot_profile_and_smoothed(I[j], res["kc"], dt_us, f"{tag}_I_max.png")
		plots.plot_I_DM_t(I, res["d"], dt_us, f"{tag}_I_DM_t.png")
		plots.plot_I_DM_t(Sx, res["d"], dt_us, f"{tag}_I_DM_t_smoothed.png")
		plots.plot_I_DM_t(I - Sx, res["d"], dt_us, f"{tag}_detrended_noise.png")

		# The noise that sigma_t^2 is taken from: the profile at d_max, from its high DCT modes.
		noise_only, k_min = high_mode_noise(I[res["j_max"]], res["kc"])
		plots.plot_noise_at_d_max(noise_only, res["sig2"], k_min, res["d_max"], dt_us,
								  f"{tag}_noise_max.png")

	def run_shrine2_sn(self,
					   run_prefix: str,
					   dm_values: np.ndarray,
					   I: np.ndarray,
					   kc: int,
					   save_all: bool = True) -> Dict:
		"""
		S/N maximisation with SHRINE-II (SHRINE2.snr.maximise_sn, SHRINE's maximise_sn.py with its
		bugs fixed) on I(d_j, t) from shrine2_profiles: the trial DM with the largest boxcar S/N of
		the smoothed profile, and the trial DMs where the S/N stays within 1 of it. Writes
		SHRINE-II's S/N outputs in shrine_logs/{run_prefix}.

		Returns
		-------
		the dict from maximise_sn, with run_dir, dm_best, and dm_low, dm_high (the first trial DMs
		where the S/N has dropped by 1; None on a side where it does not, within the trial DMs).
		"""
		dt_us = self._dt_us()
		dm_values = np.asarray(dm_values, dtype=float)
		d = dm_values - self.input_dm
		sn = maximise_sn(I, d, kc, dt_us)
		j = sn["j_max"]

		run_dir = Path("shrine_logs") / run_prefix
		run_dir.mkdir(parents=True, exist_ok=True)
		np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
		np.savetxt(run_dir / f"{run_prefix}_SNs.dat", sn["SN"])
		write_sn_summary(run_dir / f"{run_prefix}_SN_summaryfile.txt", sn, dt_us)
		np.savetxt(run_dir / f"{run_prefix}_SN_DMs.dat", sn["d"] - sn["d"][j])

		from .SHRINE2 import plots
		plots.plot_sn_vs_dm(sn["d"], sn["SN"], run_dir / f"{run_prefix}_SN.png")
		plots.plot_sn_window(sn["Sx"][j], sn["start"], sn["length"], dt_us,
							 run_dir / f"{run_prefix}_Max_SN_window.png")
		plots.plot_profile_and_smoothed(I[j], sn["kc"], dt_us,
										run_dir / f"{run_prefix}_{self._dt_label(dt_us)}us_I_SN_max.png")
		if save_all:
			np.save(run_dir / f"{run_prefix}_I_smooth.npy", sn["Sx"])
			with open(run_dir / "DM.txt", "w") as out:
				out.write(f"{j}")

		sn["run_dir"] = run_dir
		sn["dm_best"] = float(dm_values[j])
		sn["dm_low"] = float(dm_values[sn["lo"]]) if sn["lo_found"] else None
		sn["dm_high"] = float(dm_values[sn["hi"]]) if sn["hi_found"] else None
		print(f"    S/N-maximising DM = {sn['dm_best']:.6f} (S/N {sn['SN_max']:.1f}), k_c = {sn['kc']}")
		if not (sn["lo_found"] and sn["hi_found"]):
			print("    Warning: the S/N does not fall by 1 within the trial DMs on one side")
		return sn

	def structure_parameter_scan(self,
								 run_prefix: str,
								 dm_values: np.ndarray,
								 profiles: np.ndarray,
								 kc: Optional[int] = None,
								 save_all: bool = True) -> Dict:
		"""
		The structure parameter Sp(d_j) = ||D1 S x_j|| (SHRINE2.structure_parameter) of the given
		profiles, without sigma_DM: for profiles whose noise is not the zero-mean noise sigma_DM
		assumes, such as the linear polarisation L(t, DM). k_c is kc, or SHRINE-II's automatic
		choice. Writes Sp, a short summary and SHRINE-II's diagnostic plots in
		shrine_logs/{run_prefix}.

		Returns
		-------
		dict with Sp, kc, j_max, dm_best, run_dir
		"""
		profiles = np.asarray(profiles, dtype=float)
		dm_values = np.asarray(dm_values, dtype=float)
		if kc is None:
			kc = int(choose_kc(profiles))
		kc = int(kc)
		Sp = structure_parameter(profiles, kc)
		j_max = int(np.argmax(Sp))
		dt_us = self._dt_us()

		run_dir = Path("shrine_logs") / run_prefix
		run_dir.mkdir(parents=True, exist_ok=True)
		np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
		np.savetxt(run_dir / f"{run_prefix}_SPs.dat", Sp)
		with open(run_dir / f"{run_prefix}_structure_summaryfile.txt", "w") as out:
			out.write("//begin structure parameter summary//\n/*\n")
			out.write(f"Label: {run_prefix}\n")
			out.write(f"Time Resolution: {dt_us:g}us\n")
			out.write(f"kc: {kc}\n")
			out.write(f"Input DM: {self.input_dm}\n")
			out.write(f"DM of the largest structure parameter: {dm_values[j_max]}\n")
			out.write(f"Structure Parameter at Maximum: {Sp[j_max]}\n")
			out.write("\tNo sigma_DM: the noise of these profiles is not the zero-mean noise it assumes.\n")
			out.write("*/\n//end structure parameter summary//\n")
		if save_all:
			from .SHRINE2 import plots
			tag = run_dir / f"{run_prefix}_{self._dt_label(dt_us)}us"
			d = dm_values - self.input_dm
			with open(run_dir / "DM.txt", "w") as out:
				out.write(f"{j_max}")
			plots.plot_dct_spectrum(profiles, kc, f"{tag}_DCT_v_k_index.png")
			plots.plot_profile_and_smoothed(profiles[j_max], kc, dt_us, f"{tag}_I_max.png")
			plots.plot_I_DM_t(profiles, d, dt_us, f"{tag}_I_DM_t.png")
			plots.plot_I_DM_t(smooth(profiles, kc), d, dt_us, f"{tag}_I_DM_t_smoothed.png")
		return dict(Sp=Sp, kc=kc, j_max=j_max, dm_best=float(dm_values[j_max]), run_dir=run_dir)

	def build_nonshrine_L_dm_reference(self,
									   dm_values: np.ndarray,
									   data_q: np.ndarray,
									   data_u: np.ndarray,
									   output_size: int) -> np.ndarray:
		"""
		Build L(t, DM) from dedispersed linear-polarisation dynamic spectra.

		Rows are DM trials; columns are time samples (frequency-summed L).
		This matches the (delta_DM, time) layout expected by SHRINE get_kc.

		L for each DM trial is computed via sqrt(mean(Q,f)² + mean(U,f)²)
		(average Q and U over frequency FIRST, then compute L).  This avoids
		the Rice bias that plagues mean(sqrt(Q²+U²),f).
		"""
		dm_values = np.asarray(dm_values, dtype=float)
		self._nonshrine_dm_values = dm_values.copy()
		n_dm = dm_values.shape[0]
		L_dm = np.zeros((n_dm, output_size), dtype=float)

		for i, dm in enumerate(dm_values):
			dedisp_q = self.dedisperse(data_q, dm, output_size=output_size, mode=self.dedisp_mode)
			dedisp_u = self.dedisperse(data_u, dm, output_size=output_size, mode=self.dedisp_mode)
			q_ts = np.nansum(dedisp_q, axis=0)
			u_ts = np.nansum(dedisp_u, axis=0)
			L_dm[i] = np.sqrt(q_ts**2 + u_ts**2)
		self._nonshrine_L_dm_reference = L_dm
		return L_dm

	def _maybe_prepare_nonshrine_L_dm_reference(self,
												dm_values: np.ndarray,
												data_q: Optional[np.ndarray],
												data_u: Optional[np.ndarray],
												output_size: int) -> None:
		"""
		Precompute L(t, DM) for polarisation kc selection and structure-style uncertainties.
		"""
		if data_q is None or data_u is None:
			return
		if self._nonshrine_L_dm_reference is not None:
			return
		self.build_nonshrine_L_dm_reference(dm_values, data_q, data_u, output_size)

	def resolve_nonshrine_kc(self, reference_data_2d: np.ndarray) -> int:
		"""
		k_c for the non-SHRINE methods' smoothing: the resolved value if there is one, then
		--nonshrine-kc, otherwise SHRINE-II's automatic choice on the reference profiles (L(t, DM)
		when it has been built).
		"""
		if self._nonshrine_L_dm_reference is not None:
			reference_data_2d = self._nonshrine_L_dm_reference
		if self._nonshrine_resolved_kc is not None:
			if not self._nonshrine_kc_printed:
				print(f"\nNon-SHRINE kc: {self._nonshrine_resolved_kc}")
				self._nonshrine_kc_printed = True
			return int(self._nonshrine_resolved_kc)

		if self.nonshrine_kc is not None:
			self._nonshrine_resolved_kc = int(self.nonshrine_kc)
			if not self._nonshrine_kc_printed:
				print(f"\nNon-SHRINE kc: {self._nonshrine_resolved_kc}")
				self._nonshrine_kc_printed = True
			return self._nonshrine_resolved_kc

		reference = np.asarray(reference_data_2d, dtype=float)
		if reference.ndim == 1:
			reference = reference[np.newaxis, :]
		self._nonshrine_resolved_kc = int(choose_kc(reference))
		if not self._nonshrine_kc_printed:
			print(f"\nNon-SHRINE kc: {self._nonshrine_resolved_kc}")
			self._nonshrine_kc_printed = True
		return self._nonshrine_resolved_kc

	def _reset_nonshrine_kc_state(self) -> None:
		self._nonshrine_resolved_kc = None
		self._nonshrine_kc_printed = False
		self._nonshrine_L_dm_reference = None
		self._nonshrine_dm_values = None

	def maybe_kc_smooth_nonshrine(self,
									   data_i: Optional[np.ndarray],
									   data_q: Optional[np.ndarray] = None,
									   data_u: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
		"""
		Optionally apply kc smoothing to non-SHRINE method inputs.
		"""
		if not self.nonshrine_kc_smooth:
			return data_i, data_q, data_u

		if self._nonshrine_L_dm_reference is not None:
			kc = self.resolve_nonshrine_kc(self._nonshrine_L_dm_reference)
		elif data_q is not None and data_u is not None:
			l_profile = self._linear_time_profile_from_qu(data_q, data_u)
			kc = self.resolve_nonshrine_kc(l_profile[np.newaxis, :])
		else:
			reference = data_i if data_i is not None else data_q
			if reference is None:
				return data_i, data_q, data_u
			kc = self.resolve_nonshrine_kc(reference)

		sm_i = self.apply_kc_lowpass_2d(data_i, kc) if data_i is not None else None
		sm_q = self.apply_kc_lowpass_2d(data_q, kc) if data_q is not None else None
		sm_u = self.apply_kc_lowpass_2d(data_u, kc) if data_u is not None else None
		return sm_i, sm_q, sm_u

	def save_nonshrine_L_dm_diagnostics(self, label: str = "frb", segment_tag: str = "seg"):
		"""
		The structure parameter of the L(DM, t) reference profiles and SHRINE-II's diagnostic plots
		(DCT spectrum, profile at the maximum, L(DM, t)), in shrine_logs alongside the Stokes-I
		outputs. No sigma_DM: the noise of L is not the zero-mean noise it assumes.
		"""
		if self._nonshrine_L_dm_reference is None:
			print("L(DM, t) diagnostics skipped (no L reference).")
			return
		if self._nonshrine_resolved_kc is None:
			print("L(DM, t) diagnostics skipped (kc not resolved).")
			return
		if self._nonshrine_dm_values is None:
			print("L(DM, t) diagnostics skipped (no DM values stored).")
			return

		run_prefix = f"{label}_{segment_tag}_L_dm"
		self.structure_parameter_scan(
			run_prefix=run_prefix,
			dm_values=self._nonshrine_dm_values,
			profiles=self._nonshrine_L_dm_reference,
			kc=self._nonshrine_resolved_kc,
			save_all=True,
		)

	def save_nonshrine_run_outputs(self,
								   run_prefix: str,
								   method_label: str,
								   dm_values: np.ndarray,
								   metric_values: np.ndarray,
								   metric_name: str,
								   dedispersed_i: np.ndarray,
								   optimal_dm: float,
								   optimal_metric: float,
								   uncertainty: Optional[Dict[str, Optional[float]]] = None) -> Path:
		"""
		Save non-SHRINE method logs/plots in a SHRINE-like run directory.
		"""
		run_dir = Path("shrine_logs") / run_prefix
		run_dir.mkdir(parents=True, exist_ok=True)

		max_idx = int(np.argmax(metric_values))
		np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
		np.savetxt(run_dir / f"{run_prefix}_{metric_name}.dat", np.asarray(metric_values, dtype=float))
		np.save(run_dir / f"{run_prefix}_I_at_max.npy", dedispersed_i)

		# Metric-vs-DM plot
		plt.figure(figsize=pub_figsize(height_ratio=0.6))
		plt.plot(dm_values, metric_values, '-', color='tab:blue', linewidth=1.8)
		if uncertainty is not None:
			low_dm = uncertainty.get('uncertainty_low_dm')
			high_dm = uncertainty.get('uncertainty_high_dm')
			dm_left = float(np.min(dm_values))
			dm_right = float(np.max(dm_values))
			shade_low = dm_left if low_dm is None else float(low_dm)
			shade_high = dm_right if high_dm is None else float(high_dm)
			if shade_low <= shade_high:
				plt.axvspan(shade_low, shade_high, color='tab:orange', alpha=0.18, label='DM uncertainty')
		plt.axvline(optimal_dm, color='tab:red', linestyle='--', linewidth=1.2,
					label=f"max DM={optimal_dm:.6f}")
		if uncertainty is not None:
			minus = uncertainty.get('uncertainty_minus')
			plus = uncertainty.get('uncertainty_plus')
			unc_text = self._format_uncertainty(optimal_dm, minus, plus)
			plt.title(rf"{method_label}: {metric_name} vs DM\nDM = {unc_text} pc cm$^{{-3}}$")
		else:
			plt.title(rf"{method_label}: {metric_name} vs DM")
		plt.xlabel(rf"DM (pc cm$^{{-3}}$)")
		plt.ylabel(metric_name)
		plt.grid(True, alpha=0.3)
		plt.legend(loc='best')
		plt.tight_layout()
		savefig(run_dir / f"{run_prefix}_{metric_name}_v_DM.png", dpi=150, bbox_inches='tight')
		plt.close()

		# I profile at best DM
		time_series = np.nansum(dedispersed_i, axis=0)
		plt.figure(figsize=pub_figsize(height_ratio=0.6))
		plt.plot(time_series, color='k', linewidth=1.3)
		plt.xlabel('Time index')
		plt.ylabel('Stokes I (arb.)')
		plt.title(f"{method_label}: I at best DM")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		savefig(run_dir / f"{run_prefix}_I_at_max.png", dpi=150, bbox_inches='tight')
		plt.close()

		with open(run_dir / f"{run_prefix}_summaryfile.txt", "w") as summary_file:
			summary_file.write(f"//begin {run_prefix} summary//\n/*\n")
			summary_file.write(f"Method: {method_label}\n")
			summary_file.write(f"Metric name: {metric_name}\n")
			summary_file.write(f"Input DM: {self.input_dm}\n")
			summary_file.write(f"Best metric index: {max_idx}\n")
			summary_file.write(f"Best DM: {optimal_dm}\n")
			summary_file.write(f"Best metric: {optimal_metric}\n")
			if uncertainty is not None:
				summary_file.write(f"Uncertainty method: {uncertainty.get('uncertainty_method', 'unknown')}\n")
				summary_file.write(f"Uncertainty lower DM: {uncertainty.get('uncertainty_low_dm', 'unknown')}\n")
				summary_file.write(f"Uncertainty upper DM: {uncertainty.get('uncertainty_high_dm', 'unknown')}\n")
				summary_file.write(f"Uncertainty -DM: {uncertainty.get('uncertainty_minus', 'unknown')}\n")
				summary_file.write(f"Uncertainty +DM: {uncertainty.get('uncertainty_plus', 'unknown')}\n")
			if self.nonshrine_kc_smooth:
				summary_file.write(f"kc smoothing enabled: True\n")
				if self._nonshrine_resolved_kc is not None:
					summary_file.write(f"kc: {self._nonshrine_resolved_kc}\n")
			else:
				summary_file.write(f"kc smoothing enabled: False\n")
			summary_file.write("*/\n//end summary//\n")

		with open(run_dir / "DM.txt", "w") as dm_file:
			dm_file.write(str(max_idx))

		return run_dir


