"""
DMOptimiser: coordinator class for DM correction optimisation.

This class wires together the stateless module functions into a single
object that holds the data arrays and shared configuration.  Each public
method delegates to the appropriate module; no algorithm logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import dedispersion as _dedisp
from . import diagnostics as _diag
from . import noise as _noise
from . import peaks as _peaks
from . import plotting as _plotting
from . import polarisation as _pol
from . import shrine as _shrine
from . import uncertainty as _unc
from frbop.utils.plotting import savefig_rasterized


class DMOptimiser:
    """
    Optimise the dispersion measure (DM) of Fast Radio Burst data using
    multiple methods and compare their results.

    Parameters
    ----------
    stokes_i:
        2-D Stokes-I dynamic spectrum (freq × time).
    freq_mhz:
        Channel frequencies in MHz (must be sorted ascending).
    time_ms:
        Time samples in milliseconds.
    stokes_q, stokes_u:
        Optional Stokes Q/U arrays; required for PA and L/I methods.
    reference_freq:
        Dedispersion reference frequency (MHz).  Defaults to the highest
        frequency in *freq_mhz*.
    input_dm:
        DM already applied to the input data (pc cm⁻³).
    dedisp_mode:
        ``'expand'`` (fill edges with noise, default) or ``'crop'``.
    pa_fit_degree:
        Polynomial degree for the PA vs time fit (default: 1).
    pa_weight_strength:
        Exponent applied to normalised PA fit weights (default: 1.0).
    pa_fit_post_peak_only:
        Restrict PA fitting to samples at or after the Stokes-I peak.
    nonshrine_kc_smooth:
        Apply SHRINE kc low-pass smoothing inside non-SHRINE methods.
    nonshrine_shrine_like_errors:
        Use SHRINE-style relative-uncertainty error bars even when kc
        smoothing is disabled.
    nonshrine_kc:
        Fixed kc value for non-SHRINE smoothing (auto-detected when None).
    li_i_sigma_cut:
        Stokes-I sigma threshold for L/I masking (default: 2.0).
    debias_linear:
        Apply Rice-distribution debiasing to linear polarisation.
    random_seed:
        Seed for reproducible noise fill.
    """

    def __init__(
        self,
        stokes_i: np.ndarray,
        freq_mhz: np.ndarray,
        time_ms: np.ndarray,
        stokes_q: Optional[np.ndarray] = None,
        stokes_u: Optional[np.ndarray] = None,
        reference_freq: Optional[float] = None,
        input_dm: float = 0.0,
        dedisp_mode: str = "expand",
        pa_fit_degree: int = 1,
        pa_weight_strength: float = 1.0,
        pa_fit_post_peak_only: bool = False,
        nonshrine_kc_smooth: bool = False,
        nonshrine_shrine_like_errors: bool = False,
        nonshrine_kc_minimise_uncertainty: bool = False,
        nonshrine_kc: Optional[int] = None,
        li_i_sigma_cut: float = 2.0,
        debias_linear: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        # ---- validation ----
        if pa_weight_strength <= 0:
            raise ValueError("pa_weight_strength must be positive")
        if li_i_sigma_cut <= 0:
            raise ValueError("li_i_sigma_cut must be positive")
        if nonshrine_kc is not None and int(nonshrine_kc) <= 0:
            raise ValueError("nonshrine_kc must be positive")

        self.stokes_i = stokes_i
        self.stokes_q = stokes_q
        self.stokes_u = stokes_u
        self.freq_mhz = freq_mhz
        self.time_ms = time_ms
        self.n_freq, self.n_time = stokes_i.shape
        self.reference_freq = float(reference_freq) if reference_freq is not None else float(np.max(freq_mhz))
        self.input_dm = float(input_dm)
        self.dedisp_mode = dedisp_mode
        self.pa_fit_degree = int(pa_fit_degree)
        self.pa_weight_strength = float(pa_weight_strength)
        self.pa_fit_post_peak_only = bool(pa_fit_post_peak_only)
        self.nonshrine_kc_smooth = bool(nonshrine_kc_smooth)
        self.nonshrine_shrine_like_errors = bool(nonshrine_shrine_like_errors)
        self.use_shrine_like_uncertainty = bool(nonshrine_kc_smooth or nonshrine_shrine_like_errors)
        self.li_i_sigma_cut = float(li_i_sigma_cut)
        self.debias_linear = bool(debias_linear)
        self.rng = np.random.default_rng(random_seed)

        # ---- pre-computed noise statistics (full array, not sliced) ----
        full_ts = np.mean(stokes_i, axis=0)
        self.full_i_noise_median, self.full_i_noise_std = _noise.noise_stats_from_series(full_ts)

        if stokes_q is not None and stokes_u is not None:
            full_L = np.sqrt(stokes_q ** 2 + stokes_u ** 2)
            self.full_L_time = np.mean(full_L, axis=0)
            self.full_L_noise_median, self.full_L_noise_std = _noise.noise_stats_from_series(
                self.full_L_time
            )
            n_edge = max(1, int(0.05 * stokes_q.shape[1]))
            self.full_q_noise_rms = np.std(stokes_q[:, :n_edge], axis=1, keepdims=True)
            self.full_u_noise_rms = np.std(stokes_u[:, :n_edge], axis=1, keepdims=True)
            _, self.full_q_time_noise_std = _noise.noise_stats_from_series(np.mean(stokes_q, axis=0))
            _, self.full_u_time_noise_std = _noise.noise_stats_from_series(np.mean(stokes_u, axis=0))
        else:
            self.full_L_time = None
            self.full_L_noise_median = None
            self.full_L_noise_std = None
            self.full_q_noise_rms = None
            self.full_u_noise_rms = None
            self.full_q_time_noise_std = None
            self.full_u_time_noise_std = None

        # ---- SHRINE kc resolver (reset at the start of each optimisation) ----
        self._kc_resolver = _shrine.KcResolver(
            fixed_kc=nonshrine_kc,
            use_minimise_uncertainty=bool(nonshrine_kc_minimise_uncertainty),
        )

    # ================================================================
    # Private helpers
    # ================================================================

    def _noise_ref(self, data: np.ndarray) -> np.ndarray:
        """Return the full Stokes array that owns *data*'s frequency channels."""
        for full in (self.stokes_i, self.stokes_q, self.stokes_u):
            if full is not None and np.shares_memory(data, full):
                return full
        return self.stokes_i

    def _dedisperse(
        self,
        data: np.ndarray,
        dm: float,
        output_size: Optional[int] = None,
    ) -> np.ndarray:
        return _dedisp.dedisperse(
            data, dm,
            freq_mhz=self.freq_mhz,
            time_ms=self.time_ms,
            reference_freq=self.reference_freq,
            rng=self.rng,
            noise_ref=self._noise_ref(data),
            input_dm=self.input_dm,
            output_size=output_size,
            mode=self.dedisp_mode,
        )

    def _max_output_size(self, data: np.ndarray, dm_range: Tuple[float, float]) -> int:
        return _dedisp.max_output_size_for_dm_range(
            data.shape[1], self.freq_mhz, self.time_ms,
            self.reference_freq, dm_range, self.input_dm,
        )

    def _delay_samples(self, dm: float) -> np.ndarray:
        return _dedisp.get_delay_samples(
            dm, self.freq_mhz, self.time_ms, self.reference_freq, self.input_dm
        )

    def _qu_rms(self, data_q: np.ndarray, data_u: np.ndarray):
        """Return per-channel Q/U RMS from pre-computed full-array statistics."""
        if data_q.ndim == 1:
            if self.full_q_time_noise_std is not None:
                return float(self.full_q_time_noise_std), float(self.full_u_time_noise_std)
            return _pol.qu_noise_rms(data_q, data_u)
        if (self.full_q_noise_rms is not None
                and self.full_q_noise_rms.shape[0] == data_q.shape[0]):
            return self.full_q_noise_rms, self.full_u_noise_rms
        return _pol.qu_noise_rms(data_q, data_u)

    def _maybe_smooth(self, data_i, data_q, data_u):
        return _shrine.maybe_kc_smooth(
            data_i, data_q, data_u,
            kc_resolver=self._kc_resolver,
            enabled=self.nonshrine_kc_smooth,
        )

    def _pa_series(self, data_q, data_u, data_i=None):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.pa_series_deg(
            data_q, data_u, data_i, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            pa_fit_post_peak_only=self.pa_fit_post_peak_only,
            debias=self.debias_linear,
        )

    def _pa_smoothed_and_fit(self, data_q, data_u, data_i, time_axis):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.get_pa_smoothed_and_fit(
            data_q, data_u, data_i, time_axis, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            pa_fit_post_peak_only=self.pa_fit_post_peak_only,
            pa_fit_degree=self.pa_fit_degree,
            pa_weight_strength=self.pa_weight_strength,
            debias=self.debias_linear,
        )

    def _pa_shrine_smoothed_and_fit(self, data_q, data_u, data_i, time_axis, force_kc=None):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.get_pa_shrine_smoothed_and_fit(
            data_q, data_u, data_i, time_axis, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            pa_fit_post_peak_only=self.pa_fit_post_peak_only,
            pa_fit_degree=self.pa_fit_degree,
            pa_weight_strength=self.pa_weight_strength,
            apply_kc_lowpass_fn=_shrine.apply_kc_lowpass_2d,
            resolve_nonshrine_kc_fn=self._kc_resolver.resolve,
            debias=self.debias_linear,
            force_kc=force_kc,
        )

    def _pa_slope(self, data_q, data_u, data_i, time_axis):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.pa_slope_metric(
            data_q, data_u, data_i, time_axis, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            pa_fit_post_peak_only=self.pa_fit_post_peak_only,
            pa_fit_degree=self.pa_fit_degree,
            pa_weight_strength=self.pa_weight_strength,
            debias=self.debias_linear,
        )

    def _pa_slope_shrine(self, data_q, data_u, data_i, time_axis):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.pa_slope_metric_shrine(
            data_q, data_u, data_i, time_axis, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            pa_fit_post_peak_only=self.pa_fit_post_peak_only,
            pa_fit_degree=self.pa_fit_degree,
            pa_weight_strength=self.pa_weight_strength,
            apply_kc_lowpass_fn=_shrine.apply_kc_lowpass_2d,
            resolve_nonshrine_kc_fn=self._kc_resolver.resolve,
            debias=self.debias_linear,
        )

    def _li_metric(self, data_q, data_u, data_i, mode):
        q_rms, u_rms = self._qu_rms(data_q, data_u)
        return _pol.linear_to_stokes_i_metric(
            data_q, data_u, data_i, q_rms, u_rms,
            noise_median_i=self.full_i_noise_median,
            noise_std_i=self.full_i_noise_std,
            li_i_sigma_cut=self.li_i_sigma_cut,
            debias=self.debias_linear,
            mode=mode,
        )

    def _unc_from_shrine_relative(self, dm_values, metric_values, profiles, kc=None):
        return _unc.from_shrine_relative(
            dm_values, metric_values, profiles,
            shrine_get_kc=_shrine.get_kc,
            shrine_lowpass_smooth=_shrine.lowpass_smooth,
            shrine_get_ranges_above_max=_shrine.get_ranges_above_max,
            shrine_uncertainty_calc=_shrine.uncertainty_calc,
            kc=kc,
        )

    def _uncertainty(
        self,
        dm_values, metric_values, best_idx,
        profiles=None, method="half-prominence", kc=None,
    ) -> Dict:
        """Dispatch to the appropriate uncertainty estimator."""
        if self.use_shrine_like_uncertainty and profiles is not None and method != "snr-drop":
            unc = self._unc_from_shrine_relative(dm_values, metric_values, profiles, kc=kc)
        elif method == "snr-drop":
            unc = _unc.from_snr_drop(dm_values, metric_values, best_idx, drop=1.0)
        else:
            unc = _unc.from_half_prominence(dm_values, metric_values, best_idx)
        return unc

    def _save_nonshrine_outputs(
        self,
        run_prefix: str,
        method_label: str,
        dm_values: np.ndarray,
        metric_values: np.ndarray,
        metric_name: str,
        dedispersed_i: np.ndarray,
        optimal_dm: float,
        optimal_metric: float,
        uncertainty: Optional[Dict] = None,
    ) -> Path:
        """Write log files and diagnostic plots for a non-SHRINE method run."""
        run_dir = Path("shrine_logs") / run_prefix
        run_dir.mkdir(parents=True, exist_ok=True)

        max_idx = int(np.argmax(metric_values))
        np.save(run_dir / f"{run_prefix}_DMs.npy", dm_values)
        np.savetxt(run_dir / f"{run_prefix}_{metric_name}.dat", np.asarray(metric_values, dtype=float))
        np.save(run_dir / f"{run_prefix}_I_at_max.npy", dedispersed_i)

        # Metric-vs-DM plot
        plt.figure(figsize=(8, 4))
        plt.plot(dm_values, metric_values, "-", color="tab:blue", linewidth=1.8)
        if uncertainty is not None:
            low_dm = uncertainty.get("uncertainty_low_dm")
            high_dm = uncertainty.get("uncertainty_high_dm")
            shade_low = float(np.min(dm_values)) if low_dm is None else float(low_dm)
            shade_high = float(np.max(dm_values)) if high_dm is None else float(high_dm)
            if shade_low <= shade_high:
                plt.axvspan(shade_low, shade_high, color="tab:orange", alpha=0.18, label="DM uncertainty")
        plt.axvline(optimal_dm, color="tab:red", linestyle="--", linewidth=1.2,
                    label=f"max DM={optimal_dm:.6f}")
        if uncertainty is not None:
            unc_text = _unc.format_uncertainty(
                optimal_dm, uncertainty.get("uncertainty_minus"), uncertainty.get("uncertainty_plus")
            )
            plt.title(f"{method_label}: {metric_name} vs DM\nDM = {unc_text} pc cm⁻³")
        else:
            plt.title(f"{method_label}: {metric_name} vs DM")
        plt.xlabel("DM (pc cm⁻³)")
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        savefig_rasterized(run_dir / f"{run_prefix}_{metric_name}_v_DM.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Profile at best DM
        ts = np.mean(dedispersed_i, axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(ts, color="k", linewidth=1.3)
        plt.xlabel("Time index")
        plt.ylabel("Stokes I (arb.)")
        plt.title(f"{method_label}: I at best DM")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        savefig_rasterized(run_dir / f"{run_prefix}_I_at_max.png", dpi=150, bbox_inches="tight")
        plt.close()

        with open(run_dir / f"{run_prefix}_summaryfile.txt", "w") as f:
            f.write(f"//begin {run_prefix} summary//\n/*\n")
            f.write(f"Method: {method_label}\n")
            f.write(f"Metric name: {metric_name}\n")
            f.write(f"Input DM: {self.input_dm}\n")
            f.write(f"Best metric index: {max_idx}\n")
            f.write(f"Best DM: {optimal_dm}\n")
            f.write(f"Best metric: {optimal_metric}\n")
            if uncertainty:
                for key in ("uncertainty_method", "uncertainty_low_dm",
                            "uncertainty_high_dm", "uncertainty_minus", "uncertainty_plus"):
                    f.write(f"{key}: {uncertainty.get(key, 'unknown')}\n")
            f.write(f"kc smoothing enabled: {self.nonshrine_kc_smooth}\n")
            if self.nonshrine_kc_smooth and self._kc_resolver.resolved is not None:
                f.write(f"kc: {self._kc_resolver.resolved}\n")
            f.write("*/\n//end summary//\n")

        with open(run_dir / "DM.txt", "w") as f:
            f.write(str(max_idx))

        return run_dir

    def _run_shrine(self, script_name, run_prefix, dm_values, i_data,
                    include_input_dm=False, force_kc=None, save_all=True) -> Path:
        return _shrine.run_shrine_script(
            script_name=script_name,
            run_prefix=run_prefix,
            dm_values=dm_values,
            i_data=i_data,
            time_ms=self.time_ms,
            input_dm=self.input_dm,
            include_input_dm=include_input_dm,
            force_kc=force_kc,
            save_all=save_all,
        )

    def _li_uncertainty(self, dm_values, li_values, max_idx, profiles=None, kc=None):
        """L/I uncertainty with fallback chain."""
        if self.use_shrine_like_uncertainty and profiles is not None:
            unc = self._unc_from_shrine_relative(dm_values, li_values, profiles, kc=kc)
            if unc.get("uncertainty_low_dm") is None or unc.get("uncertainty_high_dm") is None:
                unc = _unc.from_local_quadratic(dm_values, li_values, max_idx)
            if unc.get("uncertainty_low_dm") is None or unc.get("uncertainty_high_dm") is None:
                unc = _unc.from_half_prominence(dm_values, li_values, max_idx)
        else:
            unc = _unc.from_half_prominence(dm_values, li_values, max_idx)
        return _unc.clamp_to_dm_bounds(
            float(dm_values[max_idx]), unc, dm_values,
            fill_missing_with_bounds=not self.use_shrine_like_uncertainty,
        )

    def _time_axis(self, n_time_out: int) -> np.ndarray:
        dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
        return self.time_ms[0] + np.arange(n_time_out) * dt

    def _nonshrine_ref_profile(self, data_i, data_q, data_u) -> np.ndarray:
        if data_i is not None:
            return np.nanmean(data_i, axis=0)
        if data_q is not None and data_u is not None:
            return np.nanmean(np.sqrt(data_q ** 2 + data_u ** 2), axis=0)
        if data_q is not None:
            return np.nanmean(data_q, axis=0)
        return np.nanmean(data_u, axis=0)

    def _li_ref_profile(self, data_q, data_u, data_i) -> np.ndarray:
        profile = self._nonshrine_ref_profile(data_i, data_q, data_u)
        profile = np.asarray(profile, dtype=float)
        profile[~np.isfinite(profile)] = 0.0
        return profile

    # ================================================================
    # Public API — dedispersion / DM utilities
    # ================================================================

    def dedisperse(
        self,
        data: np.ndarray,
        dm: float,
        output_size: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Apply dispersion correction to *data* (freq × time)."""
        if mode is None:
            mode = self.dedisp_mode
        return _dedisp.dedisperse(
            data, dm,
            freq_mhz=self.freq_mhz,
            time_ms=self.time_ms,
            reference_freq=self.reference_freq,
            rng=self.rng,
            noise_ref=self._noise_ref(data),
            input_dm=self.input_dm,
            output_size=output_size,
            mode=mode,
        )

    def recommend_lowest_dm_step(self, samples_per_step: float = 1.0) -> float:
        """Minimum DM grid step that shifts the band by *samples_per_step* samples."""
        return _dedisp.recommend_lowest_dm_step(
            self.freq_mhz, self.time_ms, self.reference_freq, samples_per_step
        )

    # ================================================================
    # Public API — peak handling
    # ================================================================

    def separate_peaks(
        self,
        min_separation_ms: float = 1.0,
        diagnostics_path: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        """Automatically detect pulse components and return their index ranges."""
        return _peaks.separate_peaks(
            self.stokes_i, self.time_ms,
            min_separation_ms=min_separation_ms,
            diagnostics_path=diagnostics_path,
        )

    def select_peaks_manual(self) -> List[Tuple[int, int]]:
        """Interactively select peak bounds by clicking on the pulse profile."""
        return _peaks.select_peaks_manual(self.stokes_i, self.time_ms)

    # ================================================================
    # Public API — individual optimisation methods
    # ================================================================

    def optimise_dm_structure(
        self,
        dm_range: Tuple[float, float],
        data: Optional[np.ndarray] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        label: str = "frb",
        segment: Optional[str] = None,
    ) -> Dict:
        """Optimise DM using SHRINE structure maximisation."""
        if data is None:
            data = self.stokes_i

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        output_size = self._max_output_size(data, dm_range)
        i_data = np.zeros((len(dm_values), output_size))
        for i, dm in enumerate(dm_values):
            i_data[i] = np.nanmean(self._dedisperse(data, dm, output_size), axis=0)

        run_prefix = f"{label}_{segment or 'segment'}_structure"
        run_dir = self._run_shrine("maximise_structure.py", run_prefix, dm_values, i_data,
                                   include_input_dm=True, save_all=True)
        structure_values = np.loadtxt(run_dir / f"{run_prefix}_SPs.dat")

        kc = None
        summary_path = run_dir / f"{run_prefix}_structure_summaryfile.txt"
        if summary_path.exists():
            with open(summary_path) as f:
                for line in f:
                    if line.strip().startswith("kc:"):
                        try:
                            kc = int(line.split(":", 1)[1].strip())
                        except Exception:
                            pass
                        break

        max_idx = int(np.argmax(structure_values))
        optimal_dm = float(dm_values[max_idx])
        unc = _unc.from_shrine_outputs(
            dm_values, structure_values, run_dir, run_prefix, max_idx,
            shrine_get_ranges_above_max=_shrine.get_ranges_above_max,
        )
        return {
            "dm": optimal_dm,
            "metric": float(structure_values[max_idx]),
            "dedispersed": self._dedisperse(data, optimal_dm),
            "method": "Structure Maximising (SHRINE)",
            "kc": kc,
            "dm_values": dm_values.copy(),
            "metric_values": np.asarray(structure_values, dtype=float).copy(),
            **unc,
        }

    def optimise_dm_snr(
        self,
        dm_range: Tuple[float, float],
        data: Optional[np.ndarray] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        label: str = "frb",
        segment: Optional[str] = None,
    ) -> Dict:
        """Optimise DM using SHRINE S/N maximisation."""
        if data is None:
            data = self.stokes_i

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        output_size = self._max_output_size(data, dm_range)
        i_data = np.zeros((len(dm_values), output_size))
        for i, dm in enumerate(dm_values):
            i_data[i] = np.nanmean(self._dedisperse(data, dm, output_size), axis=0)

        run_prefix = f"{label}_{segment or 'segment'}_snr"
        run_dir = self._run_shrine("maximise_sn.py", run_prefix, dm_values, i_data,
                                   include_input_dm=False, save_all=True)
        sn_path = run_dir / f"{run_prefix}_SNs.dat"
        if not sn_path.exists():
            raise FileNotFoundError(f"Expected SHRINE S/N output not found: {sn_path}")
        snr_values = np.loadtxt(sn_path)

        max_idx = int(np.argmax(snr_values))
        optimal_dm = float(dm_values[max_idx])
        unc = _unc.from_snr_drop(dm_values, snr_values, max_idx, drop=1.0)
        return {
            "dm": optimal_dm,
            "metric": float(snr_values[max_idx]),
            "dedispersed": self._dedisperse(data, optimal_dm),
            "method": "S/N Maximising (SHRINE)",
            "dm_values": dm_values.copy(),
            "metric_values": np.asarray(snr_values, dtype=float).copy(),
            **unc,
        }

    def optimise_dm_pa_slope(
        self,
        dm_range: Tuple[float, float],
        data_q: Optional[np.ndarray] = None,
        data_u: Optional[np.ndarray] = None,
        data_i: Optional[np.ndarray] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        label: str = "frb",
        segment: Optional[str] = None,
    ) -> Dict:
        """Optimise DM using weighted PA slope magnitude."""
        if data_q is None or data_u is None:
            raise ValueError("Stokes Q and U required for PA slope optimisation")
        self._kc_resolver.reset()

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        base = data_i if data_i is not None else data_q
        output_size = self._max_output_size(base, dm_range)
        time_axis = self._time_axis(output_size)
        pa_values = np.zeros(len(dm_values))
        unc_profiles = np.zeros((len(dm_values), output_size))

        for i, dm in enumerate(dm_values):
            dq = self._dedisperse(data_q, dm, output_size)
            du = self._dedisperse(data_u, dm, output_size)
            di = self._dedisperse(data_i, dm, output_size) if data_i is not None else None
            sm_i, sm_q, sm_u = self._maybe_smooth(di, dq, du)
            unc_profiles[i] = self._nonshrine_ref_profile(sm_i, sm_q, sm_u)
            pa_values[i] = self._pa_slope(sm_q, sm_u, sm_i, time_axis)

        max_idx = int(np.argmax(pa_values))
        optimal_dm = float(dm_values[max_idx])

        best_dq = self._dedisperse(data_q, optimal_dm, output_size)
        best_du = self._dedisperse(data_u, optimal_dm, output_size)
        best_di = self._dedisperse(data_i, optimal_dm, output_size) if data_i is not None else None
        _, best_sm_q, best_sm_u = self._maybe_smooth(best_di, best_dq, best_du)
        pa_sm, fit_line = self._pa_smoothed_and_fit(best_sm_q, best_sm_u, best_di or best_dq, time_axis)
        best_pa_deg = self._pa_series(best_sm_q, best_sm_u, best_di)

        unc = self._uncertainty(dm_values, pa_values, max_idx, profiles=unc_profiles,
                                kc=self._kc_resolver.resolved)
        display = self._dedisperse(data_i if data_i is not None else data_q, optimal_dm, output_size)
        i_for_logs = display if data_i is not None else self._dedisperse(self.stokes_i, optimal_dm, output_size)

        run_prefix = f"{label}_{segment or 'segment1'}_pa_slope"
        run_dir = self._save_nonshrine_outputs(
            run_prefix, "PA Slope Maximising", dm_values, pa_values, "PA_Slope",
            i_for_logs, optimal_dm, float(pa_values[max_idx]), unc,
        )
        return {
            "dm": optimal_dm, "metric": float(pa_values[max_idx]),
            "dedispersed": display, "method": "PA Slope Maximising",
            "pa_plot_kind": "raw", "pa_plot_time": time_axis.copy(),
            "pa_plot_series": best_pa_deg.copy(), "pa_plot_smooth": pa_sm.copy(),
            "pa_plot_fit": fit_line.copy(),
            "run_dir": str(run_dir),
            "dm_values": dm_values.copy(), "metric_values": pa_values.copy(),
            **unc,
        }

    def optimise_dm_pa_slope_shrine(
        self,
        dm_range: Tuple[float, float],
        data_q: Optional[np.ndarray] = None,
        data_u: Optional[np.ndarray] = None,
        data_i: Optional[np.ndarray] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        label: str = "frb",
        segment: Optional[str] = None,
    ) -> Dict:
        """Optimise DM using SHRINE-smoothed PA slope."""
        if data_q is None or data_u is None:
            raise ValueError("Stokes Q and U required for PA slope optimisation")
        self._kc_resolver.reset()

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        base = data_i if data_i is not None else data_q
        output_size = self._max_output_size(base, dm_range)
        time_axis = self._time_axis(output_size)
        pa_shrine_values = np.zeros(len(dm_values))
        unc_profiles = np.zeros((len(dm_values), output_size))

        for i, dm in enumerate(dm_values):
            dq = self._dedisperse(data_q, dm, output_size)
            du = self._dedisperse(data_u, dm, output_size)
            di = self._dedisperse(data_i, dm, output_size) if data_i is not None else None
            sm_i, sm_q, sm_u = self._maybe_smooth(di, dq, du)
            unc_profiles[i] = self._nonshrine_ref_profile(sm_i, sm_q, sm_u)
            pa_shrine_values[i] = self._pa_slope_shrine(sm_q, sm_u, sm_i, time_axis)

        max_idx = int(np.argmax(pa_shrine_values))
        optimal_dm = float(dm_values[max_idx])

        best_dq = self._dedisperse(data_q, optimal_dm, output_size)
        best_du = self._dedisperse(data_u, optimal_dm, output_size)
        best_di = self._dedisperse(data_i, optimal_dm, output_size) if data_i is not None else None
        _, best_sm_q, best_sm_u = self._maybe_smooth(best_di, best_dq, best_du)
        pa_sm, fit_line = self._pa_shrine_smoothed_and_fit(
            best_sm_q, best_sm_u, best_di or best_dq, time_axis,
            force_kc=self._kc_resolver.resolved,
        )
        best_pa_deg = self._pa_series(best_sm_q, best_sm_u, best_di)

        unc = self._uncertainty(dm_values, pa_shrine_values, max_idx, profiles=unc_profiles,
                                kc=self._kc_resolver.resolved)
        display = self._dedisperse(data_i if data_i is not None else data_q, optimal_dm, output_size)
        i_for_logs = display if data_i is not None else self._dedisperse(self.stokes_i, optimal_dm, output_size)

        run_prefix = f"{label}_{segment or 'segment1'}_pa_slope_shrine"
        run_dir = self._save_nonshrine_outputs(
            run_prefix, "PA Slope Maximising (SHRINE PA)", dm_values, pa_shrine_values,
            "PA_Slope_SHRINE", i_for_logs, optimal_dm, float(pa_shrine_values[max_idx]), unc,
        )
        return {
            "dm": optimal_dm, "metric": float(pa_shrine_values[max_idx]),
            "dedispersed": display, "method": "PA Slope Maximising (SHRINE PA)",
            "kc": self._kc_resolver.resolved,
            "pa_plot_kind": "shrine", "pa_plot_time": time_axis.copy(),
            "pa_plot_series": best_pa_deg.copy(), "pa_plot_smooth": pa_sm.copy(),
            "pa_plot_fit": fit_line.copy(),
            "run_dir": str(run_dir),
            "dm_values": dm_values.copy(), "metric_values": pa_shrine_values.copy(),
            **unc,
        }

    def optimise_dm_linear_to_stokes_i(
        self,
        dm_range: Tuple[float, float],
        data_q: Optional[np.ndarray] = None,
        data_u: Optional[np.ndarray] = None,
        data_i: Optional[np.ndarray] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        mode: str = "peak",
        label: str = "frb",
        segment: Optional[str] = None,
    ) -> Dict:
        """Optimise DM using L/I maximisation."""
        if data_q is None or data_u is None or data_i is None:
            raise ValueError("Stokes I, Q, and U required for L/I optimisation")
        self._kc_resolver.reset()

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        output_size = self._max_output_size(data_i, dm_range)
        li_values = np.zeros(len(dm_values))
        unc_profiles = np.zeros((len(dm_values), output_size))

        for i, dm in enumerate(dm_values):
            dq = self._dedisperse(data_q, dm, output_size)
            du = self._dedisperse(data_u, dm, output_size)
            di = self._dedisperse(data_i, dm, output_size)
            sm_i, sm_q, sm_u = self._maybe_smooth(di, dq, du)
            unc_profiles[i] = self._li_ref_profile(sm_q, sm_u, sm_i)
            li_values[i] = self._li_metric(sm_q, sm_u, sm_i, mode)

        max_idx = int(np.argmax(li_values))
        optimal_dm = float(dm_values[max_idx])
        unc = self._li_uncertainty(dm_values, li_values, max_idx, unc_profiles,
                                   kc=self._kc_resolver.resolved)
        dedispersed_i = self._dedisperse(data_i, optimal_dm)

        mode_labels = {"peak": "L/I Maximising (peak)", "mean": "L/I Maximising (mean)",
                       "max": "L/I Maximising (max)"}
        method_label = mode_labels.get(mode, f"L/I Maximising ({mode})")
        run_prefix = f"{label}_{segment or 'segment'}_l_i_{mode}"
        run_dir = self._save_nonshrine_outputs(
            run_prefix, method_label, dm_values, li_values,
            f"L_over_I_{mode}", dedispersed_i, optimal_dm, float(li_values[max_idx]), unc,
        )
        return {
            "dm": optimal_dm, "metric": float(li_values[max_idx]),
            "dedispersed": dedispersed_i, "method": method_label,
            "run_dir": str(run_dir),
            "dm_values": dm_values.copy(), "metric_values": li_values.copy(),
            **unc,
        }

    # ================================================================
    # Public API — batch comparison
    # ================================================================

    def compare_methods(
        self,
        dm_range: Tuple[float, float],
        peak_region: Optional[Tuple[int, int]] = None,
        n_points: int = 200,
        dm_step: Optional[float] = None,
        segment_tag: str = "segment1",
        label: str = "frb",
        selected_methods: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run all (or a subset of) optimisation methods on the same data region
        using a shared DM sweep for efficiency.

        Returns a dict keyed by method name with per-method result dicts.
        """
        if peak_region is not None:
            data = self.stokes_i[:, peak_region[0]:peak_region[1]]
            data_q = None if self.stokes_q is None else self.stokes_q[:, peak_region[0]:peak_region[1]]
            data_u = None if self.stokes_u is None else self.stokes_u[:, peak_region[0]:peak_region[1]]
        else:
            data = self.stokes_i
            data_q = self.stokes_q
            data_u = self.stokes_u

        print(f"Comparing methods on DM range [{dm_range[0]:.2f}, {dm_range[1]:.2f}] pc cm⁻³")

        has_qu = data_q is not None and data_u is not None
        all_keys = ["structure", "snr", "pa_slope", "pa_slope_shrine", "l_i_mean"]
        selected = set(all_keys) if selected_methods is None else set(selected_methods)

        run_structure = "structure" in selected
        run_snr = "snr" in selected
        run_pa = "pa_slope" in selected and has_qu
        run_pa_shrine = "pa_slope_shrine" in selected and has_qu
        run_li = "l_i_mean" in selected and has_qu
        run_qu = run_pa or run_pa_shrine or run_li

        if not (run_structure or run_snr or run_qu):
            print("  No methods selected; returning empty results.")
            return {}

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        output_size = self._max_output_size(data, dm_range)
        time_axis = self._time_axis(output_size)
        i_data = np.zeros((len(dm_values), output_size))

        pa_values = np.zeros(len(dm_values)) if run_pa else None
        pa_shrine_values = np.zeros(len(dm_values)) if run_pa_shrine else None
        li_mean_values = np.zeros(len(dm_values)) if run_li else None
        pa_unc_profiles = np.zeros((len(dm_values), output_size)) if run_qu else None
        li_unc_profiles = np.zeros((len(dm_values), output_size)) if run_li else None

        self._kc_resolver.reset()
        print(f"  Shared DM sweep ({len(dm_values)} trials)...")

        for i, dm in enumerate(dm_values):
            if i % 25 == 0:
                print(f"\r    Progress: {i}/{len(dm_values)}", end="", flush=True)
            di = self._dedisperse(data, dm, output_size)
            i_data[i] = np.nanmean(di, axis=0)

            if run_qu:
                dq = self._dedisperse(data_q, dm, output_size)
                du = self._dedisperse(data_u, dm, output_size)
                sm_i, sm_q, sm_u = self._maybe_smooth(di, dq, du)
                if run_pa:
                    pa_values[i] = self._pa_slope(sm_q, sm_u, sm_i, time_axis)
                if run_pa_shrine:
                    pa_shrine_values[i] = self._pa_slope_shrine(sm_q, sm_u, sm_i, time_axis)
                if run_li:
                    li_mean_values[i] = self._li_metric(sm_q, sm_u, sm_i, "mean")
                if pa_unc_profiles is not None:
                    pa_unc_profiles[i] = self._nonshrine_ref_profile(sm_i, sm_q, sm_u)
                if li_unc_profiles is not None:
                    li_unc_profiles[i] = self._li_ref_profile(sm_q, sm_u, sm_i)

        print(f"\r    Progress: {len(dm_values)}/{len(dm_values)}", flush=True)

        results: Dict[str, Dict] = {}

        # ---- Structure ----
        if run_structure:
            print("  Structure Maximising (SHRINE)...")
            run_prefix = f"{label}_{segment_tag}_structure"
            run_dir = self._run_shrine("maximise_structure.py", run_prefix, dm_values, i_data,
                                       include_input_dm=True, save_all=True)
            sv = np.loadtxt(run_dir / f"{run_prefix}_SPs.dat")
            kc = None
            sp = run_dir / f"{run_prefix}_structure_summaryfile.txt"
            if sp.exists():
                with open(sp) as f:
                    for line in f:
                        if line.strip().startswith("kc:"):
                            try:
                                kc = int(line.split(":", 1)[1].strip())
                            except Exception:
                                pass
                            break
            mi = int(np.argmax(sv))
            odm = float(dm_values[mi])
            unc = _unc.from_shrine_outputs(dm_values, sv, run_dir, run_prefix, mi,
                                           shrine_get_ranges_above_max=_shrine.get_ranges_above_max)
            r = {
                "dm": odm, "metric": float(sv[mi]),
                "dedispersed": self._dedisperse(data, odm),
                "method": "Structure Maximising (SHRINE)", "kc": kc,
                "run_dir": str(run_dir),
                "dm_values": dm_values.copy(), "metric_values": np.asarray(sv, dtype=float).copy(),
                **unc,
            }
            if has_qu:
                n_t = r["dedispersed"].shape[1]
                r["dedispersed_q"] = self._dedisperse(data_q, odm, n_t)
                r["dedispersed_u"] = self._dedisperse(data_u, odm, n_t)
            results["structure"] = r

        # ---- S/N ----
        if run_snr:
            print("  S/N Maximising (SHRINE)...")
            run_prefix = f"{label}_{segment_tag}_snr"
            run_dir = self._run_shrine("maximise_sn.py", run_prefix, dm_values, i_data,
                                       include_input_dm=False, save_all=True)
            sn_path = run_dir / f"{run_prefix}_SNs.dat"
            if not sn_path.exists():
                raise FileNotFoundError(f"SHRINE S/N output missing: {sn_path}")
            sv = np.loadtxt(sn_path)
            mi = int(np.argmax(sv))
            odm = float(dm_values[mi])
            unc = _unc.from_snr_drop(dm_values, sv, mi, drop=1.0)
            r = {
                "dm": odm, "metric": float(sv[mi]),
                "dedispersed": self._dedisperse(data, odm),
                "method": "S/N Maximising (SHRINE)",
                "run_dir": str(run_dir),
                "dm_values": dm_values.copy(), "metric_values": np.asarray(sv, dtype=float).copy(),
                **unc,
            }
            if has_qu:
                n_t = r["dedispersed"].shape[1]
                r["dedispersed_q"] = self._dedisperse(data_q, odm, n_t)
                r["dedispersed_u"] = self._dedisperse(data_u, odm, n_t)
            results["snr"] = r

        if not run_qu:
            return results

        print("  PA slope / L/I from shared sweep...")

        # ---- PA slope ----
        if run_pa and pa_values is not None:
            mi = int(np.argmax(pa_values))
            odm = float(dm_values[mi])
            best_di = self._dedisperse(data, odm, output_size)
            best_dq = self._dedisperse(data_q, odm, output_size)
            best_du = self._dedisperse(data_u, odm, output_size)
            _, sm_q, sm_u = self._maybe_smooth(best_di, best_dq, best_du)
            pa_sm, fit_line = self._pa_smoothed_and_fit(sm_q, sm_u, best_di, time_axis)
            best_pa_deg = self._pa_series(sm_q, sm_u, best_di)
            unc = self._uncertainty(dm_values, pa_values, mi, pa_unc_profiles,
                                    kc=self._kc_resolver.resolved)
            run_prefix = f"{label}_{segment_tag}_pa_slope"
            run_dir = self._save_nonshrine_outputs(
                run_prefix, "PA Slope Maximising", dm_values, pa_values, "PA_Slope",
                best_di, odm, float(pa_values[mi]), unc,
            )
            results["pa_slope"] = {
                "dm": odm, "metric": float(pa_values[mi]),
                "dedispersed": best_di, "dedispersed_q": best_dq, "dedispersed_u": best_du,
                "method": "PA Slope Maximising", "pa_plot_kind": "raw",
                "pa_plot_time": time_axis.copy(), "pa_plot_series": best_pa_deg.copy(),
                "pa_plot_smooth": pa_sm.copy(), "pa_plot_fit": fit_line.copy(),
                "run_dir": str(run_dir),
                "dm_values": dm_values.copy(), "metric_values": pa_values.copy(),
                **unc,
            }

        # ---- PA slope (SHRINE) ----
        if run_pa_shrine and pa_shrine_values is not None:
            mi = int(np.argmax(pa_shrine_values))
            odm = float(dm_values[mi])
            best_di = self._dedisperse(data, odm, output_size)
            best_dq = self._dedisperse(data_q, odm, output_size)
            best_du = self._dedisperse(data_u, odm, output_size)
            _, sm_q, sm_u = self._maybe_smooth(best_di, best_dq, best_du)
            pa_sm, fit_line = self._pa_shrine_smoothed_and_fit(
                sm_q, sm_u, best_di, time_axis, force_kc=self._kc_resolver.resolved
            )
            best_pa_deg = self._pa_series(sm_q, sm_u, best_di)
            unc = self._uncertainty(dm_values, pa_shrine_values, mi, pa_unc_profiles,
                                    kc=self._kc_resolver.resolved)
            run_prefix = f"{label}_{segment_tag}_pa_slope_shrine"
            run_dir = self._save_nonshrine_outputs(
                run_prefix, "PA Slope Maximising (SHRINE PA)", dm_values, pa_shrine_values,
                "PA_Slope_SHRINE", best_di, odm, float(pa_shrine_values[mi]), unc,
            )
            results["pa_slope_shrine"] = {
                "dm": odm, "metric": float(pa_shrine_values[mi]),
                "dedispersed": best_di, "dedispersed_q": best_dq, "dedispersed_u": best_du,
                "method": "PA Slope Maximising (SHRINE PA)",
                "kc": self._kc_resolver.resolved,
                "pa_plot_kind": "shrine",
                "pa_plot_time": time_axis.copy(), "pa_plot_series": best_pa_deg.copy(),
                "pa_plot_smooth": pa_sm.copy(), "pa_plot_fit": fit_line.copy(),
                "run_dir": str(run_dir),
                "dm_values": dm_values.copy(), "metric_values": pa_shrine_values.copy(),
                **unc,
            }

        # ---- L/I mean ----
        if run_li and li_mean_values is not None:
            mi = int(np.argmax(li_mean_values))
            odm = float(dm_values[mi])
            best_di = self._dedisperse(data, odm, output_size)
            best_dq = self._dedisperse(data_q, odm, output_size)
            best_du = self._dedisperse(data_u, odm, output_size)
            unc = self._li_uncertainty(dm_values, li_mean_values, mi, li_unc_profiles,
                                       kc=self._kc_resolver.resolved)
            run_prefix = f"{label}_{segment_tag}_l_i_mean"
            run_dir = self._save_nonshrine_outputs(
                run_prefix, "L/I Maximising (mean)", dm_values, li_mean_values,
                "L_over_I_mean", best_di, odm, float(li_mean_values[mi]), unc,
            )
            results["l_i_mean"] = {
                "dm": odm, "metric": float(li_mean_values[mi]),
                "dedispersed": best_di, "dedispersed_q": best_dq, "dedispersed_u": best_du,
                "method": "L/I Maximising (mean)",
                "run_dir": str(run_dir),
                "dm_values": dm_values.copy(), "metric_values": li_mean_values.copy(),
                **unc,
            }

        return results

    def scan_dm_space(
        self,
        dm_range: Tuple[float, float],
        n_points: int = 100,
        data: Optional[np.ndarray] = None,
        dm_step: Optional[float] = None,
        data_q: Optional[np.ndarray] = None,
        data_u: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Scan DM space and return metric arrays for all methods."""
        if data is None:
            data = self.stokes_i
        if data_q is None:
            data_q = self.stokes_q
        if data_u is None:
            data_u = self.stokes_u
        self._kc_resolver.reset()

        dm_values = _dedisp.build_dm_values(dm_range, n_points, dm_step)
        output_size = self._max_output_size(data, dm_range)
        i_data = np.zeros((len(dm_values), output_size))
        has_qu = data_q is not None and data_u is not None
        pa_values = np.zeros(len(dm_values)) if has_qu else None
        pa_shrine_values = np.zeros(len(dm_values)) if has_qu else None
        li_values = np.zeros(len(dm_values)) if has_qu else None

        dt = float(np.median(np.diff(self.time_ms))) if len(self.time_ms) > 1 else 1.0
        time_axis = np.arange(output_size) * dt

        print(f"Scanning {len(dm_values)} DM values...")
        for i, dm in enumerate(dm_values):
            if i % 20 == 0:
                print(f"\r  Progress: {i}/{len(dm_values)}", end="", flush=True)
            di = self._dedisperse(data, dm, output_size)
            i_data[i] = np.nanmean(di, axis=0)
            if has_qu:
                dq = self._dedisperse(data_q, dm, output_size)
                du = self._dedisperse(data_u, dm, output_size)
                sm_i, sm_q, sm_u = self._maybe_smooth(di, dq, du)
                pa_values[i] = self._pa_slope(sm_q, sm_u, sm_i, time_axis)
                pa_shrine_values[i] = self._pa_slope_shrine(sm_q, sm_u, sm_i, time_axis)
                li_values[i] = self._li_metric(sm_q, sm_u, sm_i, "mean")

        run_tag = f"scan_{int(round(dm_values[0]*1000))}_{int(round(dm_values[-1]*1000))}_{len(dm_values)}"

        rdir_s = self._run_shrine("maximise_structure.py", f"{run_tag}_structure",
                                  dm_values, i_data, include_input_dm=True, save_all=True)
        sv = np.loadtxt(rdir_s / f"{run_tag}_structure_SPs.dat")

        rdir_n = self._run_shrine("maximise_sn.py", f"{run_tag}_snr",
                                  dm_values, i_data, include_input_dm=False, save_all=True)
        sn_path = rdir_n / f"{run_tag}_snr_SNs.dat"
        if not sn_path.exists():
            raise FileNotFoundError(f"SHRINE S/N output missing: {sn_path}")
        snr = np.loadtxt(sn_path)

        metrics = {"structure": sv, "snr": snr}
        if has_qu:
            metrics["pa_slope"] = pa_values
            metrics["pa_slope_shrine"] = pa_shrine_values
            metrics["l_i_mean"] = li_values

        return dm_values, metrics

    # ================================================================
    # Public API — plotting (thin delegates)
    # ================================================================

    def plot_comparison(
        self,
        results: Dict,
        dm_range: Tuple[float, float],
        peak_region: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        _plotting.plot_comparison(
            results=results,
            stokes_i=self.stokes_i,
            freq_mhz=self.freq_mhz,
            time_ms=self.time_ms,
            input_dm=self.input_dm,
            dm_range=dm_range,
            dedisp_mode=self.dedisp_mode,
            get_delay_samples_fn=self._delay_samples,
            pa_series_fn=self._pa_series,
            pa_smoothed_and_fit_fn=self._pa_smoothed_and_fit,
            pa_shrine_smoothed_and_fit_fn=self._pa_shrine_smoothed_and_fit,
            stokes_q=self.stokes_q,
            stokes_u=self.stokes_u,
            peak_region=peak_region,
            save_path=save_path,
        )

    def plot_dm_scan(
        self,
        dm_values: np.ndarray,
        metrics: Dict,
        save_path: Optional[str] = None,
    ) -> None:
        _plotting.plot_dm_scan(dm_values, metrics, self.input_dm, save_path)

    def plot_component_dm_diagnostics(
        self,
        all_results: List[Dict],
        component_ids: Optional[np.ndarray] = None,
        label: str = "segment",
        save_path: Optional[str] = None,
    ) -> None:
        _plotting.plot_component_dm_diagnostics(all_results, component_ids, save_path)

    def plot_component_dne_diagnostics(
        self,
        dne_diag: Dict,
        label: str = "segment",
        save_path: Optional[str] = None,
    ) -> None:
        _plotting.plot_component_dne_diagnostics(dne_diag, save_path)

    # ================================================================
    # Public API — physical diagnostics
    # ================================================================

    def calculate_dn_e_between_components(
        self,
        all_results: List[Dict],
        component_separation_pc: Optional[float] = None,
        component_times_ms: Optional[np.ndarray] = None,
        comparison: str = "adjacent",
        reference_component: int = 0,
    ) -> Dict:
        return _diag.calculate_dn_e_between_components(
            all_results,
            component_separation_pc=component_separation_pc,
            component_times_ms=component_times_ms,
            comparison=comparison,
            reference_component=reference_component,
        )
