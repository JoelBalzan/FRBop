"""
Coherent gravitational-lensing search pipeline (Kader, Leung et al. 2022,
arXiv:2204.06014).  Five stages:

  1. Build a matched filter from the burst's intensity profile.
  2. Auto-correlate the voltage stream in the time-lag domain.
  3. Calibrate noise using off-pulse (burst-free) data.
  4. Bin the time-lag spectrum logarithmically; compute chi^2 per bin
     AGAINST AN OFF-PULSE-DERIVED NULL MODEL.
  5. Apply delay/significance/polarization vetoes to flag candidates.

Inputs are already-dedispersed voltage timestreams.  If your data is still
channelized baseband (post-PFB), invert the PFB first (App. C of the paper
/ Morrison et al. 2019 for ASKAP's oversampled PFB, e.g. via CELEBI).

Changelog vs. first draft
--------------------------
* FIX: chi^2 (Eq. 13) is now computed against mu_i, G_i estimated from
  OFF-PULSE excursions in each lag bin, not from the on-pulse excursions
  being tested. Testing candidates against their own self-derived null
  model is circular and silently deflates chi^2 / Ngauss, biasing you
  toward false confidence. See Sec. VI and Fig. 3's caption in the paper:
  "we use the off-pulse realizations to capture the instantaneous noise
  environment."
* FIX: the weighted noise power feeding Gamma (Eq. 6 / A16) is now
  computed directly from off-pulse VOLTAGE (sum_t N^2(t) W^2(t), Eq. A15),
  not from summing the already-normalized correlation output |C|, which
  has the wrong units/dependence entirely.
* ADDED: leave-one-out chi^2 over the off-pulse realizations themselves,
  giving you the off-pulse "null" chi^2 distribution per bin (the
  right-hand distributions in the paper's Figs. 9/10) as a sanity check /
  false-positive-rate estimate, alongside the on-pulse candidate search.
* ADDED: SearchConfig.resolve() so delay_veto_tol and min_lag are derived
  from dt/frame rather than hardcoding CHIME's specific numbers
  (2.56 us frame, 0.625 ns tolerance) for a different instrument.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class SearchConfig:
    dt: float = 2.97619048e-9          # native voltage sample spacing [s]
    frame: Optional[float] = None      # PFB-inversion leakage period [s];
                                        # set via detect_leakage_period() or
                                        # a known instrumental value. Delay
                                        # veto is disabled if left as None.
    min_lag: Optional[float] = None    # first log-bin edge [s]; if None,
                                        # derived as 4*frame (mirrors the
                                        # paper's Sec. IV E choice), else
                                        # falls back to CHIME's 2.56 us
                                        # (only appropriate if you actually
                                        # have CHIME-like systematics!).
    delay_veto_tol: Optional[float] = None  # if None, derived as dt/2
    n_gauss_threshold: float = 1e-2
    polarization_percentile: float = 99.0
    max_excursions_per_bin: int = 2048

    def resolve(self):
        """
        Fill in defaults that depend on dt/frame rather than hardcoding
        instrument-specific constants. Returns (delay_veto_tol, min_lag).
        """
        delay_veto_tol = (self.delay_veto_tol if self.delay_veto_tol is not None
                           else self.dt / 2.0)
        if self.min_lag is not None:
            min_lag = self.min_lag
        elif self.frame is not None:
            min_lag = 4.0 * self.frame
        else:
            min_lag = 2.56e-6  # CHIME fallback -- verify this is meaningful
                                # for your instrument before trusting it.
        return delay_veto_tol, min_lag


# ----------------------------------------------------------------------
# 1. Matched filter  (Sec. IV A, Eq. A2-A3)
# ----------------------------------------------------------------------

def make_matched_filter(Vx, Vy, downsample_factor=16, noise_floor_sigma=3.0):
    def _one_pol(V):
        intensity = np.abs(V) ** 2
        smoothed = gaussian_filter1d(intensity, sigma=downsample_factor / 2.355)
        med = np.median(smoothed)
        mad = np.median(np.abs(smoothed - med))
        sigma = 1.4826 * mad
        floor = med + noise_floor_sigma * sigma
        W2 = smoothed.copy()
        W2[W2 < floor] = 0.0
        return W2

    Wx2 = _one_pol(Vx)
    Wy2 = _one_pol(Vy)
    on_pulse_mask = (Wx2 > 0) | (Wy2 > 0)
    return Wx2, Wy2, on_pulse_mask


def shift_filter(W2, shift_samples):
    return np.roll(W2, shift_samples)


# ----------------------------------------------------------------------
# 2. Time-lag correlation  (Sec. IV D, Eq. 4 / Appendix A, Eq. A6-A11)
# ----------------------------------------------------------------------

def time_lag_correlation(V, W2, max_lag_samples=None):
    N = len(V)
    WV = W2 * V
    fV = np.fft.fft(V)
    fWV = np.fft.fft(WV)
    Cprime = np.fft.ifft(fV * np.conj(fWV)).real

    V2 = np.abs(V) ** 2
    fV2 = np.fft.fft(V2)
    fW2 = np.fft.fft(W2)
    sigma2 = np.fft.ifft(fV2 * np.conj(fW2)).real
    sigma2_0 = sigma2[0]

    denom = np.sqrt(np.clip(sigma2_0 * sigma2, 1e-300, None))
    C = Cprime / denom

    lags = np.arange(N)
    lags = np.where(lags > N // 2, lags - N, lags)

    if max_lag_samples is not None:
        keep = np.abs(lags) <= max_lag_samples
        return lags[keep], C[keep]
    return lags, C


def off_pulse_stats(V, W2, shift_samples, n_realizations, gap_samples=None):
    """
    Repeat the time-lag correlation with the matched filter shifted to
    several burst-free regions (Sec. IV A/D). Returns the per-realization
    (lags, C) list and the stacked array of shape (n_realizations, N).
    """
    if gap_samples is None:
        gap_samples = shift_samples
    realizations = []
    for i in range(n_realizations):
        shift = -(i + 1) * abs(gap_samples)
        W2_off = shift_filter(W2, shift)
        lags, C = time_lag_correlation(V, W2_off)
        realizations.append((lags, C))
    stacked = np.stack([C for _, C in realizations], axis=0)
    return realizations, stacked


def off_pulse_noise_power(V, W2, shift_samples, n_realizations, gap_samples=None):
    """
    FIXED: sum_t N^2(t) W^2(t) (Eq. A15), estimated directly from
    burst-free stretches of the VOLTAGE timestream weighted by the
    (unshifted) matched filter shape -- not from the correlation output.
    Averaged over several off-pulse shifts for stability.
    """
    if gap_samples is None:
        gap_samples = shift_samples
    powers = []
    for i in range(n_realizations):
        shift = -(i + 1) * abs(gap_samples)
        V_off = np.roll(V, shift)
        powers.append(np.sum(np.abs(V_off) ** 2 * W2))
    return float(np.mean(powers))


# ----------------------------------------------------------------------
# 3. Recovering epsilon(that) from C(that)  (Eq. 6-8)
# ----------------------------------------------------------------------

def weighted_snr_gamma(F_weighted, noise_power_weighted):
    """Gamma = F / sum_t N^2(t) W^2(t)   (Eq. 6 / A16)."""
    return F_weighted / noise_power_weighted


def recover_epsilon(C_tau, Gamma):
    """
    Eq. 8: eps^2 = C^2 (Gamma+1) / (Gamma^2 - C^2 Gamma^2 - C^2 Gamma).
    Works elementwise on arrays of any shape (used both for 1-D on-pulse
    C(that) and 2-D off-pulse [n_realizations x N] stacks).
    """
    Gamma = np.asarray(Gamma, dtype=float)
    C2 = C_tau ** 2
    denom = Gamma ** 2 - C2 * Gamma ** 2 - C2 * Gamma
    denom = np.where(denom <= 0, np.nan, denom)
    eps2 = C2 * (Gamma + 1) / denom
    eps = np.sign(C_tau) * np.sqrt(np.clip(eps2, 0, None))
    return eps


# ----------------------------------------------------------------------
# 3b. Empirically detect PFB-inversion leakage period
# ----------------------------------------------------------------------

def detect_leakage_period(lags_seconds, C_off_stack, dt, search_range=(0.5e-6, 2e-6),
                           n_candidates=5):
    """
    For an instrument where the leakage period isn't a known fixed
    constant (e.g. ASKAP/CELEBI's Fourier-based oversampled-PFB inversion,
    Morrison et al. 2019), scan the OFF-PULSE correlation stack for a
    periodic comb of excess power vs. candidate periods, and report the
    best match plus runner-ups for manual inspection. Use the result to
    set SearchConfig.frame.
    """
    mean_off = np.mean(np.abs(C_off_stack), axis=0)
    trial_periods = np.arange(search_range[0], search_range[1], dt * 4)
    scores = {}
    for period in trial_periods:
        n_mult = int(np.max(np.abs(lags_seconds)) / period)
        if n_mult < 3:
            continue
        on_comb = np.zeros(0)
        for k in range(1, n_mult + 1):
            idx = np.argmin(np.abs(np.abs(lags_seconds) - k * period))
            window = mean_off[max(0, idx - 2):idx + 3]
            on_comb = np.concatenate([on_comb, window])
        baseline = np.median(mean_off)
        scores[period] = float(np.mean(on_comb) / max(baseline, 1e-30))
    if not scores:
        return None, scores
    best_period_s = max(scores, key=scores.get)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:n_candidates]
    print("Top leakage-period candidates (period_s, excess_score):")
    for p, s in top:
        print(f"  {p:.4e} s   score={s:.3f}")
    return best_period_s, scores


# ----------------------------------------------------------------------
# 4. Logarithmic time-lag binning  (Sec. IV E)
# ----------------------------------------------------------------------

def log_lag_bins(n_bins, min_lag):
    """Bin edges at +/- min_lag * 4^i for i = 0, 1, 2, ... (Sec. IV E)."""
    edges = min_lag * 4.0 ** np.arange(0, n_bins + 1)
    return edges


def assign_lag_bin(lags_seconds, edges):
    abs_lag = np.abs(lags_seconds)
    idx = np.digitize(abs_lag, edges) - 1
    idx[abs_lag < edges[0]] = -1
    idx[abs_lag >= edges[-1]] = -1
    return idx


# ----------------------------------------------------------------------
# 5. Per-bin chi^2, Ngauss, and veto conditions  (Sec. VI, Table II, Eq. 13-15)
# ----------------------------------------------------------------------

def bin_covariance(eps_x_bin, eps_y_bin):
    """
    G_i, the 2x2 covariance of (eps_X, eps_Y) within one lag bin.
    IMPORTANT: call this with OFF-PULSE excursions to build the null
    model that on-pulse candidates get scored against (see chi2_values).
    """
    mu = np.array([np.mean(eps_x_bin), np.mean(eps_y_bin)])
    cov = np.cov(np.vstack([eps_x_bin, eps_y_bin]))
    return mu, cov


def chi2_values(eps_x, eps_y, mu, cov):
    """chi^2 = (eps - mu)^T G^-1 (eps - mu)  (Eq. 13), evaluated elementwise."""
    inv_cov = np.linalg.inv(cov)
    diff = np.vstack([eps_x - mu[0], eps_y - mu[1]])  # (2, n)
    chi2 = np.einsum('in,ij,jn->n', diff, inv_cov, diff)
    return chi2


def loo_offpulse_chi2(eps_x_off_bin, eps_y_off_bin):
    """
    Leave-one-out chi^2 distribution for the off-pulse realizations
    themselves within one bin: each realization is scored against mu_i,
    G_i built from the OTHER realizations, avoiding the circularity of
    testing a point against a null model built partly from itself.

    eps_x_off_bin, eps_y_off_bin : ndarray, shape (n_realizations, n_lags)

    Returns a flat array of chi^2 values -- this is the off-pulse "null"
    distribution to compare your on-pulse candidate chi^2 against
    (analogous to the off-pulse panels of Figs. 9/10 in the paper).
    """
    n_real = eps_x_off_bin.shape[0]
    if n_real < 2:
        return np.array([])
    chi2_all = []
    for r in range(n_real):
        mask = np.ones(n_real, dtype=bool)
        mask[r] = False
        ex_train = eps_x_off_bin[mask].ravel()
        ey_train = eps_y_off_bin[mask].ravel()
        valid = np.isfinite(ex_train) & np.isfinite(ey_train)
        if valid.sum() < 5:
            continue
        mu, cov = bin_covariance(ex_train[valid], ey_train[valid])
        ex_test, ey_test = eps_x_off_bin[r], eps_y_off_bin[r]
        valid_test = np.isfinite(ex_test) & np.isfinite(ey_test)
        if valid_test.sum() == 0:
            continue
        chi2_r = chi2_values(ex_test[valid_test], ey_test[valid_test], mu, cov)
        chi2_all.append(chi2_r)
    if not chi2_all:
        return np.array([])
    return np.concatenate(chi2_all)


def ngauss_from_chi2(chi2_max, n_trials):
    """Eq. 14-15: Ngauss,i = exp(-chi2_max/2) * N_i."""
    p_i = np.exp(-chi2_max / 2.0)
    return p_i * n_trials


def apply_vetoes(lag_seconds, chi2_max, ngauss, eps_x_at_max, eps_y_at_max,
                  bin_eps_diff_dist, frame, delay_veto_tol,
                  n_gauss_threshold, polarization_percentile):
    """
    Table II, evaluated successively:
      1. Delay condition   -- tau not within delay_veto_tol of a multiple
                               of `frame` (PFB-inversion leakage). Skipped
                               (always True) if frame is None.
      2. Significance cond -- Ngauss,i < n_gauss_threshold.
      3. Polarization cond -- |eps_X - eps_Y| at peak within the given
                               percentile of the bin's OFF-PULSE-derived
                               eps_X - eps_Y distribution.
    """
    if frame is not None:
        nearest_frame_mult = round(lag_seconds / frame) * frame
        delay_ok = abs(lag_seconds - nearest_frame_mult) > delay_veto_tol
    else:
        delay_ok = True

    significance_ok = ngauss < n_gauss_threshold

    diff = abs(eps_x_at_max - eps_y_at_max)
    if len(bin_eps_diff_dist) > 0:
        threshold = np.percentile(np.abs(bin_eps_diff_dist), polarization_percentile)
        polarization_ok = diff <= threshold
    else:
        polarization_ok = False  # no off-pulse distribution to compare against

    passed_all = delay_ok and significance_ok and polarization_ok
    return {
        "delay_ok": delay_ok,
        "significance_ok": significance_ok,
        "polarization_ok": polarization_ok,
        "candidate": passed_all,
    }


# ----------------------------------------------------------------------
# 6. Top-level driver
# ----------------------------------------------------------------------

def run_search(Vx, Vy, cfg: SearchConfig, n_off_pulse=5, off_pulse_gap_widths=5,
                n_log_bins=8, verbose=True):
    """
    End-to-end search for one FRB, given dedispersed voltage timestreams
    Vx, Vy (complex or real, same length, sampled at cfg.dt).
    """
    N = len(Vx)
    assert len(Vy) == N, "Vx and Vy must be the same length"

    delay_veto_tol, min_lag = cfg.resolve()

    # ---- Stage 1: matched filter ----
    Wx2, Wy2, on_pulse_mask = make_matched_filter(Vx, Vy)
    burst_width_samples = max(int(on_pulse_mask.sum()), 1)
    off_shift = off_pulse_gap_widths * burst_width_samples

    # ---- Stage 2: on-pulse time-lag correlation for both polarizations ----
    lags_x, Cx = time_lag_correlation(Vx, Wx2)
    lags_y, Cy = time_lag_correlation(Vy, Wy2)
    lags_seconds = lags_x * cfg.dt

    # ---- Stage 3: off-pulse noise power -> Gamma (FIXED: from voltage, not C) ----
    noise_x = off_pulse_noise_power(Vx, Wx2, off_shift, n_off_pulse)
    noise_y = off_pulse_noise_power(Vy, Wy2, off_shift, n_off_pulse)
    Fx = np.sum(np.abs(Vx) ** 2 * Wx2)  # weighted unlensed fluence, Eq. 5/A3
    Fy = np.sum(np.abs(Vy) ** 2 * Wy2)
    Gamma_x = weighted_snr_gamma(Fx, max(noise_x, 1e-30))
    Gamma_y = weighted_snr_gamma(Fy, max(noise_y, 1e-30))

    # ---- Recover eps_X(that), eps_Y(that) for the on-pulse data ----
    eps_x = recover_epsilon(Cx, Gamma_x)
    eps_y = recover_epsilon(Cy, Gamma_y)

    # ---- Off-pulse correlation realizations -> off-pulse eps, for the
    #      chi^2 null model (FIX for bug #1) ----
    _, Cx_off_stack = off_pulse_stats(Vx, Wx2, off_shift, n_off_pulse)
    _, Cy_off_stack = off_pulse_stats(Vy, Wy2, off_shift, n_off_pulse)
    eps_x_off_stack = recover_epsilon(Cx_off_stack, Gamma_x)  # (n_real, N)
    eps_y_off_stack = recover_epsilon(Cy_off_stack, Gamma_y)

    # ---- Stage 4: logarithmic binning ----
    edges = log_lag_bins(n_log_bins, min_lag)
    bin_idx = assign_lag_bin(lags_seconds, edges)

    candidates = []
    bin_diagnostics = []
    for i in range(n_log_bins):
        sel = bin_idx == i
        if sel.sum() < 10:
            continue

        # On-pulse excursions to be tested
        ex, ey = eps_x[sel], eps_y[sel]
        valid = np.isfinite(ex) & np.isfinite(ey)
        if valid.sum() < 10:
            continue
        ex, ey, lag_sel = ex[valid], ey[valid], lags_seconds[sel][valid]

        # Off-pulse excursions used to build the null model (FIX)
        off_ex_bin = eps_x_off_stack[:, sel]
        off_ey_bin = eps_y_off_stack[:, sel]
        off_valid = np.isfinite(off_ex_bin) & np.isfinite(off_ey_bin)
        off_ex_flat = off_ex_bin[off_valid]
        off_ey_flat = off_ey_bin[off_valid]
        if off_ex_flat.size < 10:
            continue  # not enough off-pulse statistics for this bin

        mu_i, cov_i = bin_covariance(off_ex_flat, off_ey_flat)

        # Score on-pulse excursions against the OFF-PULSE null model
        chi2 = chi2_values(ex, ey, mu_i, cov_i)

        top_n = min(cfg.max_excursions_per_bin, len(chi2))
        top_idx = np.argsort(chi2)[-top_n:]
        peak = top_idx[np.argmax(chi2[top_idx])]
        chi2_max = chi2[peak]
        n_trials = sel.sum()
        ngauss = ngauss_from_chi2(chi2_max, n_trials)

        # Polarization-consistency distribution, also from OFF-PULSE data
        diff_dist = off_ex_flat - off_ey_flat

        result = apply_vetoes(
            lag_seconds=lag_sel[peak],
            chi2_max=chi2_max,
            ngauss=ngauss,
            eps_x_at_max=ex[peak],
            eps_y_at_max=ey[peak],
            bin_eps_diff_dist=diff_dist,
            frame=cfg.frame,
            delay_veto_tol=delay_veto_tol,
            n_gauss_threshold=cfg.n_gauss_threshold,
            polarization_percentile=cfg.polarization_percentile,
        )
        result.update({
            "bin_index": i,
            "lag_range_s": (edges[i], edges[i + 1]),
            "tau_seconds": lag_sel[peak],
            "chi2_max": chi2_max,
            "ngauss": ngauss,
            "eps_x": ex[peak],
            "eps_y": ey[peak],
        })

        # Off-pulse null chi^2 distribution for this bin (diagnostic only,
        # not used in the pass/fail decision -- compare to chi2_max by eye
        # the way the paper's Fig. 8/9/10 dashed "noise floor" lines do)
        off_chi2 = loo_offpulse_chi2(off_ex_bin, off_ey_bin)
        bin_diag = {
            "bin_index": i,
            "lag_range_s": (edges[i], edges[i + 1]),
            "chi2_max": chi2_max,
            "off_pulse_chi2_max": float(np.max(off_chi2)) if off_chi2.size else np.nan,
        }
        bin_diagnostics.append(bin_diag)

        if verbose:
            print(f"bin {i:2d}  |lag| in [{edges[i]:.2e}, {edges[i+1]:.2e}] s  "
                  f"tau={lag_sel[peak]:.3e}s  chi2={chi2_max:.1f}  "
                  f"(off-pulse max chi2={bin_diag['off_pulse_chi2_max']:.1f})  "
                  f"Ngauss={ngauss:.2e}  candidate={result['candidate']}")

        if result["candidate"]:
            candidates.append(result)

    return {
        "candidates": candidates,
        "bin_diagnostics": bin_diagnostics,
        "lags_seconds": lags_seconds,
        "Cx": Cx,
        "Cy": Cy,
        "eps_x": eps_x,
        "eps_y": eps_y,
        "edges": edges,
        "bin_idx": bin_idx,
        "Gamma_x": Gamma_x,
        "Gamma_y": Gamma_y,
        "on_pulse_mask": on_pulse_mask,
        "off_stack_Cx": Cx_off_stack,
        "off_stack_Cy": Cy_off_stack,
        "resolved_delay_veto_tol": delay_veto_tol,
        "resolved_min_lag": min_lag,
    }