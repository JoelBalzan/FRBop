"""
Coherent gravitational-lensing search pipeline (Kader, Leung et al. 2022,
arXiv:2204.06014).  Five stages:

  1. Build a matched filter from the burst's intensity profile.
  2. Auto-correlate the voltage stream in the time-lag domain.
  3. Calibrate noise using off-pulse (burst-free) data.
  4. Bin the time-lag spectrum logarithmically; compute chi^2 per bin.
  5. Apply delay/significance/polarization vetoes to flag candidates.

Inputs are already-dedispersed voltage timestreams.  If your data is still
channelized baseband (post-PFB), invert the PFB first (App. C of the paper).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class SearchConfig:
    dt: float = 2.97619048e-9
    min_lag: float = 2.56e-6
    frame: float | None = None
    delay_veto_tol: float = 0.625e-9
    n_gauss_threshold: float = 1e-2
    polarization_percentile: float = 99.0
    max_excursions_per_bin: int = 2048


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


def weighted_snr_gamma(F_weighted, noise_power_weighted):
    return F_weighted / noise_power_weighted


def recover_epsilon(C_tau, Gamma):
    Gamma = np.asarray(Gamma, dtype=float)
    C2 = C_tau ** 2
    denom = Gamma ** 2 - C2 * Gamma ** 2 - C2 * Gamma
    denom = np.where(denom <= 0, np.nan, denom)
    eps2 = C2 * (Gamma + 1) / denom
    eps = np.sign(C_tau) * np.sqrt(np.clip(eps2, 0, None))
    return eps


def log_lag_bins(dt, n_bins, min_lag=2.56e-6):
    edges = min_lag * 4.0 ** np.arange(0, n_bins + 1)
    return edges


def assign_lag_bin(lags_seconds, edges):
    abs_lag = np.abs(lags_seconds)
    idx = np.digitize(abs_lag, edges) - 1
    idx[abs_lag < edges[0]] = -1
    idx[abs_lag >= edges[-1]] = -1
    return idx


def bin_covariance(eps_x_bin, eps_y_bin):
    mu = np.array([np.mean(eps_x_bin), np.mean(eps_y_bin)])
    cov = np.cov(np.vstack([eps_x_bin, eps_y_bin]))
    return mu, cov


def chi2_values(eps_x_bin, eps_y_bin, mu, cov):
    inv_cov = np.linalg.inv(cov)
    diff = np.vstack([eps_x_bin - mu[0], eps_y_bin - mu[1]])
    chi2 = np.einsum('in,ij,jn->n', diff, inv_cov, diff)
    return chi2


def ngauss_from_chi2(chi2_max, n_trials):
    p_i = np.exp(-chi2_max / 2.0)
    return p_i * n_trials


def apply_vetoes(lag_seconds, chi2_max, ngauss, eps_x_at_max, eps_y_at_max,
                 bin_eps_diff_dist, cfg):
    if cfg.frame is not None:
        nearest_frame_mult = round(lag_seconds / cfg.frame) * cfg.frame
        delay_ok = abs(lag_seconds - nearest_frame_mult) > cfg.delay_veto_tol
    else:
        delay_ok = True

    significance_ok = ngauss < cfg.n_gauss_threshold

    diff = abs(eps_x_at_max - eps_y_at_max)
    threshold = np.percentile(np.abs(bin_eps_diff_dist), cfg.polarization_percentile)
    polarization_ok = diff <= threshold

    passed_all = delay_ok and significance_ok and polarization_ok
    return {
        "delay_ok": delay_ok,
        "significance_ok": significance_ok,
        "polarization_ok": polarization_ok,
        "candidate": passed_all,
    }


def run_search(Vx, Vy, cfg, n_off_pulse=5, off_pulse_gap_widths=5,
               n_log_bins=8, verbose=True):
    N = len(Vx)
    assert len(Vy) == N, "Vx and Vy must be the same length"

    Wx2, Wy2, on_pulse_mask = make_matched_filter(Vx, Vy, cfg.dt)
    burst_width_samples = max(int(on_pulse_mask.sum()), 1)
    off_shift = off_pulse_gap_widths * burst_width_samples

    lags_x, Cx = time_lag_correlation(Vx, Wx2)
    lags_y, Cy = time_lag_correlation(Vy, Wy2)
    lags_seconds = lags_x * cfg.dt

    _, Cx_off_stack = off_pulse_stats(Vx, Wx2, off_shift, n_off_pulse)
    _, Cy_off_stack = off_pulse_stats(Vy, Wy2, off_shift, n_off_pulse)

    Fx = np.sum(np.abs(Vx) ** 2 * Wx2)
    Fy = np.sum(np.abs(Vy) ** 2 * Wy2)
    noise_x = np.mean(np.sum(np.abs(Cx_off_stack), axis=1))
    noise_y = np.mean(np.sum(np.abs(Cy_off_stack), axis=1))
    Gamma_x = weighted_snr_gamma(Fx, max(noise_x, 1e-30))
    Gamma_y = weighted_snr_gamma(Fy, max(noise_y, 1e-30))

    eps_x = recover_epsilon(Cx, Gamma_x)
    eps_y = recover_epsilon(Cy, Gamma_y)

    edges = log_lag_bins(cfg.dt, n_log_bins, min_lag=cfg.min_lag)
    bin_idx = assign_lag_bin(lags_seconds, edges)

    candidates = []
    for i in range(n_log_bins):
        sel = bin_idx == i
        if sel.sum() < 10:
            continue
        ex, ey = eps_x[sel], eps_y[sel]
        valid = np.isfinite(ex) & np.isfinite(ey)
        if valid.sum() < 10:
            continue
        ex, ey, lag_sel = ex[valid], ey[valid], lags_seconds[sel][valid]

        mu, cov = bin_covariance(ex, ey)
        chi2 = chi2_values(ex, ey, mu, cov)

        top_n = min(cfg.max_excursions_per_bin, len(chi2))
        top_idx = np.argsort(chi2)[-top_n:]

        peak = top_idx[np.argmax(chi2[top_idx])]
        chi2_max = chi2[peak]
        n_trials = sel.sum()
        ngauss = ngauss_from_chi2(chi2_max, n_trials)

        diff_dist = ex[top_idx] - ey[top_idx]
        result = apply_vetoes(
            lag_seconds=lag_sel[peak],
            chi2_max=chi2_max,
            ngauss=ngauss,
            eps_x_at_max=ex[peak],
            eps_y_at_max=ey[peak],
            bin_eps_diff_dist=diff_dist,
            cfg=cfg,
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
        if verbose:
            print(f"bin {i:2d}  |lag| in [{edges[i]:.2e}, {edges[i+1]:.2e}] s  "
                  f"tau={lag_sel[peak]:.3e}s  chi2={chi2_max:.1f}  "
                  f"Ngauss={ngauss:.2e}  candidate={result['candidate']}")
        if result["candidate"]:
            candidates.append(result)

    return {
        "candidates": candidates,
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
    }
