"""The profile-noise variance sigma_t^2 and the structure parameter uncertainty sigma_S.

Paper: Section 3.3.2 (sigma_t^2), Section 3.1 (Eqs. sigSgen, sigS) and Section 5.3 (noise
correlated between samples, with Cov(n) given).
"""
import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import uniform_filter1d

from .structure import filter_response


def high_mode_noise(x, kc):
    """The noise in one profile x, isolated by rebuilding x from its DCT modes k > k_min only
    (Sec. 3.3.2).

    k_min = max(N/4, 2 k_c), with 2 k_c capped at 3N/4. These modes lie above the burst's power, so
    the reconstruction holds only noise, with the fraction 1 - k_min/N of its variance.

    Args:
        x: one profile, shape (N,).
        kc: filter cutoff k_c, which sets k_min.
    Returns:
        noise_only: the reconstruction, shape (N,), in the units of x.
        k_min: the number of low modes removed.
    """
    N = x.size

    # k_min = max(N/4, 2 k_c), with 2 k_c capped at 3N/4 so that at least N/4 modes are kept.
    k_min = max(N // 4, min(2 * kc, 3 * N // 4))

    # Remove modes k = 1..k_min (array indices 0..k_min-1): what is left is noise only.
    c = dct(x, norm="ortho")
    c[:k_min] = 0
    noise_only = idct(c, norm="ortho")
    return noise_only, k_min


def noise_variance(x, kc, n_avg=15, verbose=False):
    """The noise variance sigma_t^2 at every sample of one profile x, from its high DCT modes
    (Section 3.3.2).

    1. Keep only the DCT modes k > k_min, where k_min = max(N/4, 2 k_c), at most 3N/4. These lie
       above the burst's power, so the reconstruction holds only noise.
    2. Average its square over n_avg = 15 samples centred on t (end samples repeated at the edges).
    3. Divide by 1 - k_min/N: white noise keeps only that fraction of its variance in those modes.

    Args:
        x: one profile, shape (N,) (in dm_uncertainty, the profile at d_max).
        kc: filter cutoff k_c, which sets k_min.
        n_avg: number of samples in the running mean square.
        verbose: print k_min and the range of sigma_t^2.
    Returns:
        sigma_t^2 at every sample, shape (N,), in the units of x squared.
    """
    N = x.size

    # 1. The noise only: x reconstructed from the DCT modes k > k_min.
    noise_only, k_min = high_mode_noise(x, kc)

    # 2. The local mean square of the noise-only profile.
    mean_square = uniform_filter1d(noise_only**2, size=n_avg, mode="nearest")

    # 3. Correct for the variance removed with the low modes.
    sig2 = mean_square / (1 - k_min / N)

    if verbose:
        print(f"  sigma_t^2 (Sec. 3.3.2): from the DCT modes k > k_min = {k_min} of N = {N}; "
              f"mean {np.mean(sig2):.4g}, range {sig2.min():.4g} to {sig2.max():.4g}")
    return sig2


def sigma_S(w, cov_n):
    """The 1-sigma uncertainty of the structure parameter, sigma_S = sqrt(w Cov(n) w^T)
    (Eq. sigSgen); for white noise, Cov(n) = diag(sigma_t^2) and sigma_S = sqrt( sum_t w_t^2 sigma_t^2 )
    (Eq. sigS).

    Args:
        w: coefficients at one trial DM, shape (N,) (from structure.coefficients).
        cov_n: sigma_t^2, shape (N,) (from noise_variance), for white noise; or the full Cov(n),
            shape (N, N), for noise correlated between samples (Sec. 5.3).
    Returns:
        sigma_S, a float in the units of Sp.
    """
    cov_n = np.asarray(cov_n, float)
    if cov_n.ndim == 1:
        return np.sqrt(np.sum(w**2 * cov_n))
    return np.sqrt(w @ cov_n @ w)


def filtered_noise_ratio(cov_n, kc, Sp):
    """The size of the filtered noise relative to the structure parameter, sqrt(<||e||^2>) / Sp: a
    check that the first-order expansion behind sigma_S and sigma_DM holds.

    The expansion behind the uncertainties needs ||e|| << ||a|| (Eq. small). This ratio estimates
    the typical size of ||e||/||a|| (Sec. 3.4), with the measured Sp in place of ||a||, from the mean
    square of e = D1 S n (Eq. ae). D1^T D1 has the DCT-II modes as eigenvectors, with eigenvalues
    lam_k = 2 - 2 cos[(k-1) pi/N], and S = C diag[f(k)] C^T, so
        <||e||^2> = trace(D1 S Cov(n) S D1^T) = sum_k lam_k f(k)^2 P_k,   P_k = (C^T Cov(n) C)_kk,
    the noise power in mode k.
      - Given sigma_t^2 (white noise): P_k is taken as the mean of sigma_t^2 in every mode, i.e.
        white noise whose variance is the mean of sigma_t^2 (exact for constant sigma_t^2).
      - Given the full Cov(n) (noise correlated between samples, Sec. 5.3): P_k from Cov(n), exact.
    Section 5.2 discusses where it holds (the simulations span ~0.3 at S/N 15 to ~0.07 at S/N 200).
    This is a diagnostic only: it is not used in sigma_S or sigma_DM.

    Args:
        cov_n: sigma_t^2, shape (N,), or the full Cov(n), shape (N, N).
        kc: filter cutoff k_c.
        Sp: the structure parameter at d_max.
    Returns:
        sqrt(<||e||^2>) / Sp, a float.
    """
    cov_n = np.asarray(cov_n, float)
    N = cov_n.shape[0]
    f = filter_response(N, kc)
    k = np.arange(1, N + 1)
    lam = 2 - 2 * np.cos((k - 1) * np.pi / N)

    # P_k, the noise power in each DCT mode.
    if cov_n.ndim == 1:
        P = np.full(N, np.mean(cov_n))
    else:
        # C^T Cov(n) C: the DCT of Cov(n) down its columns, then along its rows.
        P = np.diag(dct(dct(cov_n, axis=0, norm="ortho"), axis=1, norm="ortho"))

    mean_square_e = np.sum(lam * f**2 * P)
    return np.sqrt(mean_square_e) / Sp
