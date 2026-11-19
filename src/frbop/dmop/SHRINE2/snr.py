"""S/N maximisation, as SHRINE's maximise_sn.py: a standalone alternative to structure maximisation.

Use it for very low S/N bursts, where structure maximisation does not work well: there the fit
window or the parabola can fail, and the expansion behind sigma_DM breaks down (the summary's
sqrt(<||e||^2>)/S well above 0.3). The S/N needs only the burst's total fluence in a boxcar, not its
fine structure.

For each trial DM, the smoothed profile S x (Eq. smooth) less a common baseline is searched for the
boxcar, of any start and of length 1 sample to 10 ms, with the largest
S/N = (sum over the boxcar) / (sigma sqrt(length)). The S/N-maximising DM is the trial DM with the
largest boxcar S/N, and its range is where the S/N stays within 1 of that maximum (SHRINE's
criterion; unlike sigma_DM, it has not been calibrated against simulations).

SHRINE's choices are kept:
    sigma    = standard deviation of the residual I - S I over all trial DMs and samples;
    baseline = the smallest value of S I over all trial DMs.
Fixed relative to SHRINE: the S/N of a boxcar of L samples is now its sum of L samples over
sigma sqrt(L) (SHRINE summed L + 1 samples and reported a length of L - 1), boxcars may end at the
last sample, and the range stops at the ends of the trial DMs instead of running past them.
"""
import numpy as np

from .structure import smooth

# Longest boxcar, ms, as SHRINE.
MAX_WINDOW_MS = 10


def best_boxcars(X, sigma, n_max):
    """Find, for each profile (row of X), the boxcar with the largest S/N over all starts and
    lengths 1..n_max.

    Args:
        X: baseline-subtracted smoothed profiles, shape (n_d, N).
        sigma: noise standard deviation per sample.
        n_max: longest boxcar, in samples.
    Returns:
        sn, start, length: the best S/N and its boxcar's start and length (samples), each
        shape (n_d,).
    """
    n, N = X.shape

    # C[:, i] = sum of the first i samples, so a boxcar's sum is the difference of two columns.
    C = np.concatenate([np.zeros((n, 1)), np.cumsum(X, axis=1)], axis=1)

    # The best boxcar found so far in each row.
    sn_best = np.full(n, -np.inf)
    start = np.zeros(n, int)
    length = np.ones(n, int)

    rows = np.arange(n)
    for L in range(1, min(n_max, N) + 1):
        # The S/N of the boxcars of L samples at every start.
        sn = (C[:, L:] - C[:, :-L]) / (sigma * np.sqrt(L))

        # Each row's best start at this length, kept where it beats the best so far.
        i_best = np.argmax(sn, axis=1)
        sn_at_best = sn[rows, i_best]
        better = sn_at_best > sn_best
        sn_best[better] = sn_at_best[better]
        start[better] = i_best[better]
        length[better] = L
    return sn_best, start, length


def maximise_sn(I, d, kc, dt):
    """The S/N-maximising trial DM of I(d_j, t), from a boxcar search of the smoothed profiles, and
    the range of trial DMs whose S/N is within 1 of the maximum.

    Args:
        I: I(d_j, t), shape (n_d, N).
        d: the trial DMs, shape (n_d,), pc cm^-3.
        kc: filter cutoff k_c.
        dt: time resolution, us (sets the 10-ms longest boxcar).
    Returns:
        dict with
            d, SN, kc, sigma, Sx: the trial DMs, the best boxcar S/N at each, k_c, the noise
                sigma, S x;
            j_max, d_sn, SN_max: the trial DM with the largest S/N;
            start, length: its boxcar, in samples;
            lo, hi, lo_found, hi_found: indices of the first trial DMs where the S/N has dropped
                by 1, and whether it did.
    """
    # Smooth every profile. As SHRINE, sigma comes from the residual over all trial DMs and
    # samples, and the baseline is the smallest value of S x.
    Sx = smooth(I, kc)
    sigma = np.std(I - Sx)
    X = Sx - Sx.min()

    # The best boxcar at each trial DM, up to MAX_WINDOW_MS long.
    n_max = int(MAX_WINDOW_MS * 1000 / dt)
    SN, start, length = best_boxcars(X, sigma, n_max)

    # The trial DM with the largest S/N.
    j = int(np.argmax(SN))

    # Walk out on each side until the S/N has dropped by 1, or the trial DMs end.
    lo = j
    while lo > 0 and SN[j] - SN[lo] < 1:
        lo -= 1
    hi = j
    while hi < SN.size - 1 and SN[j] - SN[hi] < 1:
        hi += 1

    return dict(
        d=np.asarray(d),
        SN=SN,
        kc=kc,
        sigma=sigma,
        Sx=Sx,
        j_max=j,
        d_sn=d[j],
        SN_max=SN[j],
        start=int(start[j]),
        length=int(length[j]),
        lo=lo,
        hi=hi,
        lo_found=SN[j] - SN[lo] >= 1,
        hi_found=SN[j] - SN[hi] >= 1,
    )
