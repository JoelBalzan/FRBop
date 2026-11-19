"""The structure parameter Sp, the coefficients w and the filter cutoff k_c.

Paper: Section 2.2 (Eqs. smooth, sp), Section 3.1 (Eq. weights), Section 3.3.1 (Eq. ahat) and
Appendix 1 (Eqs. Mk, runmean).

Sp and w are computed as the paper writes them, one factor at a time and in the same order:
    C^T x   the orthonormal DCT-II of x            dct(x, norm="ortho")
    C v     the inverse transform of v             idct(v, norm="ortho")
    S x     = C diag[f(k)] C^T x (Eq. smooth)      idct(f * dct(x))
    D1      the (N-1) x N forward differences      forward_difference_matrix(N), a sparse matrix
S is applied as a function rather than stored: it is a dense N x N matrix (about 0.8 GB at
N = 10,000), whereas D1 has only 2(N - 1) non-zero entries.
"""
import numpy as np
from scipy import sparse
from scipy.fft import dct, idct

# Filter order O (Eq. smooth), as in S23.
O = 3


def filter_response(N, kc):
    """The low-pass filter's response f(k) = 1 / (1 + (k/k_c)^(2O)) to each DCT mode k = 1..N
    (Eq. smooth).

    Args:
        N: number of samples in a profile, which is also the number of DCT modes.
        kc: filter cutoff k_c.
    Returns:
        f(k) for k = 1..N, shape (N,).
    """
    k = np.arange(1, N + 1)
    return 1 / (1 + (k / kc) ** (2 * O))


def forward_difference_matrix(N):
    """D1, the (N-1) x N forward-difference matrix: row r is -1 at sample r and +1 at sample r + 1,
    so (D1 v)_r = v_(r+1) - v_r.

    It is stored as a sparse matrix, which holds only its 2(N - 1) non-zero entries but multiplies
    like an ordinary one.

    Args:
        N: number of samples in a profile.
    Returns:
        D1, a scipy.sparse array of shape (N - 1, N).
    """
    minus_ones = -np.ones(N - 1)
    ones = np.ones(N - 1)
    return sparse.diags_array([minus_ones, ones], offsets=[0, 1], shape=(N - 1, N))


def smooth(I, kc):
    """The smoothed profile S x = C diag[f(k)] C^T x (Eq. smooth). Works on one profile or on every
    row of I(d_j, t).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        S x, the same shape as I.
    """
    f = filter_response(I.shape[-1], kc)

    # C^T x: the DCT coefficients of the profile.
    c = dct(I, norm="ortho", axis=-1)

    # diag[f(k)] C^T x: each coefficient times the filter's response; then C: back to samples.
    return idct(f * c, norm="ortho", axis=-1)


def structure_parameter(I, kc):
    """The structure parameter Sp = ||D1 S x|| (Eq. sp): the length of the forward differences of
    the smoothed profile. Works on one profile or on every row of I(d_j, t).

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        Sp: one number per profile (a float for one profile, shape (n_d,) for I(d_j, t)), in the
        units of x.
    """
    # D1 S x, the structure vector of each profile. The profiles are the rows of S x, and a row
    # times D1^T is that profile's D1 S x, written as a row.
    D1 = forward_difference_matrix(I.shape[-1])
    D1Sx = smooth(I, kc) @ D1.T

    # Sp = ||D1 S x||.
    return np.linalg.norm(D1Sx, axis=-1)


def coefficients(I, kc):
    """The coefficients w = (D1 S x)^T D1 S / Sp of Eq. ahat: how much Sp changes per unit change in
    each sample of the profile (w(t) = dSp/dx_t). Works on one profile or on every row of I(d_j, t).

    This is Eq. weights, w = a_hat^T D1 S, with a_hat estimated from the data as D1 S x / Sp.
    e_par = sum_t w_t n_t is then the part of the noise that changes Sp at first order. The row
    (D1 S x)^T D1 S is multiplied out from the left, as written.

    Args:
        I: one profile x, shape (N,), or I(d_j, t), shape (n_d, N), with row j the profile at
            trial DM d_j.
        kc: filter cutoff k_c (from choose_kc or -kc).
    Returns:
        w: the same shape as I; row j is w(t; d_j).
    """
    # The structure vector D1 S x of each profile, as a row (as in structure_parameter), and its
    # length Sp (Eqs. smooth, sp).
    D1 = forward_difference_matrix(I.shape[-1])
    D1Sx = smooth(I, kc) @ D1.T
    Sp = np.linalg.norm(D1Sx, axis=-1)

    # (D1 S x)^T D1: the row times D1, as written (N numbers per profile).
    u = D1Sx @ D1

    # (D1 S x)^T D1 S = u S. S = C diag[f] C^T is symmetric, so the row u S holds the same numbers
    # as the column S u^T: the smoothed u.
    uS = smooth(u, kc)

    # Divide each profile's row by its own Sp.
    return uS / np.expand_dims(Sp, -1)


def kc_search(I):
    """Run S23's k_c search and the paper's extended search (Appendix 1), returning every
    intermediate quantity (for choose_kc and the DCT plot).

    M(k)    = largest |DCT coefficient k| over the trial DMs                 (Eq. Mk)
    Mbar(k) = mean of M over the six modes ending at k                       (Eq. runmean)
    M_n     = mean of M over the upper half of the modes                     (the noise floor)

    S23: k_c is the first mode where Mbar falls to M_n.
    Extended search: if Mbar rises above M_n + 4 sig_Mbar again between k_c and 2 k_c
    (sig_Mbar = spread of Mbar over the upper half), the spectrum has risen above the noise again;
    k_c then moves to the first mode after the last such rise at which Mbar falls back to M_n.

    Indices are 0-based array indices (see choose_kc).

    Args:
        I: I(d_j, t), shape (n_d, N), over the trial DMs.
    Returns:
        dict with
            M, Mbar: M(k) and Mbar(k), shape (N,) (the first five Mbar are incomplete windows);
            Mn, rise_level: the noise floor M_n and the level M_n + 4 sig_Mbar;
            kc_s23: S23's k_c, or None if Mbar never falls to M_n;
            last_rise: the last index between k_c and 2 k_c with Mbar above rise_level, or None;
            kc_ext: the extended search's k_c (kc_s23 if there is no rise), or None if Mbar never
                falls back to M_n after the last rise (or kc_s23 is None).
    """
    # M(k): the largest |DCT coefficient| of each mode over the trial DMs (Eq. Mk).
    M = np.max(np.abs(dct(I, norm="ortho", axis=-1)), axis=0)
    N = M.size

    # Mbar(k): the running mean of M over the six modes ending at k (Eq. runmean).
    # The first N values of the full convolution are Mbar[i] = mean(M[i-5..i]).
    Mbar = np.convolve(M, np.ones(6) / 6)[:N]

    # M_n: the noise floor, the mean of M over the upper half of the modes.
    Mn = np.mean(M[N // 2:])

    # The extended search's level, M_n + 4 sig_Mbar, with sig_Mbar the spread of Mbar over the
    # upper half, from six modes above N/2 (as in the paper's code).
    rise_level = Mn + 4 * np.std(Mbar[N // 2 + 6:])

    out = dict(M=M, Mbar=Mbar, Mn=Mn, rise_level=rise_level,
               kc_s23=None, last_rise=None, kc_ext=None)

    # S23: the first mode at which Mbar falls to the noise floor, starting at the first full
    # six-mode window (index 5).
    for i in range(5, N):
        if Mbar[i] <= Mn:
            out["kc_s23"] = i
            break
    if out["kc_s23"] is None:
        return out
    kc = out["kc_s23"]

    # The last mode between k_c and 2 k_c (and below N/2) at which Mbar is above the level.
    for i in range(kc, min(2 * kc, N // 2)):
        if Mbar[i] > rise_level:
            out["last_rise"] = i
    if out["last_rise"] is None:
        out["kc_ext"] = kc
        return out

    # The spectrum rose above the noise again: k_c moves to the first mode from the last rise on at
    # which Mbar falls back to the noise floor.
    for i in range(out["last_rise"], N):
        if Mbar[i] <= Mn:
            out["kc_ext"] = i
            break
    return out


def choose_kc(I, extend=True, verbose=False):
    """The filter cutoff k_c: the mode at which the DCT spectrum of the profiles falls to the noise
    floor (S23's procedure, with the paper's extended search unless extend=False; Appendix 1).

    The search itself is in kc_search.

    As in SHRINE's dm_processing.get_kc, k_c is a 0-based array index, used as it is in f(k) with
    k = 1..N, so it is one mode below the 1-based mode at which Mbar falls to the floor. The k_c
    values in S23, SHRINE and the paper all follow this convention, so it is kept.

    Args:
        I: I(d_j, t), shape (n_d, N), over the trial DMs (the wider the range, the better M(k) is
            sampled).
        extend: True for the paper's extended search (Appendix 1); False for S23's procedure alone.
        verbose: print the noise floor and where each step puts k_c.
    Returns:
        k_c, an int.
    Raises:
        RuntimeError if Mbar never falls to the noise floor, or (extended search) never falls back
        to it after rising above the noise again.
    """
    search = kc_search(I)
    kc = search["kc_s23"]
    Mn = search["Mn"]
    rise_level = search["rise_level"]

    if kc is None:
        raise RuntimeError("could not find k_c: the DCT spectrum never falls to the noise floor")
    if verbose:
        print(f"  k_c, S23 (App. 1, Eqs. Mk, runmean): Mbar falls to the noise floor "
              f"M_n = {Mn:.4g} at k_c = {kc}")
    if not extend:
        return kc

    last_rise = search["last_rise"]
    if last_rise is None:
        if verbose:
            print(f"  k_c, extended search (App. 1): Mbar stays below M_n + 4 sig_Mbar = "
                  f"{rise_level:.4g} between k_c and 2 k_c, so k_c = {kc}")
        return kc

    kc_ext = search["kc_ext"]
    if kc_ext is None:
        raise RuntimeError("could not find k_c: after rising above the noise again at mode "
                           f"{last_rise}, the DCT spectrum never falls back to the noise floor")
    if verbose:
        print(f"  k_c, extended search (App. 1): Mbar rises above M_n + 4 sig_Mbar = "
              f"{rise_level:.4g} again at mode {last_rise}, so k_c moves from {kc} to {kc_ext}")
    return kc_ext
