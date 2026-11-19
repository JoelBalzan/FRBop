"""The fit window and the parabola fit.

Paper: Section 3.3.3 (Eqs. parab, match, dvertex) and Section 3.3.4 (Eq. lsq).
"""
import numpy as np


def fit_window(Sp, sig_S, j_max):
    """The trial DMs used in the parabola fit: the contiguous trial DMs about the peak with
    Sp(d_j) >= Sp_max - sigma_S (Section 3.3.3).

    This is S23's interval (Eq. s23int) with sigma_S in place of theirs. It is found by stepping out
    from the peak one trial DM at a time on each side.

    Args:
        Sp: Sp(d_j) at every trial DM, shape (n_d,).
        sig_S: sigma_S at d_max.
        j_max: index of d_max in Sp.
    Returns:
        win: boolean array, shape (n_d,), True for the trial DMs in the window; or None if the
        window reaches either end of the trial DMs (then widen the DM range).
    """
    level = Sp[j_max] - sig_S

    # Step out from the peak on each side while Sp stays at or above the level.
    j_first = j_max
    while j_first > 0 and Sp[j_first - 1] >= level:
        j_first -= 1

    j_last = j_max
    while j_last < Sp.size - 1 and Sp[j_last + 1] >= level:
        j_last += 1

    # A window that reaches either end of the trial DMs may have been cut short.
    if j_first == 0 or j_last == Sp.size - 1:
        return None

    win = np.zeros(Sp.size, bool)
    win[j_first:j_last + 1] = True
    return win


def parabola_least_squares(x, y):
    """The least-squares fit of a parabola y ~ c0 + c1 x + c2 x^2: the matrix B = (A^T A)^-1 A^T
    and the parameters c = B y (Eq. lsq).

    Row j of A holds the numbers that multiply c0, c1, c2 at x_j: (1, x_j, x_j^2).
    pinv(A) equals (A^T A)^-1 A^T here (A has full column rank) and is numerically stabler.

    Args:
        x: trial DMs relative to the parabola's origin, shape (n,).
        y: Sp at those trial DMs, shape (n,).
    Returns:
        B: (A^T A)^-1 A^T, shape (3, n).
        c: (c0, c1, c2) = B y, shape (3,).
    """
    A = np.column_stack([np.ones_like(x), x, x**2])
    B = np.linalg.pinv(A)
    c = B @ y
    return B, c


def fit_parabola(d, Sp):
    """Fit a parabola to Sp over the fit window and return its vertex d_fit, its parameters about
    the vertex, Sp(d) ~ c0 + c1 (d - d_fit) + c2 (d - d_fit)^2, and B (Eqs. parab, lsq).

    Two passes:
      1. fit about the centre of the window (for numerical conditioning only) to locate the
         vertex, d_fit;
      2. refit the same parabola about d_fit, so that c1 = 0 and row 1 of B, B[1], gives c1 about
         the vertex, whose scatter between noise realisations moves the vertex by -c1/(2 c2)
         (Eq. dvertex).

    Args:
        d: the trial DMs in the fit window, shape (n,), pc cm^-3.
        Sp: Sp at those trial DMs, shape (n,).
    Returns:
        (d_fit, c, B): the vertex (pc cm^-3); c = (c0, c1, c2) about d_fit, shape (3,); B, shape
        (3, n), with c_m = sum_j B_mj Sp(d_j). None if the parabola has no maximum (c2 >= 0).
    """
    # Pass 1: about the centre of the window. 
    d_mid = d.mean()
    _, c_mid = parabola_least_squares(d - d_mid, Sp)
    # A parabola that opens upwards has no maximum.
    if c_mid[2] >= 0:
        return None

    # The vertex of the pass-1 parabola.
    d_fit = d_mid - c_mid[1] / (2 * c_mid[2])

    # Pass 2: the same parabola about the vertex (c[1] ~ 0).
    B, c = parabola_least_squares(d - d_fit, Sp)
    return d_fit, c, B
