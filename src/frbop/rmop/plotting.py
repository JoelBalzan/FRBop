"""
Plotting functions for RM analysis:
  - plot_rm_results
  - plot_burns_law_fits
  - plot_rm_time_series
  - plot_poincare_sphere
  - plot_poincare_projections

Also contains the Poincaré-sphere helper functions that are shared between
the sphere and projections plots.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.optimize import curve_fit

from frbop.utils.plotting import savefig_rasterized, set_pub_style  # noqa: F401

from .constants import (
    TWO_COLUMN_WIDTH_IN,
    SINGLE_COLUMN_WIDTH_IN,
    pub_figsize,
    plot_style,
)
from .fitter import RMFitter
from .physics import (
    sigma_rm_detection_threshold,
    sigma_rm_detection_threshold_snr,
    depolarising_medium_delta_ne_b_parallel,
)


# ---------------------------------------------------------------------------
# Internal save wrapper
# ---------------------------------------------------------------------------

def _savefig_rasterized(save_path: str,
                        dpi: int = 300,
                        bbox_inches: str = 'tight') -> None:
    """Compatibility wrapper for rasterized plot saving used in RM plotting."""
    savefig_rasterized(save_path, dpi=dpi, bbox_inches=bbox_inches)


# ---------------------------------------------------------------------------
# Poincaré-sphere helpers
# ---------------------------------------------------------------------------

def _compute_poincare_point_errors(time_series_data: Dict,
                                   point_times: np.ndarray,
                                   noise_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate q/u/v uncertainties per plotted point using off-pulse full dspec noise."""
    if time_series_data is None:
        raise ValueError("time_series_data is required for Poincare error estimates")

    I_cube = np.asarray(time_series_data['I'], dtype=float)
    Q_cube = np.asarray(time_series_data['Q'], dtype=float)
    U_cube = np.asarray(time_series_data['U'], dtype=float)
    V_cube = np.asarray(time_series_data.get('V', np.zeros_like(I_cube)), dtype=float)
    times = np.asarray(time_series_data['time'], dtype=float)
    if I_cube.ndim != 2:
        raise ValueError("time_series_data['I'] must be 2D")

    n_time = len(times)
    time_axis = 0 if I_cube.shape[0] == n_time else 1

    if time_axis == 0:
        I_t = np.nanmean(I_cube, axis=1)
        Q_t = np.nanmean(Q_cube, axis=1)
        U_t = np.nanmean(U_cube, axis=1)
        V_t = np.nanmean(V_cube, axis=1)
    else:
        I_t = np.nanmean(I_cube, axis=0)
        Q_t = np.nanmean(Q_cube, axis=0)
        U_t = np.nanmean(U_cube, axis=0)
        V_t = np.nanmean(V_cube, axis=0)

    n_off = max(1, int(n_time * noise_fraction))

    def _robust_sigma(arr: np.ndarray) -> float:
        sig = float(np.nanstd(arr[:n_off]))
        if np.isfinite(sig) and sig > 0:
            return sig
        mad = np.nanmedian(np.abs(arr[:n_off] - np.nanmedian(arr[:n_off])))
        if np.isfinite(mad) and mad > 0:
            return float(mad / 0.6745)
        return 1e-10

    sigma_i0 = _robust_sigma(I_t)
    sigma_q0 = _robust_sigma(Q_t)
    sigma_u0 = _robust_sigma(U_t)
    sigma_v0 = _robust_sigma(V_t)

    centers = np.asarray(point_times, dtype=float)
    n_pts = centers.size
    sigma_q = np.full(n_pts, np.nan, dtype=float)
    sigma_u = np.full(n_pts, np.nan, dtype=float)
    sigma_v = np.full(n_pts, np.nan, dtype=float)

    if n_pts == 0:
        return sigma_q, sigma_u, sigma_v

    if n_pts == 1:
        boundaries = np.array([-np.inf, np.inf], dtype=float)
    else:
        mids = 0.5 * (centers[:-1] + centers[1:])
        boundaries = np.concatenate(([-np.inf], mids, [np.inf]))

    for i in range(n_pts):
        left = boundaries[i]
        right = boundaries[i + 1]
        if i < n_pts - 1:
            mask_t = (times >= left) & (times < right)
        else:
            mask_t = (times >= left) & (times <= right)

        if not np.any(mask_t):
            idx = int(np.argmin(np.abs(times - centers[i])))
            mask_t = np.zeros_like(times, dtype=bool)
            mask_t[idx] = True

        n_bin = int(np.sum(mask_t))
        n_bin = max(1, n_bin)

        I_m = float(np.nanmean(I_t[mask_t]))
        Q_m = float(np.nanmean(Q_t[mask_t]))
        U_m = float(np.nanmean(U_t[mask_t]))
        V_m = float(np.nanmean(V_t[mask_t]))

        sI = sigma_i0 / np.sqrt(n_bin)
        sQ = sigma_q0 / np.sqrt(n_bin)
        sU = sigma_u0 / np.sqrt(n_bin)
        sV = sigma_v0 / np.sqrt(n_bin)

        denom = I_m + 1e-10
        sigma_q[i] = np.sqrt((sQ / denom) ** 2 + ((Q_m * sI) / (denom ** 2)) ** 2)
        sigma_u[i] = np.sqrt((sU / denom) ** 2 + ((U_m * sI) / (denom ** 2)) ** 2)
        sigma_v[i] = np.sqrt((sV / denom) ** 2 + ((V_m * sI) / (denom ** 2)) ** 2)

    return sigma_q, sigma_u, sigma_v


def _poincare_angle_errors_deg(q: np.ndarray,
                               u: np.ndarray,
                               v: np.ndarray,
                               sigma_q: np.ndarray,
                               sigma_u: np.ndarray,
                               sigma_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate q/u/v errors to lon/lat errors (degrees)."""
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    sq = np.asarray(sigma_q, dtype=float)
    su = np.asarray(sigma_u, dtype=float)
    sv = np.asarray(sigma_v, dtype=float)

    qu2 = q ** 2 + u ** 2 + 1e-20
    r = np.sqrt(q ** 2 + u ** 2 + v ** 2 + 1e-20)

    sigma_lon_rad = np.sqrt((u ** 2 * sq ** 2 + q ** 2 * su ** 2) / (qu2 ** 2))

    x = np.clip(v / r, -1.0, 1.0)
    dxdq = -v * q / (r ** 3)
    dxdu = -v * u / (r ** 3)
    dxdv = (q ** 2 + u ** 2) / (r ** 3)
    dlatdx = 1.0 / np.sqrt(1.0 - x ** 2 + 1e-20)
    sigma_lat_rad = np.sqrt(
        (dlatdx * dxdq * sq) ** 2 +
        (dlatdx * dxdu * su) ** 2 +
        (dlatdx * dxdv * sv) ** 2
    )

    return np.degrees(sigma_lon_rad), np.degrees(sigma_lat_rad)


def _build_circle_segments(n_points: int,
                           segment_pairs: Optional[List[Tuple[int, int]]],
                           filtered_indices: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Create valid inclusive [start, end] segments over plotted points."""
    if n_points <= 0:
        return []

    if segment_pairs == []:
        if filtered_indices is None or len(filtered_indices) == 0:
            return [(0, n_points - 1)]
        idx = np.asarray(filtered_indices, dtype=int)
        runs: List[Tuple[int, int]] = []
        run_start = 0
        for i in range(1, len(idx)):
            if idx[i] != (idx[i - 1] + 1):
                if (i - run_start) >= 3:
                    runs.append((run_start, i - 1))
                run_start = i
        if (len(idx) - run_start) >= 3:
            runs.append((run_start, len(idx) - 1))
        return runs if runs else [(0, n_points - 1)]

    if not segment_pairs:
        return [(0, n_points - 1)]

    out: List[Tuple[int, int]] = []
    for s_raw, e_raw in segment_pairs:
        s = max(0, min(int(s_raw), n_points - 1))
        e = max(0, min(int(e_raw), n_points - 1))
        if e < s:
            s, e = e, s
        if (e - s + 1) >= 3:
            out.append((s, e))
    return out if out else [(0, n_points - 1)]


def _fit_circle_on_sphere(points_xyz: np.ndarray,
                          mode: str = 'auto',
                          sample_points: int = 240) -> Optional[Dict[str, np.ndarray]]:
    """Fit a great/small circle to 3D points and return an arc on the unit sphere."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None

    r = np.linalg.norm(pts, axis=1)
    r[r == 0] = 1.0
    X = pts / r[:, None]

    evals_g, evecs_g = np.linalg.eigh(X.T @ X)
    n_g = evecs_g[:, np.argmin(evals_g)]
    d_g = 0.0
    res_g = np.nanstd(X @ n_g)

    mu = np.mean(X, axis=0)
    C = (X - mu).T @ (X - mu)
    evals_s, evecs_s = np.linalg.eigh(C)
    n_s = evecs_s[:, np.argmin(evals_s)]
    d_s = float(np.clip(np.dot(n_s, mu), -0.999, 0.999))
    res_s = np.nanstd((X @ n_s) - d_s)

    mode_l = str(mode).lower()
    if mode_l == 'great':
        n, d, fit_type = n_g, d_g, 'great'
    elif mode_l == 'small':
        n, d, fit_type = n_s, d_s, 'small'
    else:
        if res_g <= (res_s + 1e-12):
            n, d, fit_type = n_g, d_g, 'great'
        else:
            n, d, fit_type = n_s, d_s, 'small'

    n_norm = np.linalg.norm(n)
    if not np.isfinite(n_norm) or n_norm <= 0:
        return None
    n = n / n_norm
    d = float(np.clip(d / n_norm, -0.999, 0.999))

    center = d * n
    radius = float(np.sqrt(max(1.0 - d ** 2, 1e-10)))

    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, ref)
    e1_norm = np.linalg.norm(e1)
    if e1_norm <= 0:
        return None
    e1 = e1 / e1_norm
    e2 = np.cross(n, e1)

    proj = X - center[None, :]
    t_data = np.arctan2(proj @ e2, proj @ e1)
    t_data = np.unwrap(t_data)
    t_min = float(np.min(t_data))
    t_max = float(np.max(t_data))
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return None
    if abs(t_max - t_min) < 1e-6:
        t_min -= 0.05
        t_max += 0.05

    tt = np.linspace(t_min, t_max, max(50, int(sample_points)))
    arc = center[None, :] + radius * (np.cos(tt)[:, None] * e1[None, :] + np.sin(tt)[:, None] * e2[None, :])
    arc_r = np.linalg.norm(arc, axis=1)
    arc_r[arc_r == 0] = 1.0
    arc = arc / arc_r[:, None]

    return {
        'arc_xyz': arc,
        'fit_type': np.array([fit_type]),
    }


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_poincare_sphere(
        time_series_data: Dict,
        output_file: str = 'poincare_sphere.png',
        snr_threshold: float = 5.0,
        n_time_bins: Optional[int] = None,
        noise_fraction: float = 0.1,
        time_unit: str = 's',
        interactive: bool = False,
        force_surface: bool = False,
        rm_results: Optional[Dict] = None,
        noise_reference_data: Optional[Dict] = None,
        circle_fit_mode: Optional[str] = None,
        circle_fit_segments: Optional[List[Tuple[int, int]]] = None):
    """
    Plot polarisation states on the Poincaré sphere vs. **time only**.

    Frequency-dependent colouring and axes have been removed – this
    routine now _requires_ a time-series data dictionary and will
    average over frequency (or bins) to build the track.  The goal is to
    visualise the temporal evolution of the polarisation state.

    Parameters
    ----------
    time_series_data : dict
        Dictionary with keys ``'time'``, ``'I'``, ``'Q'``, ``'U'`` (and
        optionally ``'V'``) containing 2‑D arrays with shape
        ``(n_time, n_freq)`` or ``(n_freq, n_time)``.  Time is always used
        to colour the points.
    output_file : str
        Output filename for the plot.
    snr_threshold : float
        Minimum linear-polarisation SNR for a point to be plotted.
    n_time_bins : int, optional
        Number of time bins to average over. If ``None`` or ``<=0`` each
        time sample is used individually.
    noise_fraction : float
        Fraction of the time axis used to estimate the off‑pulse noise
        (default ``0.1`` = first 10 % of samples).
    """
    if time_series_data is None:
        raise ValueError("plot_poincare_sphere now requires ``time_series_data``; "
                         "frequency colouring has been removed.")

    I_cube = time_series_data['I']
    Q_cube = time_series_data['Q']
    U_cube = time_series_data['U']
    V_cube = time_series_data.get('V', None)
    times  = time_series_data['time']

    n_time = len(times)
    noise_ref = noise_reference_data if noise_reference_data is not None else time_series_data

    if rm_results is not None and 'q_bin' in rm_results:
        q_norm = np.array(rm_results['q_bin'])
        u_norm = np.array(rm_results['u_bin'])
        v_norm = np.array(rm_results.get('v_bin', np.zeros_like(q_norm)))
        color_axis = np.array(rm_results['time'])
        orig_idx = np.arange(q_norm.size, dtype=int)
        pol_list = np.sqrt(q_norm**2 + u_norm**2)
        mask = np.ones(q_norm.size, dtype=bool)
        if 'valid_bins' in rm_results:
            mask &= np.asarray(rm_results['valid_bins'], dtype=bool)
        if 'pa_ea_valid' in rm_results:
            mask &= np.asarray(rm_results['pa_ea_valid'], dtype=bool)
        if not np.all(mask):
            q_norm = q_norm[mask]
            u_norm = u_norm[mask]
            v_norm = v_norm[mask]
            color_axis = color_axis[mask]
            pol_list = pol_list[mask]
            orig_idx = orig_idx[mask]
        notnan = (~np.isnan(q_norm)) & (~np.isnan(u_norm)) & (~np.isnan(v_norm))
        if not np.all(notnan):
            q_norm = q_norm[notnan]
            u_norm = u_norm[notnan]
            v_norm = v_norm[notnan]
            color_axis = color_axis[notnan]
            pol_list = pol_list[notnan]
            orig_idx = orig_idx[notnan]
        if q_norm.size == 0:
            print("Warning: all Poincaré bins were masked; skipping plot.")
            return
    else:
        if I_cube.ndim == 2:
            time_axis = 0 if I_cube.shape[0] == n_time else 1
        else:
            raise ValueError("Time-series data must be 2D (time × frequency).")

        if n_time_bins is None or n_time_bins <= 0:
            bin_size = 1
            n_bins = n_time
        else:
            bin_size = max(1, n_time // n_time_bins)
            n_bins = (n_time + bin_size - 1) // bin_size

        q_list, u_list, v_list = [], [], []
        pol_list, time_list = [], []

        for i in range(n_bins):
            start = i * bin_size
            end   = min((i + 1) * bin_size, n_time)
            if end <= start:
                continue
            time_list.append(np.mean(times[start:end]))
            if time_axis == 0:
                I_sl = I_cube[start:end, :]
                Q_sl = Q_cube[start:end, :]
                U_sl = U_cube[start:end, :]
                V_sl = V_cube[start:end, :] if V_cube is not None else None
            else:
                I_sl = I_cube[:, start:end]
                Q_sl = Q_cube[:, start:end]
                U_sl = U_cube[:, start:end]
                V_sl = V_cube[:, start:end] if V_cube is not None else None
            I_m = np.nanmean(I_sl)
            Q_m = np.nanmean(Q_sl)
            U_m = np.nanmean(U_sl)
            V_m = np.nanmean(V_sl) if V_sl is not None else 0.0
            q_list.append(Q_m / (I_m + 1e-10))
            u_list.append(U_m / (I_m + 1e-10))
            v_list.append(V_m / (I_m + 1e-10))
            pol_list.append(np.sqrt(Q_m**2 + U_m**2))

        q_norm = np.array(q_list)
        u_norm = np.array(u_list)
        v_norm = np.array(v_list)
        color_axis = np.array(time_list)
        orig_idx = np.arange(q_norm.size, dtype=int)

    unit = time_unit.lower()
    if unit == 'ms':
        color_axis = color_axis * 1e3
        color_label = "Time (ms)"
    elif unit == 'us' or unit == 'µs':
        color_axis = color_axis * 1e6
        color_label = "Time (µs)"
    else:
        color_label = "Time (s)"

    n_frac = max(1, int(len(pol_list) * noise_fraction))
    sigma_pol = np.nanstd(pol_list[:n_frac])
    if sigma_pol <= 0:
        sigma_pol = 1e-10
    snr = np.array(pol_list) / (sigma_pol + 1e-10)

    if rm_results is not None and 'valid_bins' in rm_results:
        mask = np.ones(q_norm.shape[0], dtype=bool)
    else:
        mask = snr > snr_threshold
        if np.sum(mask) < 2:
            print(f"Warning: Only {np.sum(mask)} points above SNR threshold "
                  f"{snr_threshold:.1f}. Lowering to 2.0.")
            mask = snr > 2.0
        if np.sum(mask) < 2:
            print("Error: fewer than 2 points survive SNR cut. Cannot plot.")
            return

    filtered_idx = np.asarray(orig_idx[mask], dtype=int)
    q_filt = q_norm[mask]
    u_filt = u_norm[mask]
    v_filt = v_norm[mask]
    color_filt = color_axis[mask]

    if force_surface:
        vecs = np.vstack([q_filt, u_filt, v_filt])
        norms = np.linalg.norm(vecs, axis=0)
        norms[norms == 0] = 1.0
        q_filt = q_filt / norms
        u_filt = u_filt / norms
        v_filt = v_filt / norms
        q_filt *= 1.002
        u_filt *= 1.002
        v_filt *= 1.002

    sigma_q, sigma_u, sigma_v = _compute_poincare_point_errors(
        noise_ref,
        point_times=np.asarray(color_filt, dtype=float) / (1e3 if unit == 'ms' else (1e6 if unit in ('us', 'µs') else 1.0)),
        noise_fraction=noise_fraction,
    )
    sigma_lon_deg, sigma_lat_deg = _poincare_angle_errors_deg(
        q_filt, u_filt, v_filt, sigma_q, sigma_u, sigma_v
    )

    style = plot_style()

    fig = plt.figure(figsize=pub_figsize(height_ratio=0.92, min_height=6.2))
    ax  = fig.add_subplot(111, projection='3d')

    u_s = np.linspace(0, 2 * np.pi, 100)
    v_s = np.linspace(0,     np.pi, 100)
    xs  = np.outer(np.cos(u_s), np.sin(v_s))
    ys  = np.outer(np.sin(u_s), np.sin(v_s))
    zs  = np.outer(np.ones_like(u_s), np.cos(v_s))

    ax.plot_surface(xs, ys, zs, color='lightgray', alpha=0.2, rstride=4, cstride=4,
                    linewidth=0, antialiased=True, zorder=1)
    n_grid = 12
    for lat in np.linspace(0, np.pi, n_grid, endpoint=False)[1:]:
        x_lat = np.cos(u_s) * np.sin(lat)
        y_lat = np.sin(u_s) * np.sin(lat)
        z_lat = np.full_like(u_s, np.cos(lat))
        ax.plot(x_lat, y_lat, z_lat, color='gray', alpha=0.3, linewidth=0.5)
    for lon in np.linspace(0, 2*np.pi, n_grid, endpoint=False):
        x_lon = np.cos(lon) * np.sin(v_s)
        y_lon = np.sin(lon) * np.sin(v_s)
        z_lon = np.cos(v_s)
        ax.plot(x_lon, y_lon, z_lon, color='gray', alpha=0.3, linewidth=0.5)

    sc = ax.scatter(q_filt, u_filt, v_filt,
                    c=color_filt, cmap='viridis',
                    s=60, alpha=1,
                    edgecolors='black', linewidth=0.6, zorder=200,
                    depthshade=True)

    lon_deg = np.degrees(np.arctan2(u_filt, q_filt))
    r_vec = np.sqrt(q_filt**2 + u_filt**2 + v_filt**2)
    lat_deg = np.degrees(np.arcsin(np.clip(v_filt / (r_vec + 1e-20), -1.0, 1.0)))

    def _sph_to_cart(lon_d: float, lat_d: float, radius: float) -> Tuple[float, float, float]:
        lon_r = np.radians(lon_d)
        lat_r = np.radians(lat_d)
        x = radius * np.cos(lat_r) * np.cos(lon_r)
        y = radius * np.cos(lat_r) * np.sin(lon_r)
        z = radius * np.sin(lat_r)
        return float(x), float(y), float(z)

    for i in range(len(q_filt)):
        if not (np.isfinite(lon_deg[i]) and np.isfinite(lat_deg[i]) and np.isfinite(r_vec[i])):
            continue

        dlon = float(sigma_lon_deg[i]) if np.isfinite(sigma_lon_deg[i]) else 0.0
        dlat = float(sigma_lat_deg[i]) if np.isfinite(sigma_lat_deg[i]) else 0.0
        rr = float(r_vec[i])

        if dlon > 0:
            x1, y1, z1 = _sph_to_cart(lon_deg[i] - dlon, lat_deg[i], rr)
            x2, y2, z2 = _sph_to_cart(lon_deg[i] + dlon, lat_deg[i], rr)
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='0.45', linewidth=0.7, alpha=0.6, zorder=150)

        if dlat > 0:
            lat_lo = max(-89.9, lat_deg[i] - dlat)
            lat_hi = min(89.9, lat_deg[i] + dlat)
            x1, y1, z1 = _sph_to_cart(lon_deg[i], lat_lo, rr)
            x2, y2, z2 = _sph_to_cart(lon_deg[i], lat_hi, rr)
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='0.45', linewidth=0.7, alpha=0.6, zorder=150)

    if circle_fit_mode is not None and len(q_filt) >= 3:
        segments = _build_circle_segments(len(q_filt), circle_fit_segments, filtered_indices=filtered_idx)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(segments))))
        points_xyz = np.column_stack([q_filt, u_filt, v_filt])
        for i_seg, (s_idx, e_idx) in enumerate(segments):
            fit = _fit_circle_on_sphere(points_xyz[s_idx:e_idx + 1], mode=circle_fit_mode)
            if fit is None:
                continue
            arc = fit['arc_xyz']
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                    linestyle='--', linewidth=1.2, alpha=0.9,
                    color=color_cycle[i_seg], zorder=140)

    if len(q_filt) >= 1:
        mean_vec = np.array([np.mean(q_filt), np.mean(u_filt), np.mean(v_filt)])
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            azim = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
            elev = np.degrees(np.arcsin(mean_vec[2] / norm))
            ax.view_init(elev=elev, azim=azim)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02,
                        orientation='horizontal', fraction=0.04)
    cbar.set_label(color_label, fontsize=style['label'], labelpad=5)
    cbar.ax.tick_params(labelsize=style['tick'])

    ax.set_xlabel('Q', fontsize=style['label'], labelpad=-6)
    ax.set_ylabel('U', fontsize=style['label'], labelpad=-6)
    ax.set_zlabel('V', fontsize=style['label'], labelpad=-6)
    try:
        ax.xaxis.set_label_position('left')
        ax.yaxis.set_label_position('right')
    except Exception:
        pass
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
    if interactive:
        plt.show()
    _savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Poincaré sphere plot saved to {output_file}")
    plt.close()


def plot_poincare_projections(
        time_series_data: Dict,
        output_file: str = 'poincare_projections.png',
        projection_type: str = 'all',
        n_time_bins: Optional[int] = None,
        noise_fraction: float = 0.1,
        snr_threshold: float = 5.0,
        time_unit: str = 's',
        force_surface: bool = False,
        rm_results: Optional[Dict] = None,
        center: Optional[Tuple[float, float, float]] = None,
        noise_reference_data: Optional[Dict] = None,
        circle_fit_mode: Optional[str] = None,
        circle_fit_segments: Optional[List[Tuple[int, int]]] = None):
    """
    Generate a 2×2 panel of 2-D cropped projections of the Poincaré sphere.

    The four projections are:

    * **Gnomonic** (central/tangent-plane) – every great-circle arc (Faraday
      rotation path) maps to a straight line, making it ideal for measuring the
      PA rotation rate at a glance.
    * **Stereographic** – conformal (angle-preserving), so the local shape of
      the polarisation ellipse trajectory is faithfully rendered.  Points near
      the projection centre look exactly as they do on the sphere.
    * **Azimuthal equidistant** – preserves arc-length from the projection
      centre, useful for comparing radial excursions in different directions.
    * **Orthographic** – the "view from outside" hemisphere projection.
      Intuitive because it mimics a photograph of the sphere from far away; the
      equatorial Q–U plane shows the linear-polarisation disc.

    All four projections are centred on the mean Stokes vector of the data so
    the track is always near the centre where distortion is smallest.  The
    gnomonic projection is additionally cropped to ±45° from the tangent point
    (a reasonable half-sky for most FRB/pulsar PA swings); the others show the
    full visible hemisphere.

    Colour encodes time (same viridis palette as ``plot_poincare_sphere``).

    Parameters
    ----------
    time_series_data : dict
        Dictionary with keys ``'time'``, ``'I'``, ``'Q'``, ``'U'`` (and
        optionally ``'V'``) containing 2-D arrays (time × frequency or
        frequency × time).
    output_file : str
        Path for the saved PNG.
    n_time_bins : int, optional
        Number of time bins to average into before projecting.  ``None`` keeps
        all samples.
    noise_fraction : float
        Fraction of the time axis used to estimate off-pulse noise.
    snr_threshold : float
        Minimum linear-polarisation SNR threshold.
    time_unit : str
        Time unit label (``'s'``, ``'ms'``, or ``'us'``).
    force_surface : bool
        If ``True`` normalise all vectors to the unit sphere surface before
        projecting.
    rm_results : dict, optional
        Pre-computed binned Stokes parameters from ``fit_rm_time_series``.
        When supplied the function uses the pre-computed ``q_bin``/``u_bin``
        values instead of re-binning from the raw cubes.
    center : tuple(float, float, float), optional
        (x, y, z) unit vector used as the projection centre for all four
        projections.  Defaults to the mean Stokes vector of the data.
    """
    style = plot_style()

    if time_series_data is None:
        raise ValueError("plot_poincare_projections requires time_series_data.")

    I_cube = time_series_data['I']
    Q_cube = time_series_data['Q']
    U_cube = time_series_data['U']
    V_cube = time_series_data.get('V', None)
    times  = time_series_data['time']
    n_time = len(times)
    noise_ref = noise_reference_data if noise_reference_data is not None else time_series_data

    if rm_results is not None and 'q_bin' in rm_results:
        q_norm     = np.array(rm_results['q_bin'])
        u_norm     = np.array(rm_results['u_bin'])
        v_norm     = np.array(rm_results.get('v_bin', np.zeros_like(q_norm)))
        color_axis = np.array(rm_results['time'])
        orig_idx = np.arange(q_norm.size, dtype=int)
        pol_list   = np.sqrt(q_norm**2 + u_norm**2)
        mask = np.ones(q_norm.size, dtype=bool)
        if 'valid_bins' in rm_results:
            mask &= np.asarray(rm_results['valid_bins'], dtype=bool)
        if 'pa_ea_valid' in rm_results:
            mask &= np.asarray(rm_results['pa_ea_valid'], dtype=bool)
        if not np.all(mask):
            q_norm, u_norm, v_norm = q_norm[mask], u_norm[mask], v_norm[mask]
            color_axis = color_axis[mask]
            pol_list   = pol_list[mask]
            orig_idx = orig_idx[mask]
        notnan = (~np.isnan(q_norm)) & (~np.isnan(u_norm)) & (~np.isnan(v_norm))
        q_norm, u_norm, v_norm = q_norm[notnan], u_norm[notnan], v_norm[notnan]
        color_axis = color_axis[notnan]
        pol_list   = pol_list[notnan]
        orig_idx = orig_idx[notnan]
    else:
        if I_cube.ndim != 2:
            raise ValueError("Time-series data must be 2D.")
        time_axis = 0 if I_cube.shape[0] == n_time else 1
        bin_size  = max(1, n_time // n_time_bins) if (n_time_bins and n_time_bins > 0) else 1
        n_bins    = (n_time + bin_size - 1) // bin_size
        q_list, u_list, v_list, pol_list, time_list = [], [], [], [], []
        for i in range(n_bins):
            s, e = i * bin_size, min((i + 1) * bin_size, n_time)
            if e <= s:
                continue
            time_list.append(np.mean(times[s:e]))
            sl = (slice(s, e), slice(None)) if time_axis == 0 else (slice(None), slice(s, e))
            I_m = np.nanmean(I_cube[sl]);  Q_m = np.nanmean(Q_cube[sl])
            U_m = np.nanmean(U_cube[sl])
            V_m = np.nanmean(V_cube[sl]) if V_cube is not None else 0.0
            q_list.append(Q_m / (I_m + 1e-10)); u_list.append(U_m / (I_m + 1e-10))
            v_list.append(V_m / (I_m + 1e-10))
            pol_list.append(np.sqrt(Q_m**2 + U_m**2))
        q_norm     = np.array(q_list)
        u_norm     = np.array(u_list)
        v_norm     = np.array(v_list)
        color_axis = np.array(time_list)
        orig_idx = np.arange(q_norm.size, dtype=int)

    unit = time_unit.lower()
    if unit == 'ms':
        color_axis *= 1e3; color_label = "Time (ms)"
    elif unit in ('us', 'µs'):
        color_axis *= 1e6; color_label = "Time (µs)"
    else:
        color_label = "Time (s)"

    n_frac   = max(1, int(len(pol_list) * noise_fraction))
    sigma_p  = np.nanstd(pol_list[:n_frac]) or 1e-10
    snr      = np.array(pol_list) / sigma_p
    if rm_results is not None and 'valid_bins' in rm_results:
        mask = np.ones(q_norm.shape[0], dtype=bool)
    else:
        mask = snr > snr_threshold
        if np.sum(mask) < 2:
            mask = snr > 2.0
        if np.sum(mask) < 2:
            print("Error: fewer than 2 points survive SNR cut in projections.")
            return

    filtered_idx = np.asarray(orig_idx[mask], dtype=int)
    q_f = q_norm[mask]
    u_f = u_norm[mask]
    v_f = v_norm[mask]
    c_f = color_axis[mask]

    if force_surface:
        r = np.sqrt(q_f**2 + u_f**2 + v_f**2); r[r == 0] = 1.0
        q_f /= r; u_f /= r; v_f /= r

    r_f   = np.sqrt(q_f**2 + u_f**2 + v_f**2)
    r_f   = np.where(r_f < 1e-10, 1e-10, r_f)
    lon_f = np.degrees(np.arctan2(u_f, q_f))
    lat_f = np.degrees(np.arcsin(np.clip(v_f / r_f, -1.0, 1.0)))

    point_times_s = np.asarray(c_f, dtype=float) / (1e3 if unit == 'ms' else (1e6 if unit in ('us', 'µs') else 1.0))
    sigma_q, sigma_u, sigma_v = _compute_poincare_point_errors(
        noise_ref,
        point_times=point_times_s,
        noise_fraction=noise_fraction,
    )
    sigma_lon_deg, sigma_lat_deg = _poincare_angle_errors_deg(
        q_f, u_f, v_f, sigma_q, sigma_u, sigma_v
    )

    circle_fits = []
    if circle_fit_mode is not None and len(q_f) >= 3:
        segments = _build_circle_segments(len(q_f), circle_fit_segments, filtered_indices=filtered_idx)
        points_xyz = np.column_stack([q_f, u_f, v_f])
        for i_seg, (s_idx, e_idx) in enumerate(segments):
            fit = _fit_circle_on_sphere(points_xyz[s_idx:e_idx + 1], mode=circle_fit_mode)
            if fit is None:
                continue
            arc = fit['arc_xyz']
            lon_arc = np.degrees(np.arctan2(arc[:, 1], arc[:, 0]))
            lat_arc = np.degrees(np.arcsin(np.clip(arc[:, 2], -1.0, 1.0)))
            circle_fits.append((i_seg, lon_arc, lat_arc))

    if center is not None:
        cx, cy, cz = np.array(center, dtype=float)
        cn = np.sqrt(cx**2 + cy**2 + cz**2)
        cx, cy, cz = (cx/cn, cy/cn, cz/cn) if cn > 1e-10 else (0., 0., 1.)
    else:
        cx = np.mean(q_f); cy = np.mean(u_f); cz = np.mean(v_f)
        cn = np.sqrt(cx**2 + cy**2 + cz**2)
        cx, cy, cz = (cx/cn, cy/cn, cz/cn) if cn > 1e-10 else (0., 0., 1.)

    lon0 = np.degrees(np.arctan2(cy, cx))
    lat0 = np.degrees(np.arcsin(np.clip(cz, -1.0, 1.0)))

    try:
        from mpl_toolkits.basemap import Basemap as _Basemap
    except ImportError:
        print("Warning: mpl_toolkits.basemap not available; skipping projection panel.")
        return

    _btest = _Basemap(projection='gnom', lat_0=lat0, lon_0=lon0,
                      width=2e7, height=2e7, rsphere=1.0)
    mx_f, my_f = _btest(lon_f, lat_f)
    mx_f = np.array(mx_f, dtype=float); my_f = np.array(my_f, dtype=float)
    fin   = np.isfinite(mx_f) & np.isfinite(my_f)
    if not np.any(fin):
        print("Warning: no finite projected points; skipping projection panel.")
        return
    span  = max(np.ptp(mx_f[fin]), np.ptp(my_f[fin]))
    half  = max(span * 0.5 * 1.20, 0.05)
    half = max(half, np.tan(np.radians(30)))

    ang_half  = np.degrees(np.arctan(half))
    grid_step = 10 if ang_half < 45 else 30

    projection_map = {
        'gnom': ('gnom', 'Gnomonic\n(great circles → straight lines)'),
        'stere': ('stere', 'Stereographic\n(conformal / angle-preserving)'),
        'aeqd': ('aeqd', 'Azimuthal Equidistant\n(arc-length preserved)'),
        'ortho': ('ortho', 'Orthographic\n(hemisphere view)'),
    }
    proj_key = str(projection_type).lower()
    if proj_key == 'all':
        projections = [
            projection_map['gnom'],
            projection_map['stere'],
            projection_map['aeqd'],
            projection_map['ortho'],
        ]
        fig, axes = plt.subplots(2, 2, figsize=pub_figsize(height_ratio=1.0, min_height=7.0))
        axes = axes.ravel()
        is_all = True
    else:
        if proj_key not in projection_map:
            raise ValueError(
                "Invalid projection_type. Choose from: all, gnom, stere, aeqd, ortho"
            )
        projections = [projection_map[proj_key]]
        fig, ax_single = plt.subplots(1, 1, figsize=pub_figsize(height_ratio=0.75, min_height=4.8))
        axes = [ax_single]
        is_all = False

    norm = plt.Normalize(vmin=np.nanmin(c_f), vmax=np.nanmax(c_f))

    for ax, (proj, _title) in zip(axes, projections):
        if proj == 'ortho':
            bsmp = _Basemap(projection='ortho', lat_0=lat0, lon_0=lon0,
                            ax=ax, rsphere=1.0)
        else:
            bsmp = _Basemap(projection=proj, lat_0=lat0, lon_0=lon0,
                            width=2*half, height=2*half,
                            ax=ax, rsphere=1.0)

        bsmp.drawmapboundary(fill_color='white', zorder=0)
        if proj == 'ortho':
            bsmp.drawparallels(
                np.arange(-90, 91, grid_step),
                labels=[False, False, False, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
            bsmp.drawmeridians(
                np.arange(-180, 181, grid_step),
                labels=[False, False, False, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
        else:
            bsmp.drawparallels(
                np.arange(-90, 91, grid_step),
                labels=[True, False, False, True],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)
            bsmp.drawmeridians(
                np.arange(-180, 181, grid_step),
                labels=[False, True, True, False],
                fontsize=style['annotation'], linewidth=0.5, color='lightgray', zorder=1)

        pole_labels = {
            '+Q': ( 0.0,  0.0), '-Q': (180.0,  0.0),
            '+U': (90.0,  0.0), '-U': (-90.0,  0.0),
            '+V': ( 0.0, 90.0), '-V': (  0.0, -90.0),
        }
        for lbl, (plon, plat) in pole_labels.items():
            try:
                px, py = bsmp(plon, plat)
                if np.isfinite(px) and np.isfinite(py):
                    ax.annotate(lbl, (px, py), fontsize=style['annotation'], color='steelblue',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          ec='none', alpha=0.6), zorder=5)
            except Exception:
                pass

        sx, sy = bsmp(lon_f, lat_f)
        sx = np.array(sx, dtype=float); sy = np.array(sy, dtype=float)
        fin_s = np.isfinite(sx) & np.isfinite(sy)
        if np.any(fin_s):
            ax.scatter(sx[fin_s], sy[fin_s],
                       c=c_f[fin_s], cmap='viridis', norm=norm,
                       s=55, edgecolors='black', linewidths=0.6,
                       zorder=4, alpha=1.0)

            for j in np.where(fin_s)[0]:
                try:
                    x0, y0 = bsmp(lon_f[j], lat_f[j])
                    x_lon, y_lon = bsmp(lon_f[j] + sigma_lon_deg[j], lat_f[j])
                    x_lat, y_lat = bsmp(lon_f[j], lat_f[j] + sigma_lat_deg[j])
                    if not (np.isfinite(x0) and np.isfinite(y0)):
                        continue
                    dx = np.sqrt((x_lon - x0) ** 2 + (x_lat - x0) ** 2)
                    dy = np.sqrt((y_lon - y0) ** 2 + (y_lat - y0) ** 2)
                    if np.isfinite(dx) and np.isfinite(dy):
                        ax.errorbar(x0, y0, xerr=dx, yerr=dy, fmt='none',
                                    ecolor='0.45', elinewidth=0.7, alpha=0.5,
                                    capsize=0, zorder=3)
                except Exception:
                    continue

        if circle_fits:
            color_cycle = plt.cm.tab10(np.linspace(0, 1, max(1, len(circle_fits))))
            for i_seg, lon_arc, lat_arc in circle_fits:
                tx, ty = bsmp(lon_arc, lat_arc)
                tx = np.asarray(tx, dtype=float)
                ty = np.asarray(ty, dtype=float)
                ok = np.isfinite(tx) & np.isfinite(ty)
                if np.sum(ok) >= 2:
                    ax.plot(tx[ok], ty[ok], linestyle='--', linewidth=1.2,
                            color=color_cycle[i_seg], alpha=0.9, zorder=2)

        ax.tick_params(axis='both', labelsize=style['tick'])

    if is_all:
        fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.08,
                            hspace=0.15, wspace=0.15)
        cax = fig.add_axes([0.25, 0.02, 0.50, 0.016])
    else:
        fig.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.14)
        cax = fig.add_axes([0.22, 0.05, 0.56, 0.025])
    sm  = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(color_label, fontsize=style['label'])
    cb.ax.tick_params(labelsize=style['tick'])

    _savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Poincaré projection panel saved to {output_file}")
    plt.close()


def plot_rm_time_series(time_array: np.ndarray, rm_results: Dict,
                        output_file: str = 'rm_time_series.png',
                        time_profile: Optional[np.ndarray] = None,
                        separate_peaks: bool = False,
                        min_gap_bins: int = 3,
                        min_peak_bins: int = 3,
                        max_merge_gap: int = 0,
                        time_series_data: Optional[Dict] = None,
                        freq_hz: Optional[np.ndarray] = None,
                        n_rm_bins: int = 20,
                        n_pa_bins: int = 50,
                        noise_fraction: float = 0.1):
    """
    Plot RM as a function of time.

    Parameters:
    -----------
    ...
    n_pa_bins : int
        Number of bins for PA/EA plot (0 = full resolution, default: 0)
    ...
    """
    from .data_io import find_peak_regions  # local import to avoid circular

    style = plot_style()

    if separate_peaks and time_profile is not None:
        snr_array = rm_results.get('snr', None)
        peak_regions = find_peak_regions(time_profile, snr_array, min_gap_bins,
                                         min_peak_bins, max_merge_gap)
        n_peaks = len(peak_regions)
        print(f"\nIdentified {n_peaks} separate peak region(s):")
        for i, (start, end) in enumerate(peak_regions):
            print(f"  Peak {i+1}: bins {start}-{end} ({end-start+1} bins)")
    else:
        peak_regions = [(0, len(time_array) - 1)]
        n_peaks = 1

    rm_ts_height = max(6.8, 2.25 * 3)
    rm_ts_width = SINGLE_COLUMN_WIDTH_IN if n_peaks == 1 else min(TWO_COLUMN_WIDTH_IN, SINGLE_COLUMN_WIDTH_IN * n_peaks)
    fig = plt.figure(figsize=(rm_ts_width, rm_ts_height))
    gs = GridSpec(3, n_peaks, figure=fig, hspace=0, wspace=0.3)
    axes = np.empty((3, n_peaks), dtype=object)
    for col in range(n_peaks):
        for row in range(3):
            if row == 0:
                axes[row, col] = fig.add_subplot(gs[row, col])
            else:
                axes[row, col] = fig.add_subplot(gs[row, col], sharex=axes[0, col])

    TICK_LENGTH = 3

    if 'snr' in rm_results and np.any(rm_results['snr'] > 0):
        snr_threshold = 5.0
        good_signal = rm_results['snr'] >= snr_threshold
    else:
        good_signal = np.ones(len(time_array), dtype=bool)

    for peak_idx, (start_idx, end_idx) in enumerate(peak_regions):
        peak_mask = np.zeros(len(time_array), dtype=bool)
        peak_mask[start_idx:end_idx+1] = True

        time_peak = time_array[peak_mask]
        good_signal_peak = good_signal[peak_mask]

        full_time = None
        snr_full = None
        P_frac_full = None
        L_frac_full = None
        V_frac_full = None
        if time_series_data is not None and 'time' in time_series_data:
            full_time = np.asarray(time_series_data['time'])
            if time_series_data['I'].ndim == 2:
                if time_series_data['I'].shape[0] == len(full_time):
                    time_axis_dim = 0
                elif time_series_data['I'].shape[1] == len(full_time):
                    time_axis_dim = 1
                else:
                    time_axis_dim = 0
            else:
                time_axis_dim = 0

            if time_axis_dim == 0:
                I_full = np.mean(time_series_data['I'], axis=1)
                Q_full = np.mean(time_series_data['Q'], axis=1)
                U_full = np.mean(time_series_data['U'], axis=1)
                V_full = np.mean(time_series_data['V'], axis=1) if 'V' in time_series_data else np.zeros_like(I_full)
            else:
                I_full = np.mean(time_series_data['I'], axis=0)
                Q_full = np.mean(time_series_data['Q'], axis=0)
                U_full = np.mean(time_series_data['U'], axis=0)
                V_full = np.mean(time_series_data['V'], axis=0) if 'V' in time_series_data else np.zeros_like(I_full)

            pol_int_full = np.sqrt(Q_full**2 + U_full**2)
            P_frac_full = np.sqrt(Q_full**2 + U_full**2 + V_full**2) / (I_full + 1e-10)
            L_full = np.sqrt(Q_full**2 + U_full**2)
            L_frac_full = L_full / (I_full + 1e-10)
            V_frac_full = V_full / (I_full + 1e-10)

            if len(time_peak) > 0:
                tmin = time_peak.min()
                tmax = time_peak.max()
                if len(full_time) > 1:
                    dt = np.median(np.diff(full_time))
                    pad = dt / 2.0
                else:
                    pad = 0.0
                full_mask = (full_time >= (tmin - pad)) & (full_time <= (tmax + pad))
            else:
                full_mask = np.ones_like(full_time, dtype=bool)

            if not np.any(full_mask):
                if len(full_time) > 0:
                    centre = 0.5 * (tmin + tmax) if len(time_peak) > 0 else full_time[0]
                    idx = int(np.argmin(np.abs(full_time - centre)))
                    full_mask = np.zeros_like(full_time, dtype=bool)
                    full_mask[idx] = True
                else:
                    full_mask = np.ones_like(full_time, dtype=bool)

            n_frac = max(1, int(len(I_full) * noise_fraction))
            noise_est = np.nanstd(I_full[:n_frac])
            if noise_est <= 0:
                mad = np.nanmedian(np.abs(I_full - np.nanmedian(I_full)))
                if mad > 0:
                    noise_est = mad / 0.6745
                else:
                    noise_est = max(np.nanmedian(I_full) * 0.1, 1e-10)

            snr_full = pol_int_full / (noise_est + 1e-10)

        # ── Row 0: PA / EA (TOP panel) ─────────────────────────────────────
        ax_pa = axes[0, peak_idx]
        if time_series_data is not None and full_time is not None:
            Q_vals = Q_full[full_mask]
            U_vals = U_full[full_mask]
            V_vals = V_full[full_mask] if 'V' in time_series_data else np.zeros_like(Q_vals)
            times_ms = full_time[full_mask] * 1e3

            pa_rad = 0.5 * np.arctan2(U_vals, Q_vals)
            pa_deg = np.degrees(np.unwrap(pa_rad))

            P_amp = np.sqrt(Q_vals**2 + U_vals**2 + V_vals**2) + 1e-10
            sin_arg = np.clip(V_vals / P_amp, -1.0, 1.0)
            ea_rad = 0.5 * np.arcsin(sin_arg)
            ea_deg = np.degrees(ea_rad)

            n_frac_pa = max(1, int(len(I_full) * noise_fraction))
            sigma_Q = np.nanstd(Q_full[:n_frac_pa])
            sigma_U = np.nanstd(U_full[:n_frac_pa])
            sigma_V = np.nanstd(V_full[:n_frac_pa]) if 'V' in time_series_data else 0.0
            if sigma_Q <= 0:
                mad_q = np.nanmedian(np.abs(Q_full - np.nanmedian(Q_full)))
                sigma_Q = mad_q / 0.6745 if mad_q > 0 else 1e-10
            if sigma_U <= 0:
                mad_u = np.nanmedian(np.abs(U_full - np.nanmedian(U_full)))
                sigma_U = mad_u / 0.6745 if mad_u > 0 else 1e-10
            if sigma_V <= 0 and 'V' in time_series_data:
                mad_v = np.nanmedian(np.abs(V_full - np.nanmedian(V_full)))
                sigma_V = mad_v / 0.6745 if mad_v > 0 else 1e-10

            P_lin_sq = Q_vals**2 + U_vals**2 + 1e-20
            pa_sigma_rad = 0.5 * np.sqrt((U_vals**2 * sigma_Q**2 + Q_vals**2 * sigma_U**2) / (P_lin_sq**2))
            pa_sigma_deg = np.degrees(pa_sigma_rad)

            sigma_P = np.sqrt((Q_vals**2 * sigma_Q**2 + U_vals**2 * sigma_U**2)) / (P_amp + 1e-10)
            sigma_VoverP = np.sqrt((sigma_V**2 / (P_amp**2)) + ((V_vals**2) * (sigma_P**2) / (P_amp**2 + 1e-20)))
            denom = np.sqrt(1.0 - (V_vals / P_amp)**2 + 1e-20)
            ea_sigma_rad = 0.5 * (sigma_VoverP / denom)
            ea_sigma_deg = np.degrees(ea_sigma_rad)

            snr_i_full_pa = I_full / (noise_est + 1e-10)
            badi_pa = snr_i_full_pa < 2.0
            combined_pa = ~badi_pa[full_mask]
            bad_pa_err = pa_sigma_deg > 50.0
            bad_ea_err = ea_sigma_deg > 50.0
            mask_pa = combined_pa & ~bad_pa_err & ~bad_ea_err

            if n_pa_bins > 0 and np.any(mask_pa):
                t_good = times_ms[mask_pa]
                pa_good = pa_deg[mask_pa]
                ea_good = ea_deg[mask_pa]
                pa_sig_good = pa_sigma_deg[mask_pa]
                ea_sig_good = ea_sigma_deg[mask_pa]

                bin_edges = np.linspace(t_good.min(), t_good.max(), n_pa_bins + 1)
                bin_centres, pa_binned, pa_binned_err = [], [], []
                ea_binned, ea_binned_err = [], []
                for b in range(n_pa_bins):
                    sel = (t_good >= bin_edges[b]) & (t_good < bin_edges[b + 1])
                    if b == n_pa_bins - 1:
                        sel = (t_good >= bin_edges[b]) & (t_good <= bin_edges[b + 1])
                    if not np.any(sel):
                        continue
                    w_pa = 1.0 / (pa_sig_good[sel]**2 + 1e-20)
                    w_ea = 1.0 / (ea_sig_good[sel]**2 + 1e-20)
                    bin_centres.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                    pa_binned.append(np.sum(w_pa * pa_good[sel]) / np.sum(w_pa))
                    pa_binned_err.append(1.0 / np.sqrt(np.sum(w_pa)))
                    ea_binned.append(np.sum(w_ea * ea_good[sel]) / np.sum(w_ea))
                    ea_binned_err.append(1.0 / np.sqrt(np.sum(w_ea)))

                bc = np.array(bin_centres)
                pa_b = np.array(pa_binned)
                pa_be = np.array(pa_binned_err)
                ea_b = np.array(ea_binned)
                ea_be = np.array(ea_binned_err)
                bin_ok = (
                    np.isfinite(bc)
                    & np.isfinite(pa_b) & np.isfinite(pa_be)
                    & np.isfinite(ea_b) & np.isfinite(ea_be)
                    & (pa_be <= 50.0) & (ea_be <= 50.0)
                )
                if np.any(bin_ok):
                    ax_pa.errorbar(bc[bin_ok], ea_b[bin_ok], yerr=ea_be[bin_ok], fmt='s', color='b',
                                   markersize=4, capsize=2, label='EA', zorder=2)
                    ax_pa.errorbar(bc[bin_ok], pa_b[bin_ok], yerr=pa_be[bin_ok], fmt='o', color='r',
                                   markersize=4, capsize=2, label='PA', zorder=2)
            else:
                def scatter_runs(x, y, axis, **kwargs):
                    if len(x) == 0:
                        return
                    idx = np.arange(len(x))
                    splits = np.where(np.diff(idx) != 1)[0] + 1
                    for seg in np.split(idx, splits):
                        if len(seg) > 0:
                            axis.scatter(x[seg], y[seg], **kwargs)

                scatter_runs(times_ms[mask_pa], pa_deg[mask_pa], ax_pa, color='r', s=8, label='PA', zorder=2)
                scatter_runs(times_ms[mask_pa], ea_deg[mask_pa], ax_pa, color='b', s=8, label='EA', zorder=2)
                if np.any(mask_pa):
                    ax_pa.errorbar(times_ms[mask_pa], ea_deg[mask_pa], yerr=ea_sigma_deg[mask_pa],
                                   fmt='none', ecolor='gray', alpha=0.6, capsize=2, zorder=1)
                    ax_pa.errorbar(times_ms[mask_pa], pa_deg[mask_pa], yerr=pa_sigma_deg[mask_pa],
                                   fmt='none', ecolor='gray', alpha=0.6, capsize=2, zorder=1)

        ax_pa.set_ylabel('Angle (deg)', fontsize=style['label'])
        if n_peaks > 1:
            ax_pa.set_title(f'Peak {peak_idx+1}: PA & EA', fontsize=style['title'], fontweight='bold')
        ax_pa.grid(True, alpha=0.3)
        ax_pa.legend(loc='best', fontsize=style['legend'])
        ax_pa.tick_params(axis='both', labelsize=style['tick'], length=TICK_LENGTH, labelbottom=False)

        # ── Row 1: Pulse Profile + RM (MIDDLE panel) ────────────────────────
        ax_top = axes[1, peak_idx]
        ax_top_twin = ax_top.twinx()
        ax_top_twin.yaxis.set_label_position('right')
        ax_top_twin.yaxis.tick_right()
        ax_top_twin.set_ylabel('RM (rad/m²)', fontsize=style['label'], color='darkorange')
        ax_top_twin.tick_params(axis='y', colors='darkorange', labelsize=style['tick'], length=TICK_LENGTH)

        rm_peak = rm_results['rm'][peak_mask]
        tms = time_peak * 1e3
        rm_err_peak = rm_results.get('rm_err', None)
        if rm_err_peak is not None:
            rm_err_peak = rm_err_peak[peak_mask]
            ax_top_twin.errorbar(tms, rm_peak, yerr=rm_err_peak,
                                 fmt='o-', color='darkorange', markersize=3,
                                 linewidth=1.5, capsize=2, alpha=1,
                                 label='RM')
        else:
            ax_top_twin.plot(tms, rm_peak, 'darkorange-o', linewidth=1.5, markersize=3, label='RM')

        ax_top.plot(full_time[full_mask] * 1e3, I_full[full_mask], 'k-', linewidth=1.5, label='I')
        ax_top.set_ylabel(r'$S$ (arb.)', fontsize=style['label'])
        ax_top.tick_params(axis='y', labelsize=style['tick'], length=TICK_LENGTH)
        ax_top.plot(full_time[full_mask] * 1e3, L_full[full_mask], 'r-', linewidth=1.5, label='L')
        ax_top.plot(full_time[full_mask] * 1e3, V_full[full_mask], 'b-', linewidth=1.5, label='V')

        if time_series_data is not None and freq_hz is not None and not rm_results.get('is_binned', False):
            if time_series_data['I'].shape[0] == len(time_array):
                time_axis_dim = 0
            else:
                time_axis_dim = 1

            n_time = np.sum(peak_mask)
            bin_size = max(1, n_time // n_rm_bins)
            n_bins_actual = (n_time + bin_size - 1) // bin_size

            binned_rm = []
            binned_rm_err = []
            binned_time = []

            if time_axis_dim == 0:
                I_full_tmp = np.mean(time_series_data['I'], axis=1)
                Q_tmp = time_series_data['Q']
                U_tmp = time_series_data['U']
            else:
                I_full_tmp = np.mean(time_series_data['I'], axis=0)
                Q_tmp = time_series_data['Q']
                U_tmp = time_series_data['U']
            n_frac_tmp = max(1, int(len(I_full_tmp) * noise_fraction))
            if time_axis_dim == 0:
                q_off_tmp = Q_tmp[:n_frac_tmp, :]
                u_off_tmp = U_tmp[:n_frac_tmp, :]
            else:
                q_off_tmp = Q_tmp[:, :n_frac_tmp]
                u_off_tmp = U_tmp[:, :n_frac_tmp]
            q_std_chan_tmp = np.nanstd(q_off_tmp, axis=0 if time_axis_dim == 0 else 1)
            u_std_chan_tmp = np.nanstd(u_off_tmp, axis=0 if time_axis_dim == 0 else 1)
            noise_q_tmp = np.nanmedian(q_std_chan_tmp) if np.nanmedian(q_std_chan_tmp) > 0 else (np.nanmean(q_std_chan_tmp) if np.nanmean(q_std_chan_tmp) > 0 else 1e-10)
            noise_u_tmp = np.nanmedian(u_std_chan_tmp) if np.nanmedian(u_std_chan_tmp) > 0 else (np.nanmean(u_std_chan_tmp) if np.nanmean(u_std_chan_tmp) > 0 else 1e-10)
            noise_i_tmp = np.nanstd(I_full_tmp[:n_frac_tmp])
            if noise_i_tmp <= 0:
                mad_tmp = np.nanmedian(np.abs(I_full_tmp - np.nanmedian(I_full_tmp)))
                if mad_tmp > 0:
                    noise_i_tmp = mad_tmp / 0.6745
                else:
                    noise_i_tmp = max(np.nanmedian(I_full_tmp) * 0.1, 1e-10)

            print(f"  Computing binned RM with {n_bins_actual} bins...")

            for bin_idx in range(n_bins_actual):
                bin_start = start_idx + bin_idx * bin_size
                bin_end = min(start_idx + (bin_idx + 1) * bin_size, end_idx + 1)
                if bin_end <= bin_start:
                    continue
                if time_axis_dim == 0:
                    i_avg = np.mean(time_series_data['I'][bin_start:bin_end, :], axis=0)
                    q_avg = np.mean(time_series_data['Q'][bin_start:bin_end, :], axis=0)
                    u_avg = np.mean(time_series_data['U'][bin_start:bin_end, :], axis=0)
                    v_avg = np.mean(time_series_data['V'][bin_start:bin_end, :], axis=0) if 'V' in time_series_data else None
                else:
                    i_avg = np.mean(time_series_data['I'][:, bin_start:bin_end], axis=1)
                    q_avg = np.mean(time_series_data['Q'][:, bin_start:bin_end], axis=1)
                    u_avg = np.mean(time_series_data['U'][:, bin_start:bin_end], axis=1)
                    v_avg = np.mean(time_series_data['V'][:, bin_start:bin_end], axis=1) if 'V' in time_series_data else None

                fitter = RMFitter(freq_hz, i_avg, q_avg, u_avg, v_avg)
                result = fitter._fit_rm_with_rmtools(rm_range=(-1000, 1000), n_rm=500,
                                                     noise_i=noise_i_tmp,
                                                     noise_q=noise_q_tmp, noise_u=noise_u_tmp)
                rm_fit = result.get('rm_clean_peak', result.get('rm_peak', np.nan))
                rm_err = result.get('rm_clean_err', result.get('rm_err', result.get('noise_estimate', 0) * 2))
                binned_rm.append(rm_fit)
                binned_rm_err.append(rm_err)
                binned_time.append(time_array[(bin_start + bin_end) // 2])

            ax_top_twin.errorbar(np.array(binned_time) * 1e3, binned_rm, yerr=binned_rm_err,
                                 fmt='o-', color='red', markersize=5, linewidth=2, capsize=3,
                                 label=f'Binned RM ({n_bins_actual} bins)')
            ax_top_twin.set_ylabel('RM (rad/m²)', fontsize=style['label'], color='red')
            ax_top_twin.tick_params(axis='y', labelcolor='red', labelsize=style['tick'], length=TICK_LENGTH)

        if n_peaks > 1:
            ax_top.set_title(f'Peak {peak_idx+1}: Pulse Profile & RM', fontsize=style['title'], fontweight='bold')
        ax_top.grid(True, alpha=0.3)
        if ax_top.get_legend() is not None:
            ax_top.get_legend().remove()
        if ax_top_twin.get_legend() is not None:
            ax_top_twin.get_legend().remove()

        handles1, labels1 = ax_top.get_legend_handles_labels()
        handles2, labels2 = ax_top_twin.get_legend_handles_labels()
        if peak_idx == 0:
            all_handles = list(handles1) + list(handles2)
            all_labels = list(labels1) + list(labels2)
            final_handles = []
            final_labels = []
            seen = set()
            for h, lab in zip(all_handles, all_labels):
                if not lab or lab in seen:
                    continue
                if 'Binned RM' in lab or 'Binned' in lab:
                    continue
                if lab.strip() == 'RM':
                    keep = False
                    try:
                        col = getattr(h, 'get_color', lambda: None)()
                    except Exception:
                        col = None
                    if isinstance(col, str) and col.lower() in ('lawngreen',):
                        keep = True
                    else:
                        try:
                            import numpy as _np
                            colarr = _np.asarray(col)
                            if colarr.size >= 3 and _np.allclose(colarr[:3], _np.array([0.486, 0.988, 0.0]), atol=0.05):
                                keep = True
                        except Exception:
                            keep = False
                    if not keep:
                        continue
                final_handles.append(h)
                final_labels.append(lab)
                seen.add(lab)
            if final_handles:
                ax_top.legend(final_handles, final_labels, fontsize=style['legend'], loc='best')
        ax_top.tick_params(axis='x', labelbottom=False, length=TICK_LENGTH)

        # ── Row 2: Polarisation fractions (BOTTOM panel) ────────────────────
        ax3 = axes[2, peak_idx]

        if time_series_data is not None and full_time is not None:
            n_frac = max(1, int(len(I_full) * noise_fraction))
            noise_est = np.nanstd(I_full[:n_frac])
            if noise_est <= 0:
                mad = np.nanmedian(np.abs(I_full - np.nanmedian(I_full)))
                if mad > 0:
                    noise_est = mad / 0.6745
                else:
                    noise_est = max(np.nanmedian(I_full) * 0.1, 1e-10)
            err_frac = noise_est / (I_full + 1e-10)

            times_ms = full_time[full_mask] * 1e3
            snr_i_full = I_full / (noise_est + 1e-10)
            badi_full = snr_i_full < 2.0
            P_frac_full[badi_full] = np.nan
            L_frac_full[badi_full] = np.nan
            V_frac_full[badi_full] = np.nan

            signal_mask = ~badi_full[full_mask]
            if not np.any(signal_mask):
                signal_mask = np.ones_like(signal_mask, dtype=bool)
            rm_mask = np.ones_like(times_ms, dtype=bool)
            if np.any(good_signal_peak):
                good_times = time_peak[good_signal_peak] * 1e3
                dt = np.median(np.diff(time_peak)) * 1e3 if len(time_peak) > 1 else 0
                tol = dt/2 + 1e-9
                rm_mask = np.array([np.any(np.abs(t - good_times) <= tol) for t in times_ms])
            combined = signal_mask

            full_indices = np.where(full_mask)[0]
            bin_mask = None

            if 'P_frac_bin' in rm_results:
                bt = np.asarray(rm_results['time']) * 1e3
                pf_bin = np.asarray(rm_results['P_frac_bin'])
                lf_bin = np.asarray(rm_results['L_frac_bin'])
                vf_bin = np.asarray(rm_results.get('V_frac_bin', []))

                if 'valid_bins' in rm_results:
                    bin_mask = np.asarray(rm_results['valid_bins'], dtype=bool)
                else:
                    if full_time is not None:
                        idx = np.argmin(np.abs(full_time[:, None] - (bt/1e3)[None, :]), axis=0)
                        bin_mask = combined[idx]
                    else:
                        bin_mask = np.ones_like(bt, dtype=bool)

                if np.any(bin_mask):
                    ax3.plot(bt[bin_mask], pf_bin[bin_mask], 'k--', linewidth=2, zorder=1, label='P/I')
                    ax3.plot(bt[bin_mask], lf_bin[bin_mask], 'r--', linewidth=2, zorder=1, label='L/I')
                    ax3.scatter(bt[bin_mask], lf_bin[bin_mask], 25, 'r', label=None, zorder=20)
                    if 'V_frac_bin' in rm_results and vf_bin.size:
                        ax3.plot(bt[bin_mask], vf_bin[bin_mask], 'b--', linewidth=2, zorder=1, label='V/I')
                        ax3.scatter(bt[bin_mask], vf_bin[bin_mask], 25, 'b', label=None, zorder=20)
                else:
                    combined_idx = full_indices[combined]
                    if combined_idx.size > 0:
                        seg_splits = np.where(np.diff(combined_idx) != 1)[0] + 1
                        segs = np.split(combined_idx, seg_splits) if len(seg_splits) > 0 else [combined_idx]
                        for seg in segs:
                            if seg.size == 0:
                                continue
                            ax3.plot(full_time[seg] * 1e3, P_frac_full[seg], color='k', linewidth=1.5)
                            ax3.plot(full_time[seg] * 1e3, L_frac_full[seg], color='r', linewidth=1.5)
                            if 'V' in time_series_data:
                                ax3.plot(full_time[seg] * 1e3, V_frac_full[seg], color='b', linewidth=1.5)
                        ax3.plot([], [], color='k', linewidth=1.5)
                        ax3.plot([], [], color='r', linewidth=1.5)
                        if 'V' in time_series_data:
                            ax3.plot([], [], color='b', linewidth=1.5)

            def _finite_minmax(arrs: List[np.ndarray]) -> Optional[Tuple[float, float]]:
                vals = []
                for arr in arrs:
                    arr = np.asarray(arr, dtype=float)
                    if arr.size == 0:
                        continue
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        vals.append(finite)
                if not vals:
                    return None
                flat = np.concatenate(vals)
                return float(np.nanmin(flat)), float(np.nanmax(flat))

            use_arrays: List[np.ndarray] = []
            if bin_mask is not None and np.any(bin_mask):
                use_arrays.extend([pf_bin[bin_mask], lf_bin[bin_mask]])
                if 'V_frac_bin' in rm_results and vf_bin.size:
                    use_arrays.append(vf_bin[bin_mask])
            else:
                sig_idx = full_indices[signal_mask]
                use_arrays.extend([P_frac_full[sig_idx], L_frac_full[sig_idx]])
                if 'V' in time_series_data:
                    use_arrays.append(V_frac_full[sig_idx])

            minmax = _finite_minmax(use_arrays)
            if minmax is not None:
                min_frac, max_frac = minmax
                ax3.set_ylim(min_frac - 0.05, max_frac + 0.05)

        ax3.set_xlabel('Time (ms)', fontsize=style['label'])
        ax3.set_ylabel('Polarisation Fraction', fontsize=style['label'])
        if n_peaks > 1:
            ax3.set_title(f'Peak {peak_idx+1}: Polarisation Fractions', fontsize=style['title'], fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=style['legend'])
        ax3.tick_params(axis='both', labelsize=style['tick'], length=TICK_LENGTH)

    plt.tight_layout()
    _savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Time series plot saved to {output_file}")
    plt.close()


def plot_rm_results(fitter: RMFitter, rm_synthesis_result: Dict,
                    output_file: str = 'rm_fitting_results.png',
                    pol_frac_err: Optional[np.ndarray] = None,
                    valid_mask: Optional[np.ndarray] = None,
                    circ_frac_err: Optional[np.ndarray] = None,
                    circ_valid_mask: Optional[np.ndarray] = None,
                    show_frac_panel: bool = True):
    """
    Create comprehensive plots of RM fitting results.

    Parameters:
    -----------
    fitter : RMFitter
        RMFitter object with data
    rm_synthesis_result : dict
        Results from RM synthesis
    output_file : str
        Output filename for plot
    pol_frac_err : array, optional
        Per-channel uncertainty for linear fraction (L/I).
    valid_mask : array, optional
        Boolean mask for valid linear-fraction channels.
    circ_frac_err : array, optional
        Per-channel uncertainty for circular fraction (V/I).
    circ_valid_mask : array, optional
        Boolean mask for valid circular-fraction channels.
    show_frac_panel : bool
        If False, omit the third (polarisation-fraction) panel.

    Note
    ----
    The Poincaré sphere is no longer produced automatically from here – it
    depends only on time-series information.  Callers who require one should
    invoke :func:`plot_poincare_sphere` explicitly with the appropriate
    ``time_series_data``.
    """
    style = plot_style()
    n_rows = 3 if show_frac_panel else 2
    fig_height = 7.4 if show_frac_panel else 5.2
    fig, axes = plt.subplots(n_rows, 1, figsize=(TWO_COLUMN_WIDTH_IN, fig_height), sharex=False)
    pol_frac_err_arr = None if pol_frac_err is None else np.asarray(pol_frac_err, dtype=float)
    valid_mask_arr = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    circ_frac_err_arr = None if circ_frac_err is None else np.asarray(circ_frac_err, dtype=float)
    circ_valid_mask_arr = None if circ_valid_mask is None else np.asarray(circ_valid_mask, dtype=bool)

    def _robust_channel_noise(series: np.ndarray) -> float:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return float(np.nanstd(arr)) if arr.size > 0 else np.nan
        diffs = np.diff(arr)
        mad = np.nanmedian(np.abs(diffs - np.nanmedian(diffs)))
        if np.isfinite(mad) and mad > 0:
            return float((1.4826 * mad) / np.sqrt(2.0))
        return float(np.nanstd(diffs) / np.sqrt(2.0))

    sigma_i = _robust_channel_noise(fitter.stokes_i)
    sigma_q = _robust_channel_noise(fitter.stokes_q)
    sigma_u = _robust_channel_noise(fitter.stokes_u)
    sigma_v = _robust_channel_noise(fitter.stokes_v) if fitter.stokes_v is not None else 0.0
    if not np.isfinite(sigma_i) or sigma_i <= 0:
        sigma_i = float(rm_synthesis_result.get('noise_estimate', 1e-10))
    if not np.isfinite(sigma_q) or sigma_q <= 0:
        sigma_q = sigma_i
    if not np.isfinite(sigma_u) or sigma_u <= 0:
        sigma_u = sigma_i
    if fitter.stokes_v is not None and (not np.isfinite(sigma_v) or sigma_v <= 0):
        sigma_v = sigma_i

    used_external_li_err = pol_frac_err_arr is not None and pol_frac_err_arr.shape == np.asarray(fitter.stokes_i).shape
    used_external_vi_err = (
        fitter.stokes_v is not None
        and circ_frac_err_arr is not None
        and circ_frac_err_arr.shape == np.asarray(fitter.stokes_i).shape
    )
    print(
        f"  RM plot uncertainty source: L/I={'Burn-law propagated' if used_external_li_err else 'local fallback'}"
        f", V/I={'Burn-law propagated' if used_external_vi_err else ('local fallback' if fitter.stokes_v is not None else 'N/A')}"
    )

    # Plot 1: Polarisation angle vs λ²
    ax1 = axes[0]
    pol_angle_deg = np.degrees(np.unwrap(fitter.pol_angle))
    pol_angle_deg = ((pol_angle_deg + 90.0) % 180.0) - 90.0
    q_vals = np.asarray(fitter.stokes_q, dtype=float)
    u_vals = np.asarray(fitter.stokes_u, dtype=float)
    i_vals = np.asarray(fitter.stokes_i, dtype=float)
    l_meas = np.sqrt(q_vals**2 + u_vals**2)
    lin_sq = q_vals**2 + u_vals**2 + 1e-20
    sigma_l = np.sqrt((q_vals**2 * sigma_q**2 + u_vals**2 * sigma_u**2) / (lin_sq + 1e-20))
    pa_sigma_rad = 0.5 * np.sqrt((u_vals**2 * sigma_q**2 + q_vals**2 * sigma_u**2) / (lin_sq**2))
    pa_sigma_deg = np.degrees(pa_sigma_rad)

    pa_mask = np.isfinite(pol_angle_deg) & np.isfinite(pa_sigma_deg)
    pa_mask &= np.isfinite(l_meas) & np.isfinite(sigma_l) & (sigma_l > 0)
    pa_mask &= l_meas >= (2.0 * sigma_l)
    if valid_mask_arr is not None and valid_mask_arr.shape == i_vals.shape:
        pa_mask &= valid_mask_arr
    else:
        pa_mask &= i_vals > 0
    pa_mask &= pa_sigma_deg <= 50.0

    if np.any(pa_mask):
        ax1.errorbar(fitter.lambda_sq[pa_mask], pol_angle_deg[pa_mask], yerr=pa_sigma_deg[pa_mask],
                     fmt='o', color='k', markersize=4, ecolor='gray',
                     elinewidth=0.9, capsize=2, alpha=0.8)

    rm_peak = rm_synthesis_result.get('rm_clean_peak', rm_synthesis_result.get('rm_peak'))
    lambda_sq_model = np.linspace(fitter.lambda_sq.min(), fitter.lambda_sq.max(), 100)

    if np.sum(pa_mask) >= 2:
        coeffs = np.polyfit(fitter.lambda_sq[pa_mask], np.unwrap(fitter.pol_angle)[pa_mask], 1)
        rm_fit = coeffs[0]
        pol_angle_0 = coeffs[1]
        pol_angle_model = np.degrees(pol_angle_0 + rm_fit * lambda_sq_model)
        pol_angle_model = ((pol_angle_model + 90.0) % 180.0) - 90.0
    else:
        rm_fit = np.nan
        pol_angle_model = None
    rm_err = rm_synthesis_result.get('rm_clean_err', rm_synthesis_result.get('noise_estimate', 0) * 2)
    if pol_angle_model is not None:
        if rm_err > 0:
            label_line = f'RM = {rm_peak:.2f} ± {rm_err:.2f} rad/m² (fit: {rm_fit:.2f})'
        else:
            label_line = f'RM = {rm_peak:.2f} rad/m² (fit: {rm_fit:.2f})'
        ax1.plot(lambda_sq_model, pol_angle_model, 'r-', linewidth=2, label=label_line)
    else:
        if rm_err > 0:
            label_line = f'RM = {rm_peak:.2f} ± {rm_err:.2f} rad/m² (insufficient masked PA points for fit)'
        else:
            label_line = f'RM = {rm_peak:.2f} rad/m² (insufficient masked PA points for fit)'
        ax1.plot([], [], 'r-', linewidth=2, label=label_line)

    ax1.set_xlabel('λ² (m²)', fontsize=style['label'])
    ax1.set_ylabel('Polarisation Angle (deg.)', fontsize=style['label'])
    ax1.legend(fontsize=style['legend'])
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=style['tick'])

    # Plot 2: RM Spectrum
    ax2 = axes[1]
    ax2.plot(rm_synthesis_result['rm_values'],
             rm_synthesis_result['rm_amplitude'],
             'k-', linewidth=1.5)
    rm_err = rm_synthesis_result.get('rm_clean_err', rm_synthesis_result.get('noise_estimate', 0) * 2)
    if rm_err > 0:
        ax2.axvspan(rm_peak - rm_err, rm_peak + rm_err, color='red', alpha=0.1,
                    label=f'RM error ≈ ±{rm_err:.2f}')
    ax2.axvline(rm_peak, color='r', linestyle='--', linewidth=2,
                label=f'Peak RM = {rm_peak:.2f} rad/m²')
    ax2.axhline(rm_synthesis_result['noise_estimate'], color='gray',
                linestyle=':', linewidth=1, label='Noise level')

    ax2.set_xlabel('RM (rad/m²)', fontsize=style['label'])
    ax2.set_ylabel('|F(RM)|', fontsize=style['label'])
    ax2.legend(fontsize=style['legend'])
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=style['tick'])

    if show_frac_panel:
        ax3 = axes[2]
        freq_mhz = fitter.freq_hz / 1e6
        l_vals = np.sqrt(q_vals**2 + u_vals**2)
        l_over_i = l_vals / (i_vals + 1e-10)
        if pol_frac_err_arr is not None and pol_frac_err_arr.shape == l_over_i.shape:
            sigma_l_over_i = pol_frac_err_arr
        else:
            sigma_l = np.sqrt((q_vals**2 * sigma_q**2 + u_vals**2 * sigma_u**2) / (l_vals**2 + 1e-20))
            sigma_l_over_i = np.sqrt((sigma_l / (i_vals + 1e-10))**2 +
                                     ((l_vals * sigma_i) / ((i_vals + 1e-10)**2))**2)

        l_plot_mask = np.isfinite(freq_mhz) & np.isfinite(l_over_i) & np.isfinite(sigma_l_over_i)
        if valid_mask_arr is not None and valid_mask_arr.shape == l_over_i.shape:
            l_plot_mask &= valid_mask_arr
        order = np.argsort(freq_mhz)
        freq_sorted = freq_mhz[order]
        l_sorted = l_over_i[order]
        sigma_l_sorted = sigma_l_over_i[order]
        l_mask_sorted = l_plot_mask[order]
        l_plot = np.where(l_mask_sorted, l_sorted, np.nan)
        l_low = np.where(l_mask_sorted, l_sorted - sigma_l_sorted, np.nan)
        l_high = np.where(l_mask_sorted, l_sorted + sigma_l_sorted, np.nan)

        ax3.plot(freq_sorted, l_plot, 'r-', linewidth=2, label='L/I')
        ax3.fill_between(freq_sorted, l_low, l_high, color='r', alpha=0.18, linewidth=0)

        if fitter.stokes_v is not None:
            v_vals = np.asarray(fitter.stokes_v, dtype=float)
            circ_pol_fraction = v_vals / (i_vals + 1e-10)
            if circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_pol_fraction.shape:
                sigma_v_over_i = circ_frac_err_arr
            else:
                sigma_v_over_i = np.sqrt((sigma_v / (i_vals + 1e-10))**2 +
                                         ((np.abs(v_vals) * sigma_i) / ((i_vals + 1e-10)**2))**2)

            v_plot_mask = np.isfinite(freq_mhz) & np.isfinite(circ_pol_fraction) & np.isfinite(sigma_v_over_i)
            if circ_valid_mask_arr is not None and circ_valid_mask_arr.shape == circ_pol_fraction.shape:
                v_plot_mask &= circ_valid_mask_arr
            elif valid_mask_arr is not None and valid_mask_arr.shape == circ_pol_fraction.shape:
                v_plot_mask &= valid_mask_arr

            v_sorted = circ_pol_fraction[order]
            sigma_v_sorted = sigma_v_over_i[order]
            v_mask_sorted = v_plot_mask[order]
            v_plot = np.where(v_mask_sorted, v_sorted, np.nan)
            v_low = np.where(v_mask_sorted, v_sorted - sigma_v_sorted, np.nan)
            v_high = np.where(v_mask_sorted, v_sorted + sigma_v_sorted, np.nan)

            ax3.plot(freq_sorted, v_plot, 'b-', linewidth=2, label='V/I')
            ax3.fill_between(freq_sorted, v_low, v_high, color='b', alpha=0.14, linewidth=0)

        ax3.set_xlabel('Frequency (MHz)', fontsize=style['label'])
        ax3.set_ylabel('Polarisation Fraction', fontsize=style['label'])
        ax3.legend(fontsize=style['legend'])
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=style['tick'])

    plt.tight_layout()
    _savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_burns_law_fits(fitter: RMFitter,
                        output_file: str = 'burns_law_fit.png',
                        pol_frac_err: Optional[np.ndarray] = None,
                        valid_mask: Optional[np.ndarray] = None,
                        circ_frac_err: Optional[np.ndarray] = None,
                        circ_valid_mask: Optional[np.ndarray] = None,
                        turbulent_radius_pc: float = 21.0,
                        screen_scale_cm: float = 1e15):
    """
    Fit and plot Burn-law depolarisation models for the linear polarisation spectrum.

    Models:
    - P_Burn(λ)       = exp(-2 * sigma_RM^2 * λ^4)
    - P_mod-Burn(λ)   = P_i * exp(-2 * sigma_RM'^2 * λ^4)
    - P_const(λ)      = P_i
    where P is the linear polarisation fraction (L/I).
    """
    from scipy.constants import c as _c  # avoid shadowing the module-level name

    style = plot_style()

    lambda_sq = np.asarray(fitter.lambda_sq, dtype=float)
    freq_hz_arr = np.asarray(fitter.freq_hz, dtype=float)
    pol_frac = np.asarray(fitter.pol_fraction, dtype=float)
    circ_frac = None
    if fitter.stokes_v is not None:
        circ_frac = np.asarray(fitter.stokes_v / (fitter.stokes_i + 1e-10), dtype=float)
    pol_frac_err_arr = None if pol_frac_err is None else np.asarray(pol_frac_err, dtype=float)
    valid_mask_arr = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    circ_frac_err_arr = None if circ_frac_err is None else np.asarray(circ_frac_err, dtype=float)
    circ_valid_mask_arr = None if circ_valid_mask is None else np.asarray(circ_valid_mask, dtype=bool)

    valid = np.isfinite(lambda_sq) & np.isfinite(freq_hz_arr) & (freq_hz_arr > 0) & np.isfinite(pol_frac) & (pol_frac > 0)
    if valid_mask_arr is not None and valid_mask_arr.shape == pol_frac.shape:
        valid &= valid_mask_arr
    if pol_frac_err_arr is not None and pol_frac_err_arr.shape == pol_frac.shape:
        valid &= np.isfinite(pol_frac_err_arr) & (pol_frac_err_arr > 0)
    if np.sum(valid) < 5:
        print("Warning: insufficient valid points for Burn-law fitting; skipping plot.")
        return

    sigma_rm_thresh = None
    sigma_rm_thresh_snr = None
    pol_snr_eff = np.nan
    meas_nsigma = 3.0
    freq_center_mhz = np.nan
    try:
        freq_valid = freq_hz_arr[valid]
        freq_center_hz = float(np.nanmedian(freq_valid))
        freq_center_mhz = freq_center_hz / 1e6
        sigma_rm_thresh = sigma_rm_detection_threshold(freq_center_hz)
    except Exception:
        sigma_rm_thresh = None

    x = lambda_sq[valid]
    freq_mhz = freq_hz_arr[valid] / 1e6
    y = pol_frac[valid]
    yerr = pol_frac_err_arr[valid] if (pol_frac_err_arr is not None and pol_frac_err_arr.shape == pol_frac.shape) else None
    order = np.argsort(freq_mhz)
    x = x[order]
    freq_mhz = freq_mhz[order]
    y = y[order]
    if yerr is not None:
        yerr = yerr[order]

    if yerr is not None:
        snr_arr = y / (yerr + 1e-20)
        snr_arr = snr_arr[np.isfinite(snr_arr) & (snr_arr > 0)]
        if snr_arr.size > 0:
            pol_snr_eff = float(np.nanmedian(snr_arr))
    else:
        dy = np.diff(y)
        sigma_y = np.nanstd(dy) / np.sqrt(2.0) if dy.size > 1 else np.nanstd(y)
        if np.isfinite(sigma_y) and sigma_y > 0:
            pol_snr_eff = float(np.nanmedian(y) / sigma_y)

    if sigma_rm_thresh is not None and np.isfinite(pol_snr_eff) and pol_snr_eff > 0:
        try:
            sigma_rm_thresh_snr = sigma_rm_detection_threshold_snr(
                freq_center_hz=freq_center_hz,
                pol_snr=pol_snr_eff,
                nsigma=meas_nsigma,
            )
        except Exception:
            sigma_rm_thresh_snr = None

    x_full = lambda_sq
    freq_mhz_full = freq_hz_arr / 1e6

    def burn_model(l2, sigma_rm):
        return np.exp(-2.0 * (sigma_rm ** 2) * (l2 ** 2))

    def modified_burn_model(l2, p_i, sigma_rm_prime):
        return p_i * np.exp(-2.0 * (sigma_rm_prime ** 2) * (l2 ** 2))

    def constant_model(l2, p_i):
        return np.full_like(l2, p_i, dtype=float)

    def circ_const_model(l2, c0):
        return np.full_like(l2, c0, dtype=float)

    def circ_linear_model(l2, c0, c1):
        return c0 + c1 * l2

    def circ_sine_model(l2, c0, amp, phi0, beta):
        return c0 + amp * np.sin(2.0 * (phi0 + beta * l2))

    def _log10_evidence_bic(y_obs: np.ndarray,
                            y_model: np.ndarray,
                            n_params: int,
                            sigma_obs: Optional[np.ndarray] = None) -> float:
        n = len(y_obs)
        if n <= max(1, n_params):
            return np.nan

        residual = y_obs - y_model
        if sigma_obs is None:
            sigma_level = np.nanstd(y_obs)
            if not np.isfinite(sigma_level) or sigma_level <= 0:
                sigma_level = 1e-10
            sigma = np.full_like(y_obs, sigma_level, dtype=float)
        else:
            sigma = np.array(sigma_obs, dtype=float)
            finite = np.isfinite(sigma) & (sigma > 0)
            if not np.any(finite):
                sigma = np.full_like(y_obs, 1e-10, dtype=float)
            else:
                fallback = np.nanmedian(sigma[finite])
                if not np.isfinite(fallback) or fallback <= 0:
                    fallback = 1e-10
                sigma[~finite] = fallback

        ln_like = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))
        bic = n_params * np.log(n) - 2.0 * ln_like
        ln_z = -0.5 * bic
        return ln_z / np.log(10.0)

    def _trotta_strength(delta_log10_z: float) -> str:
        if delta_log10_z < 0.5:
            return 'inconclusive'
        if delta_log10_z < 1.0:
            return 'substantial'
        if delta_log10_z < 2.0:
            return 'strong'
        return 'decisive'

    p0_guess = float(np.nanmax(y)) if np.nanmax(y) > 0 else 0.1
    p0_guess = min(max(p0_guess, 1e-6), 1.0)
    sigma_guess = 10.0

    burn_popt = burn_perr = mod_popt = mod_perr = None
    const_popt = const_perr = None
    circ_const_popt = circ_const_perr = None
    circ_lin_popt = circ_lin_perr = None
    circ_sin_popt = circ_sin_perr = None

    try:
        burn_popt, burn_pcov = curve_fit(burn_model, x, y, p0=[sigma_guess],
                                         bounds=([0.0], [1e5]), maxfev=20000)
        burn_perr = np.sqrt(np.diag(burn_pcov))
    except Exception:
        burn_popt = None

    try:
        mod_popt, mod_pcov = curve_fit(modified_burn_model, x, y,
                                        p0=[p0_guess, sigma_guess],
                                        bounds=([0.0, 0.0], [1.02, 1e5]), maxfev=30000)
        mod_perr = np.sqrt(np.diag(mod_pcov))
    except Exception:
        mod_popt = None

    try:
        const_popt, const_pcov = curve_fit(constant_model, x, y, p0=[p0_guess],
                                            bounds=([0.0], [1.02]), maxfev=10000)
        const_perr = np.sqrt(np.diag(const_pcov))
    except Exception:
        const_popt = None

    burn_y_fit = burn_model(x, *burn_popt) if burn_popt is not None else None
    mod_y_fit = modified_burn_model(x, *mod_popt) if mod_popt is not None else None
    const_y_fit = constant_model(x, *const_popt) if const_popt is not None else None

    x_c = freq_c = y_c = yerr_c = None
    if circ_frac is not None:
        valid_c = np.isfinite(x_full) & np.isfinite(freq_mhz_full) & np.isfinite(circ_frac)
        if circ_valid_mask_arr is not None and circ_valid_mask_arr.shape == circ_frac.shape:
            valid_c &= circ_valid_mask_arr
        if circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_frac.shape:
            valid_c &= np.isfinite(circ_frac_err_arr) & (circ_frac_err_arr > 0)
        if np.sum(valid_c) >= 5:
            x_c = x_full[valid_c]
            freq_c = freq_mhz_full[valid_c]
            y_c = circ_frac[valid_c]
            yerr_c = circ_frac_err_arr[valid_c] if (circ_frac_err_arr is not None and circ_frac_err_arr.shape == circ_frac.shape) else None
            order_c = np.argsort(freq_c)
            x_c = x_c[order_c]; freq_c = freq_c[order_c]; y_c = y_c[order_c]
            if yerr_c is not None:
                yerr_c = yerr_c[order_c]

            c0_guess = float(np.nanmedian(y_c))
            amp_guess = max(0.01, 0.5 * float(np.nanmax(y_c) - np.nanmin(y_c)))
            beta_guess = 100.0

            try:
                circ_const_popt, circ_const_pcov = curve_fit(circ_const_model, x_c, y_c,
                                                              p0=[c0_guess], bounds=([-1.0], [1.0]), maxfev=20000)
                circ_const_perr = np.sqrt(np.diag(circ_const_pcov))
            except Exception:
                circ_const_popt = None

            try:
                circ_lin_popt, circ_lin_pcov = curve_fit(circ_linear_model, x_c, y_c,
                                                          p0=[c0_guess, 0.0],
                                                          bounds=([-1.0, -1e5], [1.0, 1e5]), maxfev=30000)
                circ_lin_perr = np.sqrt(np.diag(circ_lin_pcov))
            except Exception:
                circ_lin_popt = None

            if np.sum(np.isfinite(y_c)) >= 8:
                try:
                    circ_sin_popt, circ_sin_pcov = curve_fit(
                        circ_sine_model, x_c, y_c,
                        p0=[c0_guess, amp_guess, 0.0, beta_guess],
                        bounds=([-1.0, 0.0, -np.pi, -1e5], [1.0, 1.0, np.pi, 1e5]), maxfev=60000)
                    circ_sin_perr = np.sqrt(np.diag(circ_sin_pcov))
                except Exception:
                    circ_sin_popt = None

    circ_const_y_fit = circ_const_model(x_c, *circ_const_popt) if (x_c is not None and circ_const_popt is not None) else None
    circ_lin_y_fit = circ_linear_model(x_c, *circ_lin_popt) if (x_c is not None and circ_lin_popt is not None) else None
    circ_sin_y_fit = circ_sine_model(x_c, *circ_sin_popt) if (x_c is not None and circ_sin_popt is not None) else None

    # ---- Terminal summary + text file ----
    summary_lines: List[str] = []
    summary_lines.append("Depolarisation fit summary:")
    summary_lines.append(f"  Output plot: {output_file}")
    print("\nDepolarisation fit summary:")
    print(f"  Output plot: {output_file}")

    def _print_and_store(line: str) -> None:
        print(line)
        summary_lines.append(line)

    if burn_popt is not None:
        if burn_perr is not None and burn_perr.size == 1:
            _print_and_store(f"  P_Burn: sigma_RM = {burn_popt[0]:.6f} ± {burn_perr[0]:.6f} rad/m^2")
        else:
            _print_and_store(f"  P_Burn: sigma_RM = {burn_popt[0]:.6f} rad/m^2")
    else:
        _print_and_store("  P_Burn: fit failed")

    if burn_popt is not None:
        try:
            delta_burn = depolarising_medium_delta_ne_b_parallel(
                float(burn_popt[0]), turbulent_radius_pc, screen_scale_cm)
            if burn_perr is not None and burn_perr.size == 1 and np.isfinite(burn_perr[0]) and burn_popt[0] > 0:
                frac_err = float(burn_perr[0] / burn_popt[0])
                delta_burn_err = abs(delta_burn) * frac_err
                _print_and_store(f"    delta(n_e, B_parallel) = {delta_burn:.6e} ± {delta_burn_err:.6e} uG/cm^3 "
                                  f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
            else:
                _print_and_store(f"    delta(n_e, B_parallel) = {delta_burn:.6e} uG/cm^3 "
                                  f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
        except Exception as exc:
            _print_and_store(f"    delta(n_e, B_parallel): not computed ({exc})")

    if sigma_rm_thresh_snr is not None and burn_popt is not None:
        if np.isfinite(sigma_rm_thresh_snr):
            burn_measurable = bool(burn_popt[0] >= sigma_rm_thresh_snr)
            _print_and_store(f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                              f"SNR_eff={pol_snr_eff:.2f}, nsigma={meas_nsigma:.1f}, "
                              f"threshold={sigma_rm_thresh_snr:.6f} rad/m^2, "
                              f"fitted={burn_popt[0]:.6f} -> {'measurable' if burn_measurable else 'not measurable'}")
        else:
            _print_and_store(f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                              f"SNR_eff={pol_snr_eff:.2f} is too low for a {meas_nsigma:.1f}σ depolarisation detection")
    elif sigma_rm_thresh is not None and burn_popt is not None:
        burn_measurable = bool(burn_popt[0] >= sigma_rm_thresh)
        _print_and_store(f"    measurability (fallback e-fold @ {freq_center_mhz:.2f} MHz): "
                          f"threshold={sigma_rm_thresh:.6f} rad/m^2, "
                          f"fitted={burn_popt[0]:.6f} -> {'measurable' if burn_measurable else 'not measurable'}")

    if mod_popt is not None:
        if mod_perr is not None and mod_perr.size == 2:
            _print_and_store(f"  P_mod-Burn: P_i = {mod_popt[0]:.6f} ± {mod_perr[0]:.6f}, "
                              f"sigma_RM' = {mod_popt[1]:.6f} ± {mod_perr[1]:.6f} rad/m^2")
        else:
            _print_and_store(f"  P_mod-Burn: P_i = {mod_popt[0]:.6f}, sigma_RM' = {mod_popt[1]:.6f} rad/m^2")
    else:
        _print_and_store("  P_mod-Burn: fit failed")

    if mod_popt is not None:
        try:
            delta_mod = depolarising_medium_delta_ne_b_parallel(
                float(mod_popt[1]), turbulent_radius_pc, screen_scale_cm)
            if mod_perr is not None and mod_perr.size == 2 and np.isfinite(mod_perr[1]) and mod_popt[1] > 0:
                frac_err = float(mod_perr[1] / mod_popt[1])
                delta_mod_err = abs(delta_mod) * frac_err
                _print_and_store(f"    delta(n_e, B_parallel) = {delta_mod:.6e} ± {delta_mod_err:.6e} uG/cm^3 "
                                  f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
            else:
                _print_and_store(f"    delta(n_e, B_parallel) = {delta_mod:.6e} uG/cm^3 "
                                  f"(R={turbulent_radius_pc:.3g} pc, l_screen={screen_scale_cm:.3e} cm)")
        except Exception as exc:
            _print_and_store(f"    delta(n_e, B_parallel): not computed ({exc})")

    if sigma_rm_thresh_snr is not None and mod_popt is not None:
        if np.isfinite(sigma_rm_thresh_snr):
            mod_measurable = bool(mod_popt[1] >= sigma_rm_thresh_snr)
            _print_and_store(f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                              f"SNR_eff={pol_snr_eff:.2f}, nsigma={meas_nsigma:.1f}, "
                              f"threshold={sigma_rm_thresh_snr:.6f} rad/m^2, "
                              f"fitted={mod_popt[1]:.6f} -> {'measurable' if mod_measurable else 'not measurable'}")
        else:
            _print_and_store(f"    measurability (S/N-aware @ {freq_center_mhz:.2f} MHz): "
                              f"SNR_eff={pol_snr_eff:.2f} is too low for a {meas_nsigma:.1f}σ depolarisation detection")
    elif sigma_rm_thresh is not None and mod_popt is not None:
        mod_measurable = bool(mod_popt[1] >= sigma_rm_thresh)
        _print_and_store(f"    measurability (fallback e-fold @ {freq_center_mhz:.2f} MHz): "
                          f"threshold={sigma_rm_thresh:.6f} rad/m^2, "
                          f"fitted={mod_popt[1]:.6f} -> {'measurable' if mod_measurable else 'not measurable'}")

    if const_popt is not None:
        if const_perr is not None and const_perr.size == 1:
            _print_and_store(f"  P_const: P_i = {const_popt[0]:.6f} ± {const_perr[0]:.6f}")
        else:
            _print_and_store(f"  P_const: P_i = {const_popt[0]:.6f}")
    else:
        _print_and_store("  P_const: fit failed")

    if y_c is None:
        _print_and_store("  Circular fraction models: skipped (no valid V/I data)")
    else:
        if circ_const_popt is not None:
            if circ_const_perr is not None and circ_const_perr.size == 1:
                _print_and_store(f"  mC const: C0 = {circ_const_popt[0]:.6f} ± {circ_const_perr[0]:.6f}")
            else:
                _print_and_store(f"  mC const: C0 = {circ_const_popt[0]:.6f}")
        else:
            _print_and_store("  mC const: fit failed")

        if circ_lin_popt is not None:
            if circ_lin_perr is not None and circ_lin_perr.size == 2:
                _print_and_store(f"  mC linear: C0 = {circ_lin_popt[0]:.6f} ± {circ_lin_perr[0]:.6f}, "
                                  f"C1 = {circ_lin_popt[1]:.6f} ± {circ_lin_perr[1]:.6f}")
            else:
                _print_and_store(f"  mC linear: C0 = {circ_lin_popt[0]:.6f}, C1 = {circ_lin_popt[1]:.6f}")
        else:
            _print_and_store("  mC linear: fit failed")

        if circ_sin_popt is not None:
            if circ_sin_perr is not None and circ_sin_perr.size == 4:
                _print_and_store(f"  mC sinusoid: C0 = {circ_sin_popt[0]:.6f} ± {circ_sin_perr[0]:.6f}, "
                                  f"A = {circ_sin_popt[1]:.6f} ± {circ_sin_perr[1]:.6f}, "
                                  f"phi0 = {circ_sin_popt[2]:.6f} ± {circ_sin_perr[2]:.6f}, "
                                  f"beta = {circ_sin_popt[3]:.6f} ± {circ_sin_perr[3]:.6f}")
            else:
                _print_and_store(f"  mC sinusoid: C0 = {circ_sin_popt[0]:.6f}, A = {circ_sin_popt[1]:.6f}, "
                                  f"phi0 = {circ_sin_popt[2]:.6f}, beta = {circ_sin_popt[3]:.6f}")
        else:
            _print_and_store("  mC sinusoid: fit failed")

    # Model comparison via BIC
    linear_log10z = {}
    if burn_y_fit is not None:
        linear_log10z['P_Burn'] = _log10_evidence_bic(y, burn_y_fit, 1, yerr)
    if mod_y_fit is not None:
        linear_log10z['P_mod-Burn'] = _log10_evidence_bic(y, mod_y_fit, 2, yerr)
    if const_y_fit is not None:
        linear_log10z['P_const'] = _log10_evidence_bic(y, const_y_fit, 1, yerr)

    if len(linear_log10z) > 0:
        _print_and_store("  Linear models log10 evidence (BIC approximation):")
        for name, val in linear_log10z.items():
            _print_and_store(f"    {name}: log10(Z) ≈ {val:.6f}")
        if len(linear_log10z) >= 2:
            ranking = sorted(linear_log10z.items(), key=lambda item: item[1], reverse=True)
            best_name, best_val = ranking[0]
            second_name, second_val = ranking[1]
            delta = best_val - second_val
            strength = _trotta_strength(delta)
            _print_and_store(f"  Preferred linear model: {best_name} over {second_name} (Δlog10Z={delta:.3f}, {strength})")

    best_linear_model = None
    if len(linear_log10z) > 0:
        best_linear_model = max(linear_log10z, key=linear_log10z.get)

    circular_log10z = {}
    if y_c is not None:
        if circ_const_y_fit is not None:
            circular_log10z['mC_const'] = _log10_evidence_bic(y_c, circ_const_y_fit, 1, yerr_c)
        if circ_lin_y_fit is not None:
            circular_log10z['mC_linear'] = _log10_evidence_bic(y_c, circ_lin_y_fit, 2, yerr_c)
        if circ_sin_y_fit is not None:
            circular_log10z['mC_sinusoid'] = _log10_evidence_bic(y_c, circ_sin_y_fit, 4, yerr_c)

    if len(circular_log10z) > 0:
        _print_and_store("  Circular models log10 evidence (BIC approximation):")
        for name, val in circular_log10z.items():
            _print_and_store(f"    {name}: log10(Z) ≈ {val:.6f}")
        if len(circular_log10z) >= 2:
            ranking_c = sorted(circular_log10z.items(), key=lambda item: item[1], reverse=True)
            best_name_c, best_val_c = ranking_c[0]
            second_name_c, second_val_c = ranking_c[1]
            delta_c = best_val_c - second_val_c
            strength_c = _trotta_strength(delta_c)
            _print_and_store(f"  Preferred circular model: {best_name_c} over {second_name_c} (Δlog10Z={delta_c:.3f}, {strength_c})")

    best_circular_model = None
    if len(circular_log10z) > 0:
        best_circular_model = max(circular_log10z, key=circular_log10z.get)

    import os as _os
    summary_txt = _os.path.splitext(output_file)[0] + "_fit_summary.txt"
    with open(summary_txt, 'w', encoding='utf-8') as summary_file:
        summary_file.write("\n".join(summary_lines) + "\n")
    print(f"  Fit summary saved to {summary_txt}")

    freq_model_mhz = np.linspace(np.nanmin(freq_mhz), np.nanmax(freq_mhz), 500)
    freq_model_hz = freq_model_mhz * 1e6
    x_model = (_c / freq_model_hz) ** 2

    fig_height = max(3.2, SINGLE_COLUMN_WIDTH_IN * 0.75)
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH_IN, fig_height))
    if yerr is not None:
        ax.errorbar(freq_mhz, y, yerr=yerr, fmt='o', markersize=4,
                    color='tab:red', marker='s', ecolor='gray', elinewidth=1, capsize=2,
                    alpha=0.8, label=r'$L/I$')
    else:
        ax.scatter(freq_mhz, y, s=28, c='tab:red', marker='s', alpha=0.8, label=r'$L/I$')

    if burn_popt is not None and best_linear_model == 'P_Burn':
        y_burn = burn_model(x_model, *burn_popt)
        burn_label = r"$P_{\mathrm{Burn}}(\lambda)=\exp\left(-2\sigma_{\mathrm{RM}}^2\lambda^4\right)$"
        ax.plot(freq_model_mhz, y_burn, color='tab:cyan', linewidth=2, label=burn_label)

    if mod_popt is not None and best_linear_model == 'P_mod-Burn':
        y_mod = modified_burn_model(x_model, *mod_popt)
        mod_label = r"$P_{\mathrm{mod-Burn}}(\lambda)=P_i\exp\left(-2\sigma_{\mathrm{RM}}'^{\,2}\lambda^4\right)$"
        ax.plot(freq_model_mhz, y_mod, color='tab:cyan', linewidth=2, linestyle='--', label=mod_label)

    if const_popt is not None and best_linear_model == 'P_const':
        y_const = constant_model(x_model, *const_popt)
        label_const = r"$P_{\mathrm{const}}(\lambda)=P_i$"
        ax.plot(freq_model_mhz, y_const, color='0.25', linewidth=2, linestyle=':', label=label_const)

    if y_c is not None and freq_c is not None:
        if yerr_c is not None:
            ax.errorbar(freq_c, y_c, yerr=yerr_c, fmt='s', markersize=3,
                        color='tab:blue', ecolor='tab:blue', elinewidth=0.8,
                        capsize=2, alpha=0.8, label=r'$V/I$')
        else:
            ax.scatter(freq_c, y_c, s=16, c='tab:blue', marker='s', alpha=0.8, label=r'$V/I$')

        if circ_lin_popt is not None and best_circular_model == 'mC_linear':
            y_cl = circ_linear_model(x_model, *circ_lin_popt)
            lbl = r"$m_C(\lambda^2)=C_0 + C_1\lambda^2$"
            ax.plot(freq_model_mhz, y_cl, color='tab:green', linewidth=1.8, linestyle='--', label=lbl)

        if circ_sin_popt is not None and best_circular_model == 'mC_sinusoid':
            y_cs = circ_sine_model(x_model, *circ_sin_popt)
            lbl = r"$m_C(\lambda^2)=C_0 + A\sin\left(2\left(\phi_0 + \beta\lambda^2\right)\right)$"
            ax.plot(freq_model_mhz, y_cs, color='tab:olive', linewidth=1.8, linestyle='-.', label=lbl)

        if circ_const_popt is not None and best_circular_model == 'mC_const':
            y_cc = circ_const_model(x_model, *circ_const_popt)
            lbl = r"$m_C(\lambda^2)=C_0$"
            ax.plot(freq_model_mhz, y_cc, color='tab:green', linewidth=1.8, linestyle='-', label=lbl)

    ax.set_xlabel('Frequency (MHz)', fontsize=style['label'])
    ax.set_ylabel('Polarisation Fraction', fontsize=style['label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=style['legend'], loc='best')
    ax.tick_params(axis='both', labelsize=style['tick'])

    plt.tight_layout()
    _savefig_rasterized(output_file, dpi=600, bbox_inches='tight')
    print(f"Burn-law fit plot saved to {output_file}")
    plt.close()
