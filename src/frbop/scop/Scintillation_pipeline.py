"""
FRB scintillation pipeline

- Reads FRB metadata from YAMLs in the current directory, loads X/Y voltages from
DM-tagged .npy files, channelizes with overlapping FFTs to form Stokes-I dynamic
spectra, selects ON/OFF windows around t_ref, applies zap-channel masking,
normalizes by OFF statistics, fits an ACF Lorentzian for scintillation bandwidth,
fits the power spectrum with an exponential, estimates modulation index from a Gaussian core fit to the ON ACF,
and saves a big multi-panel PNG.

Outputs:
- OUTPUT_COLLECTION/<FRBNAME>/<FRBNAME>_big_panel_multipanel.png
- STATISTICS2/_errors/<yaml>.<timestamp>.log
"""

import argparse
import glob
import os
import re
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import correlate, get_window

#Configuration
FS = 336e6
TIME_DOWNSAMPLE = 10

RESOLUTIONS = [0.1e6]
RES_LABELS = ["100kHz"]

ACF_EXCLUDE_NCHAN_CENTER = 3 #Do not fit the Lorentzian in the very central ACF lags 
ACF_DISPLAY_LAG_CAP_MHZ = 16


#Some style for the text 
BIG_TEXT = {"suptitle": 22, "title": 14, "legend": 10, "ticks": 11}


# NaN Safe 
def nanmean_cols(A):
    A = np.asarray(A, float)
    out = np.full(A.shape[1], np.nan)
    finite = np.isfinite(A)
    cnt = finite.sum(axis=0)
    if np.any(cnt > 0):
        s = np.nansum(np.where(finite, A, 0.0), axis=0)
        out[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
    return out

def nanstd_cols(A, ddof=1):
    A = np.asarray(A, float)
    out = np.full(A.shape[1], np.nan)
    finite = np.isfinite(A)
    cnt = finite.sum(axis=0)
    mu = nanmean_cols(A)
    diff2 = np.where(finite, (A - mu[None, :]) ** 2, 0.0)
    denom = np.maximum(cnt - ddof, 1)
    var = np.nansum(diff2, axis=0) / denom
    std = np.sqrt(var)
    std[cnt == 0] = np.nan
    return std


# Plot styling 
def style_axes(fig):
    for ax in fig.get_axes():
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(1.1)


# Find zap-channels in the YAML
def get_zap_field(meta):
    for path in (("zapchan",), ("par", "zapchan"), ("RFI", "zapchan")):
        d = meta
        ok = True
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok and d is not None:
            if isinstance(d, str) and d.strip().upper() in {"NONE", "NULL", ""}:
                continue
            return d

    stack = [meta]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "zapchan" in cur:
                z = cur["zapchan"]
                if not (isinstance(z, str) and z.strip().upper() in {"NONE", "NULL", ""}):
                    if z is not None:
                        return z
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


# zapchan can be "a:b,c:d" or a list; normalize to [(lo,hi), ...] in MHz
def parse_zap_from_yaml(zap_field):
    ranges = []
    if zap_field is None:
        return ranges
    if isinstance(zap_field, str) and zap_field.strip().upper() in {"NONE", "NULL", ""}:
        return ranges

    def add_tokens(s):
        for tok in re.split(r"[,\s]+", str(s).strip()):
            if not tok:
                continue
            if ":" in tok:
                a, b = tok.split(":", 1)
                a = float(a)
                b = float(b)
                ranges.append((min(a, b), max(a, b)))
            else:
                x = float(tok)
                ranges.append((x, x))

    if isinstance(zap_field, (int, float)):
        x = float(zap_field)
        ranges.append((x, x))
    elif isinstance(zap_field, str):
        add_tokens(zap_field)
    elif isinstance(zap_field, list):
        for it in zap_field:
            add_tokens(it)
    else:
        try:
            x = float(zap_field)
            ranges.append((x, x))
        except Exception:
            pass

    return ranges


# Load X/Y time series .npy files, trying DM with 1, 2 or 3 decimals
def load_XY_with_dm_fallback(base_path, folder_name, dm):
    dm = float(dm)
    for fmt in ("{:.1f}", "{:.2f}", "{:.3f}"):
        dmstr = fmt.format(dm)
        xp = os.path.join(base_path, f"{folder_name}_X_t_{dmstr}.npy")
        yp = os.path.join(base_path, f"{folder_name}_Y_t_{dmstr}.npy")
        if os.path.exists(xp) and os.path.exists(yp):
            return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r"), dmstr
    raise FileNotFoundError(
        f"Could not find X/Y npy files for DM {dm} (1/2/3 decimals) in {base_path}"
    )


def _resolve_input_mode(meta, cli_mode):
    if cli_mode != "auto":
        return cli_mode

    data = meta.get("data", {}) if isinstance(meta, dict) else {}
    mode_hint = str(data.get("mode", "")).strip().lower()
    if mode_hint in {"voltages", "stokes"}:
        return mode_hint

    if any(k in data for k in ("stokes_i", "I")):
        return "stokes"
    return "voltages"


def _ensure_time_freq_2d(arr2d, time_axis=0):
    arr2d = np.asarray(arr2d, float)
    if arr2d.ndim != 2:
        raise ValueError(f"Expected 2D array [time,freq] or [freq,time], got shape={arr2d.shape}")
    if time_axis not in (0, 1):
        raise ValueError(f"time_axis must be 0 or 1, got {time_axis}")
    if time_axis == 1:
        return np.swapaxes(arr2d, 0, 1)
    return arr2d


def _coerce_optional_float(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return float(x)


def _load_time_axis_from_npy(path, yaml_path, time_unit="ms"):
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), p)

    t = np.asarray(np.load(p, mmap_mode="r")).squeeze()
    if t.ndim != 1:
        raise ValueError(f"time_file must contain a 1D array, got shape={t.shape}")

    unit = str(time_unit).strip().lower()
    if unit == "s":
        t = t * 1e3
    elif unit != "ms":
        raise ValueError(f"Unsupported time_unit '{time_unit}'. Use 'ms' or 's'.")

    return np.asarray(t, float)


def _estimate_dt_ms_from_time_axis(t_ms):
    t_ms = np.asarray(t_ms, float)
    if t_ms.ndim != 1:
        raise ValueError(f"Expected 1D time axis, got shape={t_ms.shape}")
    if t_ms.size < 2:
        return None

    dt = np.diff(t_ms)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return None

    dt_med = float(np.nanmedian(dt))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return None
    return dt_med


def _peak_ref_time_ms(I_dyn, t_ms=None, dt_ms=None):
    I_dyn = np.asarray(I_dyn, float)
    if I_dyn.ndim != 2 or I_dyn.shape[0] == 0:
        raise ValueError("Cannot estimate peak t_ref from empty/non-2D dynamic spectrum")

    It = np.nanmean(I_dyn, axis=1)
    if not np.isfinite(It).any():
        raise ValueError("Cannot estimate peak t_ref: all time samples are non-finite")

    idx = int(np.nanargmax(It))
    if t_ms is not None:
        if len(t_ms) != I_dyn.shape[0]:
            raise ValueError(
                f"time axis length ({len(t_ms)}) does not match stokes time axis ({I_dyn.shape[0]})"
            )
        return float(t_ms[idx])
    if dt_ms is None:
        raise ValueError("Need dt_ms or a time_file to estimate peak t_ref")
    return float(idx * float(dt_ms))


def estimate_t_ref_ms_from_voltages(X, Y, fft_size, step_size):
    t_bin_ms = step_size / FS * 1e3
    acc_I = None
    acc_n = 0
    block = 0

    t_ds_list = []
    It_list = []

    for xf, yf in zip(streaming_fft(X, fft_size, step_size), streaming_fft(Y, fft_size, step_size)):
        I = (np.abs(xf) ** 2 + np.abs(yf) ** 2).astype(np.float32, copy=False)

        if acc_I is None:
            acc_I = np.zeros_like(I, dtype=np.float32)

        acc_I += I
        acc_n += 1

        if acc_n == TIME_DOWNSAMPLE:
            I_ds = acc_I / float(TIME_DOWNSAMPLE)
            t_ds = (block - (TIME_DOWNSAMPLE - 1) / 2.0) * t_bin_ms
            t_ds_list.append(float(t_ds))
            It_list.append(float(np.nanmean(I_ds)))
            acc_I.fill(0.0)
            acc_n = 0

        block += 1

    if not It_list:
        raise RuntimeError("Could not estimate t_ref from voltages: no downsampled blocks produced")

    It = np.asarray(It_list, float)
    if not np.isfinite(It).any():
        raise RuntimeError("Could not estimate t_ref from voltages: all samples are non-finite")

    i_peak = int(np.nanargmax(It))
    return float(t_ds_list[i_peak])


def load_stokes_i_from_meta(meta, yaml_path, cli_time_file=None, cli_time_unit=None):
    data = meta.get("data", {})
    if not isinstance(data, dict):
        raise KeyError("YAML key 'data' must be a mapping")

    time_axis = int(data.get("time_axis", 0))

    if "stokes_i" in data or "I" in data:
        key = "stokes_i" if "stokes_i" in data else "I"
        p = os.path.expanduser(str(data[key]))
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), p)
        I2d = np.load(p, mmap_mode="r")
        I2d = _ensure_time_freq_2d(I2d, time_axis=time_axis)
    else:
        raise KeyError("For stokes mode, provide data.stokes_i (or data.I)")

    time_file = cli_time_file if cli_time_file else data.get("time_file", None)
    t_ms = None
    if time_file:
        time_unit = cli_time_unit if cli_time_unit else data.get("time_unit", "ms")
        t_ms = _load_time_axis_from_npy(time_file, yaml_path=yaml_path, time_unit=time_unit)
        if len(t_ms) != I2d.shape[0]:
            raise ValueError(
                f"time_file length ({len(t_ms)}) does not match stokes time axis ({I2d.shape[0]})"
            )

    dt_ms = data.get("dt_ms", meta.get("par", {}).get("dt_ms", None))
    dt_ms = _coerce_optional_float(dt_ms)
    if t_ms is not None:
        dt_from_time = _estimate_dt_ms_from_time_axis(t_ms)
        if dt_from_time is not None:
            dt_ms = dt_from_time

    if t_ms is None and dt_ms is None:
        raise KeyError("For stokes mode, provide data.time_file or data.dt_ms (or par.dt_ms)")
    if t_ms is not None and dt_ms is None:
        raise KeyError("Could not derive dt_ms from time_file; provide at least two valid time samples or set data.dt_ms")

    return I2d, t_ms, dt_ms


# Channellisation: overlapping FFT 
def streaming_fft(t_ser, fft_size, step_size, window_name="hamming"):
    w = get_window(window_name, fft_size)
    n_blocks = (len(t_ser) - fft_size) // step_size + 1
    for i in range(n_blocks):
        s = i * step_size
        seg = t_ser[s : s + fft_size]
        if len(seg) == fft_size:
            yield np.fft.fft(seg * w)


# Build ON/OFF dynamic spectra around t_ref 
def build_dynspec_onoff(X, Y, fft_size, step_size, t_ref_ms, on_half_width_ms=4.0, off_first_frac=0.10):
    t_bin_ms = step_size / FS * 1e3

    all_rows, all_t = [], []

    acc_I = None
    acc_n = 0
    block = 0

    for xf, yf in zip(streaming_fft(X, fft_size, step_size),
                      streaming_fft(Y, fft_size, step_size)):

        I = (np.abs(xf) ** 2 + np.abs(yf) ** 2).astype(np.float32, copy=False)

        if acc_I is None:
            acc_I = np.zeros_like(I, dtype=np.float32)

        acc_I += I
        acc_n += 1

        if acc_n == TIME_DOWNSAMPLE:
            I_ds = acc_I / float(TIME_DOWNSAMPLE)
            t_ds = (block - (TIME_DOWNSAMPLE - 1) / 2.0) * t_bin_ms

            all_rows.append(I_ds.copy())
            all_t.append(float(t_ds))

            acc_I.fill(0.0)
            acc_n = 0

        block += 1

    if not all_rows:
        return (np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=float))

    full_dyn = np.asarray(all_rows, dtype=float)
    full_t = np.asarray(all_t, dtype=float)

    on_mask = (t_ref_ms - on_half_width_ms <= full_t) & (full_t <= t_ref_ms + on_half_width_ms)
    n_total = full_dyn.shape[0]
    n_off = max(1, int(np.ceil(float(off_first_frac) * n_total)))
    off_mask = np.zeros(n_total, dtype=bool)
    off_mask[:n_off] = True

    on_rows = full_dyn[on_mask, :]
    off_rows = full_dyn[off_mask, :]
    t_on = full_t[on_mask]
    t_off = full_t[off_mask]

    return (np.asarray(on_rows, dtype=float),
            np.asarray(off_rows, dtype=float),
            np.asarray(t_on, dtype=float),
            np.asarray(t_off, dtype=float))


def build_dynspec_onoff_from_stokes(
    I_dyn,
    t_ref_ms,
    dt_ms=None,
    t_ms=None,
    on_half_width_ms=4.0,
    off_first_frac=0.10,
):
    I_dyn = np.asarray(I_dyn, float)
    if I_dyn.ndim != 2:
        raise ValueError(f"Expected 2D stokes-I dynamic spectrum [time,freq], got {I_dyn.shape}")

    nt = I_dyn.shape[0]
    if t_ms is not None:
        t = np.asarray(t_ms, float)
        if t.ndim != 1 or len(t) != nt:
            raise ValueError(f"Expected t_ms shape ({nt},), got {t.shape}")
    else:
        if dt_ms is None:
            raise ValueError("Need dt_ms or t_ms for stokes ON/OFF selection")
        t = np.arange(nt, dtype=float) * float(dt_ms)

    on_mask = (t_ref_ms - on_half_width_ms <= t) & (t <= t_ref_ms + on_half_width_ms)
    n_total = int(nt)
    n_off = max(1, int(np.ceil(float(off_first_frac) * n_total)))
    off_mask = np.zeros(n_total, dtype=bool)
    off_mask[:n_off] = True

    dyn_on = I_dyn[on_mask, :]
    dyn_off = I_dyn[off_mask, :]
    t_on = t[on_mask]
    t_off = t[off_mask]

    return dyn_on, dyn_off, t_on, t_off

 
def match_off_to_on(dyn_on, dyn_off):
    need = dyn_on.shape[0] if dyn_on.size else 0
    if need == 0:
        return dyn_off
    if dyn_off.shape[0] == 0:
        return dyn_on.copy()
    if dyn_off.shape[0] < need:
        reps = int(np.ceil(need / dyn_off.shape[0]))
        dyn_off = np.tile(dyn_off, (reps, 1))[:need]
    elif dyn_off.shape[0] > need:
        dyn_off = dyn_off[:need]
    return dyn_off


# Make a frequency axis (MHz) and channel spacing df (MHz)
def freq_axis(nfft, cfreq, bw):
    f = np.linspace(cfreq - bw / 2.0, cfreq + bw / 2.0, int(nfft))
    df = float(abs(f[1] - f[0])) if len(f) > 1 else np.nan
    return f, df


# Zap frequency ranges 
def build_zap_mask(f_data, zap_ranges_mhz, df, display_flips_y=True):
    order = np.argsort(f_data)
    f_plot = f_data[order]
    half = max(df / 2.0, 1e-9)

    zap_plot_asc = np.zeros_like(f_plot, bool)
    for lo, hi in zap_ranges_mhz:
        lo, hi = min(lo, hi), max(lo, hi)
        zap_plot_asc |= (f_plot >= lo - half) & (f_plot <= hi + half)

    zap_plot_disp = zap_plot_asc[::-1] if display_flips_y else zap_plot_asc

    zap_data = np.zeros_like(f_data, bool)
    zap_data[order] = zap_plot_disp
    return order, f_plot, zap_plot_disp, zap_data


# Apply zap mask, normalize ON/OFF by OFF statistics, and build masked plot array
def zap_and_normalize(dyn_on, dyn_off, order, zap_plot_disp, zap_data):
    dyn_on = dyn_on.copy()
    dyn_off = dyn_off.copy()

    dyn_on[:, zap_data] = np.nan
    dyn_off[:, zap_data] = np.nan

    mu_off = nanmean_cols(dyn_off)
    sd_off = nanstd_cols(dyn_off, ddof=1)
    sd = np.where(np.isfinite(sd_off) & (sd_off > 0), sd_off, 1.0)

    norm_on = (dyn_on - mu_off[None, :]) / (sd[None, :] + 1e-12)
    norm_off = (dyn_off - mu_off[None, :]) / (sd[None, :] + 1e-12)

    valid_on = np.isfinite(norm_on).any(axis=0)
    valid_off = np.isfinite(norm_off).any(axis=0)
    keep = (~zap_data) & valid_on & valid_off

    dyn_plot = np.ma.masked_invalid(norm_on[:, order]).T
    dyn_plot.mask[zap_plot_disp, :] = True
    return norm_on, norm_off, keep, dyn_plot


def analyze_dynspec_and_plot(
    dyn_on,
    dyn_off,
    t_on,
    f_data,
    df,
    zap_ranges,
    label,
    frbname,
    outdir,
    verbose=False,
    acf_fit_lag_cap_mhz=8.0,
    acf_fit_center_mhz=0.0,
    acf_lorentzian_components=1,
):
    dyn_off = match_off_to_on(dyn_on, dyn_off)
    order, f_plot, zap_plot_disp, zap_data = build_zap_mask(f_data, zap_ranges, df, display_flips_y=True)
    norm_on, norm_off, keep, dyn_plot = zap_and_normalize(dyn_on, dyn_off, order, zap_plot_disp, zap_data)

    I_on_mean_raw = nanmean_cols(dyn_on)
    S_hat_m = float(np.nanmean(I_on_mean_raw[keep])) if np.any(keep) else np.nan
    N_on = max(int(dyn_on.shape[0]), 1)

    I_display = nanmean_cols(norm_on)
    I_f_display_plot = I_display[order]
    I_f_display_plot[zap_plot_disp] = np.nan

    Ion_model = np.array([])
    Ioff_model = np.array([])
    have_model = False
    if np.any(keep):
        Ion_model = nanmean_cols(norm_on[:, keep])
        Ioff_model = nanmean_cols(norm_off[:, keep])
        have_model = np.isfinite(Ion_model).any() and (Ion_model.size >= 8)

    dnu_fit = np.array([0.0])
    acf_raw = np.array([np.nan])
    acf_fit_model = np.array([np.nan])
    acf_fit_component_models = np.empty((0, 1), dtype=float)
    acf_fit_nu_s_components = np.array([], dtype=float)
    acf_fit_n_components = 1
    acf_fit_off = np.nan
    nu_s_acf = np.nan

    mod_freq = np.array([0.0])
    ps_on = np.array([np.nan])
    ps_fit_freqs = np.array([])
    ps_fit_model = np.array([])
    nu_s_ps = np.nan

    acf_on = np.array([np.nan])
    dnu_on = np.array([0.0])
    acf_smooth = np.array([np.nan])
    m_corr = np.nan

    if have_model:
        on_mu = float(np.nanmean(Ion_model)) if np.isfinite(Ion_model).any() else 0.0
        Ion_c = Ion_model - on_mu
        scale = float(np.nanmax(np.abs(Ion_c)))
        scale = 1.0 if (not np.isfinite(scale) or scale == 0) else scale
        Ion_c = Ion_c / scale
        Ioff_c = Ioff_model / scale

        N = len(Ion_c)
        mod_freq = np.fft.fftshift(np.fft.fftfreq(N, d=df))
        ps_on = np.abs(np.fft.fftshift(np.fft.fft(Ion_c))) ** 2
        _ps_off = np.abs(np.fft.fftshift(np.fft.fft(Ioff_c))) ** 2

        acf_raw, acf_used, dnu_fit, sigma_n, S_hat, _bias0 = fractional_acf_noise_sub(
            Ion_model, norm_on, norm_off, keep, df
        )
        acf_fit = fit_acf_lorentzian(
            dnu_fit,
            acf_used,
            fit_lag_cap_mhz=acf_fit_lag_cap_mhz,
            fit_lag_center_mhz=acf_fit_center_mhz,
            n_components=acf_lorentzian_components,
            verbose=verbose,
        )
        nu_s_acf = acf_fit["nu_s"]
        acf_fit_model = acf_fit["model"]
        acf_fit_component_models = acf_fit.get("model_components", np.empty((0, len(dnu_fit)), dtype=float))
        acf_fit_nu_s_components = acf_fit.get("nu_s_components", np.array([], dtype=float))
        acf_fit_n_components = int(acf_fit.get("n_components", 1))
        acf_fit_off = float(acf_fit.get("off", np.nan))

        ps_fit = fit_power_spectrum(mod_freq, ps_on, nu_s_hint=nu_s_acf, nu_prior_weight=1.0, verbose=verbose)
        nu_s_ps = ps_fit["nu_s"]
        ps_fit_freqs = ps_fit["freqs"]
        ps_fit_model = ps_fit["model"]

        if np.isfinite(nu_s_acf) and nu_s_acf > 0:
            sigma_ch = 5.0 * (nu_s_acf / df)
        else:
            sigma_ch = 0.0
        I_smooth = gaussian_filter1d(Ion_model, sigma=sigma_ch) if sigma_ch > 0 else Ion_model.copy()
        acf_smooth = fully_normalized_acf(I_smooth)

        acf_on = fully_normalized_acf(Ion_model)
        c2 = len(Ion_model) // 2
        lags = np.arange(-c2, len(Ion_model) - c2)
        dnu_on = lags * df

        xfit, yfit, gpar = fit_gaussian_central_lags(dnu_on, acf_on, n_lags=10)
        if gpar is not None:
            G0 = float(gpar[3])

            if np.isfinite(S_hat_m) and S_hat_m > 0:
                noise_term = 1.0 / np.sqrt(N_on * S_hat_m)
                m2 = max(G0 - noise_term, 0.0)
                m_corr = float(np.sqrt(m2))
            else:
                m_corr = np.nan

    r = {
        "label": label,
        "frbname": frbname,
        "t_on": t_on,
        "f_plot": f_plot,
        "dyn_plot": dyn_plot,
        "dyn_on_raw": dyn_on,
        "I_f_display_plot": I_f_display_plot,

        "dnu_fit": dnu_fit,
        "acf_raw": acf_raw,
        "acf_fit_model": acf_fit_model,
        "acf_fit_component_models": acf_fit_component_models,
        "acf_fit_nu_s_components": acf_fit_nu_s_components,
        "acf_fit_n_components": acf_fit_n_components,
        "acf_fit_off": acf_fit_off,
        "nu_s_acf": nu_s_acf,

        "mod_freq": mod_freq,
        "ps_on": ps_on,
        "ps_fit_freqs": ps_fit_freqs,
        "ps_fit_model": ps_fit_model,
        "nu_s_ps": nu_s_ps,

        "acf_on": acf_on,
        "dnu_on": dnu_on,
        "acf_smooth": acf_smooth,
        "m_corr": m_corr,
    }

    make_big_multipanel(r, outdir)





####ACTUAL ANALYSIS 

# Fully-normalized autocorrelation function
def fully_normalized_acf(spec):
    spec = np.asarray(spec, float)
    m = np.nanmean(spec)
    denom = m if (np.isfinite(m) and m != 0) else 1.0
    x = (spec - m) / denom

    w = np.isfinite(x).astype(float)
    x = np.where(np.isfinite(x), x, 0.0)

    num = correlate(x, x, mode="same")
    den = correlate(w, w, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        acf = num / den
    acf[den == 0] = np.nan
    return acf


# Estimate and subtract the ACF zero-lag noise bias from OFF statistics
def fractional_acf_noise_sub(Ion_model, norm_on, norm_off, keep, df):
    acf_raw = fully_normalized_acf(Ion_model)

    N_on = max(int(norm_on.shape[0]), 1)
    sigma_off_ch = nanstd_cols(norm_off[:, keep], ddof=1)
    sigma_n = float(np.nanmean(sigma_off_ch)) / np.sqrt(N_on)

    S_hat = float(np.nanmean(Ion_model))
    bias0 = (sigma_n ** 2) / (S_hat ** 2) if (np.isfinite(S_hat) and S_hat != 0) else 0.0

    acf_corr = acf_raw.copy()
    c0 = len(acf_corr) // 2
    if np.isfinite(bias0):
        acf_corr[c0] = acf_corr[c0] - bias0

    lags = np.arange(-c0, len(acf_corr) - c0)
    dnu = lags * df

    return acf_raw, acf_corr, dnu, sigma_n, S_hat, bias0


# Simple Gaussian model for fitting ACF core
def _gauss(x, A, sigma, C):
    sigma = max(float(sigma), 1e-12)
    x = np.asarray(x, float)
    return A * np.exp(-(x * x) / (2.0 * sigma * sigma)) + C


# Fit a Gaussian to the central ACF lags (for modulation index estimate; still we have to subtract the noise)
def fit_gaussian_central_lags(x, y, n_lags=10):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < n_lags or y.size < n_lags or x.size != y.size:
        return None, None, None

    c = int(np.nanargmin(np.abs(x)))
    half = n_lags // 2
    i0, i1 = max(c - half, 0), min(c + half, x.size)
    while (i1 - i0) < n_lags and (i0 > 0 or i1 < x.size):
        if i0 > 0:
            i0 -= 1
        if (i1 - i0) < n_lags and i1 < x.size:
            i1 += 1

    xs, ys = x[i0:i1], y[i0:i1]
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None, None, None
    xs, ys = xs[m], ys[m]

    C0 = float(np.nanmedian(ys))
    A0 = float(np.nanmax(ys) - C0)
    if (not np.isfinite(A0)) or A0 <= 0:
        A0 = 1e-3

    ux = np.sort(np.unique(np.abs(xs)))
    dx = np.diff(ux)
    dx = dx[(dx > 0) & np.isfinite(dx)]
    step = float(np.nanmin(dx)) if dx.size else 1.0
    sigma0 = max(2.0 * step, 1e-3)

    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    span = max(ymax - ymin, 1e-3)
    lb = np.array([0.0, 0.2 * step, ymin - 0.5 * span], float)
    ub = np.array([50.0 * span, 50.0 * max(abs(xs).max(), step), ymax + 0.5 * span], float)

    p0 = np.array([A0, sigma0, C0], float)
    p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)

    def resid(p):
        A, sig, C = map(float, p)
        return _gauss(xs, A, sig, C) - ys

    f_scale = max(float(np.nanstd(ys)), 1e-3)
    sol = least_squares(resid, x0=p0, bounds=(lb, ub), loss="soft_l1", f_scale=f_scale, max_nfev=4000)
    if not sol.success:
        return None, None, None

    A_hat, sig_hat, C_hat = map(float, sol.x)
    yfit = _gauss(xs, A_hat, sig_hat, C_hat)
    G0 = A_hat + C_hat
    return xs, yfit, (A_hat, sig_hat, C_hat, G0)


# Lorentzian model for ACF fit
def lorentzian(dnu, nu_s, A, Off):
    nu_s = max(float(nu_s), 1e-12)
    x = np.asarray(dnu, float) / nu_s
    return A / (1.0 + x * x) + Off


def _lorentzian_sum(dnu, nu_s_arr, amp_arr, off):
    dnu = np.asarray(dnu, float)
    out = np.full_like(dnu, float(off), dtype=float)
    for ns, amp in zip(np.asarray(nu_s_arr, float), np.asarray(amp_arr, float)):
        ns = max(float(ns), 1e-12)
        out += float(amp) / (1.0 + (dnu / ns) ** 2)
    return out


# Seeds for width estimate are data driven: from the half-maximum point on positive lags
def _halfmax_seed(x, y, Off0, A0, W):
    target = Off0 + 0.5 * A0
    pos = x > 0
    if np.any(pos):
        i = np.nanargmin(np.abs(y[pos] - target))
        est = float(abs(x[pos][i]))
        if np.isfinite(est) and est > 0:
            return est
    return max(min(W / 3.0, W), 0.2)


# Fit a Lorentzian to the (noise-corrected) ACF to get scintillation bandwidth
def fit_acf_lorentzian(
    Dnu,
    ACF,
    fit_lag_cap_mhz=8.0,
    fit_lag_center_mhz=0.0,
    n_components=1,
    verbose=False,
):
    Dnu = np.asarray(Dnu, float)
    ACF = np.asarray(ACF, float)
    lag_center = float(fit_lag_center_mhz)
    Dnu_centered = Dnu - lag_center
    n_components = int(n_components)
    if n_components not in (1, 2, 3):
        raise ValueError(f"n_components must be 1, 2, or 3, got {n_components}")

    def _fail():
        return {
            "nu_s": np.nan,
            "amp": np.nan,
            "off": np.nan,
            "model": np.full_like(Dnu, np.nan),
            "nu_s_components": np.array([], dtype=float),
            "amp_components": np.array([], dtype=float),
            "model_components": np.empty((0, Dnu.size), dtype=float),
            "n_components": n_components,
        }

    mfin = np.isfinite(Dnu) & np.isfinite(ACF)
    if mfin.sum() < 7:
        return _fail()

    if fit_lag_cap_mhz and np.isfinite(fit_lag_cap_mhz) and fit_lag_cap_mhz > 0:
        mfin &= (np.abs(Dnu_centered) <= float(fit_lag_cap_mhz))
        if mfin.sum() < 7:
            return _fail()

    x_full = Dnu_centered[mfin]
    y_full = ACF[mfin]

    ux = np.sort(np.unique(np.abs(x_full)))
    if ux.size > 1:
        d = np.diff(ux)
        d = d[(d > 0) & np.isfinite(d)]
        dnu_min = float(np.nanmin(d)) if d.size else 0.0
    else:
        dnu_min = 0.0

    max_abs_x = float(np.nanmax(np.abs(x_full)))
    center_exclude = (ACF_EXCLUDE_NCHAN_CENTER * dnu_min) if dnu_min > 0 else 0.0
    center_exclude = min(center_exclude, 0.5 * max_abs_x)
    center_exclude = max(center_exclude, 2.0 * dnu_min)

    for _ in range(4):
        mfit = (np.abs(x_full) >= center_exclude)
        if mfit.sum() >= 7:
            break
        center_exclude *= 0.5
    else:
        return _fail()

    x = x_full[mfit]
    y = y_full[mfit]

    absx = np.abs(x)
    thr = np.nanquantile(absx, 0.8) if np.isfinite(absx).any() else np.nan
    if np.isfinite(thr):
        outer = absx >= thr
        Off0 = float(np.nanmedian(y[outer])) if outer.sum() >= 5 else float(np.nanmedian(y))
    else:
        Off0 = float(np.nanmedian(y))
    if not np.isfinite(Off0):
        Off0 = 0.0

    A0 = float(np.nanmax(y) - Off0)
    if (not np.isfinite(A0)) or A0 <= 0:
        A0 = 1e-3

    ns0 = _halfmax_seed(x, y, Off0, A0, max_abs_x)
    if (not np.isfinite(ns0)) or ns0 <= 0:
        ns0 = 1.0

    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dyn = (ymax - ymin) if (ymax > ymin) else 1.0

    nu_lo = max(1.5 * dnu_min, 0.05)
    nu_hi = min(max(2.0 * max_abs_x, 2.0 * nu_lo), 50.0)

    amp_lo = 1e-6
    amp_hi = max(10.0 * dyn, 1e-3)

    off_lo = float(np.nanpercentile(y, 30)) if np.isfinite(y).any() else ymin
    off_hi = float(np.nanpercentile(y, 90)) if np.isfinite(y).any() else ymax
    if not np.isfinite(off_lo) or not np.isfinite(off_hi) or off_lo >= off_hi:
        pad = max(0.1 * dyn, 1e-3)
        off_lo, off_hi = ymin - pad, ymax + pad

    ns_seeds = np.array([max(ns0 * (2.0 ** i), nu_lo) for i in range(n_components)], float)
    amp_seeds = np.array([max(A0 / float(i + 1), amp_lo) for i in range(n_components)], float)

    lb = np.concatenate([
        np.full(n_components, nu_lo, float),
        np.full(n_components, amp_lo, float),
        np.array([off_lo], float),
    ])
    ub = np.concatenate([
        np.full(n_components, nu_hi, float),
        np.full(n_components, amp_hi, float),
        np.array([off_hi], float),
    ])

    p0 = np.concatenate([ns_seeds, amp_seeds, np.array([Off0], float)])
    p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)

    W = 4.0 * (ns0 if (np.isfinite(ns0) and ns0 > 0) else 1.0)
    w = 1.0 - (np.abs(x) / max(W, 1e-12))
    w = np.clip(w, 0.0, 1.0)
    w = np.where(np.isfinite(w), w, 0.0)

    f_scale = max(float(np.nanstd(y)), 1e-3)

    def resid(p):
        p = np.asarray(p, float)
        ns = p[:n_components]
        amps = p[n_components:2 * n_components]
        Off = float(p[-1])
        r = (_lorentzian_sum(x, ns, amps, Off) - y) * w
        r[~np.isfinite(r)] = 0.0
        return r

    sol = least_squares(resid, x0=p0, bounds=(lb, ub), loss="cauchy", f_scale=f_scale, max_nfev=6000)
    if not sol.success:
        return _fail()

    p_hat = np.asarray(sol.x, float)
    ns_hat = p_hat[:n_components]
    A_hat = p_hat[n_components:2 * n_components]
    Off_hat = float(p_hat[-1])

    order = np.argsort(-A_hat)
    ns_hat = ns_hat[order]
    A_hat = A_hat[order]

    model = np.full_like(Dnu, np.nan, float)
    ok = np.isfinite(Dnu)
    model[ok] = _lorentzian_sum(Dnu_centered[ok], ns_hat, A_hat, Off_hat)

    model_components = np.full((n_components, Dnu.size), np.nan, float)
    for i in range(n_components):
        if np.any(ok):
            model_components[i, ok] = lorentzian(Dnu_centered[ok], ns_hat[i], A_hat[i], 0.0)

    ns_primary = float(ns_hat[0]) if ns_hat.size else np.nan
    amp_primary = float(A_hat[0]) if A_hat.size else np.nan

    if verbose:
        chi2 = np.nanmean(resid(sol.x) ** 2)
        ns_txt = ",".join([f"{v:.4f}" for v in ns_hat])
        a_txt = ",".join([f"{v:.3g}" for v in A_hat])
        print(f"[ACF fit] components={n_components}, nu_s=[{ns_txt}] MHz, A=[{a_txt}], Off={Off_hat:.3g}, "
              f"window center={lag_center:.3f} MHz, half-width=±{fit_lag_cap_mhz} MHz, "
              f"center_exclude≈{center_exclude:.3f} MHz, N={x.size}, <r^2>≈{chi2:.3g}")

    return {
        "nu_s": ns_primary,
        "amp": amp_primary,
        "off": Off_hat,
        "model": model,
        "nu_s_components": ns_hat,
        "amp_components": A_hat,
        "model_components": model_components,
        "n_components": n_components,
    }


# Exponential model used in the power-spectrum fit
def _ps_model(fv, ns, A, O):
    return A * np.exp(-2 * np.pi * ns * np.abs(fv)) + O


# Fit an exponential model to the modulation power spectrum 
def fit_power_spectrum(freqs, power, nu_s_hint=None, nu_prior_weight=1.0, verbose=False):
    f = np.asarray(freqs, float)
    P = np.asarray(power, float)

    pos = (f > 0) & np.isfinite(f) & np.isfinite(P)
    if pos.sum() < 5:
        return {"nu_s": np.nan, "freqs": f[pos], "model": np.full(pos.sum(), np.nan)}

    f = f[pos]
    P = P[pos]

    fmin = 0.005
    m = (f >= fmin)
    if m.sum() < 5:
        return {"nu_s": np.nan, "freqs": f, "model": np.full_like(f, np.nan)}

    fx, Py = f[m], P[m]

    Off0 = float(np.nanmedian(Py))
    A0 = max(float(np.nanmax(Py) - Off0), 1e-9)
    ns0 = 1.0

    ymin, ymax = float(np.nanmin(Py)), float(np.nanmax(Py))
    pad = max(0.1 * (ymax - ymin), 1e-3)
    f_scale = max(float(np.nanstd(Py)), 1.0)

    # Keep ns positive and effectively unbounded above for practical data ranges.
    lb = np.array([0.1, 1e-9, ymin - 2 * pad])
    ub = np.array([1.0e6, 1e12, ymax + 2 * pad])
    p0 = np.array([ns0, A0, Off0], float)
    p0 = np.minimum(np.maximum(p0, lb + 1e-12), ub - 1e-12)

    def resid(p):
        ns, A, O = map(float, p)
        r = _ps_model(fx, ns, A, O) - Py
        if nu_s_hint is not None and np.isfinite(nu_s_hint) and nu_prior_weight is not None:
            r = np.hstack([r, np.sqrt(float(nu_prior_weight)) * (ns - float(nu_s_hint))])
        return r

    sol = least_squares(resid, x0=p0, bounds=(lb, ub), loss="soft_l1", f_scale=f_scale, max_nfev=2000)
    if not sol.success:
        return {"nu_s": np.nan, "freqs": f, "model": np.full_like(f, np.nan)}

    ns_fit, A_hat, O_hat = map(float, sol.x)
    model = _ps_model(fx, ns_fit, A_hat, O_hat)

    if verbose:
        print(f"[PS fit] nu_s={ns_fit:.4f} MHz^-1  A={A_hat:.3g}  Off={O_hat:.3g}")

    return {"nu_s": ns_fit, "freqs": fx, "model": model}


# Create and save the big multi-panel summary figure
def make_big_multipanel(r, outdir):
    fig, axs = plt.subplots(
        2, 5, figsize=(26, 9.5),
        gridspec_kw={"width_ratios": [1, 1, 1.2, 0.5, 1],
                     "height_ratios": [0.75, 1.0]}
    )

    It = np.nanmean(r["dyn_on_raw"], axis=1) if r["dyn_on_raw"].size else np.array([])
    t_on = r["t_on"]
    axs[0, 2].plot(t_on, It, color="darkblue")
    axs[0, 2].set_title("Iₜ vs t", fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[0, 2].set_ylabel("I(t)")
    if t_on.size:
        axs[0, 2].set_xlim(t_on[0], t_on[-1])
    axs[0, 2].set_xticklabels([])
    axs[0, 2].tick_params(axis="both", labelsize=BIG_TEXT["ticks"])

    dnu_full = np.asarray(r["dnu_fit"], float)
    acf_full = np.asarray(r["acf_raw"], float)
    fit_full = np.asarray(r["acf_fit_model"], float)
    fit_comp_full = np.asarray(r.get("acf_fit_component_models", np.empty((0, dnu_full.size))), float)
    fit_comp_nu = np.asarray(r.get("acf_fit_nu_s_components", np.array([])), float)
    fit_ncomp = int(r.get("acf_fit_n_components", 1))
    fit_off = float(r.get("acf_fit_off", np.nan))
    off_plot = fit_off if np.isfinite(fit_off) else 0.0

    acf_full_plot = acf_full - off_plot
    fit_full_plot = fit_full - off_plot

    good = np.isfinite(dnu_full)
    if good.sum() > 2:
        df_mhz = float(np.nanmedian(np.diff(dnu_full[good])))
        if not np.isfinite(df_mhz) or df_mhz <= 0:
            df_mhz = 0.1
    else:
        df_mhz = 0.1
    cap_mhz = 100.0 * df_mhz

    mwin = (np.isfinite(dnu_full) & np.isfinite(acf_full_plot) & np.isfinite(fit_full_plot) &
            (dnu_full >= -cap_mhz) & (dnu_full <= cap_mhz))
    Dx30, Ay30, Fy30 = dnu_full[mwin], acf_full_plot[mwin], fit_full_plot[mwin]

    axs[0, 1].plot(Dx30, Ay30, color="black", label="Raw ACF")
    axs[0, 1].plot(Dx30, Fy30, "--", color="red",
                   label=fr"Lorentzian (νₛ={r['nu_s_acf']:.2f} MHz)")
    if fit_ncomp > 1 and fit_comp_full.ndim == 2 and fit_comp_full.shape[1] == dnu_full.size:
        # Comp 1 often visually overlaps the total fit; show additional components only.
        for i in range(1, min(fit_comp_full.shape[0], fit_comp_nu.size)):
            comp = fit_comp_full[i]
            mc = np.isfinite(dnu_full) & np.isfinite(comp) & (dnu_full >= -cap_mhz) & (dnu_full <= cap_mhz)
            if np.any(mc):
                axs[0, 1].plot(
                    dnu_full[mc],
                    comp[mc],
                    ":",
                    linewidth=1.2,
                    label=fr"Comp {i+1} (νₛ={fit_comp_nu[i]:.2f} MHz)",
                )
    axs[0, 1].axvline(0.0, ls=":", lw=1.0, color="k")
    axs[0, 1].set_title("ACF of I(f) (±30 lags)", fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[0, 1].set_xlabel("Freq lag [MHz]")
    axs[0, 1].set_ylabel("ACF")
    axs[0, 1].legend(prop={"size": BIG_TEXT["legend"]})
    axs[0, 1].tick_params(axis="both", labelsize=BIG_TEXT["ticks"])

    for col in [0, 3, 4]:
        fig.delaxes(axs[0, col])

    mf = r["mod_freq"]
    Pon = r["ps_on"]
    pf = r["ps_fit_freqs"]
    pm = r["ps_fit_model"]

    m_base = (mf > 0) & np.isfinite(mf) & np.isfinite(Pon) & (mf >= 0.005)
    m_zoom = m_base.copy()
    if np.isfinite(r["nu_s_ps"]) and (r["nu_s_ps"] > 0):
        zw_ps = 3.0 / (2.0 * np.pi * r["nu_s_ps"])
        m_zoom = m_base & (mf <= zw_ps)

    # Keep the nu_s-driven zoom when possible, but avoid nearly-empty plots.
    min_zoom_pts = 12
    if np.sum(m_zoom) < min_zoom_pts and np.any(m_base):
        idx = np.where(m_base)[0]
        idx = idx[np.argsort(mf[idx])]
        k = min(min_zoom_pts, idx.size)
        m_zoom = np.zeros_like(mf, dtype=bool)
        m_zoom[idx[:k]] = True

    if np.any(m_zoom):
        axs[1, 0].scatter(mf[m_zoom], Pon[m_zoom], s=10, label="ON (zoom)")
        if pf.size and pm.size:
            mfit_base = (pf > 0) & np.isfinite(pf) & np.isfinite(pm) & (pf >= 0.005)
            mfit = mfit_base.copy()
            if np.isfinite(r["nu_s_ps"]) and (r["nu_s_ps"] > 0):
                mfit &= (pf <= zw_ps)
            if np.sum(mfit) < min_zoom_pts and np.any(mfit_base):
                idxf = np.where(mfit_base)[0]
                idxf = idxf[np.argsort(pf[idxf])]
                kf = min(min_zoom_pts, idxf.size)
                mfit = np.zeros_like(pf, dtype=bool)
                mfit[idxf[:kf]] = True
            if np.any(mfit):
                axs[1, 0].plot(pf[mfit], pm[mfit], "--", linewidth=1.5,
                               label=f"Exp fit νₛ={r['nu_s_ps']:.3f} MHz")
        axs[1, 0].set_title(f"{r['label']} Power Spectrum", fontsize=BIG_TEXT["title"], fontweight="bold")
        axs[1, 0].set_xlabel(r"Modulation frequency [MHz$^{-1}$]")
        axs[1, 0].set_ylabel("Power")
        axs[1, 0].legend(prop={"size": BIG_TEXT["legend"]})

        mw = (np.isfinite(dnu_full) & np.isfinite(acf_full_plot) & np.isfinite(fit_full_plot) &
                    (dnu_full >= -ACF_DISPLAY_LAG_CAP_MHZ) & (dnu_full <= ACF_DISPLAY_LAG_CAP_MHZ))
        Dx, Ay, Fy = dnu_full[mw], acf_full_plot[mw], fit_full_plot[mw]

    axs[1, 1].plot(Dx, Ay, color="black", label="Raw ACF")
    axs[1, 1].plot(Dx, Fy, "--", color="red", label=f"νₛ={r['nu_s_acf']:.2f} MHz")
    if fit_ncomp > 1 and fit_comp_full.ndim == 2 and fit_comp_full.shape[1] == dnu_full.size:
        # Comp 1 often visually overlaps the total fit; show additional components only.
        for i in range(1, min(fit_comp_full.shape[0], fit_comp_nu.size)):
            comp = fit_comp_full[i]
            mc = np.isfinite(dnu_full) & np.isfinite(comp) & (dnu_full >= -ACF_DISPLAY_LAG_CAP_MHZ) & (dnu_full <= ACF_DISPLAY_LAG_CAP_MHZ)
            if np.any(mc):
                axs[1, 1].plot(
                    dnu_full[mc],
                    comp[mc],
                    ":",
                    linewidth=1.2,
                    label=f"Comp {i+1} νₛ={fit_comp_nu[i]:.2f}",
                )
    axs[1, 1].legend(prop={"size": BIG_TEXT["legend"]})
    axs[1, 1].set_title(f"{r['label']} Raw ACF + Fit", fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[1, 1].set_ylabel("ACF")
    axs[1, 1].set_xlabel("Freq lag [MHz]")
    axs[1, 1].tick_params(axis="both", labelsize=BIG_TEXT["ticks"])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e60073")
    dp = r["dyn_plot"]
    data = np.array(dp, dtype=float)
    if isinstance(dp, np.ma.MaskedArray):
        data[dp.mask] = np.nan
    vmin, vmax = np.nanpercentile(data, [2, 98]) if np.isfinite(data).any() else (-3, 3)

    if r["t_on"].size:
        axs[1, 2].imshow(
            dp, aspect="auto", origin="upper", cmap=cmap,
            extent=[r["t_on"][0], r["t_on"][-1], r["f_plot"][0], r["f_plot"][-1]],
            vmin=vmin, vmax=vmax, interpolation="none"
        )
    axs[1, 2].set_ylim(r["f_plot"][0], r["f_plot"][-1])
    axs[1, 2].set_title(f"{r['label']} Dynamic Spectrum", fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[1, 2].set_xlabel("Time [ms]")
    axs[1, 2].set_ylabel("Freq [MHz]")

    I, F = r["I_f_display_plot"], r["f_plot"]
    m = np.isfinite(I) & np.isfinite(F)
    axs[1, 3].plot(I[m], F[m], color="darkblue", linewidth=1.5)
    axs[1, 3].set_title(f"{r['label']} I(f)", fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[1, 3].set_ylim(F[0], F[-1])
    axs[1, 3].invert_yaxis()
    axs[1, 3].set_xlabel("I(f)")
    axs[1, 3].set_yticks([])

    xacf = np.asarray(r["dnu_on"], float)
    yacf = np.asarray(r["acf_on"], float)
    axs[1, 4].plot(xacf, yacf, color="blue", linewidth=2.5, label="ON")

    xfit, yfit, gpar = fit_gaussian_central_lags(xacf, yacf, n_lags=10)
    if gpar is not None:
        axs[1, 4].plot(xfit, yfit, "--", color="red", linewidth=2.5,
                       label=f"Gaussian; m={r['m_corr']:.3g}")

    axs[1, 4].axvline(0.0, ls=":", lw=1.0, color="k")
    if np.isfinite(ACF_DISPLAY_LAG_CAP_MHZ) and ACF_DISPLAY_LAG_CAP_MHZ > 0:
        axs[1, 4].set_xlim(-ACF_DISPLAY_LAG_CAP_MHZ, ACF_DISPLAY_LAG_CAP_MHZ)
    elif np.any(np.isfinite(xacf)):
        xfin = xacf[np.isfinite(xacf)]
        axs[1, 4].set_xlim(np.nanmin(xfin), np.nanmax(xfin))
    axs[1, 4].set_title(f"{r['label']} ACF of I(f) (ON, core)",
                        fontsize=BIG_TEXT["title"], fontweight="bold")
    axs[1, 4].set_xlabel("Freq lag [MHz]")
    axs[1, 4].set_ylabel("ACF")
    axs[1, 4].legend(prop={"size": BIG_TEXT["legend"]})
    axs[1, 4].tick_params(axis="both", labelsize=BIG_TEXT["ticks"])

    fig.suptitle(f"FRB {r['frbname']} – Big Multipanel Output",
                 fontsize=BIG_TEXT["suptitle"], fontweight="bold")

    style_axes(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # New, cleaner filename
    fig.savefig(os.path.join(outdir, f"{r['frbname']}_big_panel_multipanel.png"), dpi=400)
    plt.close(fig)


# Process one YAML 
def process_yaml(
    yaml_path,
    input_mode="auto",
    verbose=False,
    cli_t_ref_ms=None,
    cli_time_file=None,
    cli_time_unit=None,
    on_half_width_ms=4.0,
    acf_fit_lag_cap_mhz=8.0,
    acf_fit_center_mhz=0.0,
    acf_lorentzian_components=1,
):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    par = meta["par"]
    frbname = str(par["name"]).strip()

    # New output root folder
    outdir = os.path.join(os.getcwd(), "OUTPUT_COLLECTION", frbname)
    os.makedirs(outdir, exist_ok=True)

    bw = float(par["bw"])
    cfreq = float(par["cfreq"])
    par_t_ref_ms = _coerce_optional_float(par.get("t_ref", None))
    chosen_t_ref_ms = _coerce_optional_float(cli_t_ref_ms)

    zap_ranges = parse_zap_from_yaml(get_zap_field(meta))
    print(f"Parsed zap ranges (MHz): {zap_ranges}")
    mode = _resolve_input_mode(meta, input_mode)
    print(f"\nProcessing FRB {frbname} (mode={mode})")

    if mode == "voltages":
        base_path = os.path.dirname(meta["data"]["dsI"])
        folder_name = os.path.basename(os.path.dirname(base_path))
        X, Y, dmstr = load_XY_with_dm_fallback(base_path, folder_name, par["DM"])
        print(f"Using DM string for file load: {dmstr}")

        for res, label in zip(RESOLUTIONS, RES_LABELS):
            nfft = int(FS / res)
            step = nfft // 2

            t_ref_ms = chosen_t_ref_ms
            if t_ref_ms is None:
                t_ref_ms = par_t_ref_ms
            if t_ref_ms is None:
                t_ref_ms = estimate_t_ref_ms_from_voltages(X, Y, fft_size=nfft, step_size=step)
                print(f"Auto-selected t_ref from peak (voltages): {t_ref_ms:.6f} ms")
            else:
                print(f"Using provided t_ref: {float(t_ref_ms):.6f} ms")

            dyn_on, dyn_off, t_on, _t_off = build_dynspec_onoff(
                X,
                Y,
                nfft,
                step,
                float(t_ref_ms),
                on_half_width_ms=on_half_width_ms,
            )

            if dyn_on.size == 0:
                raise RuntimeError(f"No ON samples selected for {frbname} at resolution {label}")

            f_data, df = freq_axis(nfft, cfreq, bw)
            analyze_dynspec_and_plot(
                dyn_on=dyn_on,
                dyn_off=dyn_off,
                t_on=t_on,
                f_data=f_data,
                df=df,
                zap_ranges=zap_ranges,
                label=label,
                frbname=frbname,
                outdir=outdir,
                verbose=verbose,
                acf_fit_lag_cap_mhz=acf_fit_lag_cap_mhz,
                acf_fit_center_mhz=acf_fit_center_mhz,
                acf_lorentzian_components=acf_lorentzian_components,
            )
    elif mode == "stokes":
        I_dyn, t_ms, dt_ms = load_stokes_i_from_meta(
            meta,
            yaml_path,
            cli_time_file=cli_time_file,
            cli_time_unit=cli_time_unit,
        )

        t_ref_ms = chosen_t_ref_ms
        if t_ref_ms is None:
            t_ref_ms = par_t_ref_ms
        if t_ref_ms is None:
            t_ref_ms = _peak_ref_time_ms(I_dyn, t_ms=t_ms, dt_ms=dt_ms)
            print(f"Auto-selected t_ref from peak (stokes): {t_ref_ms:.6f} ms")
        else:
            print(f"Using provided t_ref: {float(t_ref_ms):.6f} ms")

        dyn_on, dyn_off, t_on, _t_off = build_dynspec_onoff_from_stokes(
            I_dyn,
            t_ref_ms=float(t_ref_ms),
            dt_ms=dt_ms,
            t_ms=t_ms,
            on_half_width_ms=on_half_width_ms,
        )
        if dyn_on.size == 0:
            raise RuntimeError(f"No ON samples selected for {frbname} in stokes mode")

        nchan = int(I_dyn.shape[1])
        f_data, df = freq_axis(nchan, cfreq, bw)
        analyze_dynspec_and_plot(
            dyn_on=dyn_on,
            dyn_off=dyn_off,
            t_on=t_on,
            f_data=f_data,
            df=df,
            zap_ranges=zap_ranges,
            label="Stokes-I",
            frbname=frbname,
            outdir=outdir,
            verbose=verbose,
            acf_fit_lag_cap_mhz=acf_fit_lag_cap_mhz,
            acf_fit_center_mhz=acf_fit_center_mhz,
            acf_lorentzian_components=acf_lorentzian_components,
        )
    else:
        raise ValueError(f"Unsupported mode '{mode}'")


# Process all .yaml files - the script runs on all the yaml files in the folder 
def _parse_args():
    p = argparse.ArgumentParser(description="Run the FRB scintillation pipeline on YAML metadata files")
    p.add_argument("yaml_files", nargs="*", help="Specific YAML files to process")
    p.add_argument("--yaml-glob", default="*.yaml", help="Glob used when no positional YAML files are provided")
    p.add_argument(
        "--input-mode",
        choices=["auto", "voltages", "stokes"],
        default="auto",
        help="Choose input path. 'auto' uses YAML hints or inferred mode.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose fit diagnostics")
    p.add_argument("--t-ref-ms", type=float, default=None, help="Override t_ref in ms. If omitted, YAML t_ref is used, else peak is auto-selected.")
    p.add_argument("--time-file", default=None, help="Optional .npy file of time samples (used in stokes mode).")
    p.add_argument(
        "--time-unit",
        choices=["ms", "s"],
        default=None,
        help="Unit for --time-file samples (default comes from YAML data.time_unit or ms).",
    )
    p.add_argument(
        "--acf-time-half-width-ms",
        type=float,
        default=4.0,
        help="Half-width of the ON time window around t_ref in ms (uses t_ref ± value).",
    )
    p.add_argument(
        "--acf-freq-half-width-mhz",
        type=float,
        default=8.0,
        help="Half-width of the ACF fit window in frequency lag (MHz).",
    )
    p.add_argument(
        "--acf-freq-center-mhz",
        type=float,
        default=0.0,
        help="Center of the ACF fit frequency-lag window in MHz.",
    )
    p.add_argument(
        "--acf-lorentzian-components",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Number of Lorentzian components used in ACF fitting (single/double/triple).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    yaml_files = args.yaml_files if args.yaml_files else sorted(glob.glob(args.yaml_glob))
    out_root = os.path.join(os.getcwd(), "STATISTICS2")
    err_dir = os.path.join(out_root, "_errors")
    os.makedirs(err_dir, exist_ok=True)

    if not yaml_files:
        print(f"No YAML files found (glob={args.yaml_glob})")
        return

    for yfile in yaml_files:
        print(f"\n=== Processing {yfile} ===")
        try:
            process_yaml(
                yfile,
                input_mode=args.input_mode,
                verbose=args.verbose,
                cli_t_ref_ms=args.t_ref_ms,
                cli_time_file=args.time_file,
                cli_time_unit=args.time_unit,
                on_half_width_ms=args.acf_time_half_width_ms,
                acf_fit_lag_cap_mhz=args.acf_freq_half_width_mhz,
                acf_fit_center_mhz=args.acf_freq_center_mhz,
                acf_lorentzian_components=args.acf_lorentzian_components,
            )
            print(f"Done: {yfile}")
        except Exception as e:
            print(f"ERROR in {yfile}: {e.__class__.__name__}: {e}")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logp = os.path.join(err_dir, f"{os.path.basename(yfile)}.{stamp}.log")
            with open(logp, "w") as fh:
                fh.write(f"YAML: {yfile}\n\n")
                fh.write(traceback.format_exc())


if __name__ == "__main__":
    main()



