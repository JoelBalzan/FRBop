# snop_cli.py - Generalized version
import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ilex.frb import FRB


def auto_set_pa0_main(frb, Ldebias_threshold: float = 2.0, logger=None, **kwargs):
    """
    Local helper to compute the circular mean PA from FRB time-series and
    update `frb.par.pa0` so the mean PA becomes zero.

    This mirrors the previous helper that lived inside the installed `ilex` package
    but keeps the change local to this repository.
    """
    # Update instance params / crops
    try:
        frb._load_new_params(**kwargs)
    except Exception:
        pass

    if frb.this_metapar.terr_crop is None:
        msg = "Need terr_crop to estimate off-pulse rms for PA masking"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    frb.get_data(["tI", "tQ", "tU"], **kwargs)
    if not frb._isdata():
        msg = "Unable to retrieve time-series data to compute PA"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    Q = frb._t["Q"]
    U = frb._t["U"]
    Ierr = frb._t.get("Ierr", None)

    # compute PA and linear pol amplitude
    PA = 0.5 * np.arctan2(U, Q)
    L = np.sqrt(Q**2 + U**2)

    # mask low S/N using Ierr if available
    if Ierr is not None:
        valid = (~np.isnan(PA)) & (L > Ldebias_threshold * Ierr)
    else:
        valid = ~np.isnan(PA)

    if np.count_nonzero(valid) == 0:
        # fallback to any finite PA
        valid = ~np.isnan(PA)
        if np.count_nonzero(valid) == 0:
            msg = "No valid PA samples found to compute mean"
            if logger:
                logger.error(msg)
            else:
                print(msg)
            return None

    mean_two = np.angle(np.mean(np.exp(2j * PA[valid])))
    mean_pa = mean_two / 2.0
    new_pa0 = -mean_pa

    # normalise into [-pi/2, pi/2]
    while new_pa0 <= -np.pi/2:
        new_pa0 += np.pi
    while new_pa0 > np.pi/2:
        new_pa0 -= np.pi

    frb.set(pa0=new_pa0)
    if logger:
        logger.info("Auto-set pa0 to %.3f deg", np.rad2deg(new_pa0))
    else:
        print(f"Auto-set pa0 to {np.rad2deg(new_pa0):.3f} deg")

    return new_pa0


def _setup_logging(outdir, label):
    os.makedirs(outdir, exist_ok=True)
    safe_label = str(label).replace(" ", "_") if label else "snop"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(outdir, f"{safe_label}_{timestamp}.log")

    logger = logging.getLogger("snop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def _run_command(cmd, logger):
    logger.info("Executing: %s", " ".join(cmd))
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            logger.info("[subprocess] %s", line.rstrip())
        process.wait()
        return process.returncode


def _apply_supplied_dm_shift(args, logger, loaded_dt, cfreq, bw):
    """Apply a user-supplied delta DM shift to out_*.npy using ILEX script."""
    dedisp_script = "/home/joel/Documents/ILEX/scripts/incoherent_dedisperse.py"
    if not os.path.exists(dedisp_script):
        raise FileNotFoundError(f"DM shift script not found: {dedisp_script}")

    stokes_files = ["out_I.npy", "out_Q.npy", "out_U.npy", "out_V.npy"]
    for infile in stokes_files:
        if not os.path.exists(infile):
            raise FileNotFoundError(f"Required dynamic spectrum not found: {infile}")

    logger.info(
        "Applying supplied DM shift delDM=%.6f pc/cm^3 to out_*.npy",
        float(args.dm_shift),
    )

    for infile in stokes_files:
        tmp_out = infile.replace(".npy", "_dedisp_tmp.npy")
        cmd = [
            sys.executable,
            dedisp_script,
            "-i",
            infile,
            "--dt",
            str(float(loaded_dt)),
            "--cfreq",
            str(float(cfreq)),
            "--bw",
            str(float(bw)),
            "--delDM",
            str(float(args.dm_shift)),
            "-o",
            tmp_out,
        ]
        if args.dm_lower:
            cmd.append("--lower")

        returncode = _run_command(cmd, logger)
        if returncode != 0:
            raise RuntimeError(
                f"incoherent_dedisperse.py failed for {infile} with exit code {returncode}"
            )
        os.replace(tmp_out, infile)

    logger.info("DM shift application complete for out_I/Q/U/V.npy")




def load_config(filepath):
    """Load configuration file with sections marked by ****Section****"""
    config = {}
    section = "General"
    config[section] = {}
    keyval_re = re.compile(r'^([a-zA-Z0-9_]+)\s*=\s*(.*)$')
    section_re = re.compile(r'^\*{4}([a-zA-Z0-9_ ]+)\*{4}$')  # Added underscore here
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sec_match = section_re.match(line)
            if sec_match:
                section = sec_match.group(1).strip()
                config[section] = {}
                continue
            kv_match = keyval_re.match(line)
            if kv_match:
                key, val = kv_match.group(1), kv_match.group(2)
                if "#" in val:
                    val = val.split("#", 1)[0].strip()
                # Try to parse lists
                if val.startswith("[") and val.endswith("]"):
                    val = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                    # Try to convert to int/float if possible
                    parsed_val = []
                    for v in val:
                        try:
                            parsed_val.append(int(v) if v.isdigit() else float(v))
                        except ValueError:
                            parsed_val.append(v)
                    val = parsed_val
                # Try to parse booleans
                elif val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                # Try to parse None
                elif val.lower() == "none":
                    val = None
                # Try to parse numbers
                else:
                    try:
                        if "." in val or "e" in val.lower():
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
                config[section][key] = val
    return config


def main():
    parser = argparse.ArgumentParser(description="Optimise S/N dynamic spectrum from X Y polarisation data")
    parser.add_argument(
        "-x", "--xdata",
        type=str,
        required=True,
        help="Path to x polarisation data file"
    )
    parser.add_argument(
        "-y", "--ydata",
        type=str,
        required=True,
        help="Path to y polarisation data file"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory (default: current working directory)"
    )
    parser.add_argument(
        "-p", "--parameters",
        type=str,
        default="parameters.txt",
        help="Path to parameters.txt file (default: parameters.txt)"
    )
    parser.add_argument(
        "--fires-config-dir",
        type=str,
        default=None,
        help="Config directory for FIRES (default: ~/Documents/GitHub/FIRES/paper/<frbname>/)"
    )
    parser.add_argument(
        "--skip-dynspec",
        action="store_true",
        help="Skip make_dynspec.py step (use existing out_*.npy files)"
    )
    parser.add_argument(
        "--skip-rm",
        action="store_true",
        help="Skip RM fitting (use RM=0 and skip second pass)"
    )
    parser.add_argument(
        "--nFFT",
        type=int,
        default=336,
        help="nFFT value to pass to make_dynspec.py (default: 336)"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show interactive plots during processing"
    )
    parser.add_argument(
        "--dm-shift",
        "--delDM",
        dest="dm_shift",
        type=float,
        default=None,
        help="Apply a supplied delta-DM shift (pc/cm^3) using incoherent_dedisperse.py before FRB analysis"
    )
    parser.add_argument(
        "--dm-lower",
        action="store_true",
        help="Pass --lower to incoherent_dedisperse.py (set if first channel is bottom of band)"
    )
    parser.add_argument(
        "--fires",
        action="store_true",
        help="Run FIRES analysis after processing"
    )
    parser.add_argument(
        "--tN",
        type=int,
        default=None,
        help="time averaging factor (overridden by --time-res-ms if provided)"
    )
    parser.add_argument(
        "--time-res-ms",
        type=float,
        default=None,
        help="desired dynspec time resolution in ms; converted to nearest valid tN"
    )
    args = parser.parse_args()

    # Load parameters
    parameters = load_config(args.parameters)
    cfreq = float(parameters['FRB']['centre_freq_frb'])
    bw = float(parameters['General']['bw'])
    label = str(parameters['General']['label']).replace(" ", "_")
    stk_debias = parameters['General'].get('stk_debias', False)
    # base time resolution (ms per voltage sample) -- fixed by the recorder
    # note: for the 336-MHz system this is ≈2.98e-6 ms; it does *not* scale
    # with bw.  Earlier versions multiplied by bw, which gave the wrong
    # behaviour when bw changed.
    raw_dt = 2.98e-6
    postfft_dt = raw_dt * args.nFFT

    if args.time_res_ms is not None and args.time_res_ms <= 0:
        raise ValueError("--time-res-ms must be > 0")
    if args.tN is not None and args.tN < 1:
        raise ValueError("--tN must be >= 1")
    if args.time_res_ms is not None and args.tN is not None:
        raise ValueError("Specify only one of --tN or --time-res-ms")

    if args.time_res_ms is not None:
        effective_tN = max(1, int(round(args.time_res_ms / postfft_dt)))
        target_dt = postfft_dt * effective_tN
    elif args.tN is not None:
        effective_tN = int(args.tN)
        target_dt = postfft_dt * effective_tN
    else:
        effective_tN = 50
        target_dt = postfft_dt * effective_tN

    # ILEX out_*.npy are loaded at post-FFT cadence; requested tN is applied in FRB analysis.
    loaded_dt = postfft_dt

    df = None
    # raw_dt tracks the instrument sample interval.
    # loaded_dt is the cadence of out_*.npy when loaded into FRB.
    # target_dt is the requested final cadence after FRB-side averaging by effective_tN.
    
    # Load FRB-specific settings with defaults
    frb_config = parameters.get('FRB_Config', {})
    f0 = frb_config.get('f0', 1000)
    # If pa0 provided in config it's degrees; otherwise we'll auto-set later
    if 'pa0' in frb_config:
        pa0_deg = frb_config['pa0']
        pa0 = np.deg2rad(pa0_deg)
        use_auto_pa0 = False
    else:
        pa0 = None
        use_auto_pa0 = True
    t_crop = frb_config.get('t_crop', None)
    terr_crop = frb_config.get('terr_crop', None)
    f_crop = frb_config.get('f_crop', None)
    ferr_crop = frb_config.get('ferr_crop', None)
    
    # find_frb parameters (support both legacy and pass-based config sections)
    find_frb_config = {}
    find_frb_section = None
    for candidate in ('FindFRB_Pass1', 'FindFRB', 'FindFRB_Pass2'):
        if candidate in parameters:
            find_frb_config = parameters.get(candidate, {})
            find_frb_section = candidate
            break

    padding = find_frb_config.get('padding', 0.4)
    dt_from_peak_sigma = find_frb_config.get('dt_from_peak_sigma', 20)
    rms_guard = find_frb_config.get('rms_guard', None)
    rms_width = find_frb_config.get('rms_width', None)
    rms_offset = find_frb_config.get('rms_offset', None)
    
    # Plot settings
    plot_config = parameters.get('Plot', {})
    plot_pa = plot_config.get('plot_pa', True)
    plot_ds_intermediate = plot_config.get('plot_ds_intermediate', False)
    plot_ds_final = plot_config.get('plot_ds_final', False)

    logger, log_path = _setup_logging(args.outdir, label)
    logger.info("Logging to %s", log_path)
    logger.info(
        "Using find_frb config section: %s (dt_from_peak_sigma=%s, padding=%s)",
        find_frb_section or "defaults",
        dt_from_peak_sigma,
        padding,
    )
    if args.time_res_ms is not None:
        logger.info(
            "Using --time-res-ms=%.6f ms -> effective tN=%s (target final dt=%.6f ms)",
            float(args.time_res_ms),
            effective_tN,
            target_dt,
        )
    elif args.tN is not None:
        logger.info("Using user-supplied tN=%s (target final dt=%.6f ms)", effective_tN, target_dt)
    else:
        logger.info("Using default tN=%s (target final dt=%.6f ms)", effective_tN, target_dt)

    # Apply requested temporal averaging during FRB analysis/saving.
    analysis_tN = int(effective_tN)

    # Step 1: Generate dynamic spectrum from X/Y polarisation data
    if not args.skip_dynspec:
        script_path = os.path.expanduser("~/Documents/ILEX/scripts/make_dynspec.py")
        cmd = [
            sys.executable,
            script_path,
            "-x",
            args.xdata,
            "-y",
            args.ydata,
            "--bline",
            "--QUV",
            "--do_chanflag",
            "--nFFT",
            str(args.nFFT),
            "--tN",
            str(effective_tN)
        ]
        returncode = _run_command(cmd, logger)
        if returncode != 0:
            logger.error("Dynamic spectrum generation failed with exit code %s", returncode)
            sys.exit(returncode)

    if args.dm_shift is not None:
        try:
            _apply_supplied_dm_shift(args, logger, loaded_dt=loaded_dt, cfreq=cfreq, bw=bw)
        except Exception as e:
            logger.error("Failed to apply supplied DM shift: %s", e)
            sys.exit(1)

    # Load generated data
    I = np.load("out_I.npy")
    Q = np.load("out_Q.npy")
    U = np.load("out_U.npy")
    V = np.load("out_V.npy")

    # Infer channel width from dynamic spectrum shape
    nchan = I.shape[0]
    nsamp_loaded = I.shape[1]
    df = bw / nchan
    # Loaded out_*.npy cadence before FRB-side averaging
    logger.info("=== Data loaded: %s channels × %s time samples ===", nchan, nsamp_loaded)
    logger.info("Loaded dynspec column width (post-FFT): %.6f ms", loaded_dt)
    logger.info("Target final column width after FRB averaging: %.6f ms", target_dt)
    
    # Check if loaded data matches expected nFFT
    if args.skip_dynspec and nchan != args.nFFT:
        logger.warning("Loaded data has %s channels, but --nFFT %s was specified.", nchan, args.nFFT)
        logger.warning("The existing out_*.npy files were generated with a different nFFT value.")
        logger.warning("To use %s channels, remove --skip-dynspec flag to regenerate the files.", args.nFFT)
    elif nchan == args.nFFT:
        logger.info("Data has expected %s channels", args.nFFT)
    
    logger.info("Channel width: df = %.6f MHz", df)
    logger.info("Note: Matplotlib may downsample the display, but the full %s channels are in the data", nchan)
    
    # Calculate total observation time for automatic rms parameter calculation
    nsamp = I.shape[1]
    total_time = nsamp * loaded_dt  # Total time in milliseconds
    
    # Set automatic defaults for rms parameters if not specified
    # These define the off-pulse region used for noise estimation
    if rms_guard is None:
        rms_guard = total_time * 0.025  # 2.5% of total time
        logger.info("Auto-setting rms_guard = %.2f ms (2.5%% of total time)", rms_guard)
    if rms_width is None:
        rms_width = total_time * 0.1  # 10% of total time
        logger.info("Auto-setting rms_width = %.2f ms (10%% of total time)", rms_width)
    if rms_offset is None:
        rms_offset = total_time * 0.15  # 15% of total time
        logger.info("Auto-setting rms_offset = %.2f ms (15%% of total time)", rms_offset)
    
    # Don't set terr_crop yet - let find_frb() set it relative to the burst peak
    # Setting it now causes issues because there's no t_ref yet

    # Step 2: First pass - find FRB and fit RM
    logger.info("=== First pass: Finding FRB and fitting RM ===")
    frb_kwargs = {
        'name': label,
        'cfreq': cfreq,
        'bw': bw,
        'dt': loaded_dt,
        'df': df,
        'f0': f0,
    }
    if pa0 is not None:
        frb_kwargs['pa0'] = pa0
    if t_crop is not None:
        frb_kwargs['t_crop'] = t_crop
    # Don't pass terr_crop to FRB init - find_frb will set it
    if f_crop is not None:
        frb_kwargs['f_crop'] = f_crop

    frb = FRB(**frb_kwargs)
    frb.load_data(dsI="out_I.npy", dsQ="out_Q.npy", dsU="out_U.npy", dsV="out_V.npy")
    
    # Find FRB
    find_frb_kwargs = {
        'method': 'fluence',
        'mode': 'min',
        'padding': padding,
        'dt_from_peak_sigma': dt_from_peak_sigma,
        'tN': analysis_tN
    }
    if rms_guard is not None:
        find_frb_kwargs['rms_guard'] = rms_guard
    if rms_width is not None:
        find_frb_kwargs['rms_width'] = rms_width
    if rms_offset is not None:
        find_frb_kwargs['rms_offset'] = rms_offset
    
    frb.set(tN=int(analysis_tN))
    frb.find_frb(**find_frb_kwargs)
    frb.set(tN=int(analysis_tN))
    logger.info(
        "Requested averaging: make_dynspec tN=%s, FRB analysis tN=%s",
        int(effective_tN),
        int(analysis_tN),
    )
  
    if plot_ds_intermediate or args.show_plots:
        frb.plot_data("dsI", show_plots=True)
    
    # Fit RM (unless skipped)
    if not args.skip_rm:
        try:
            _, rmDict = frb.fit_RM(method="RMsynth", show_plots=False, tN=int(analysis_tN))
            if rmDict is not None:
                logger.info("Found RM = %.2f rad/m^2", rmDict['rm'])
                
                # Apply RM correction
                logger.info("=== Applying RM correction ===")
                frb.set(RM=rmDict['rm'])
            else:
                logger.warning("RM fitting returned None, using RM=0")
        except Exception as e:
            logger.warning("RM fitting failed: %s", e)
            logger.warning("Using RM=0. Consider using --skip-rm flag in future.")
    else:
        logger.info("Skipping RM fitting (using RM=0)")
    
    # Disable terr_crop after RM fitting to prevent crop parameter corruption
    # (ILEX has a bug where terr_crop causes negative array dimensions in some operations)
    #frb.metapar.terr_crop = None

    # Auto-center PA only if pa0 wasn't provided in config
    if use_auto_pa0:
        try:
            new_pa0_rad = auto_set_pa0_main(frb, logger=logger, tN=int(analysis_tN))
            if new_pa0_rad is not None:
                logger.info("Auto-set pa0 = %.3f deg", np.rad2deg(new_pa0_rad))
        except Exception as e:
            logger.warning("auto_set_pa0 failed: %s", e)
    
    if plot_pa:
        frb.plot_PA()
    
    if plot_ds_final or args.show_plots:
        frb.plot_data("dsI", show_plots=True)
    
    active_tN = int(getattr(frb.metapar, "tN", 1) or 1)
    active_fN = int(getattr(frb.metapar, "fN", 1) or 1)
    base_dt = float(getattr(frb.par, "dt", loaded_dt))
    base_df = float(getattr(frb.par, "df", df))
    frb_dt = float(getattr(frb.this_par, "dt", base_dt))  # sampling interval of FRB time series (final resolution)
    # compute time resolutions for reporting
    postfft_dt = raw_dt * args.nFFT  # time per FFT bin (before temporal averaging in make_dynspec)
    # target_dt = raw_dt * nFFT * effective_tN (requested final time resolution)
    logger.info(
        "Time resolution chain: raw=%.6e ms → post-FFT loaded=%.6f ms (nFFT=%s) → target=%.6f ms (requested tN=%s) → FRB series=%.6f ms (active_tN=%s)",
        raw_dt,
        postfft_dt,
        args.nFFT,
        target_dt,
        effective_tN,
        frb_dt,
        active_tN,
    )
    logger.info(
        "Active frequency resolution: fN=%s, df=%.6f MHz (base %.6f MHz)",
        active_fN,
        base_df * active_fN,
        base_df,
    )
    logger.info("FRB parameters (base): %s", frb.par)
    
    # Step 4: Save processed data
    logger.info("=== Saving processed data ===")
    if stk_debias:
        logger.info("Saving with stk_debias=True (debiasing applied to Stokes parameters)")
    frb.save_data(
        data_list=['dsI', 'dsQ', 'dsU', 'dsV'],
        name=f"{label}_htr",
        stk_debias=stk_debias
    )
   
    del frb
    I = np.load(f"{label}_htr_dsI.npy")
    Q = np.load(f"{label}_htr_dsQ.npy")
    U = np.load(f"{label}_htr_dsU.npy")
    V = np.load(f"{label}_htr_dsV.npy")
    ds = np.array([I, Q, U, V])
    np.save(os.path.join(args.outdir, f"{label}_ds.npy"), ds)

    # Step 5: Run FIRES analysis if requested
    if not args.fires:
        sys.exit(0)

    # Else: Run FIRES analysis
    logger.info("=== Running FIRES analysis ===")
    config_dir = args.fires_config_dir
    if config_dir is None:
        # Default to ~/Documents/GitHub/FIRES/paper/<frbname>/
        frbname_short = label.replace("FRB_", "").replace("FRB", "").lstrip("_").lstrip()
        config_dir = f"~/Documents/GitHub/FIRES/examples/20240318A/"#{frbname_short}/"
    
    cmd = [
        sys.executable, "-m", "fires",
        "-f", f"{label}_htr",
        "-p", "lvpa",
        "--obs-data", ".",
        "-v",
        "-o", args.outdir,
        "--config-dir", config_dir,
        "--override-param", "tau=0",
        "--plot-config", config_dir
    ]
    returncode = _run_command(cmd, logger)
    if returncode != 0:
        logger.error("FIRES analysis failed with exit code %s", returncode)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
