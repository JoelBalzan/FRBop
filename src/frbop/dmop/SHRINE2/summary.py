"""The summary files written by the scripts, in SHRINE's layout and with SHRINE's key lines.

write_structure_summary writes {label}_structure_summaryfile.txt (maximise_structure.py,
optimise_dm.py); write_sn_summary writes {label}_SN_summaryfile.txt (maximise_sn.py).
"""


def write_structure_summary(path, res, label, dt, DM, kc_note, warnings, grid, saving=False,
                            zap=()):
    """Write the structure-maximisation summary file, {label}_structure_summaryfile.txt, for a
    result of estimate.dm_uncertainty: k_c, d_fit +- sigma_DM, Sp_max +- sigma_S, the diagnostics
    and the warnings.

    Args:
        path: output file name.
        res: the dict returned by estimate.dm_uncertainty.
        label: FRB label (-l).
        dt: time resolution, us (-t).
        DM: the DM the input was dedispersed to (-d).
        kc_note: how k_c was chosen.
        warnings: list of str, from check_trial_dms and the k_c check.
        grid: (d_low, d_high, step) from check_trial_dms, or None.
        saving: whether plots were saved (-s).
        zap: the channels left out of the profiles.
    """
    # The zapped channel indices as text, separated by spaces ("none" if there are none).
    zap_words = []
    for c in zap:
        zap_words.append(str(c))
    zap_text = " ".join(zap_words)
    if not zap_text:
        zap_text = "none"

    with open(path, "w") as out:
        out.write("//begin maximise_structure summary//\n/*\n")

        # The run: SHRINE's key lines first.
        out.write(f"FRB Label: {label}\t//should match params.label\n")
        out.write(f"Time Resolution: {dt}us\t//should match params.timescale\n")
        out.write(f"Saving: {saving}\t//should match params.saving\n")
        out.write(f"kc: {res['kc']}\t//{kc_note}\n")
        out.write(f"Initial DM: {DM}\n")
        out.write(f"Zapped channels: {zap_text}\n")

        # The structure-maximising delta DM and its uncertainty, with the fit's diagnostics.
        if res["ok"]:
            out.write(f"Structure Maximising Delta DM: {res['d_fit']}\n")
            out.write(f"Uncertainty in Structure Maximising Delta DM: +/-{res['sig_DM']}"
                      "\t//1 sigma\n")
            out.write(f"\tVertex of a parabola fitted over delta DM {res['d_lo']} to {res['d_hi']} "
                      f"({res['n_win']} trial DMs).\n")
            out.write("\tDelta DM of the largest structure parameter on the grid: "
                      f"{res['d_max']}\n")
            out.write(f"\tRelative curvature error std[c2]/|c2|: {res['c2_err']:.3f}"
                      "\t//sigma_DM is unreliable as it approaches 1\n")
        else:
            out.write(f"Structure Maximising Delta DM: {res['d_max']}"
                      "\t//largest structure parameter on the grid; no fit\n")
            out.write("Uncertainty in Structure Maximising Delta DM: unknown\n")

        # The structure parameter and the validity check of the first-order expansion.
        out.write(f"Structure Parameter at Maximum: {res['Sp_max']} +/- {res['sig_S']}"
                  "\t//1 sigma\n")
        out.write("\tFiltered noise / structure parameter: sqrt(<||e||^2>)/S = "
                  f"{res['e_ratio']:.3f}\t//first order holds for <~ 0.3\n")

        for message in warnings:
            out.write(f"\tWarning: {message}.\n")

        if grid is not None:
            d_low, d_high, step = grid
            out.write("\tSuggested trial DMs for generate_profiles.py: "
                      f"-L {d_low:.6g} -H {d_high:.6g} --dDM {step:.3g}\n")

        out.write("*/\n//end maximise_structure summary//\n\n\n")


def write_sn_summary(path, sn, dt):
    """Write the S/N-maximisation summary file, {label}_SN_summaryfile.txt, for a result of
    snr.maximise_sn, in SHRINE's maximise_sn layout.

    Args:
        path: output file name.
        sn: the dict returned by snr.maximise_sn.
        dt: time resolution, us (-t).
    """
    d = sn["d"]
    j = sn["j_max"]
    lo = sn["lo"]
    hi = sn["hi"]

    # The boxcar's first sample and the sample after its last.
    boxcar_start = sn["start"]
    boxcar_end = sn["start"] + sn["length"]

    with open(path, "w") as out:
        out.write("//begin maximise_sn summary//\n/*\n")
        out.write(f"Max SNR at index {j}\n")
        out.write(f"Uncertainty bounds at indexes {lo} - {hi}\n")
        out.write(f"Corresponds to a delta DM of {d[j]} +{d[hi] - d[j]} -{d[j] - d[lo]}\n")
        if not (sn["lo_found"] and sn["hi_found"]):
            out.write("\tThe S/N does not fall by 1 within the trial DMs on that side: "
                      "the bound is a lower limit.\n")
        out.write(f"Calculated with window from {boxcar_start} to {boxcar_end}\n")
        out.write(f"Corresponds to a time window from {boxcar_start * dt / 1000}ms "
                  f"to {boxcar_end * dt / 1000}ms\n")
        out.write(f"Max SNR was {sn['SN_max']}\n")
        out.write(f"\t//noise sigma = std of I - S I over all trial DMs; k_c = {sn['kc']}\n")
        out.write("*/\n//end maximise_sn summary//\n\n\n")
