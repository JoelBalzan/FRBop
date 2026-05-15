"""
Physical diagnostics: delta-DM and electron-density contrast between pulse components.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Speed of light in pc/ms (c = 9.715611890180196e-12 pc/ms)
_C_PC_PER_MS: float = 9.715611890180196e-12


def calculate_dn_e_between_components(
    all_results: List[Dict],
    component_separation_pc: Optional[float] = None,
    component_times_ms: Optional[np.ndarray] = None,
    comparison: str = "adjacent",
    reference_component: int = 0,
) -> Dict:
    """
    Calculate delta-DM and dn_e between burst components.

    Parameters
    ----------
    all_results:
        Per-component result dictionaries from ``compare_methods``.
    component_separation_pc:
        Fixed physical separation (pc) used for all pairs when
        *component_times_ms* is not provided.
    component_times_ms:
        Component peak arrival times (ms).  When given, the physical
        separation for each pair is derived as ``L ~ c * |Δt|``.
    comparison:
        ``'adjacent'``  — component i → i+1.
        ``'reference'`` — *reference_component* → every other component.
    reference_component:
        Index into *all_results* used as the reference when
        ``comparison='reference'``.

    Returns
    -------
    diagnostics dict with keys:
        ``comparison``, ``component_separation_pc``, ``component_times_ms``,
        ``pair_indices``, ``pair_labels``, ``pair_separations_pc``,
        ``methods`` (nested per-method arrays).
    """
    n_components = len(all_results)
    if n_components < 2:
        raise ValueError("Need at least two components to calculate dn_e")

    if component_times_ms is not None:
        component_times_ms = np.asarray(component_times_ms, dtype=float)
        if component_times_ms.ndim != 1 or component_times_ms.shape[0] != n_components:
            raise ValueError("component_times_ms must be 1-D with one value per component")
        if not np.all(np.isfinite(component_times_ms)):
            raise ValueError("component_times_ms must be finite")
    elif component_separation_pc is None:
        raise ValueError("Provide either component_times_ms or component_separation_pc")
    elif component_separation_pc <= 0:
        raise ValueError("component_separation_pc must be positive")

    if comparison not in ("adjacent", "reference"):
        raise ValueError("comparison must be 'adjacent' or 'reference'")

    first_methods = list(all_results[0].keys())
    common_methods = [m for m in first_methods if all(m in comp for comp in all_results)]
    if not common_methods:
        raise ValueError("No common methods across components")

    if comparison == "adjacent":
        pair_indices: List[Tuple[int, int]] = [(i, i + 1) for i in range(n_components - 1)]
    else:
        if reference_component < 0 or reference_component >= n_components:
            raise ValueError(f"reference_component must be in [0, {n_components - 1}]")
        pair_indices = [
            (reference_component, j)
            for j in range(n_components)
            if j != reference_component
        ]

    pair_labels = [f"comp{a + 1}->comp{b + 1}" for a, b in pair_indices]

    pair_separations_pc = np.zeros(len(pair_indices), dtype=float)
    for i, (idx_a, idx_b) in enumerate(pair_indices):
        if component_times_ms is not None:
            delta_t_ms = abs(float(component_times_ms[idx_b]) - float(component_times_ms[idx_a]))
            sep_pc = _C_PC_PER_MS * delta_t_ms
            if sep_pc <= 0:
                raise ValueError("Component times imply zero separation for at least one pair")
            pair_separations_pc[i] = sep_pc
        else:
            pair_separations_pc[i] = float(component_separation_pc)

    method_diagnostics: Dict[str, Dict[str, np.ndarray]] = {}
    for method_name in common_methods:
        delta_dm = np.zeros(len(pair_indices), dtype=float)
        delta_dm_low = np.zeros(len(pair_indices), dtype=float)
        delta_dm_high = np.zeros(len(pair_indices), dtype=float)
        dn_e = np.zeros(len(pair_indices), dtype=float)
        dn_e_low = np.zeros(len(pair_indices), dtype=float)
        dn_e_high = np.zeros(len(pair_indices), dtype=float)

        for i, (idx_a, idx_b) in enumerate(pair_indices):
            sep_pc = float(pair_separations_pc[i])
            res_a = all_results[idx_a][method_name]
            res_b = all_results[idx_b][method_name]

            dm_a = float(res_a["dm"])
            dm_b = float(res_b["dm"])
            minus_a = max(0.0, float(res_a.get("uncertainty_minus") or 0.0))
            plus_a = max(0.0, float(res_a.get("uncertainty_plus") or 0.0))
            minus_b = max(0.0, float(res_b.get("uncertainty_minus") or 0.0))
            plus_b = max(0.0, float(res_b.get("uncertainty_plus") or 0.0))

            # Conservative asymmetric interval propagation: delta = DM_b - DM_a
            delta = dm_b - dm_a
            delta_low = (dm_b - minus_b) - (dm_a + plus_a)
            delta_high = (dm_b + plus_b) - (dm_a - minus_a)

            delta_dm[i] = delta
            delta_dm_low[i] = delta_low
            delta_dm_high[i] = delta_high
            dn_e[i] = delta / sep_pc
            dn_e_low[i] = delta_low / sep_pc
            dn_e_high[i] = delta_high / sep_pc

        method_diagnostics[method_name] = {
            "delta_dm": delta_dm,
            "delta_dm_low": delta_dm_low,
            "delta_dm_high": delta_dm_high,
            "dn_e": dn_e,
            "dn_e_low": dn_e_low,
            "dn_e_high": dn_e_high,
        }

    return {
        "comparison": comparison,
        "component_separation_pc": (
            None if component_separation_pc is None else float(component_separation_pc)
        ),
        "component_times_ms": (
            None if component_times_ms is None else component_times_ms.copy()
        ),
        "pair_indices": pair_indices,
        "pair_labels": pair_labels,
        "pair_separations_pc": pair_separations_pc,
        "methods": method_diagnostics,
    }
