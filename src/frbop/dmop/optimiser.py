"""
DM optimiser class composed from functional mixins.
"""

from typing import List, Optional, Tuple

import numpy as np

from .comparison import ComparisonMixin
from .components import ComponentsMixin
from .dedispersion import DedispersionMixin
from .metrics import MetricsMixin
from .optimisation import OptimisationMixin
from .peaks import PeaksMixin
from .plotting import PlottingMixin
from .polarization import PolarizationMixin
from .shrine import ShrineMixin
from .uncertainty import UncertaintyMixin


class DMOptimiser(
    ShrineMixin,
    UncertaintyMixin,
    DedispersionMixin,
    PolarizationMixin,
    MetricsMixin,
    OptimisationMixin,
    PeaksMixin,
    ComparisonMixin,
    PlottingMixin,
    ComponentsMixin,
):
    """
    Class for optimising DM correction using various methods.
    """

    def __init__(self, stokes_i: np.ndarray, freq_mhz: np.ndarray, time_ms: np.ndarray,
                 stokes_q: Optional[np.ndarray] = None, stokes_u: Optional[np.ndarray] = None,
                 reference_freq: Optional[float] = None,
                 input_dm: float = 0.0,
                 dedisp_mode: str = 'expand',
                 pa_fit_degree: int = 1,
                 pa_weight_strength: float = 1.0,
                 pa_fit_post_peak_only: bool = False,
                 nonshrine_kc_smooth: bool = False,
                 nonshrine_shrine_like_errors: bool = False,
                 nonshrine_kc_minimise_uncertainty: bool = False,
                 nonshrine_kc: Optional[int] = None,
                 li_i_sigma_cut: float = 2.0,
                 debias_linear: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize the DM optimiser.
        
        Parameters:
        -----------
        stokes_i : np.ndarray
            2D array of Stokes I data (freq x time)
        freq_mhz : np.ndarray
            Frequency array in MHz
        time_ms : np.ndarray
            Time array in ms
        """
        self.stokes_i = stokes_i
        self.stokes_q = stokes_q
        self.stokes_u = stokes_u
        self.freq_mhz = freq_mhz
        self.time_ms = time_ms
        self.n_freq, self.n_time = stokes_i.shape
        self.reference_freq = reference_freq if reference_freq is not None else np.max(freq_mhz)
        self.input_dm = float(input_dm)
        self.dedisp_mode = dedisp_mode
        self.pa_fit_degree = int(pa_fit_degree)
        self.pa_weight_strength = float(pa_weight_strength)
        if self.pa_weight_strength <= 0:
            raise ValueError("pa_weight_strength must be positive")
        self.pa_fit_post_peak_only = bool(pa_fit_post_peak_only)
        self.nonshrine_kc_smooth = bool(nonshrine_kc_smooth)
        self.nonshrine_shrine_like_errors = bool(nonshrine_shrine_like_errors)
        self.use_nonshrine_shrine_like_uncertainty = bool(
            self.nonshrine_kc_smooth or self.nonshrine_shrine_like_errors
        )
        self.nonshrine_kc_minimise_uncertainty = bool(nonshrine_kc_minimise_uncertainty)
        self.nonshrine_kc = None if nonshrine_kc is None else int(nonshrine_kc)
        if self.nonshrine_kc is not None and self.nonshrine_kc <= 0:
            raise ValueError("nonshrine_kc must be positive")
        self._nonshrine_resolved_kc: Optional[int] = None
        self._nonshrine_kc_printed = False
        self.li_i_sigma_cut = float(li_i_sigma_cut)
        if self.li_i_sigma_cut <= 0:
            raise ValueError("li_i_sigma_cut must be positive")
        self.debias_linear = bool(debias_linear)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.full_i_time_series = np.mean(self.stokes_i, axis=0)
        self.full_i_noise_median, self.full_i_noise_std = self._noise_stats_from_series(self.full_i_time_series)
        if self.stokes_q is not None and self.stokes_u is not None:
            full_L = np.sqrt(self.stokes_q**2 + self.stokes_u**2)
            self.full_L_time = np.mean(full_L, axis=0)
            self.full_L_noise_median, self.full_L_noise_std = self._noise_stats_from_series(self.full_L_time)
            self.full_q_time_series = np.mean(self.stokes_q, axis=0)
            self.full_u_time_series = np.mean(self.stokes_u, axis=0)
            _, self.full_q_time_noise_std = self._noise_stats_from_series(self.full_q_time_series)
            _, self.full_u_time_noise_std = self._noise_stats_from_series(self.full_u_time_series)
            n_edge_full = max(1, int(0.05 * self.stokes_q.shape[1]))
            self.full_q_noise_rms = np.std(self.stokes_q[:, :n_edge_full], axis=1, keepdims=True)
            self.full_u_noise_rms = np.std(self.stokes_u[:, :n_edge_full], axis=1, keepdims=True)
        else:
            self.full_L_time = None
            self.full_L_noise_median = None
            self.full_L_noise_std = None
            self.full_q_time_series = None
            self.full_u_time_series = None
            self.full_q_time_noise_std = None
            self.full_u_time_noise_std = None
            self.full_q_noise_rms = None
            self.full_u_noise_rms = None
        
        # DM constant: k = 4.148808e6 ms MHz^2 pc^-1 cm^3 (delay between frequencies)
        # From pulsar handbook: dt = 4.15 × 10^6 ms × (f1^-2 - f2^-2) × DM
        self.DM_CONSTANT = 4.148808e6

