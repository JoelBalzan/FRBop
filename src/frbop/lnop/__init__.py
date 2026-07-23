"""Gravitational lensing search from dedispersed voltage timestreams.

Pipeline (Kader, Leung et al. 2022, arXiv:2204.06014):
  1. Build a matched filter from the burst's intensity profile.
  2. Auto-correlate the voltage stream in the time-lag domain.
  3. Calibrate noise using off-pulse (burst-free) data.
  4. Bin the time-lag spectrum logarithmically; compute chi^2 per bin.
  5. Apply delay/significance/polarization vetoes to flag candidates.

The ``min_lag`` parameter controls the first logarithmic bin edge (the
effective reflection cutoff).  The ``frame`` parameter is used only for
the PFB-based delay veto and defaults to ``None`` (veto disabled).
"""
