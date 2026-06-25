"""Public API for rmt_dynamics."""
from __future__ import annotations

from .correlation import build_C, velocity_autocorr_integrals
from .hbond import (
    hbond_adjacency_luzar_chandler,
    hbond_degree,
    time_averaged_hbond_degree,
)
from .io import load_C, load_velocities, save_C
from .peaks import find_peak_in_window, fwhm_in_window
from .rmt_null import (
    T_eff_from_trajectory,
    estimate_tau_int,
    ks_distance,
    mp_cdf,
    mp_density,
    mp_edges,
)
from .spectrum import (
    eigendecomposition,
    eigenvalues,
    participation_ratio,
    trace_normalised,
    vn_entropy,
)
from .transport import (
    contact_graph,
    coordination_number,
    green_kubo_integral,
    radial_distribution,
    time_averaged_rdf,
)

__version__ = "0.1.0"

__all__ = [
    "build_C",
    "velocity_autocorr_integrals",
    "eigenvalues",
    "eigendecomposition",
    "vn_entropy",
    "participation_ratio",
    "trace_normalised",
    "mp_density",
    "mp_cdf",
    "mp_edges",
    "estimate_tau_int",
    "T_eff_from_trajectory",
    "ks_distance",
    "save_C",
    "load_C",
    "load_velocities",
    "find_peak_in_window",
    "fwhm_in_window",
    "hbond_adjacency_luzar_chandler",
    "hbond_degree",
    "time_averaged_hbond_degree",
    "green_kubo_integral",
    "radial_distribution",
    "time_averaged_rdf",
    "coordination_number",
    "contact_graph",
]
