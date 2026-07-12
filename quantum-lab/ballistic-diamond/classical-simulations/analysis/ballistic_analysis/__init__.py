"""Analysis utilities for the ballistic-diamond implantation sweep.

Reads the per-ensemble LAMMPS outputs produced by
classical-simulations/tersoff-sweep[-pb]/ and aggregates them into a single
HDF5 summary keyed by (species, energy, angle, temperature, ensemble).
"""
from ballistic_analysis.reader import (
    parse_dirname,
    read_final_state,
    read_ion_from_data,
    read_ion_trajectory,
)
from ballistic_analysis.aggregate import (
    walk_sweep_tree,
    build_summary,
    save_summary,
    load_summary,
)
from ballistic_analysis import viz

__all__ = [
    "parse_dirname",
    "read_final_state",
    "read_ion_from_data",
    "read_ion_trajectory",
    "walk_sweep_tree",
    "build_summary",
    "save_summary",
    "load_summary",
    "viz",
]
