"""Analysis utilities for the ballistic-diamond implantation sweep.

Reads the per-ensemble LAMMPS outputs produced by
classical-simulations/tersoff-sweep[-pb]/ and aggregates them into a single
HDF5 summary keyed by (species, energy, angle, temperature, ensemble).
"""
from ballistic_analysis.reader import read_ion_from_data, parse_dirname
from ballistic_analysis.aggregate import (
    walk_sweep_tree,
    build_summary,
    save_summary,
    load_summary,
)
from ballistic_analysis import viz

__all__ = [
    "read_ion_from_data",
    "parse_dirname",
    "walk_sweep_tree",
    "build_summary",
    "save_summary",
    "load_summary",
    "viz",
]
