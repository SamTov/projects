"""Wigner-Seitz defect analysis of final.data snapshots.

Produces, per ensemble:
  - vacancy positions (empty lattice sites) -> vacancy DEPTH distribution
  - interstitial count (atoms outside any site's capture radius, plus
    excess occupants of multiply-occupied sites)
  - sp2 count (3-coordinated carbons: graphitised / amorphous pockets)
  - sputter loss (expected site count minus surviving carbons)

Method
------
The reference lattice is generated analytically: every production run uses
the SAME orientation (x=[110], y=[00-1], z=[-110], a=3.567) and slab region,
so one site list per box geometry serves all ensembles.  Atoms are assigned
to their nearest reference site (the Wigner-Seitz cell assignment); a site
with zero occupants is a vacancy.

Registration: before assignment, the site lattice is rigidly shifted by the
median displacement of a mid-slab probe slice (iterated twice).  This
absorbs any origin-convention mismatch between this generator and LAMMPS
create_atoms without needing a stored pre-strike snapshot.  Diamond's
thermal displacements (u_rms <= 0.07 A at 900 K) are tiny against the
0.77 A half-nearest-neighbour distance, so assignment on the un-minimised
hot snapshot is unambiguous.

Acceptance check on real data (run once after the test jobs): a pristine
slab far from the ion track must yield ~0 vacancies away from the surfaces.
If it doesn't, the registration failed -- don't trust the histograms.

Sanity anchor for totals: NRT with E_d ~ 40 eV predicts ~0.8*E_damage/(2E_d)
~ 300-450 Frenkel pairs for a 60 keV Sn cascade; MD after in-cascade
recombination typically retains 1/3 - 1/2 of NRT.  Expect O(100-250)
vacancies per 60 keV ion, scaling roughly linearly with damage energy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Diamond conventional-cell basis (fractional coordinates, 8 atoms).
_DIAMOND_BASIS = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])

# Rotation rows = unit vectors of the box axes in cubic-crystal coordinates,
# one triad per sweep orientation (must match simulate.lmp's lattice orient).
def _normalise(rows):
    r = np.array(rows, dtype=float)
    return r / np.linalg.norm(r, axis=1)[:, None]


_R_MAP = {
    100: _normalise([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    110: _normalise([[1, 1, 0], [0, 0, -1], [-1, 1, 0]]),
    111: _normalise([[1, -1, 0], [1, 1, -2], [1, 1, 1]]),
}

A_DIAMOND = 3.5678   # measured a0 of potentials/C.tersoff.zbl (matches simulate.lmp alat)
HALF_NN = 0.77          # half the C-C nearest-neighbour distance (1.545/2)
SP3_CUT = 1.85          # bond-count cutoff for coordination analysis


def _wrap(points: np.ndarray, box: dict) -> np.ndarray:
    """Shift to box-origin frame and wrap x,y onto the lateral torus."""
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    out = points - np.array([box["xlo"], box["ylo"], box["zlo"]])
    out[:, 0] %= lx
    out[:, 1] %= ly
    # fp edges cKDTree(boxsize=...) rejects: `v % L` returns exactly L for a
    # tiny negative v, and coords must be non-negative.
    out[out[:, 0] >= lx, 0] -= lx
    out[out[:, 1] >= ly, 1] -= ly
    out[:, 2] = np.maximum(out[:, 2], 0.0)
    return out


def _tree(points_wrapped: np.ndarray, box: dict):
    """cKDTree periodic in x,y only (z period padded out to open)."""
    from scipy.spatial import cKDTree

    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    return cKDTree(points_wrapped, boxsize=[lx, ly, 1.0e9])


def generate_reference_sites(
    box: dict,
    z_bottom: float,
    z_top: float,
    a: float = A_DIAMOND,
    orientation: int = 110,
) -> np.ndarray:
    """Perfect-lattice site positions (box frame) for the standard slab.

    box            : {xlo,xhi,ylo,yhi,zlo,zhi} from final.data
    z_bottom/z_top : material extent in box coordinates (use the measured
                     values from reader.read_final_state).
    orientation    : 100 / 110 / 111 -- selects the box-axes triad.
    """
    R = _R_MAP[int(orientation)]

    # Corners of the region of interest, in cubic-crystal coordinates.
    # (R is orthonormal: box->cubic is r @ R, cubic->box is r @ R.T)
    corners = np.array([
        [x, y, z]
        for x in (box["xlo"], box["xhi"])
        for y in (box["ylo"], box["yhi"])
        for z in (z_bottom - 2 * a, z_top + 2 * a)
    ]) @ R

    lo = np.floor(corners.min(axis=0) / a).astype(int) - 1
    hi = np.ceil(corners.max(axis=0) / a).astype(int) + 1

    ii, jj, kk = np.meshgrid(
        np.arange(lo[0], hi[0] + 1),
        np.arange(lo[1], hi[1] + 1),
        np.arange(lo[2], hi[2] + 1),
        indexing="ij",
    )
    cells = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(float)

    sites_cubic = (cells[:, None, :] + _DIAMOND_BASIS[None, :, :]).reshape(-1, 3) * a
    sites = sites_cubic @ R.T

    # z-range padded so the outermost real atom planes always have home
    # sites even when the measured surface/bottom is a hair off; the face
    # margins in analyse_damage keep these out of the statistics.
    eps = 1e-6
    keep = (
        (sites[:, 0] >= box["xlo"] - eps) & (sites[:, 0] < box["xhi"] - eps)
        & (sites[:, 1] >= box["ylo"] - eps) & (sites[:, 1] < box["yhi"] - eps)
        & (sites[:, 2] >= z_bottom - 2.0) & (sites[:, 2] <= z_top + 2.0)
    )
    sites = sites[keep]

    # Dedupe: with irrational rotation components, float jitter at the
    # wrapped lateral faces can keep BOTH periodic images of a boundary
    # plane (phantom duplicate sites -> phantom "vacancies" spanning all z).
    # Tolerance-based pair matching on the wrapped torus: distinct sites are
    # >= 1.54 A apart, so any pair within 0.5 A is a duplicate.  (A rounding
    # -grid key leaks when jitter straddles a cell boundary -- observed as
    # +0.4% phantom sites on production-size 111 boxes.)
    from scipy.spatial import cKDTree

    wrapped = _wrap(sites, box)
    tree = cKDTree(wrapped, boxsize=[box["xhi"] - box["xlo"],
                                     box["yhi"] - box["ylo"], 1.0e9])
    drop = np.zeros(len(sites), dtype=bool)
    for i, j in tree.query_pairs(0.5):
        drop[max(i, j)] = True
    return sites[~drop]


@dataclass
class DamageRecord:
    species: str
    orientation: int
    E_keV: float
    angle_deg: float
    T_K: float
    ensemble: int
    n_sites: int
    n_carbon: int
    n_vac: int
    n_int: int
    n_lost: int          # sputtered / transmitted carbons (sites - carbons)
    n_sp2: int           # 3-coordinated carbons (graphitisation proxy)
    ok: int


def analyse_damage(
    state: dict,
    positions: np.ndarray,
    types: np.ndarray,
    coordination: bool = True,
    sites: Optional[np.ndarray] = None,
    orientation: int = 110,
    a: float = A_DIAMOND,
) -> dict:
    """Wigner-Seitz analysis of one parsed final.data.

    state           : reader.read_final_state output (box + surface/bottom)
    positions/types : full atom arrays from reader.read_all_atoms
    sites           : optional pre-generated reference sites (reused across
                      ensembles of identical geometry for speed); shifted
                      copies are made internally, the input is not mutated.
    orientation     : 100 / 110 / 111 -- reference-lattice triad (ignored
                      when explicit `sites` are supplied).

    Returns counts plus the vacancy depth array (A below measured surface).
    """
    box = state["box"]
    carbon = positions[types == 1]
    carbon_w = _wrap(carbon, box)

    if sites is None:
        sites = generate_reference_sites(
            box, state["z_bottom"], state["z_surface"], a=a,
            orientation=orientation,
        )
    sites = np.array(sites, dtype=float, copy=True)

    # --- Registration: rigid shift + uniaxial z-strain, iterated ---
    # The reference is generated at a = 3.567 exactly, but the real slab is
    # laterally clamped by the periodic box and FREE along z, so it carries
    # a z-strain (Tersoff a0 != 3.567, plus thermal expansion) of order 1e-4.
    # A rigid shift alone leaves a displacement that grows linearly with
    # distance from the anchor and crosses the 0.77 A capture radius roughly
    # 1.5-2 kA out, mass-misassigning the slab ends.  Fit and remove
    # disp_z = c + eps*(z - zmid) alongside the lateral median shift.
    zb, zs = state["z_bottom"], state["z_surface"]
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    stride = max(1, len(carbon_w) // 100000)
    probe = carbon_w[::stride]                      # spread over full depth
    zmid = 0.5 * (zb + zs) - box["zlo"]
    for _ in range(4):
        sites_w = _wrap(sites, box)
        tree = _tree(sites_w, box)
        d, idx = tree.query(probe, workers=-1)
        ok = d < 1.2          # exclude cascade-displaced / not-yet-captured
        if ok.sum() < 1000:
            ok = d < np.percentile(d, 80)
        disp = probe[ok] - sites_w[idx[ok]]
        disp[:, 0] -= np.round(disp[:, 0] / lx) * lx
        disp[:, 1] -= np.round(disp[:, 1] / ly) * ly
        shift_xy = np.median(disp[:, :2], axis=0)
        z = probe[ok][:, 2] - zmid
        A = np.column_stack([np.ones_like(z), z])
        (c_z, eps), *_ = np.linalg.lstsq(A, disp[:, 2], rcond=None)
        sites[:, 0] += shift_xy[0]
        sites[:, 1] += shift_xy[1]
        sz = sites[:, 2] - box["zlo"] - zmid
        sites[:, 2] += c_z + eps * sz
        if abs(c_z) < 0.01 and np.hypot(*shift_xy) < 0.01 \
                and abs(eps) * max(abs(sz.max()), abs(sz.min())) < 0.02:
            break

    # --- Occupancy assignment (Wigner-Seitz) ---
    sites_w = _wrap(sites, box)
    tree = _tree(sites_w, box)
    dist, nearest = tree.query(carbon_w, workers=-1)

    n_sites = len(sites)
    # TRUE Wigner-Seitz occupancy: every atom claims its nearest site, with
    # NO distance cutoff.  Elastic strain fields (thermal expansion against
    # the pinned anchor, surface relaxation) displace whole regions
    # coherently by up to ~1 A; a capture-radius variant misreads those as
    # vacancy+interstitial carpets, while nearest-site assignment keeps the
    # atom<->site bijection intact.  A site is vacant only if NO atom maps
    # to it; interstitials are excess occupants of multiply-claimed sites.
    occupancy = np.bincount(nearest, minlength=n_sites)
    # Exclude ~3 A at both faces from the statistics: face bookkeeping and
    # surface reconstruction (Tersoff (111) Pandey-like chains especially)
    # make occupancy ambiguous there.  Sputter-crater vacancies deeper than
    # the margin still count.
    # Bottom margin 25 A: covers the pinned anchor + Langevin scaffold layers,
    # whose frozen/step displacement field is simulation plumbing, not damage.
    # Top margin 8 A: covers surface reconstruction (Tersoff (111) especially).
    site_interior = (sites[:, 2] > zb + 25.0) & (sites[:, 2] < zs - 8.0)
    vac_mask = (occupancy == 0) & site_interior
    n_vac = int(vac_mask.sum())
    n_int = int((occupancy[(occupancy > 1) & site_interior] - 1).sum())
    n_lost = n_sites - len(carbon)

    vac_depths = state["z_surface"] - sites[vac_mask][:, 2]

    n_sp2 = -1
    if coordination:
        ctree = _tree(carbon_w, box)
        ncoord = np.asarray(ctree.query_ball_point(
            carbon_w, r=SP3_CUT, return_length=True, workers=-1
        )) - 1  # subtract self
        n_sp2 = int((ncoord == 3).sum())

    return {
        "n_sites": n_sites,
        "n_carbon": int(len(carbon)),
        "n_vac": n_vac,
        "n_int": n_int,
        "n_lost": int(n_lost),
        "n_sp2": n_sp2,
        "vac_depths": np.asarray(vac_depths, dtype=np.float64),
    }


def save_damage(
    records: Iterable[DamageRecord],
    vac_depth_lists: list[np.ndarray],
    out_path: Path,
) -> None:
    """Write metrics table + ragged vacancy-depth arrays (concat + offsets)."""
    import h5py

    records = list(records)
    if len(records) != len(vac_depth_lists):
        raise ValueError("records / vac_depth_lists length mismatch")
    if not records:
        raise ValueError("No records to save.")

    cols = list(asdict(records[0]).keys())
    dtype = []
    for c in cols:
        if c == "species":
            dtype.append((c, "S2"))
        elif c in ("orientation", "ensemble", "n_sites", "n_carbon", "n_vac",
                   "n_int", "n_lost", "n_sp2", "ok"):
            dtype.append((c, "i8"))
        else:
            dtype.append((c, "f8"))
    arr = np.zeros(len(records), dtype=dtype)
    for i, r in enumerate(records):
        for c in cols:
            v = getattr(r, c)
            arr[c][i] = v.encode() if isinstance(v, str) else v

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for i, v in enumerate(vac_depth_lists):
        offsets[i + 1] = offsets[i] + len(v)
    concat = (np.concatenate(vac_depth_lists)
              if offsets[-1] else np.zeros(0, dtype=np.float64))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("metrics", data=arr, compression="gzip")
        f.create_dataset("vac_depths", data=concat, compression="gzip")
        f.create_dataset("vac_offsets", data=offsets)


def load_damage(path: Path):
    """Return (metrics structured array, list of per-ensemble depth arrays)."""
    import h5py

    with h5py.File(path, "r") as f:
        arr = f["metrics"][...]
        concat = f["vac_depths"][...]
        offsets = f["vac_offsets"][...]

    out_dtype = [
        (n, "U8" if arr.dtype[n].kind == "S" else arr.dtype[n])
        for n in arr.dtype.names
    ]
    out = np.zeros(arr.shape, dtype=out_dtype)
    for n in arr.dtype.names:
        out[n] = ([s.decode() for s in arr[n]]
                  if arr.dtype[n].kind == "S" else arr[n])
    depths = [concat[offsets[i]:offsets[i + 1]] for i in range(len(arr))]
    return out, depths
