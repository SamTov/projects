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


# ---------------------------------------------------------------------------
# Reference-slab Wigner-Seitz (the method to use on production slabs)
# ---------------------------------------------------------------------------
#
# Why the analytic lattice above is not enough.  A production slab is ~3300 A
# thick, laterally clamped by the periodic box, pinned at the bottom and free
# at the top.  It is created at the 0 K lattice constant and then heated, so
# it has to expand along z -- and it does so by ringing like a bar (period
# ~4L/c ~ 70 ps, barely damped because only the boundary skins are
# thermostatted).  Two snapshots taken at different times therefore differ by
# a smooth z-displacement field that grows from zero at the anchor to >1 A at
# the surface.  The interplanar spacing along <110> is 1.26 A, so anything
# beyond 0.63 A re-assigns whole planes to the wrong sites: on 35 keV
# production runs a global (rigid-shift + linear-strain) registration left
# 10% of PRISTINE sites flagged as vacancies.
#
# The fix has two parts:
#   1. sites come from a pristine slab produced by make-reference.lmp (same
#      lattice, box, potential and warmup, no ion), so surfaces, the anchor
#      scaffold and the lattice origin match production exactly;
#   2. the residual displacement field is removed by UNWRAPPING it slice by
#      slice from the pinned anchor upward: each ~12 A slice is registered
#      starting from the offset of the slice below, so the step between
#      slices is << half a plane spacing and nearest-site matching is never
#      ambiguous.  The per-slice median is robust to cascade-displaced atoms.
#
# Acceptance check (reported per run as `control_vac`): the crystal well
# below the ion's deepest point is pristine and must contain ~0 vacancies.

def load_reference_sites(path: Path, cache: bool = True) -> np.ndarray:
    """Carbon positions of a pristine reference slab (make-reference.lmp).

    Parsing 4.2M atoms from ASCII takes a while, so the array is cached as
    `<name>.sites.npy` beside the data file.
    """
    from .reader import read_all_atoms

    path = Path(path)
    npy = path.with_suffix(".sites.npy")
    if cache and npy.exists() and npy.stat().st_mtime >= path.stat().st_mtime:
        return np.load(npy)
    atoms = read_all_atoms(path)
    if atoms is None:
        raise FileNotFoundError(f"cannot read reference slab {path}")
    types, pos = atoms
    sites = np.ascontiguousarray(pos[types == 1])
    # Drop strays that left the slab (not lattice sites): keep the dense
    # z-range only.
    hist, edges = np.histogram(sites[:, 2], bins=np.arange(
        sites[:, 2].min() - 1.0, sites[:, 2].max() + 2.0, 1.0))
    dense = np.nonzero(hist >= 0.5 * np.median(hist[hist > 0]))[0]
    lo, hi = edges[dense[0]] - 1.5, edges[dense[-1] + 1] + 1.5
    sites = sites[(sites[:, 2] >= lo) & (sites[:, 2] <= hi)]
    if cache:
        try:
            np.save(npy, sites)
        except OSError:
            pass
    return sites


def _fold_xy(points: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Wrap x,y into [0, L) in place (cKDTree boxsize needs strict bounds)."""
    points[:, 0] %= lx
    points[:, 1] %= ly
    points[points[:, 0] >= lx, 0] -= lx
    points[points[:, 1] >= ly, 1] -= ly
    points[:, 2] = np.maximum(points[:, 2], 0.0)
    return points


def register_reference(
    sites_w: np.ndarray,
    atoms_w: np.ndarray,
    box: dict,
    slice_width: float = 12.0,
    capture: float = 0.40,
    max_iter: int = 6,
) -> tuple[np.ndarray, dict]:
    """Remove the smooth slab displacement field between reference and run.

    sites_w / atoms_w : wrapped coordinates (see _wrap) of the reference
                        sites and of the run's carbon atoms.

    The unwrapping starts at the TOP surface and walks down.  The top is the
    one place where plane-for-plane correspondence is known without any
    assumption (no planes are created or destroyed, and the surface is
    clean), whereas the bottom carries the anchor scaffold, which is
    disordered in the 2026 production runs (see the README).  The starting
    offset comes from rank-matching the uppermost atoms, then a short scan
    over u_z picks the registry with the best match before the per-slice
    median refinement takes over.

    Returns (registered sites, profile); profile holds the unwrapped
    displacement u(z) and quality numbers:
        z, u         : slice centres (ascending) and displacement vectors (A)
        good_frac    : per-slice fraction of sites matched within `capture`
        max_step     : largest |u| change between adjacent TRUSTED slices --
                       must stay well below half a plane spacing (0.63 A)
    """
    from scipy.spatial import cKDTree

    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]

    s_sorted = sites_w[np.argsort(sites_w[:, 2], kind="stable")]
    a_sorted = atoms_w[np.argsort(atoms_w[:, 2], kind="stable")]
    sz, az = s_sorted[:, 2], a_sorted[:, 2]

    # Rank-matched offset of the top few planes (robust to a handful of
    # sputtered atoms / adatoms shifting the ranks).
    ranks = np.array([1500, 3000, 4500, 6000, 7500, 9000])
    ranks = ranks[ranks < min(len(sz), len(az)) // 4]
    u = np.array([0.0, 0.0, float(np.median(az[-ranks] - sz[-ranks]))])

    def match(rs, tree, cand, shift, cap):
        q = _fold_xy(rs + shift, lx, ly)
        d, idx = tree.query(q)
        good = d < cap
        return q, idx, good

    centres, disp, good_frac, trusted = [], [], [], []
    hi = sz[-1] + 0.5
    first = True
    while hi > sz[0]:
        lo = hi - slice_width
        i0, i1 = np.searchsorted(sz, [lo, hi])
        if i1 - i0 >= 500:
            rs = s_sorted[i0:i1]
            j0, j1 = np.searchsorted(az, [lo + u[2] - 3.0, hi + u[2] + 3.0])
            frac, ok = 0.0, False
            if j1 - j0 >= 200:
                cand = a_sorted[j0:j1]
                tree = cKDTree(cand, boxsize=[lx, ly, 1.0e9])
                trial = u.copy()
                if first:
                    # pick the registry: scan u_z about the rank estimate
                    best = (-1.0, 0.0)
                    for dz in np.arange(-0.6, 0.6001, 0.05):
                        _, _, good = match(rs, tree, cand, trial + [0, 0, dz], 0.30)
                        if good.mean() > best[0]:
                            best = (float(good.mean()), float(dz))
                    trial[2] += best[1]
                for _ in range(max_iter):
                    q, idx, good = match(rs, tree, cand, trial, capture)
                    frac = float(good.mean())
                    if good.sum() < 200:
                        break
                    delta = cand[idx[good]] - q[good]
                    delta[:, 0] -= np.round(delta[:, 0] / lx) * lx
                    delta[:, 1] -= np.round(delta[:, 1] / ly) * ly
                    du = np.median(delta, axis=0)
                    trial = trial + du
                    if np.abs(du).max() < 2.0e-3:
                        break
                # Only a slice that is mostly crystalline may move the
                # running offset; a wrecked slice (cascade core, anchor
                # scaffold) inherits the offset of the slice above.
                ok = frac >= 0.5
                if ok:
                    u = trial
            centres.append(0.5 * (lo + hi))
            disp.append(u.copy())
            good_frac.append(frac)
            trusted.append(ok)
            first = False
        hi = lo

    centres = np.asarray(centres)[::-1]
    disp = np.asarray(disp)[::-1]
    good_frac = np.asarray(good_frac)[::-1]
    trusted = np.asarray(trusted)[::-1]
    if len(centres) < 2 or not trusted.any():
        raise RuntimeError("reference registration found no usable slices")

    shift = np.column_stack([
        np.interp(sites_w[:, 2], centres, disp[:, k]) for k in range(3)
    ])
    registered = _fold_xy(sites_w + shift, lx, ly)
    tu = disp[trusted]
    steps = np.abs(np.diff(tu, axis=0)).max(axis=1) if len(tu) > 1 else np.zeros(1)
    profile = {
        "z": centres,
        "u": disp,
        "good_frac": good_frac,
        "trusted": trusted,
        "max_step": float(steps.max()),
    }
    return registered, profile


def analyse_damage_reference(
    state: dict,
    positions: np.ndarray,
    types: np.ndarray,
    ref_sites: np.ndarray,
    bottom_margin: float = 60.0,
    top_margin: float = 5.0,
    control_gap: float = 300.0,
    workers: int = -1,
) -> dict:
    """Wigner-Seitz defect analysis against a pristine reference slab.

    state / positions / types : reader.read_final_state / read_all_atoms
    ref_sites                 : load_reference_sites(...) for the SAME
                                orientation and box geometry
    bottom_margin             : MINIMUM height above the slab bottom excluded
                                from the statistics.  The anchor scaffold of
                                the 2026 runs is disordered (thermostatted
                                pinned atoms, see README), so the excluded
                                height is max(bottom_margin, scaffold_top+20)
                                with scaffold_top measured per run.
    top_margin                : A below the surface excluded (surface-layer
                                bookkeeping; deeper sputter damage counts)
    control_gap               : the pristine control region is everything
                                more than this far BELOW the ion (A)

    Returns counts, defect coordinates (box frame of the analysed run) and
    registration quality.  A vacancy is a reference site no atom maps to;
    an interstitial is every occupant beyond the first of a site (the
    occupant nearest the site is taken as the lattice atom).
    """
    box = state["box"]
    origin = np.array([box["xlo"], box["ylo"], box["zlo"]])
    carbon = positions[types == 1]
    carbon_w = _wrap(carbon, box)
    sites_w = _wrap(np.array(ref_sites, dtype=float, copy=True), box)

    sites_r, profile = register_reference(sites_w, carbon_w, box)

    tree = _tree(sites_r, box)
    dist, nearest = tree.query(carbon_w, workers=workers)
    n_sites = len(sites_r)
    occupancy = np.bincount(nearest, minlength=n_sites)

    zb = state["z_bottom"] - box["zlo"]
    zs = state["z_surface"] - box["zlo"]

    # Height of the disordered anchor scaffold: contiguous run of 5 A slices,
    # counted from the slab bottom, holding more than a handful of vacancies.
    edges = zb + np.arange(0.0, 405.0, 5.0)
    hist, _ = np.histogram(sites_r[occupancy == 0][:, 2], bins=edges)
    clean = np.nonzero(hist <= 20)[0]
    scaffold_top = float(edges[clean[0]] - zb) if len(clean) else 400.0
    z_lo = zb + max(bottom_margin, scaffold_top + 20.0)
    site_interior = (sites_r[:, 2] > z_lo) & (sites_r[:, 2] < zs - top_margin)

    vac_mask = (occupancy == 0) & site_interior
    vac_pos = sites_r[vac_mask] + origin

    # Interstitials: excess occupants of multiply-claimed interior sites.
    multi = (occupancy > 1) & site_interior
    on_multi = np.nonzero(multi[nearest])[0]
    order = np.lexsort((dist[on_multi], nearest[on_multi]))
    on_multi = on_multi[order]
    host = nearest[on_multi]
    is_first = np.ones(len(on_multi), dtype=bool)
    is_first[1:] = host[1:] != host[:-1]
    excess = on_multi[~is_first]
    int_pos = carbon_w[excess] + origin
    int_site = sites_r[nearest[excess]] + origin

    ion = None
    ion_site = None
    control_vac, control_sites = -1, 0
    if state.get("ion") is not None:
        ion = np.array([state["ion"]["x"], state["ion"]["y"], state["ion"]["z"]])
        ion_w = _wrap(ion[None, :].copy(), box)
        _, j = tree.query(ion_w)
        ion_site = sites_r[int(j[0])] + origin
        # undo the lateral wrap so ion_site sits next to the (unwrapped) ion
        lx, ly = box["xhi"] - box["xlo"], box["yhi"] - box["ylo"]
        ion_site[0] += np.round((ion[0] - ion_site[0]) / lx) * lx
        ion_site[1] += np.round((ion[1] - ion_site[1]) / ly) * ly
        z_ctrl = (ion[2] - box["zlo"]) - control_gap
        ctrl = (sites_r[:, 2] > z_lo) & (sites_r[:, 2] < z_ctrl)
        control_sites = int(ctrl.sum())
        control_vac = int(((occupancy == 0) & ctrl).sum())

    trusted_above = profile["good_frac"][profile["z"] > z_lo]
    return {
        "n_sites": n_sites,
        "n_carbon": int(len(carbon)),
        "n_vac": int(vac_mask.sum()),
        "n_int": int(len(excess)),
        "n_offsite": int(((dist > 0.5) & site_interior[nearest]).sum()),
        "vac_pos": vac_pos,
        "vac_depths": state["z_surface"] - vac_pos[:, 2],
        "int_pos": int_pos,
        "int_site": int_site,
        "ion": ion,
        "ion_site": ion_site,
        "scaffold_top": scaffold_top,
        "z_interior_lo": float(z_lo + box["zlo"]),
        "control_vac": control_vac,
        "control_sites": control_sites,
        "resid_median": float(np.median(dist)),
        "reg_max_step": profile["max_step"],
        "reg_u_top": profile["u"][-1],
        "reg_u_range": float(np.ptp(profile["u"][:, 2])),
        "reg_min_good": float(trusted_above.min()) if len(trusted_above) else 0.0,
        "profile": profile,
    }


def periodic_delta(points: np.ndarray, centre: np.ndarray, box: dict) -> np.ndarray:
    """points - centre with the lateral (x,y) minimum-image convention."""
    d = np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)
    if d.size == 0:
        return d.reshape(0, 3)
    lx, ly = box["xhi"] - box["xlo"], box["yhi"] - box["ylo"]
    d[:, 0] -= np.round(d[:, 0] / lx) * lx
    d[:, 1] -= np.round(d[:, 1] / ly) * ly
    return d


def lattice_indices(
    delta: np.ndarray,
    orientation: int = 110,
    a: float = A_DIAMOND,
    sites_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap displacement vectors (box frame, measured FROM a lattice site)
    onto the ideal diamond lattice.

    Returns (n, resid): integer coordinates in units of a/4 along the cubic
    crystal axes, and the distance (A) between each point and its snapped
    position.  Every diamond site sits on the a/4 grid, so `n` is what a
    lattice KMC code needs.

    sites_only=True snaps to the nearest actual LATTICE SITE rather than the
    nearest grid point (only 1 grid point in 8 is a site), which tolerates
    ~0.75 A of accumulated strain / thermal offset instead of ~0.45 A.  With
    the origin on a site, sites are the grid points with all-even n and
    sum(n) % 4 == 0, plus the other sublattice: all-odd n with sum(n) % 4
    equal to 1 or 3 depending on which sublattice the origin sits on -- that
    is decided from the data (whichever fits better).
    """
    import itertools

    R = _R_MAP[int(orientation)]
    cubic = np.asarray(delta, dtype=float).reshape(-1, 3) @ R
    grid = 4.0 * cubic / a
    n0 = np.rint(grid).astype(int)
    if not sites_only or len(n0) == 0:
        return n0, np.linalg.norm(cubic - n0 * (a / 4.0), axis=1)

    offs = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    cand = n0[:, None, :] + offs[None, :, :]
    dist = np.linalg.norm(cand - grid[:, None, :], axis=2) * (a / 4.0)
    even = (cand % 2 == 0).all(axis=2)
    odd = (cand % 2 != 0).all(axis=2)
    s4 = cand.sum(axis=2) % 4

    best = None
    for other in (1, 3):
        valid = (even & (s4 == 0)) | (odd & (s4 == other))
        d = np.where(valid, dist, np.inf)
        pick = d.argmin(axis=1)
        resid = d[np.arange(len(d)), pick]
        if best is None or resid.sum() < best[0]:
            best = (resid.sum(), cand[np.arange(len(d)), pick], resid)
    return best[1], best[2]


def cluster_sizes(points: np.ndarray, cutoff: float) -> np.ndarray:
    """Sizes of single-linkage clusters of `points` at the given cutoff."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(n, dtype=int)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.bincount(labels)


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
