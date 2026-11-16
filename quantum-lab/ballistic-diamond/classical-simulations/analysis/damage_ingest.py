#!/usr/bin/env python3
"""CLI: reference-slab Wigner-Seitz damage analysis over completed runs.

For every run directory under --root (…/orient-O/energy-E/temperature-T/
angle-A-N with a final.data) this writes

    <out>/summary.csv                 one row per impact: depth, entry point,
                                      vacancy / interstitial counts, damage
                                      within 2 / 5 / 10 nm of the stopped ion,
                                      vacancy-cluster statistics, Sn site
                                      environment, registration quality
    <out>/defects/<run>.npz           every bulk vacancy and interstitial of
                                      the impact (box frame, A)
    <out>/snippets/<run>.xyz          the 3D damage snippet around the stopped
                                      ion (default 10 nm radius) for lattice
                                      KMC: Sn, vacancies, interstitials, with
                                      ideal-lattice integer coordinates

Usage (on the cluster; ~40 s and ~3 GB per impact per worker):

    python damage_ingest.py \\
        --root /work/stovey/ballistic-diamond/tersoff-sweep-maps \\
        --references /work/stovey/ballistic-diamond/references \\
        --out /work/stovey/ballistic-diamond/analysis/damage-maps \\
        --workers 6

The reference slabs come from tersoff-sweep/make-reference.lmp (one per
orientation; the matching temperature is preferred, the nearest is used
otherwise).  See ballistic_analysis/damage.py for why a pristine reference
plus an unwrapped registration is required on production-size slabs.

Trust check: `control_vac` is the vacancy count in the pristine crystal more
than 30 nm below the ion.  It should be ~0 unless the ion stopped shallow
(fast carbon recoils do travel further than the ion); `reg_max_step` should
stay far below 0.63 A and `resid_median` near 0.06 A at 300 K.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ballistic_analysis.damage import (
    A_DIAMOND,
    analyse_damage_reference,
    cluster_sizes,
    lattice_indices,
    load_reference_sites,
    periodic_delta,
)
from ballistic_analysis.reader import (
    parse_dirname,
    read_all_atoms,
    read_final_state,
    read_ion_trajectory,
)

# Geometry of simulate.lmp: slab top in lattice units and the z lattice
# spacing (bounding-box extent of the rotated conventional cell) in units of a.
SLAB_HI = {100: 940, 110: 670, 111: 550}
Z_FACTOR = {100: 1.0, 110: 2.0 ** 0.5, 111: 3.0 ** 0.5}
AXES = {
    100: "x=[100] y=[010] z=[001]",
    110: "x=[110] y=[00-1] z=[-110]",
    111: "x=[1-10] y=[11-2] z=[111]",
}

_REF_CACHE: dict[str, np.ndarray] = {}


def reference_for(ref_dir: Path, orientation: int, temperature: float) -> Path:
    """Reference slab for this orientation; nearest temperature wins."""
    best = None
    for cand in ref_dir.glob(f"reference-orient{orientation}-T*.data"):
        m = re.search(r"-T([0-9.]+)\.data$", cand.name)
        if not m:
            continue
        gap = abs(float(m.group(1)) - temperature)
        if best is None or gap < best[0]:
            best = (gap, cand)
    if best is None:
        raise FileNotFoundError(
            f"no reference-orient{orientation}-T*.data in {ref_dir} "
            "(generate one with tersoff-sweep/make-reference.lmp)"
        )
    return best[1]


def reference_sites(path: Path) -> np.ndarray:
    key = str(path)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = load_reference_sites(path)
    return _REF_CACHE[key]


def write_snippet(path: Path, header: dict, rows: list[tuple]) -> None:
    """Extended-XYZ damage snippet.  Species: Sn, X (vacancy site), C
    (interstitial carbon).  Positions are relative to the stopped ion."""
    comment = " ".join(
        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in header.items()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"{len(rows)}\n")
        fh.write('Properties=species:S:1:pos:R:3:lattice_n:I:3:kind:S:1 pbc="F F F" ' + comment + "\n")
        for sp, pos, n, kind in rows:
            fh.write(f"{sp:<2} {pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f} "
                     f"{n[0]:5d} {n[1]:5d} {n[2]:5d} {kind}\n")


def process_run(task: tuple) -> dict:
    run_dir, root, ref_dir, out_dir, radius, species, tree_workers = task
    run_dir = Path(run_dir)
    params = parse_dirname(run_dir)
    rel = run_dir.relative_to(root)
    row: dict = {"run": str(rel), "species": species, "ok": 0, "status": "", **params}
    try:
        state = read_final_state(run_dir / "final.data")
        atoms = read_all_atoms(run_dir / "final.data") if state else None
        if state is None or atoms is None:
            row["status"] = "no_data"
            return row
        if state["ion"] is None:
            row["status"] = "ion_not_in_slab"
            return row
        types, positions = atoms
        orientation = params["orientation"]
        a = A_DIAMOND
        try:
            pj = json.loads((run_dir / "params.json").read_text())
            a = float(pj.get("a_lattice", a))
            row["azimuth_deg"] = float(pj.get("azimuth_deg", float("nan")))
        except (OSError, ValueError):
            row["azimuth_deg"] = float("nan")

        ref_path = reference_for(Path(ref_dir), orientation, params["T_K"])
        res = analyse_damage_reference(
            state, positions, types, reference_sites(ref_path), workers=tree_workers
        )
        box = state["box"]
        ion, ion_site = res["ion"], res["ion_site"]

        # --- entry point and stop-time depth from the ion trajectory ---
        surf = SLAB_HI[orientation] * a * Z_FACTOR[orientation]
        row.update(entry_x=float("nan"), entry_y=float("nan"), depth_stop_nm=float("nan"))
        tr = read_ion_trajectory(run_dir / "collision-ion.lammpstraj")
        if tr is not None and len(tr["pos"]) >= 2 and tr["vel"][0][2] < 0:
            p0, v0 = tr["pos"][0], tr["vel"][0]
            t_hit = (p0[2] - surf) / (-v0[2])
            row["entry_x"] = float(p0[0] + v0[0] * t_hit)
            row["entry_y"] = float(p0[1] + v0[1] * t_hit)
            row["depth_stop_nm"] = float((surf - tr["pos"][-1, 2]) / 10.0)
        row["depth_nm"] = float((state["z_surface"] - ion[2]) / 10.0)
        row["ion_above_bottom_nm"] = float((ion[2] - state["z_bottom"]) / 10.0)
        # the snippet sphere must not reach into the excluded anchor scaffold
        row["snippet_clean"] = int(ion[2] - radius > res["z_interior_lo"])

        # --- damage relative to the stopped ion ---
        dv = periodic_delta(res["vac_pos"], ion, box)
        di = periodic_delta(res["int_pos"], ion, box)
        dis = periodic_delta(res["int_site"], ion, box)
        rv = np.linalg.norm(dv, axis=1)
        ri = np.linalg.norm(di, axis=1)
        nn = a * 3.0 ** 0.5 / 4.0
        row.update(
            n_vac=res["n_vac"], n_int=res["n_int"], n_offsite=res["n_offsite"],
            vac_r2=int((rv < 20).sum()), vac_r5=int((rv < 50).sum()),
            vac_r10=int((rv < 100).sum()), int_r10=int((ri < 100).sum()),
        )
        # nearest-neighbour clusters (the divacancy is two adjacent sites);
        # a looser second-neighbour linkage is reported alongside.
        for tag, sel in (("all", np.ones(len(rv), bool)), ("r10", rv < 100), ("r5", rv < 50)):
            cs = cluster_sizes(dv[sel], 1.2 * nn)
            row[f"V1_{tag}"] = int((cs == 1).sum())
            row[f"V2_{tag}"] = int((cs == 2).sum())
            row[f"V3p_{tag}"] = int((cs >= 3).sum())
            row[f"Vmax_{tag}"] = int(cs.max()) if len(cs) else 0
        cs2 = cluster_sizes(dv[rv < 100], 1.09 * a / 2.0 ** 0.5)
        row["V2p_r10_2nn"] = int((cs2 >= 2).sum())

        # --- where the Sn sits ---
        off = ion - ion_site
        row["sn_site_dist"] = float(np.linalg.norm(off))
        row["sn_nvac_2A"] = int((rv < 2.0).sum())
        row["sn_nvac_3A"] = int((rv < 3.0).sum())
        row["sn_nint_3A"] = int((ri < 3.0).sum())

        # --- registration quality ---
        row.update(
            control_vac=res["control_vac"], control_sites=res["control_sites"],
            resid_median=res["resid_median"], reg_max_step=res["reg_max_step"],
            reg_u_range=res["reg_u_range"], reg_min_good=res["reg_min_good"],
            scaffold_top=res["scaffold_top"],
        )

        # --- per-impact defect file ---
        out_dir = Path(out_dir)
        npz = out_dir / "defects" / rel.with_suffix(".npz")
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz, vac_pos=res["vac_pos"], int_pos=res["int_pos"], int_site=res["int_site"],
            ion=ion, ion_site=ion_site, z_surface=state["z_surface"],
            z_bottom=state["z_bottom"], box=np.array([box[k] for k in
                                                      ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")]),
            profile_z=res["profile"]["z"], profile_u=res["profile"]["u"],
        )

        # --- KMC snippet: everything within `radius` of the stopped ion ---
        in_v, in_i = rv < radius, ri < radius
        n_v, resid_v = lattice_indices(dv[in_v] + off, orientation, a)
        n_i, _ = lattice_indices(dis[in_i] + off, orientation, a)
        rows = [("Sn", np.zeros(3), np.zeros(3, dtype=int), "ion")]
        rows += [("X", p, n, "vacancy") for p, n in zip(dv[in_v], n_v)]
        rows += [("C", p, n, "interstitial") for p, n in zip(di[in_i], n_i)]
        row["snip_resid_max"] = float(resid_v.max()) if len(resid_v) else 0.0
        header = {
            "origin": "final Sn position; positions in A, box frame",
            "axes_box": AXES[orientation],
            "lattice_n": "ideal diamond lattice site in units of a/4 along the cubic axes, "
                         "relative to the lattice site nearest the Sn (Sn and interstitials: "
                         "nearest / host site)",
            "a_A": round(a, 5), "radius_A": radius, "species_ion": species,
            "E_keV": params["E_keV"], "angle_deg": params["angle_deg"],
            "T_K": params["T_K"], "ensemble": params["ensemble"],
            "depth_nm": round(row["depth_nm"], 2),
            "entry_x_A": round(row["entry_x"], 4), "entry_y_A": round(row["entry_y"], 4),
            "Sn_offset_from_site_A": " ".join(f"{v:.3f}" for v in off),
            "n_vac": int(in_v.sum()), "n_int": int(in_i.sum()),
            "V1": row["V1_r10"], "V2": row["V2_r10"], "V3plus": row["V3p_r10"],
            "state": "as-quenched MD, 50 ps after the cascade",
        }
        write_snippet(out_dir / "snippets" / rel.with_suffix(".xyz"), header, rows)

        row["ok"], row["status"] = 1, "ok"
    except Exception as exc:                       # keep the batch alive
        row["status"] = f"error: {exc}"
        traceback.print_exc()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="sweep tree to analyse")
    ap.add_argument("--references", type=Path, required=True,
                    help="directory of reference-orient<O>-T<T>.data slabs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--species", default="sn")
    ap.add_argument("--radius", type=float, default=100.0, help="snippet radius [A]")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tree-workers", type=int, default=1)
    ap.add_argument("--only", default=None, help="substring filter on the run path")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    runs = sorted(p.parent for p in args.root.rglob("final.data")
                  if parse_dirname(p.parent) is not None)
    if args.only:
        runs = [r for r in runs if args.only in str(r)]
    if args.limit:
        runs = runs[: args.limit]
    print(f"{len(runs)} impacts under {args.root}; workers={args.workers}", flush=True)
    if not runs:
        return 1

    tasks = [(r, args.root, args.references, args.out, args.radius, args.species,
              args.tree_workers) for r in runs]
    t0 = time.time()
    rows = []
    if args.workers <= 1:
        for i, task in enumerate(tasks):
            rows.append(process_run(task))
            print(f"  {i + 1}/{len(tasks)}  {rows[-1]['run']}  {rows[-1]['status']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(process_run, tasks)):
                rows.append(row)
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print(f"  {i + 1}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)

    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, restval="")
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(r["ok"] for r in rows)
    print(f"analysed {len(rows)} impacts ({n_ok} ok) in {time.time() - t0:.0f}s "
          f"-> {args.out / 'summary.csv'}")
    for row in rows:
        if not row["ok"]:
            print(f"  NOT OK: {row['run']}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
