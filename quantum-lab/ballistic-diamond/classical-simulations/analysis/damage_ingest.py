#!/usr/bin/env python3
"""CLI: Wigner-Seitz damage analysis over a completed sweep.

Run AFTER (or independently of) ingest.py -- this walks the same tree but
does the heavy per-snapshot defect analysis:

    cd classical-simulations/analysis
    python damage_ingest.py \\
        --sn  /work/stovey/ballistic-diamond/tersoff-sweep \\
        --pb  /work/stovey/ballistic-diamond/tersoff-sweep-pb \\
        --out /work/stovey/ballistic-diamond/analysis/damage-summary.h5 \\
        --workers 16

Cost: ~1-2 min per ensemble (4.2M-atom parse + KDTree queries); 5400
ensembles on 16 workers ~ 6-11 h.  --no-coordination drops the sp2 count
and roughly halves the time.

Acceptance check before trusting output: on a TEST run (pristine slab, one
ion), n_vac should be O(10-300) and vac_depths clustered along the track --
NOT thousands scattered uniformly (that means lattice registration failed).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ballistic_analysis.aggregate import walk_sweep_tree
from ballistic_analysis.damage import DamageRecord, analyse_damage, save_damage
from ballistic_analysis.reader import parse_dirname, read_all_atoms, read_final_state


def process_one(args):
    species, ens_dir, coordination = args
    params = parse_dirname(ens_dir)
    if params is None:
        return None

    base = dict(
        species=species,
        orientation=params["orientation"],
        E_keV=params["E_keV"], angle_deg=params["angle_deg"],
        T_K=params["T_K"], ensemble=params["ensemble"],
    )
    state = read_final_state(ens_dir / "final.data")
    atoms = read_all_atoms(ens_dir / "final.data") if state else None
    if state is None or atoms is None:
        rec = DamageRecord(**base, n_sites=0, n_carbon=0, n_vac=0, n_int=0,
                           n_lost=0, n_sp2=-1, ok=0)
        return rec, __import__("numpy").zeros(0)

    types, positions = atoms
    import json
    try:
        with open(ens_dir / "params.json") as fh:
            a_lat = float(json.load(fh).get("a_lattice", 3.5656))
    except (OSError, ValueError, json.JSONDecodeError):
        a_lat = 3.5656
    result = analyse_damage(state, positions, types, coordination=coordination,
                            orientation=params["orientation"], a=a_lat)
    rec = DamageRecord(
        **base,
        n_sites=result["n_sites"], n_carbon=result["n_carbon"],
        n_vac=result["n_vac"], n_int=result["n_int"],
        n_lost=result["n_lost"], n_sp2=result["n_sp2"], ok=1,
    )
    return rec, result["vac_depths"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sn", type=Path, default=None)
    p.add_argument("--pb", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-coordination", action="store_true",
                   help="Skip the sp2/coordination count (roughly 2x faster).")
    args = p.parse_args()

    roots = {}
    if args.sn is not None:
        roots["sn"] = args.sn
    if args.pb is not None:
        roots["pb"] = args.pb
    if not roots:
        p.error("Provide at least one of --sn / --pb.")

    work = []
    for species, root in roots.items():
        for ens_dir in walk_sweep_tree(root, species):
            work.append((species, ens_dir, not args.no_coordination))
    print(f"{len(work)} ensembles to analyse; workers={args.workers}")

    t0 = time.time()
    results = []
    if args.workers <= 1:
        for i, item in enumerate(work):
            r = process_one(item)
            if r is not None:
                results.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(work)}  ({time.time()-t0:.0f}s)")
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(process_one, work, chunksize=4)):
                if r is not None:
                    results.append(r)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(work)}  ({time.time()-t0:.0f}s)")

    records = [r for r, _ in results]
    depths = [d for _, d in results]
    n_ok = sum(r.ok for r in records)
    print(f"Analysed {len(records)} ensembles ({n_ok} ok) in {time.time()-t0:.0f}s")

    save_damage(records, depths, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
