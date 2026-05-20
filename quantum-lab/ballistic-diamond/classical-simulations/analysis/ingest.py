#!/usr/bin/env python3
"""CLI for ingesting a completed sweep into a single HDF5 summary file.

Typical usage on the cluster, after all jobs land:

    cd classical-simulations/analysis
    python ingest.py \\
        --sn  /work/stovey/ballistic-diamond/tersoff-sweep \\
        --pb  /work/stovey/ballistic-diamond/tersoff-sweep-pb \\
        --out /work/stovey/ballistic-diamond/analysis/sweep-summary.h5 \\
        --workers 16

The two --sn / --pb roots are optional; supply whichever you have data for.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly from the analysis/ directory without installing.
sys.path.insert(0, str(Path(__file__).parent))

from ballistic_analysis import build_summary, save_summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sn", type=Path, default=None,
                   help="Root of the Sn sweep (contains energy-*/temperature-*/angle-*-N/).")
    p.add_argument("--pb", type=Path, default=None,
                   help="Root of the Pb sweep.")
    p.add_argument("--out", type=Path, required=True,
                   help="Destination HDF5 file.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for file parsing.  16 is a good default on a fat node.")
    args = p.parse_args()

    sweep_roots: dict[str, Path] = {}
    if args.sn is not None:
        sweep_roots["sn"] = args.sn
    if args.pb is not None:
        sweep_roots["pb"] = args.pb
    if not sweep_roots:
        p.error("Provide at least one of --sn / --pb.")

    print(f"Ingesting sweep roots: {sweep_roots}")
    print(f"Workers: {args.workers}")

    t0 = time.time()
    records = build_summary(sweep_roots, n_workers=args.workers)
    n_ok = sum(r.ok for r in records)
    elapsed = time.time() - t0

    print(f"Parsed {len(records)} ensemble directories  ({n_ok} with usable final.data)  "
          f"in {elapsed:.1f}s.")

    save_summary(records, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
