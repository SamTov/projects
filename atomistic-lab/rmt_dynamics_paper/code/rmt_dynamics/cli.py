"""Shared argparse parent for experiment `run.py` entry points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parent_parser() -> argparse.ArgumentParser:
    """Argparse parent with --seed, --config, --output-dir.

    Experiments compose this with `parents=[build_parent_parser()]` and add
    their own arguments on top.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed threaded through MD and numpy (default: 0).",
    )
    parser.add_argument(
        "--config", type=Path, required=False,
        help="YAML config file (see each experiment's config.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Where to write artifacts (default: ./outputs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Placeholder console-script entry point.

    Per-experiment drivers live in `experiments/NN_*/run.py` (not yet
    created). This stub tells the user where to look.
    """
    parser = argparse.ArgumentParser(
        prog="rmt-run",
        description="rmt_dynamics analysis shared infrastructure.",
    )
    parser.add_argument(
        "experiment", nargs="?", default=None,
        help="experiment id, e.g. 01_noise_convergence",
    )
    args = parser.parse_args(argv)

    print(
        "rmt_dynamics infrastructure is installed.\n"
        "Experiment drivers land in `experiments/NN_*/run.py` in follow-up sessions.\n"
        "See experiments.md for the full specification."
    )
    if args.experiment is not None:
        print(f"Requested: {args.experiment} — driver not yet implemented.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
