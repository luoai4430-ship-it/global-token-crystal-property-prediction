#!/usr/bin/env python3
"""Run the unchanged paired benchmark for a fresh seed-42 or seed-43 run."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--training_seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--help", action="store_true")
    if not sys.argv[1:] or sys.argv[1:] == ["--help"]:
        parser.print_help()
        print("\nCommands: preflight, train, recover-valid, freeze, test")
        print("Use summarize_paper_predictions.py to combine saved predictions.")
        return
    options, remaining = parser.parse_known_args()
    if not remaining:
        parser.error("a benchmark command is required")
    if remaining[0] == "summarize":
        parser.error("use summarize_paper_predictions.py; the historical summary uses frozen baselines")

    # Retain the server runner verbatim. Only the run seed and dataset location
    # are adapted; model construction, losses, scheduler and freeze stay intact.
    import run_seed43_benchmark as benchmark

    benchmark.SEEDS = (options.training_seed,)
    data_root = options.data_root or Path(
        os.environ.get("CRYSTAL_DATA_ROOT", str(benchmark.DATA_ROOT))
    )
    old_root = benchmark.DATA_ROOT
    benchmark.TASKS = {
        name: replace(spec, source_dir=str(data_root / Path(spec.source_dir).relative_to(old_root)))
        for name, spec in benchmark.TASKS.items()
    }
    benchmark.DATA_ROOT = data_root
    if options.help:
        remaining.append("--help")
    sys.argv = [sys.argv[0], *remaining]
    benchmark.main()


if __name__ == "__main__":
    main()
