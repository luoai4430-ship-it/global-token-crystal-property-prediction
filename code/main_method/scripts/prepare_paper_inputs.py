#!/usr/bin/env python3
"""Build task graph caches and select descriptions by the processed split IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    import run_seed43_benchmark as benchmark
    from data.token_sequence import graph_dataset_ids
    from dataclasses import replace

    if args.task not in benchmark.TASKS:
        parser.error(f"unknown task {args.task}")
    spec = benchmark.TASKS[args.task]
    spec = replace(spec, source_dir=str(args.data_root / Path(spec.source_dir).relative_to(benchmark.DATA_ROOT)))
    benchmark.TASKS[args.task] = spec
    # The loader selects MP 60000/5000/remainder and JARVIS 80/10/remainder.
    # It generates no property-model predictions.
    train, valid, test, mean, std = benchmark.load_graph_splits(args.task)
    split_ids = {name: graph_dataset_ids(dataset) for name, dataset in (
        ("train", train), ("valid", valid), ("test", test)
    )}
    ordered = [item for values in split_ids.values() for item in values]
    if len(ordered) != len(set(ordered)):
        raise ValueError("processed graph splits overlap or contain duplicate IDs")
    descriptions = {}
    with spec.description_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row[spec.id_column]).strip()
            if sample_id in descriptions:
                raise ValueError(f"duplicate description ID {sample_id}")
            descriptions[sample_id] = str(row[spec.text_column]).strip()
    missing = [item for item in ordered if not descriptions.get(item)]
    if missing:
        raise ValueError(f"missing descriptions for {len(missing)} processed IDs, e.g. {missing[:10]}; obtain the matching snapshot")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "text"])
        writer.writeheader()
        writer.writerows({"sample_id": item, "text": descriptions[item]} for item in ordered)
    metadata = {
        "task": args.task, "split_seed": 123,
        "sizes": {name: len(ids) for name, ids in split_ids.items()},
        "split_ids": split_ids, "mean": mean, "std": std,
        "source_description": str(spec.description_csv),
        "test_forward_count": 0,
    }
    args.output_csv.with_suffix(".split.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "split_ids"}, indent=2))


if __name__ == "__main__":
    main()
