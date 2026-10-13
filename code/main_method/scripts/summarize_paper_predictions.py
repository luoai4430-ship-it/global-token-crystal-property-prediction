#!/usr/bin/env python3
"""Summarize saved paired predictions; no model or inference is loaded."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TASKS = (
    "Jarvis-Bandgap_MBJ", "Jarvis-Bandgap_OPT", "Jarvis-FormationEnergy",
    "Jarvis-TotalEnergy", "Jarvis-BulkModulusKv", "Jarvis-ShearModulusGv",
    "MP-Bandgap", "MP-FormationEnergy", "MP-BulkModuli", "MP-ShearModuli",
)


def read_predictions(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        sample_id = row["sample_id"]
        target, prediction = float(row["target"]), float(row["prediction"])
        if sample_id in values or not all(math.isfinite(value) for value in (target, prediction)):
            raise ValueError(f"invalid predictions in {path}")
        values[sample_id] = (target, prediction)
    if not values:
        raise ValueError(f"empty predictions in {path}")
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=ROOT")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for value in args.run:
        seed, path = value.split("=", 1)
        seed = int(seed)
        if seed not in (42, 43) or seed in roots:
            parser.error("each seed must be 42 or 43 and appear only once")
        roots[seed] = Path(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed, root in roots.items():
        if args.split == "test":
            freeze = json.loads((root / "valid_freeze_manifest.json").read_text())
            entries = {(item["task"], item["model_name"], int(item["seed"])) for item in freeze["entries"]}
            expected = {(task, model, seed) for task in TASKS for model in ("original_sftmat", "ours")}
            if freeze["status"] != "valid_selection_frozen" or entries != expected or len(freeze["entries"]) != 20:
                raise ValueError(f"incomplete validation freeze for seed {seed}")
        for task in TASKS:
            frames = {}
            for model in ("original_sftmat", "ours"):
                run = root / "runs" / task / model / f"seed_{seed}"
                valid = json.loads((run / "summary.json").read_text())
                if valid["test_forward_count"] != 0:
                    raise ValueError("validation summary contains test forwards")
                if args.split == "test":
                    summary = json.loads((run / "test_summary.json").read_text())
                    if summary["test_forward_count"] != 1 or summary["selected_by"] != "Valid only":
                        raise ValueError("test evaluation protocol mismatch")
                frames[model] = read_predictions(run / f"{args.split}_predictions.csv")
            global_values, token_values = frames["original_sftmat"], frames["ours"]
            if set(global_values) != set(token_values):
                raise ValueError(f"{task}: Global/Token ID sets differ")
            pairs = []
            for sample_id, (target, global_pred) in global_values.items():
                token_target, token_pred = token_values[sample_id]
                if abs(target - token_target) > 1e-6:
                    raise ValueError(f"{task}: targets differ")
                pairs.append({
                    "sample_id": sample_id, "target": target,
                    "prediction_global": global_pred, "prediction_token": token_pred,
                    "prediction_dual": 0.5 * (global_pred + token_pred),
                })
            write_csv(args.output_dir / f"{task}--seed{seed}--paired_{args.split}.csv", pairs)
            maes = {view: statistics.mean(abs(row[f"prediction_{view}"] - row["target"]) for row in pairs)
                    for view in ("global", "token", "dual")}
            results.append({
                "task": task, "seed": seed, "split": args.split, "count": len(pairs),
                **{f"{view}_mae": value for view, value in maes.items()},
                "dual_gain_vs_global_percent": 100 * (maes["global"] - maes["dual"]) / maes["global"],
                "dual_beats_global": maes["dual"] < maes["global"],
                "dual_beats_both": maes["dual"] < min(maes["global"], maes["token"]),
            })
    write_csv(args.output_dir / "per_seed_task_results.csv", results)
    means = []
    for task in TASKS:
        group = [row for row in results if row["task"] == task]
        row = {"task": task, "split": args.split, "runs": len(group)}
        for metric in ("global_mae", "token_mae", "dual_mae", "dual_gain_vs_global_percent"):
            values = [item[metric] for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) > 1 else ""
        means.append(row)
    write_csv(args.output_dir / "two_run_task_summary.csv", means)
    summary = {"split": args.split, "runs": {}}
    for seed in roots:
        group = [row for row in results if row["seed"] == seed]
        summary["runs"][str(seed)] = {
            "macro_gain_percent": statistics.mean(row["dual_gain_vs_global_percent"] for row in group),
            "jarvis_macro_gain_percent": statistics.mean(row["dual_gain_vs_global_percent"] for row in group if row["task"].startswith("Jarvis-")),
            "mp_macro_gain_percent": statistics.mean(row["dual_gain_vs_global_percent"] for row in group if row["task"].startswith("MP-")),
            "wins_vs_global": sum(row["dual_beats_global"] for row in group),
            "wins_vs_both": sum(row["dual_beats_both"] for row in group),
        }
    (args.output_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
