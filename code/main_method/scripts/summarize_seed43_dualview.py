#!/usr/bin/env python3
"""Summarize paired seed-43 Global, TrueToken, and equal-weight predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = (
    "Jarvis-Bandgap_MBJ",
    "Jarvis-Bandgap_OPT",
    "Jarvis-FormationEnergy",
    "Jarvis-TotalEnergy",
    "Jarvis-BulkModulusKv",
    "Jarvis-ShearModulusGv",
    "MP-Bandgap",
    "MP-FormationEnergy",
    "MP-BulkModuli",
    "MP-ShearModuli",
)
SEED = 43


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    run_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for task in TASKS:
        frames: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict[str, object]] = {}
        for model in ("original_sftmat", "ours"):
            run = root / "runs" / task / model / f"seed_{SEED}"
            valid = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            test = json.loads((run / "test_summary.json").read_text(encoding="utf-8"))
            if valid["test_forward_count"] != 0 or test["test_forward_count"] != 1:
                raise AssertionError(f"{task}/{model}: invalid test-forward count")
            if test["selected_by"] != "Valid only":
                raise AssertionError(f"{task}/{model}: invalid checkpoint selection")
            frame = pd.read_csv(run / "test_predictions.csv", dtype={"sample_id": str})
            if frame.sample_id.duplicated().any():
                raise AssertionError(f"{task}/{model}: duplicate test IDs")
            frames[model] = frame[["sample_id", "target", "prediction"]]
            metrics[model] = test
            run_rows.append({
                "task": task,
                "dataset": "JARVIS" if task.startswith("Jarvis-") else "MP",
                "model_name": model,
                "seed": SEED,
                "best_valid_epoch": valid["best_valid_epoch"],
                "valid_mae": valid["valid_mae"],
                "test_mae": test["test_mae"],
                "training_time_seconds": valid["training_time_seconds"],
                "parameter_count": valid["parameter_count"],
                "test_forward_count": test["test_forward_count"],
                "checkpoint_sha256": valid["checkpoint_sha256"],
                "protocol_sha256": valid["protocol_sha256"],
            })

        paired = frames["original_sftmat"].merge(
            frames["ours"], on="sample_id", suffixes=("_global", "_token"),
            validate="one_to_one",
        )
        if len(paired) != len(frames["original_sftmat"]) or len(paired) != len(frames["ours"]):
            raise AssertionError(f"{task}: test sample-ID sets differ")
        target_delta = float(np.max(np.abs(paired.target_global - paired.target_token)))
        if target_delta > 1e-6:
            raise AssertionError(f"{task}: paired targets differ by {target_delta}")
        paired["prediction_dual"] = 0.5 * (
            paired.prediction_global + paired.prediction_token
        )
        paired["absolute_error_global"] = (
            paired.prediction_global - paired.target_global
        ).abs()
        paired["absolute_error_token"] = (
            paired.prediction_token - paired.target_global
        ).abs()
        paired["absolute_error_dual"] = (
            paired.prediction_dual - paired.target_global
        ).abs()
        paired.to_csv(root / f"{task}--paired_test_predictions.csv", index=False)
        global_mae = float(paired.absolute_error_global.mean())
        token_mae = float(paired.absolute_error_token.mean())
        dual_mae = float(paired.absolute_error_dual.mean())
        paired_rows.append({
            "task": task,
            "dataset": "JARVIS" if task.startswith("Jarvis-") else "MP",
            "seed": SEED,
            "test_size": len(paired),
            "global_test_mae": global_mae,
            "token_test_mae": token_mae,
            "dual_test_mae": dual_mae,
            "dual_gain_vs_global_percent": gain(global_mae, dual_mae),
            "dual_gain_vs_token_percent": gain(token_mae, dual_mae),
            "token_gain_vs_global_percent": gain(global_mae, token_mae),
            "winner": min(
                (global_mae, "Global"),
                (token_mae, "Token"),
                (dual_mae, "Dual"),
            )[1],
            "target_max_abs_delta": target_delta,
        })

    runs = pd.DataFrame(run_rows)
    results = pd.DataFrame(paired_rows)
    runs.to_csv(root / "seed43_per_run_results.csv", index=False)
    results.to_csv(root / "seed43_dualview_results.csv", index=False)
    groups = {
        "all": results,
        "JARVIS": results[results.dataset == "JARVIS"],
        "MP": results[results.dataset == "MP"],
    }
    macro = {}
    for name, frame in groups.items():
        macro[name] = {
            "task_count": len(frame),
            "dual_wins_vs_global": int((frame.dual_gain_vs_global_percent > 0).sum()),
            "dual_wins_vs_both": int(
                ((frame.dual_test_mae < frame.global_test_mae)
                 & (frame.dual_test_mae < frame.token_test_mae)).sum()
            ),
            "macro_dual_gain_vs_global_percent": float(
                frame.dual_gain_vs_global_percent.mean()
            ),
            "macro_dual_gain_vs_token_percent": float(
                frame.dual_gain_vs_token_percent.mean()
            ),
            "macro_token_gain_vs_global_percent": float(
                frame.token_gain_vs_global_percent.mean()
            ),
        }
    selection = {
        "status": "SEED43_TEST_COMPLETE",
        "seed": SEED,
        "weights": {"global": 0.5, "token": 0.5},
        "selected_by": "validation MAE only",
        "test_used_for_training_or_selection": False,
        "test_forward_count": int(runs.test_forward_count.sum()),
        "macro": macro,
    }
    atomic_json(root / "seed43_selection.json", selection)
    report = [
        "# Seed 43 dual-view result",
        "",
        f"All-task macro gain vs Global: {macro['all']['macro_dual_gain_vs_global_percent']:.3f}%",
        f"JARVIS macro gain vs Global: {macro['JARVIS']['macro_dual_gain_vs_global_percent']:.3f}%",
        f"MP macro gain vs Global: {macro['MP']['macro_dual_gain_vs_global_percent']:.3f}%",
        f"Dual wins vs Global: {macro['all']['dual_wins_vs_global']}/10",
        f"Dual wins vs both views: {macro['all']['dual_wins_vs_both']}/10",
        "",
        "Test was evaluated once per validation-selected checkpoint.",
    ]
    (root / "seed43_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
