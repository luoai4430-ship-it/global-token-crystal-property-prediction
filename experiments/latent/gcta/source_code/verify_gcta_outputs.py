"""Independently verify the completed GCTA validation screen and copied files."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


TASKS = ("Jarvis-Bandgap_MBJ", "MP-Bandgap")
VARIANTS = ("B1_NaiveConcat", "P2_GCTA")
DELIVERABLES = (
    "gcta_fast_screen.csv", "gcta_attention_diagnostics.csv",
    "gcta_query_controls.csv", "gcta_parameter_count.json", "gcta_report.md",
)
DEFINITION = "qc=0; q=normalize(qg); prediction_head_cls=correct"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(a, b, tolerance=1e-7):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a) - float(b)) <= tolerance, (a, b)


def zero_counts(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "test_forward_count":
                assert value == 0, value
            zero_counts(value)
    elif isinstance(obj, list):
        for value in obj:
            zero_counts(value)


def verify(root):
    state = json.loads((root / "queue_state.json").read_text())
    assert state["status"] == "COMPLETE" and state["completed_training_runs"] == 4
    assert not state["active"] and not state["failed"]
    for name in DELIVERABLES:
        assert (root / name).is_file() and (root / name).stat().st_size > 0
    for path in root.rglob("*.json"):
        zero_counts(json.loads(path.read_text(encoding="utf-8")))
    forbidden = {"test_predictions.csv", "test_summary.json", "test_forward_started.json"}
    assert not [p for p in root.rglob("*") if p.is_file() and p.name.lower() in forbidden]

    results = read_csv(root / "gcta_fast_screen.csv")
    controls = read_csv(root / "gcta_query_controls.csv")
    assert len(results) == 4 and len(controls) == 6
    index = {(r["task"], r["variant"]): r for r in results}
    assert set(index) == {(t, v) for t in TASKS for v in VARIANTS}
    assert len(list(root.glob("runs/*/*/seed_42/summary.json"))) == 4
    task_gains = {}
    attention_l1 = {}
    corrected = {}
    for task in TASKS:
        predictions = {}
        for variant in VARIANTS:
            directory = root / "runs" / task / variant / "seed_42"
            summary = json.loads((directory / "summary.json").read_text())
            assert summary["status"] == "valid_complete"
            assert summary["training_runs"] == 1 and summary["checkpoint_reselection"] == 0
            assert summary["seed"] == 42 and summary["temperature"] == 0.5
            assert summary["stopped_epoch"] <= 100
            assert digest(directory / "best_valid.pt") == summary["checkpoint_sha256"]
            rows = read_csv(directory / "valid_predictions.csv")
            assert len(rows) == summary["valid_size"]
            by_id = {r["sample_id"]: r for r in rows}
            assert len(rows) == len(by_id)
            errors = [abs(float(r["prediction"]) - float(r["target"])) for r in rows]
            close(mean(errors), summary["valid_mae"])
            close(summary["valid_mae"], index[task, variant]["valid_mae"])
            for r, error in zip(rows, errors):
                close(r["absolute_error"], error)
            predictions[variant] = by_id
            if variant == "P2_GCTA":
                correction = json.loads((directory / "query_controls_corrected.json").read_text())
                assert correction["zero_control_definition"] == DEFINITION
                assert correction["zero_qc_qg_ratio_mean"] == 0
                assert correction["checkpoint_sha256"] == summary["checkpoint_sha256"]
                assert correction["training_runs"] == 0 and correction["checkpoint_reselection"] == 0
                for mode in ("normal", "query_cls_shuffled", "query_cls_zero"):
                    rows = read_csv(directory / f"{mode}_control_corrected.csv")
                    assert len(rows) == len(by_id)
                    assert {r["sample_id"] for r in rows} == set(by_id)
                    errors = []
                    for r in rows:
                        close(r["target"], by_id[r["sample_id"]]["target"])
                        errors.append(abs(float(r["prediction"]) - float(r["target"])))
                    close(mean(errors), correction[f"{mode}_mae"])
                    entry = [r for r in controls if r["task"] == task and r["control"] == mode]
                    assert len(entry) == 1
                    close(entry[0]["valid_mae"], correction[f"{mode}_mae"])
                corrected[task] = {m: correction[f"{m}_mae"] for m in (
                    "normal", "query_cls_shuffled", "query_cls_zero"
                )}
        b1, p2 = (predictions[v] for v in VARIANTS)
        assert set(b1) == set(p2)
        distances = []
        for sample_id in b1:
            close(b1[sample_id]["target"], p2[sample_id]["target"])
            left, right = (json.loads(d[sample_id]["attention_json"]) for d in (b1, p2))
            assert len(left) == len(right) == 128
            distances.append(sum(abs(a - b) for a, b in zip(left, right)))
        attention_l1[task] = mean(distances)
        b, p = (float(index[task, v]["valid_mae"]) for v in VARIANTS)
        task_gains[task] = 100 * (b - p) / b
    selection = json.loads((root / "gcta_selection.json").read_text())
    close(mean(task_gains.values()), selection["macro_gain_vs_b1_pct"])
    assert selection["wins_vs_b1"] == sum(g > 0 for g in task_gains.values())
    assert selection["query_semantic_both_tasks"] == all(
        c["normal"] < c["query_cls_shuffled"] and c["normal"] < c["query_cls_zero"]
        for c in corrected.values()
    )
    parameters = json.loads((root / "gcta_parameter_count.json").read_text())
    counts = parameters["parameter_counts"]
    close(100 * (counts["P2_GCTA"] - counts["B1_NaiveConcat"]) / counts["B1_NaiveConcat"],
          parameters["p2_added_params_vs_b1_pct"])
    return {
        "status": "verified", "validation_runs": 4, "test_forward_count": 0,
        "task_gains_pct": task_gains, "macro_gain_pct": mean(task_gains.values()),
        "corrected_query_controls": corrected,
        "mean_summed_attention_L1": attention_l1, "decision": selection["decision"],
        "added_parameters_pct": parameters["p2_added_params_vs_b1_pct"],
        "files_sha256": {
            p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix not in (".tgz", ".gz")
            and p.name not in ("gcta_verification.json", "interim_analysis_20260905.md")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = verify(args.root)
    if args.write:
        (args.root / "gcta_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2))
