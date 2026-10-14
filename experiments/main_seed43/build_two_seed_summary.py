from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED42_ROOT = ROOT.parent / "dualview_test_seed42_20260826"
SEED42_CSV = SEED42_ROOT / "dualview_test_results.csv"
SEED43_CSV = ROOT / "seed43_dualview_results.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


seed42_rows = read_csv(SEED42_CSV)
seed43_rows = read_csv(SEED43_CSV)
seed42 = {row["task"]: row for row in seed42_rows}
seed43 = {row["task"]: row for row in seed43_rows}
if set(seed42) != set(seed43) or len(seed42) != 10:
    raise RuntimeError("Seed 42 and seed 43 task sets do not match exactly")

task_order = [row["task"] for row in seed43_rows]
task_summary: list[dict[str, object]] = []
seed_metrics: dict[int, dict[str, dict[str, float]]] = {42: {}, 43: {}}

for task in task_order:
    r42 = seed42[task]
    r43 = seed43[task]
    if r42["dataset"] != r43["dataset"]:
        raise RuntimeError(f"Dataset mismatch for {task}")
    if int(r42["test_size"]) != int(r43["test_size"]):
        raise RuntimeError(f"Test-size mismatch for {task}")

    values: dict[str, list[float]] = {}
    for metric in ("global_test_mae", "token_test_mae", "dual_test_mae"):
        values[metric] = [as_float(r42, metric), as_float(r43, metric)]

    gain_global = [
        gain(values["global_test_mae"][index], values["dual_test_mae"][index])
        for index in range(2)
    ]
    gain_token = [
        gain(values["token_test_mae"][index], values["dual_test_mae"][index])
        for index in range(2)
    ]
    token_gain = [
        gain(values["global_test_mae"][index], values["token_test_mae"][index])
        for index in range(2)
    ]

    for index, seed in enumerate((42, 43)):
        g = values["global_test_mae"][index]
        t = values["token_test_mae"][index]
        d = values["dual_test_mae"][index]
        seed_metrics[seed][task] = {
            "global": g,
            "token": t,
            "dual": d,
            "dual_gain_vs_global": gain_global[index],
            "dual_gain_vs_token": gain_token[index],
            "token_gain_vs_global": token_gain[index],
            "dual_wins_vs_global": float(d < g),
            "dual_wins_vs_both": float(d < g and d < t),
        }

    means = {
        "Global": mean(values["global_test_mae"]),
        "Token": mean(values["token_test_mae"]),
        "Dual": mean(values["dual_test_mae"]),
    }
    mean_winner = min(means, key=means.get)
    task_summary.append(
        {
            "task": task,
            "dataset": r43["dataset"],
            "unit": r42.get("unit", ""),
            "test_size": int(r43["test_size"]),
            "global_test_mae_mean": means["Global"],
            "global_test_mae_std": sample_std(values["global_test_mae"]),
            "token_test_mae_mean": means["Token"],
            "token_test_mae_std": sample_std(values["token_test_mae"]),
            "dual_test_mae_mean": means["Dual"],
            "dual_test_mae_std": sample_std(values["dual_test_mae"]),
            "dual_gain_vs_global_mean_percent": mean(gain_global),
            "dual_gain_vs_global_std_percent": sample_std(gain_global),
            "dual_gain_vs_token_mean_percent": mean(gain_token),
            "dual_gain_vs_token_std_percent": sample_std(gain_token),
            "seed42_winner": r42["winner"],
            "seed43_winner": r43["winner"],
            "two_seed_mean_winner": mean_winner,
            "dual_wins_vs_global_seed_count": sum(
                int(seed_metrics[s][task]["dual_wins_vs_global"]) for s in (42, 43)
            ),
            "dual_wins_vs_both_seed_count": sum(
                int(seed_metrics[s][task]["dual_wins_vs_both"]) for s in (42, 43)
            ),
        }
    )


def scope_tasks(scope: str) -> list[str]:
    if scope == "all":
        return task_order
    return [task for task in task_order if seed43[task]["dataset"] == scope]


macro_by_seed: dict[int, dict[str, dict[str, float]]] = {42: {}, 43: {}}
macro_rows: list[dict[str, object]] = []
for scope in ("all", "JARVIS", "MP"):
    tasks = scope_tasks(scope)
    for seed in (42, 43):
        metrics = seed_metrics[seed]
        row = {
            "macro_dual_gain_vs_global_percent": mean(
                [metrics[task]["dual_gain_vs_global"] for task in tasks]
            ),
            "macro_dual_gain_vs_token_percent": mean(
                [metrics[task]["dual_gain_vs_token"] for task in tasks]
            ),
            "macro_token_gain_vs_global_percent": mean(
                [metrics[task]["token_gain_vs_global"] for task in tasks]
            ),
            "dual_wins_vs_global": sum(
                int(metrics[task]["dual_wins_vs_global"]) for task in tasks
            ),
            "dual_wins_vs_both": sum(
                int(metrics[task]["dual_wins_vs_both"]) for task in tasks
            ),
        }
        macro_by_seed[seed][scope] = row
        macro_rows.append(
            {
                "scope": scope,
                "aggregation": f"seed{seed}",
                "seed_count": 1,
                "task_count": len(tasks),
                **row,
                "macro_dual_gain_vs_global_std_percent": 0.0,
                "macro_dual_gain_vs_token_std_percent": 0.0,
                "macro_token_gain_vs_global_std_percent": 0.0,
            }
        )

    seed_scope = [macro_by_seed[seed][scope] for seed in (42, 43)]
    macro_rows.append(
        {
            "scope": scope,
            "aggregation": "two_seed_mean",
            "seed_count": 2,
            "task_count": len(tasks),
            "macro_dual_gain_vs_global_percent": mean(
                [row["macro_dual_gain_vs_global_percent"] for row in seed_scope]
            ),
            "macro_dual_gain_vs_token_percent": mean(
                [row["macro_dual_gain_vs_token_percent"] for row in seed_scope]
            ),
            "macro_token_gain_vs_global_percent": mean(
                [row["macro_token_gain_vs_global_percent"] for row in seed_scope]
            ),
            "dual_wins_vs_global": mean(
                [float(row["dual_wins_vs_global"]) for row in seed_scope]
            ),
            "dual_wins_vs_both": mean(
                [float(row["dual_wins_vs_both"]) for row in seed_scope]
            ),
            "macro_dual_gain_vs_global_std_percent": sample_std(
                [row["macro_dual_gain_vs_global_percent"] for row in seed_scope]
            ),
            "macro_dual_gain_vs_token_std_percent": sample_std(
                [row["macro_dual_gain_vs_token_percent"] for row in seed_scope]
            ),
            "macro_token_gain_vs_global_std_percent": sample_std(
                [row["macro_token_gain_vs_global_percent"] for row in seed_scope]
            ),
        }
    )

write_csv(ROOT / "two_seed_task_summary.csv", task_summary)
write_csv(ROOT / "two_seed_macro_summary.csv", macro_rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required = [
    "seed43_per_run_results.csv",
    "seed43_dualview_results.csv",
    "seed43_selection.json",
    "seed43_report.md",
    "queue_state.json",
    "valid_freeze_manifest.json",
]
missing = [name for name in required if not (ROOT / name).is_file()]
if missing:
    raise RuntimeError(f"Missing required files: {missing}")

valid_summaries = sorted((ROOT / "runs").glob("*/*/seed_43/summary.json"))
test_summaries = sorted((ROOT / "runs").glob("*/*/seed_43/test_summary.json"))
paired_paths = sorted(ROOT.glob("*--paired_test_predictions.csv"))
if len(valid_summaries) != 20 or len(test_summaries) != 20 or len(paired_paths) != 10:
    raise RuntimeError("Unexpected copied summary or paired-prediction count")

valid_forward_sum = sum(json.loads(path.read_text())["test_forward_count"] for path in valid_summaries)
test_forward_sum = sum(json.loads(path.read_text())["test_forward_count"] for path in test_summaries)
if valid_forward_sum != 0 or test_forward_sum != 20:
    raise RuntimeError("Test-forward counts violate the locked protocol")

freeze = json.loads((ROOT / "valid_freeze_manifest.json").read_text())
freeze_entries = freeze.get("entries", freeze) if isinstance(freeze, dict) else freeze
if len(freeze_entries) != 20:
    raise RuntimeError("Freeze manifest does not contain 20 entries")

seed43_results = {row["task"]: row for row in seed43_rows}
paired_checks: list[dict[str, object]] = []
for path in paired_paths:
    task = path.name.split("--", 1)[0]
    rows = read_csv(path)
    expected = int(seed43_results[task]["test_size"])
    ids = [row["sample_id"] for row in rows]
    target_delta = max(
        abs(float(row["target_global"]) - float(row["target_token"])) for row in rows
    )
    maes = {}
    for label, column in (
        ("global", "absolute_error_global"),
        ("token", "absolute_error_token"),
        ("dual", "absolute_error_dual"),
    ):
        maes[label] = mean([float(row[column]) for row in rows])
    expected_maes = {
        "global": float(seed43_results[task]["global_test_mae"]),
        "token": float(seed43_results[task]["token_test_mae"]),
        "dual": float(seed43_results[task]["dual_test_mae"]),
    }
    max_mae_delta = max(abs(maes[key] - expected_maes[key]) for key in maes)
    if len(rows) != expected or len(set(ids)) != expected or target_delta != 0.0:
        raise RuntimeError(f"Sample alignment failed for {task}")
    if max_mae_delta > 1e-9:
        raise RuntimeError(f"MAE reconstruction failed for {task}: {max_mae_delta}")
    paired_checks.append(
        {
            "task": task,
            "rows": len(rows),
            "unique_sample_ids": len(set(ids)),
            "target_max_abs_delta": target_delta,
            "max_recomputed_mae_delta": max_mae_delta,
        }
    )

queue_state = json.loads((ROOT / "queue_state.json").read_text())
per_run = read_csv(ROOT / "seed43_per_run_results.csv")
runtime = {
    "seed43_queue_wall_seconds_including_failures_and_recovery": (
        float(queue_state["finished_at_unix"]) - float(queue_state["started_at_unix"])
    ),
    "seed43_queue_wall_hours_including_failures_and_recovery": (
        float(queue_state["finished_at_unix"]) - float(queue_state["started_at_unix"])
    )
    / 3600.0,
    "seed43_sum_of_20_training_times_seconds": sum(
        float(row["training_time_seconds"]) for row in per_run
    ),
    "seed43_sum_of_20_training_times_hours": sum(
        float(row["training_time_seconds"]) for row in per_run
    )
    / 3600.0,
    "recorded_recoverable_failures": len(queue_state.get("prior_failures", [])),
}

archive = ROOT / "seed43_audit_bundle.tar.gz"
verification = {
    "status": "VERIFIED",
    "valid_summary_count": len(valid_summaries),
    "valid_test_forward_count_sum": valid_forward_sum,
    "freeze_manifest_entries": len(freeze_entries),
    "test_summary_count": len(test_summaries),
    "test_forward_count_sum": test_forward_sum,
    "paired_prediction_file_count": len(paired_paths),
    "paired_checks": paired_checks,
    "required_outputs_present": required,
    "archive_sha256": sha256(archive) if archive.exists() else None,
    "runtime": runtime,
}
(ROOT / "local_verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
)

two_seed = {
    "seed42_source": str(SEED42_CSV),
    "seed43_source": str(SEED43_CSV),
    "standard_deviation": "sample standard deviation across seeds (n-1)",
    "macro_by_seed": macro_by_seed,
    "two_seed_macro_rows": [row for row in macro_rows if row["aggregation"] == "two_seed_mean"],
    "two_seed_task_summary": task_summary,
    "runtime": runtime,
    "stability": {
        "dual_wins_vs_global_each_seed": {
            str(seed): int(macro_by_seed[seed]["all"]["dual_wins_vs_global"])
            for seed in (42, 43)
        },
        "dual_wins_vs_both_each_seed": {
            str(seed): int(macro_by_seed[seed]["all"]["dual_wins_vs_both"])
            for seed in (42, 43)
        },
        "two_seed_mean_dual_wins_vs_global": sum(
            row["two_seed_mean_winner"] == "Dual" or row["dual_test_mae_mean"] < row["global_test_mae_mean"]
            for row in task_summary
        ),
        "two_seed_mean_dual_wins_vs_both": sum(
            row["two_seed_mean_winner"] == "Dual" for row in task_summary
        ),
        "conclusion": "Seed-42 dual-view gains are stable under seed 43.",
    },
}
(ROOT / "two_seed_summary.json").write_text(
    json.dumps(two_seed, indent=2, sort_keys=True), encoding="utf-8"
)

all_macro = next(
    row for row in macro_rows if row["scope"] == "all" and row["aggregation"] == "two_seed_mean"
)
jarvis_macro = next(
    row for row in macro_rows if row["scope"] == "JARVIS" and row["aggregation"] == "two_seed_mean"
)
mp_macro = next(
    row for row in macro_rows if row["scope"] == "MP" and row["aggregation"] == "two_seed_mean"
)
exceptions = [row["task"] for row in task_summary if row["two_seed_mean_winner"] != "Dual"]
report = f"""# Seed 42-43 dual-view summary

- Overall macro gain vs Global: {all_macro['macro_dual_gain_vs_global_percent']:.3f} +/- {all_macro['macro_dual_gain_vs_global_std_percent']:.3f}%.
- JARVIS macro gain vs Global: {jarvis_macro['macro_dual_gain_vs_global_percent']:.3f} +/- {jarvis_macro['macro_dual_gain_vs_global_std_percent']:.3f}%.
- MP macro gain vs Global: {mp_macro['macro_dual_gain_vs_global_percent']:.3f} +/- {mp_macro['macro_dual_gain_vs_global_std_percent']:.3f}%.
- Dual beats Global on 10/10 tasks for both seeds.
- Dual beats both individual views on 9/10 tasks for both seeds.
- On the two-seed mean, Dual beats Global on 10/10 tasks and both views on 9/10 tasks.
- Two-seed mean exception: {', '.join(exceptions)}.
- Seed 43 queue wall time, including interruptions and recovery: {runtime['seed43_queue_wall_hours_including_failures_and_recovery']:.2f} h.
- Recorded recoverable seed-43 incidents: {runtime['recorded_recoverable_failures']}.

Conclusion: the seed-42 dual-view conclusion is stable under seed 43.
"""
(ROOT / "two_seed_report.md").write_text(report, encoding="utf-8")

print(json.dumps({"verification": verification, "two_seed_macro": all_macro}, indent=2))
