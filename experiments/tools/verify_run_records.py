"""Verify imported run evidence; no models or experiment inference are loaded."""

import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    experiments = root / "experiments"
    checked = set()
    for manifest in sorted((experiments / "manifests").glob("*_export.json")):
        for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            path = root / entry["path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"source hash mismatch: {entry['path']}")
            checked.add(entry["path"])
    secret = re.compile(
        r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----|"
        r"(?:password|passwd|api_key)\s*[:=]\s*['\"][^'\"]{6,}['\"])", re.I,
    )
    for name in checked:
        if secret.search((root / name).read_text(encoding="utf-8-sig")):
            raise ValueError(f"credential marker requires review: {name}")
    matched = list((experiments / "matched_text_ablation").glob("*/*/summary.json"))
    assert len(matched) == 18, "expected 18 matched validation summaries"
    values = {}
    for path in matched:
        data = json.loads(path.read_text())
        assert data["seed"] == 42 and data["test_forward_count"] == 0
        assert len(data["history"]) == 300
        assert path.with_name("run.log").is_file()
        values.setdefault(data["task"], {})[data["variant"]] = data["best_valid_mae"]
    assert len(values) == 6
    normalized = {variant: statistics.mean(row[variant] / row["CLS"] for row in values.values())
                  for variant in ("CLS", "MaskedMean", "CosineAttention")}
    # Published macros used rounded source MAEs; verify without rewriting either.
    assert abs(normalized["MaskedMean"] - 0.9979018957) < 1e-7
    assert abs(normalized["CosineAttention"] - 1.0290914366) < 1e-7
    pairs = []
    for seed, directory, result_file in (
        (42, experiments / "main_seed42/paired_test", root / "results/seed42/dualview_test_results.csv"),
        (43, experiments / "main_seed43", root / "results/seed43/seed43_dualview_results.csv"),
    ):
        with result_file.open(newline="", encoding="utf-8-sig") as handle:
            references = {row["task"]: row for row in csv.DictReader(handle)}
        files = list(directory.glob("*--paired_test_predictions.csv"))
        assert len(files) == 10
        for path in files:
            task = path.name.split("--", 1)[0]
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            assert len({row["sample_id"] for row in rows}) == len(rows)
            for row in rows:
                assert abs(float(row["target_global"]) - float(row["target_token"])) < 1e-6
                assert abs(float(row["prediction_dual"]) -
                           0.5 * (float(row["prediction_global"]) + float(row["prediction_token"]))) < 1e-6
            delta = max(abs(statistics.mean(abs(float(row[f"prediction_{view}"]) -
                       float(row["target_global"])) for row in rows) -
                       float(references[task][f"{view}_test_mae"])) for view in ("global", "token", "dual"))
            assert delta < 1e-6
            pairs.append({"seed": seed, "task": task, "rows": len(rows), "max_mae_delta": delta})
    summaries = list((experiments / "main_seed43/runs").glob("*/*/seed_43/summary.json"))
    tests = list((experiments / "main_seed43/runs").glob("*/*/seed_43/test_summary.json"))
    assert len(summaries) == len(tests) == 20
    assert sum(json.loads(path.read_text())["test_forward_count"] for path in tests) == 20
    freeze = json.loads((experiments / "main_seed43/valid_freeze_manifest.json").read_text())
    assert len(freeze["entries"]) == 20
    report = {"status": "VERIFIED", "date": "2026-10-14", "source_hashes_verified": len(checked),
              "matched_runs": len(matched), "matched_normalized_macro": normalized,
              "seed43_valid_summaries": len(summaries), "seed43_test_summaries": len(tests),
              "saved_test_pair_checks": pairs,
              "extensions": dict(Counter((root / name).suffix for name in checked)),
              "training_runs_started": 0, "test_evaluations_started": 0}
    (experiments / "manifests/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "saved_test_pair_checks"}, indent=2))


if __name__ == "__main__":
    main()
