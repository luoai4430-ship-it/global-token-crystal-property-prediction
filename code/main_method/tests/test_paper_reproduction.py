"""Saved-file pairing tests, without loading models or experiment datasets."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_paper_predictions.py"
spec = importlib.util.spec_from_file_location("summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

wrapper_spec = importlib.util.spec_from_file_location(
    "wrapper", SCRIPT.with_name("run_paper_benchmark.py")
)
wrapper = importlib.util.module_from_spec(wrapper_spec)
wrapper_spec.loader.exec_module(wrapper)


class PairedWrapperTests(unittest.TestCase):
    def test_adapts_each_seed_and_data_root_without_changing_models(self):
        @dataclass(frozen=True)
        class Task:
            source_dir: str

        for seed in (42, 43):
            old_root = Path("/old/data")
            new_root = Path("/new/data")
            captured = {}
            benchmark = SimpleNamespace(
                SEEDS=(43,), MODELS=("original_sftmat", "ours"),
                DATA_ROOT=old_root, TASKS={"task": Task(str(old_root / "task"))},
            )

            def fake_main():
                captured["argv"] = list(sys.argv)

            benchmark.main = fake_main
            with patch.dict(sys.modules, {"run_seed43_benchmark": benchmark}), patch.object(
                sys, "argv", ["wrapper", "--training_seed", str(seed),
                              "--data_root", str(new_root), "freeze", "--output_root", "runs"],
            ):
                wrapper.main()
            self.assertEqual(benchmark.SEEDS, (seed,))
            self.assertEqual(benchmark.MODELS, ("original_sftmat", "ours"))
            self.assertEqual(benchmark.TASKS["task"].source_dir, str(new_root / "task"))
            self.assertEqual(captured["argv"][1:], ["freeze", "--output_root", "runs"])

    def test_rejects_historical_summarizer(self):
        with patch.object(sys, "argv", ["wrapper", "--training_seed", "42", "summarize"]):
            with self.assertRaises(SystemExit) as error:
                wrapper.main()
        self.assertEqual(error.exception.code, 2)


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for task in summary.TASKS:
            for model in ("original_sftmat", "ours"):
                run = self.root / "runs" / task / model / "seed_42"
                run.mkdir(parents=True)
                (run / "summary.json").write_text(json.dumps({"test_forward_count": 0}))
                (run / "test_summary.json").write_text(json.dumps({"test_forward_count": 1, "selected_by": "Valid only"}))
                rows = [("a", 1.0, 0.0), ("b", 2.0, 1.0)]
                if model == "ours":
                    rows = [("b", 2.0, 3.0), ("a", 1.0, 2.0)]
                with (run / "test_predictions.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_id", "target", "prediction"])
                    writer.writerows(rows)
        entries = [{"task": task, "model_name": model, "seed": 42}
                   for task in summary.TASKS for model in ("original_sftmat", "ours")]
        (self.root / "valid_freeze_manifest.json").write_text(json.dumps({
            "status": "valid_selection_frozen", "entries": entries,
        }))

    def tearDown(self):
        self.temp.cleanup()

    def run_summary(self):
        return subprocess.run([
            sys.executable, str(SCRIPT), "--run", f"42={self.root}",
            "--output_dir", str(self.root / "output"),
        ], text=True, capture_output=True)

    def test_aligns_reordered_ids_and_computes_fixed_mean(self):
        result = self.run_summary()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads((self.root / "output/paper_summary.json").read_text())
        self.assertEqual(output["runs"]["42"]["macro_gain_percent"], 100.0)
        self.assertEqual(output["runs"]["42"]["wins_vs_both"], 10)

    def test_rejects_changed_targets(self):
        path = self.root / "runs" / summary.TASKS[0] / "ours/seed_42/test_predictions.csv"
        path.write_text("sample_id,target,prediction\nb,2,3\na,9,2\n")
        result = self.run_summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("targets differ", result.stderr)

    def test_rejects_incomplete_freeze(self):
        path = self.root / "valid_freeze_manifest.json"
        data = json.loads(path.read_text())
        data["entries"].pop()
        path.write_text(json.dumps(data))
        result = self.run_summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete validation freeze", result.stderr)

    def test_rejects_duplicate_ids(self):
        path = self.root / "duplicate.csv"
        path.write_text("sample_id,target,prediction\na,1,2\na,1,2\n")
        with self.assertRaises(ValueError):
            summary.read_predictions(path)


if __name__ == "__main__":
    unittest.main()
