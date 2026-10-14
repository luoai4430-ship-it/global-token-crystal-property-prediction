#!/usr/bin/env python3
"""Two-process validation-only queue for the GCTA fast screen."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/TextResidualSHFMat")
PYTHON = Path("/root/autodl-tmp/micromamba/envs/sftmat/bin/python")
SCRIPT = PROJECT / "scripts/run_gcta_fast_screen.py"
MODEL_CHECKPOINT = Path("/root/autodl-tmp/matbert-base-cased")
OUTPUT = Path("/root/gcta_fast_screen_seed42_20260902")
SCRATCH = Path("/dev/shm/gcta_fast_screen_seed42_20260902")
LOGS = OUTPUT / "logs"
STATE = OUTPUT / "queue_state.json"
SEED = 42
TASKS = ("Jarvis-Bandgap_MBJ", "MP-Bandgap")
VARIANTS = ("B1_NaiveConcat", "P2_GCTA")
MAX_TRAINING_PROCESSES = 2

sys.path.insert(0, str(PROJECT / "scripts"))
import run_final_10task_queue as cache_common  # noqa: E402

cache_common.OUTPUT_ROOT = OUTPUT
cache_common.SCRATCH_ROOT = SCRATCH
cache_common.STATE_PATH = STATE
cache_common.LOG_ROOT = LOGS / "cache"
cache_common.SCRIPT = SCRIPT
cache_common.PYTHON = PYTHON
cache_common.MODEL_CHECKPOINT = MODEL_CHECKPOINT


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_dir(task: str, variant: str) -> Path:
    return OUTPUT / "runs" / task / variant / f"seed_{SEED}"


def summary_path(task: str, variant: str) -> Path:
    return run_dir(task, variant) / "summary.json"


def completed_count() -> int:
    return len(list(OUTPUT.glob("runs/*/*/seed_42/summary.json")))


def read_state() -> dict[str, object]:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {
        "status": "RUNNING",
        "phase": "initializing",
        "seed": SEED,
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "expected_training_runs": len(TASKS) * len(VARIANTS),
        "completed_training_runs": 0,
        "test_forward_count": 0,
        "max_training_processes": MAX_TRAINING_PROCESSES,
        "active": [],
        "failed": [],
        "started_at_unix": time.time(),
    }


def update_state(**changes: object) -> None:
    state = read_state()
    state.update(changes)
    state["completed_training_runs"] = completed_count()
    state["test_forward_count"] = 0
    state["updated_at_unix"] = time.time()
    atomic_json(STATE, state)


def prepare(task: str) -> tuple[Path, Path]:
    update_state(
        status="RUNNING",
        phase="preparing",
        current_task=task,
        current_variant=None,
        active=[],
    )
    return cache_common.prepare_task(task, "valid")


def command_for(task: str, variant: str, token_cache: Path) -> list[str]:
    directory = run_dir(task, variant)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        str(PYTHON),
        str(SCRIPT),
        "train",
        "--task",
        task,
        "--variant",
        variant,
        "--seed",
        str(SEED),
        "--token_cache",
        str(token_cache),
        "--output_dir",
        str(directory),
        "--model_checkpoint",
        str(MODEL_CHECKPOINT),
        "--device",
        "cuda",
        "--num_workers",
        "2",
        "--epochs",
        "100",
        "--patience",
        "20",
    ]


def execute_variants(task: str, token_cache: Path) -> None:
    pending = [variant for variant in VARIANTS if not summary_path(task, variant).is_file()]
    active: list[dict[str, object]] = []
    while pending or active:
        while pending and len(active) < MAX_TRAINING_PROCESSES:
            variant = pending.pop(0)
            directory = run_dir(task, variant)
            if (directory / "test_forward_started.json").exists():
                raise PermissionError(f"Test marker exists for {task}/{variant}")
            log_path = LOGS / f"{task}_{variant}_seed{SEED}_valid.log"
            LOGS.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a", encoding="utf-8", buffering=1)
            command = command_for(task, variant, token_cache)
            log.write("COMMAND " + " ".join(command) + "\n")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active.append(
                {
                    "task": task,
                    "variant": variant,
                    "pid": process.pid,
                    "process": process,
                    "log_handle": log,
                    "log": str(log_path),
                    "started_at_unix": time.time(),
                }
            )
        public_active = [
            {
                "task": item["task"],
                "variant": item["variant"],
                "pid": item["pid"],
                "log": item["log"],
                "started_at_unix": item["started_at_unix"],
            }
            for item in active
        ]
        update_state(
            status="RUNNING",
            phase="training",
            current_task=task,
            current_variant=None,
            active=public_active,
        )
        time.sleep(15)
        remaining = []
        for item in active:
            process = item["process"]
            code = process.poll()
            if code is None:
                remaining.append(item)
                continue
            item["log_handle"].close()
            if code != 0:
                for sibling in active:
                    sibling_process = sibling["process"]
                    if sibling_process.poll() is None:
                        sibling_process.terminate()
                        sibling_process.wait(timeout=30)
                    if not sibling["log_handle"].closed:
                        sibling["log_handle"].close()
                raise RuntimeError(
                    f"{task}/{item['variant']} exited with {code}; see {item['log']}"
                )
            expected = summary_path(task, str(item["variant"]))
            if not expected.is_file():
                raise FileNotFoundError(expected)
        active = remaining


def run_checked(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def verify_outputs() -> None:
    expected = {
        "gcta_fast_screen.csv",
        "gcta_attention_diagnostics.csv",
        "gcta_query_controls.csv",
        "gcta_parameter_count.json",
        "gcta_report.md",
    }
    missing = [name for name in sorted(expected) if not (OUTPUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing GCTA outputs: {missing}")
    if completed_count() != 4:
        raise AssertionError(f"expected four completed runs, found {completed_count()}")
    for path in OUTPUT.glob("runs/*/*/seed_42/summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "valid_complete":
            raise AssertionError(f"incomplete summary: {path}")
        if int(payload.get("test_forward_count", -1)) != 0:
            raise AssertionError(f"nonzero Test count: {path}")
    forbidden_names = {
        "test_summary.json",
        "test_predictions.csv",
        "test_forward_started.json",
    }
    forbidden_artifacts = [
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.name.lower() in forbidden_names
    ]
    if forbidden_artifacts:
        raise AssertionError(f"Test prediction artifacts exist: {forbidden_artifacts}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if STATE.is_file() and read_state().get("status") == "FAILED":
        raise RuntimeError("refusing automatic restart of a failed GCTA queue")
    update_state(status="RUNNING", phase="valid_only", active=[])
    try:
        for task in TASKS:
            if all(summary_path(task, variant).is_file() for variant in VARIANTS):
                continue
            scratch, token_cache = prepare(task)
            try:
                execute_variants(task, token_cache)
            finally:
                if scratch.exists():
                    shutil.rmtree(scratch)
        if completed_count() != 4:
            raise AssertionError(f"expected four validation runs, found {completed_count()}")
        update_state(status="RUNNING", phase="summarizing", active=[])
        run_checked(
            [str(PYTHON), str(SCRIPT), "summarize", "--output_root", str(OUTPUT)],
            LOGS / "summarize.log",
        )
        verify_outputs()
        update_state(
            status="COMPLETE",
            phase="complete",
            current_task=None,
            current_variant=None,
            active=[],
            finished_at_unix=time.time(),
        )
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_unix": time.time(),
        }
        atomic_json(OUTPUT / "queue_failure.json", failure)
        update_state(status="FAILED", phase="failed", active=[], failed=[failure])
        raise


if __name__ == "__main__":
    main()
