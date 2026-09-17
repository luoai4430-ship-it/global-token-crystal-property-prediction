#!/usr/bin/env python3
"""Strict two-way queue for the final ten-task benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

from jarvis.core.atoms import Atoms


PROJECT_ROOT = Path("/root/autodl-tmp/TextResidualSHFMat")
SCRIPT = PROJECT_ROOT / "scripts/run_final_10task_benchmark.py"
CACHE_SCRIPT = PROJECT_ROOT / "scripts/cache_matbert_token_features.py"
PYTHON = Path("/root/autodl-tmp/micromamba/envs/sftmat/bin/python")
MODEL_CHECKPOINT = Path("/root/autodl-tmp/matbert-base-cased")
OUTPUT_ROOT = Path("/root/final_10task_benchmark_20260809_tau05")
SCRATCH_ROOT = Path("/dev/shm/final_10task_benchmark_current")
STATE_PATH = OUTPUT_ROOT / "parallel_queue_state.json"
LOG_ROOT = OUTPUT_ROOT / "logs"
SEEDS = (42, 7, 1234, 2025, 2026)
MODELS = ("original_sftmat", "ours")
TASKS = {
    "Jarvis-Bandgap_MBJ": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/mbj_bandgap/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/mbj_bandgap/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "Jarvis-Bandgap_OPT": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_bandgap/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_bandgap/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "Jarvis-FormationEnergy": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/formation_energy_peratom/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/formation_energy_peratom/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "Jarvis-TotalEnergy": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_total_energy/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/optb88vdw_total_energy/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "Jarvis-BulkModulusKv": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/bulk_modulus_kv/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/bulk_modulus_kv/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "Jarvis-ShearModulusGv": {
        "source": "/root/autodl-tmp/dataset1/dataset/jarvis/shear_modulus_gv/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/jarvis/shear_modulus_gv/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "MP-Bandgap": {
        "source": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-bandgap/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-bandgap/cif",
        "id_column": "id", "text_column": "text",
    },
    "MP-FormationEnergy": {
        "source": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-formation/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-formation/cif",
        "id_column": "id", "text_column": "text",
    },
    "MP-BulkModuli": {
        "source": "/root/autodl-tmp/dataset1/dataset/mp_2018_small/bulk/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/mp_2018_small/bulk/cif",
        "id_column": "Id", "text_column": "Description",
    },
    "MP-ShearModuli": {
        "source": "/root/autodl-tmp/dataset1/dataset/mp_2018_small/shear/description.csv",
        "cif_dir": "/root/autodl-tmp/dataset1/dataset/mp_2018_small/shear/cif",
        "id_column": "Id", "text_column": "Description",
    },
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_state(**changes) -> None:
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {
            "status": "RUNNING",
            "phase": "initializing",
            "completed_valid": 0,
            "completed_test": 0,
            "expected_valid": 100,
            "expected_test": 100,
            "active": [],
            "failed": [],
            "started_at_unix": time.time(),
        }
    state.update(changes)
    state["updated_at_unix"] = time.time()
    atomic_json(STATE_PATH, state)


def describe(cif_path: Path) -> tuple[str, str]:
    atoms = Atoms.from_cif(cif_path, use_cif2cell=False)
    composition = atoms.composition.reduced_formula
    counts = Counter(atoms.elements)
    species = ", ".join(f"{element}:{counts[element]}" for element in counts)
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    positions = "; ".join(
        f"{element}({coord[0]:.5f},{coord[1]:.5f},{coord[2]:.5f})"
        for element, coord in zip(atoms.elements, atoms.frac_coords)
    )
    text = (
        f"{composition} is a crystalline material with {atoms.num_atoms} atoms in "
        f"the unit cell. The lattice lengths are {lengths[0]:.5f}, "
        f"{lengths[1]:.5f}, and {lengths[2]:.5f} angstrom, and the lattice "
        f"angles are {angles[0]:.5f}, {angles[1]:.5f}, and {angles[2]:.5f} "
        f"degrees. The elemental counts are {species}. Fractional atomic "
        f"positions are {positions}."
    )
    return composition, text


def complete_description(task: str, scratch: Path) -> tuple[Path, dict[str, object]]:
    spec = TASKS[task]
    source = Path(spec["source"])
    cif_dir = Path(spec["cif_dir"])
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"{source} has no header")
    id_column = spec["id_column"]
    text_column = spec["text_column"]
    existing = {str(row[id_column]).strip(): row for row in rows}
    if len(existing) != len(rows):
        raise ValueError(f"{task} source contains duplicate IDs")
    expected: list[tuple[str, str]] = []
    with (cif_dir / "id_prop.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                expected.append((str(row[0]).strip(), str(row[1]).strip()))
    expected_ids = {sample_id for sample_id, _ in expected}
    if len(expected_ids) != len(expected):
        raise ValueError(f"{task} id_prop.csv contains duplicate IDs")
    output_rows = []
    generated_ids = []
    for sample_id, target in expected:
        row = existing.get(sample_id)
        if row is None or not str(row.get(text_column, "")).strip():
            cif_path = cif_dir / f"{sample_id}.cif"
            if not cif_path.is_file():
                raise FileNotFoundError(cif_path)
            composition, text = describe(cif_path)
            row = {field: "" for field in fieldnames}
            row[id_column] = sample_id
            row[text_column] = text
            for candidate in ("Composition", "composition"):
                if candidate in row:
                    row[candidate] = composition
            if "prop" in row:
                row["prop"] = target
            if "File_Name" in row:
                row["File_Name"] = f"generated_from_{sample_id}.cif"
            generated_ids.append(sample_id)
        output_rows.append(row)
    complete = scratch / "description_complete.csv"
    with complete.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "task": task,
        "source_csv": str(source.resolve()),
        "source_sha256": sha256(source),
        "complete_sha256": sha256(complete),
        "id_column": id_column,
        "text_column": text_column,
        "expected_samples": len(expected),
        "source_samples": len(rows),
        "generated_missing_count": len(generated_ids),
        "generated_missing_ids": generated_ids,
        "dropped_source_count": len(set(existing) - expected_ids),
        "complete_unique_coverage": len(output_rows) == len(expected_ids),
        "description_generator": "original source plus deterministic CIF fallback only for missing rows",
    }
    manifest_path = OUTPUT_ROOT / "source_manifests" / f"{task}.json"
    if manifest_path.is_file():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "source_sha256", "complete_sha256", "expected_samples",
            "generated_missing_ids", "id_column", "text_column",
        ):
            if frozen.get(key) != manifest.get(key):
                raise AssertionError(f"{task} description provenance changed at {key}")
    else:
        atomic_json(manifest_path, manifest)
    return complete, manifest


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def prepare_task(task: str, phase: str) -> tuple[Path, Path]:
    scratch = SCRATCH_ROOT / task
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    update_state(phase=f"{phase}_preparing", current_task=task, active=[])
    complete, manifest = complete_description(task, scratch)
    token_cache = scratch / "token_cache"
    cache_log = LOG_ROOT / f"{task}_{phase}_token_cache.log"
    run_checked(
        [
            str(PYTHON), str(CACHE_SCRIPT), "--input", str(complete),
            "--output_dir", str(token_cache), "--model_name", str(MODEL_CHECKPOINT),
            "--id_column", str(manifest["id_column"]),
            "--text_column", str(manifest["text_column"]),
            "--max_length", "128", "--batch_size", "64", "--device", "cuda",
            "--progress_every", "1024", "--overwrite",
        ],
        cache_log,
    )
    complete.unlink()
    preflight_log = LOG_ROOT / f"{task}_{phase}_preflight.log"
    run_checked(
        [
            str(PYTHON), str(SCRIPT), "preflight", "--task", task,
            "--token_cache", str(token_cache), "--scratch", str(scratch),
            "--output_root", str(OUTPUT_ROOT),
            "--model_checkpoint", str(MODEL_CHECKPOINT), "--device", "cuda",
        ],
        preflight_log,
    )
    payload = json.loads((OUTPUT_ROOT / "preflight" / f"{task}.json").read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise AssertionError(f"{task} preflight did not pass")
    return scratch, token_cache


def completed_valid_count() -> int:
    return len(list(OUTPUT_ROOT.glob("runs/*/*/seed_*/summary.json")))


def completed_test_count() -> int:
    return len(list(OUTPUT_ROOT.glob("runs/*/*/seed_*/test_summary.json")))


def execute_pairs(task: str, token_cache: Path, phase: str) -> None:
    pending = []
    for seed in SEEDS:
        for model_name in MODELS:
            run_dir = OUTPUT_ROOT / "runs" / task / model_name / f"seed_{seed}"
            completion = run_dir / ("summary.json" if phase == "valid" else "test_summary.json")
            if completion.is_file():
                continue
            if phase == "valid" and (run_dir / "test_forward_started.json").exists():
                raise AssertionError("Test marker exists during Valid phase")
            pending.append((model_name, seed, run_dir))
    active: list[dict[str, object]] = []
    while pending or active:
        while pending and len(active) < 2:
            model_name, seed, run_dir = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = LOG_ROOT / f"{task}_{model_name}_seed{seed}_{phase}.log"
            if phase == "valid":
                command = [
                    str(PYTHON), str(SCRIPT), "train", "--task", task,
                    "--model_name", model_name, "--seed", str(seed),
                    "--token_cache", str(token_cache), "--output_dir", str(run_dir),
                    "--model_checkpoint", str(MODEL_CHECKPOINT), "--device", "cuda",
                    "--num_workers", "0", "--epochs", "300", "--batch_size", "32",
                ]
            else:
                command = [
                    str(PYTHON), str(SCRIPT), "test", "--task", task,
                    "--model_name", model_name, "--seed", str(seed),
                    "--token_cache", str(token_cache), "--output_root", str(OUTPUT_ROOT),
                    "--model_checkpoint", str(MODEL_CHECKPOINT), "--device", "cuda",
                    "--num_workers", "0",
                ]
            handle = log_path.open("a", encoding="utf-8")
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            active.append(
                {
                    "process": process, "handle": handle, "model_name": model_name,
                    "seed": seed, "run_dir": run_dir, "log_path": log_path,
                }
            )
        update_state(
            phase=phase,
            current_task=task,
            completed_valid=completed_valid_count(),
            completed_test=completed_test_count(),
            active=[
                {"task": task, "model_name": item["model_name"], "seed": item["seed"],
                 "pid": item["process"].pid, "phase": phase}
                for item in active
            ],
        )
        time.sleep(15)
        still_active = []
        for item in active:
            code = item["process"].poll()
            if code is None:
                still_active.append(item)
                continue
            item["handle"].close()
            if code != 0:
                raise RuntimeError(
                    f"{phase} failed for {task}/{item['model_name']}/seed{item['seed']} "
                    f"with exit {code}; see {item['log_path']}"
                )
            completion = item["run_dir"] / ("summary.json" if phase == "valid" else "test_summary.json")
            if not completion.is_file():
                raise FileNotFoundError(completion)
        active = still_active


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        old = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if old.get("status") == "FAILED":
            raise RuntimeError("refusing to auto-recover a failed formal queue")
    update_state(status="RUNNING", phase="valid", current_task=None)
    try:
        for task in TASKS:
            task_complete = all(
                (OUTPUT_ROOT / "runs" / task / model_name / f"seed_{seed}" / "summary.json").is_file()
                for model_name in MODELS for seed in SEEDS
            )
            if task_complete:
                continue
            scratch, token_cache = prepare_task(task, "valid")
            execute_pairs(task, token_cache, "valid")
            shutil.rmtree(scratch)
        if completed_valid_count() != 100:
            raise AssertionError(f"expected 100 Valid summaries, got {completed_valid_count()}")
        update_state(phase="freezing_valid", active=[])
        run_checked(
            [str(PYTHON), str(SCRIPT), "freeze", "--output_root", str(OUTPUT_ROOT)],
            LOG_ROOT / "freeze_valid.log",
        )
        update_state(phase="test", valid_selection_frozen=True)
        for task in TASKS:
            task_complete = all(
                (OUTPUT_ROOT / "runs" / task / model_name / f"seed_{seed}" / "test_summary.json").is_file()
                for model_name in MODELS for seed in SEEDS
            )
            if task_complete:
                continue
            scratch, token_cache = prepare_task(task, "test")
            execute_pairs(task, token_cache, "test")
            shutil.rmtree(scratch)
        if completed_test_count() != 100:
            raise AssertionError(f"expected 100 Test summaries, got {completed_test_count()}")
        update_state(phase="summarizing", active=[])
        run_checked(
            [str(PYTHON), str(SCRIPT), "summarize", "--output_root", str(OUTPUT_ROOT)],
            LOG_ROOT / "summarize.log",
        )
        selection = json.loads((OUTPUT_ROOT / "final_selection.json").read_text(encoding="utf-8"))
        status = selection["status"]
        (OUTPUT_ROOT / status).touch()
        update_state(
            status=status,
            phase="complete",
            current_task=None,
            active=[],
            completed_valid=100,
            completed_test=100,
            finished_at_unix=time.time(),
        )
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at_unix": time.time(),
        }
        atomic_json(OUTPUT_ROOT / "queue_failure.json", failure)
        update_state(status="FAILED", active=[], failed=[failure])
        raise


if __name__ == "__main__":
    main()
