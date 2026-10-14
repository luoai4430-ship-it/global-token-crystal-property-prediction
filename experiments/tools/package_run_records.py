"""Package existing manuscript run records without loading models or weights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path


SERVER = {
    "main_seed42/paired_test": "/root/dualview_test_seed42_20260826",
    "main_seed42/token_training": "/root/final_10task_ours_seed42_20260810_tau05",
    "main_seed42/token_training_4090d": "/root/final_10task_ours_seed42_20260810_tau05_gpu4090d",
    "main_seed42/global_completion": "/root/e1_missing_original_seed42_20260822",
    "historical_references/early_benchmark": "/root/final_10task_benchmark_20260809_tau05",
    "historical_references/paper_tables": "/root/autodl-tmp/TextResidualSHFMat/paper_tables",
    "historical_references/strict_mbj": "/root/autodl-tmp/TextResidualSHFMat/results/strict_mbj",
    "matched_text_ablation": "/root/autodl-tmp/TextResidualSHFMat/matched_ablation_runs",
    "latent/pvin_fast": "/root/pvin_fast_screen_seed42_20260908",
    "latent/pvin_e2": "/root/pvin_stage_e2_seed42_20260909",
    "latent/pvin_diagnostics": "/root/pvin_10task_interaction_diagnostic_seed42_20260910_full",
    "latent/gt_pvin_e3": "/root/gt_pvin_stage_e3_seed42_20260910",
    "latent/gt_pvin_confirmation": "/root/gt_pvin_overnight_confirmation_20260911",
    "message/sec_sft": "/root/sec_sft_stage_s1_seed42_20260911",
    "message/residual_recovery": "/root/sec_residual_mp_mechanical_seed42_20260912",
    "message/residual_preservation": "/root/sec_residual_positive_preservation_seed42_20260912",
    "shared_backbone/six_task": "/root/sbdvf_6task_seed42_20260912",
    "shared_backbone/extra_mp_bulk": "/root/sbdvf_extra_mpbulk_seed42_20260913",
    "shared_backbone/extra_remaining": "/root/sbdvf_extra_remaining_seed42_20260913",
}
LOCAL = {
    "main_seed43": "dualview_seed43_20260830",
    "latent/tagif": "tagif_fast_screen_seed42_20260902",
    "latent/gcta": "gcta_fast_screen_seed42_20260902",
    "latent/gctr": "gctr_fast_screen_seed42_20260906",
    "validation_complementarity_seed42": "error_complementarity_seed42_20260822",
    "validation_prediction_mean_seed42": "e1_simple_ensemble_seed42_20260822",
}
SERVER_SCRIPTS = {
    "matched_text_ablation/source/matched_text_ablation_runner.py": "/root/autodl-tmp/TextResidualSHFMat/matched_text_ablation_runner.py",
    "matched_text_ablation/source/prepare_matched_inputs.py": "/root/autodl-tmp/TextResidualSHFMat/prepare_matched_inputs.py",
    "matched_text_ablation/source/matched_ablation_queue.sh": "/root/autodl-tmp/TextResidualSHFMat/matched_ablation_queue.sh",
    "matched_text_ablation/queue.log": "/root/matched_ablation_queue.log",
    "matched_text_ablation/queue_launcher.log": "/root/matched_ablation_queue_launcher.log",
    "matched_text_ablation/recovery.log": "/root/matched_ablation_recovery.log",
    "matched_text_ablation/source/matched_ablation_recovery.sh": "/root/matched_ablation_recovery.sh",
}
SUPPLEMENT_FILES = (
    "same_view_ensemble_control.csv", "two_run_error_complementarity.csv",
    "attention_ablation.csv", "literature_benchmark_provenance.csv",
    "benchmark_split_comparison.csv", "numerical_provenance_audit.csv",
    "sbdvf_extra_tasks_baseline_audit.md", "review_20260914/test_same_view_recomputed.csv",
)
TEXT_EXTENSIONS = {".json", ".csv", ".tsv", ".log", ".out", ".txt", ".md",
                   ".yaml", ".yml", ".sh", ".py", ".diff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("server", "local", "server_scripts", "supplements"), required=True)
    parser.add_argument("--local_audit_root", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.profile in ("local", "supplements") and args.local_audit_root is None:
        parser.error("local profile requires --local_audit_root")
    if args.profile == "server":
        roots = SERVER
    elif args.profile == "server_scripts":
        roots = SERVER_SCRIPTS
    elif args.profile == "supplements":
        roots = {f"paper_analysis/{name}": str(args.local_audit_root.parent / name)
                 for name in SUPPLEMENT_FILES}
    else:
        roots = {label: str(args.local_audit_root / name) for label, name in LOCAL.items()}
    manifest = {"date": "2026-10-14", "profile": args.profile,
                "training_runs_started": 0, "test_evaluations_started": 0,
                "roots": [], "files": [], "excluded": []}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        for label, name in roots.items():
            root = Path(name)
            record = {"destination": label, "source_root": name,
                      "exists": root.exists(), "exported_files": 0}
            manifest["roots"].append(record)
            if not root.exists():
                continue
            walking = [(str(root.parent), [], [root.name])] if root.is_file() else os.walk(root, followlinks=False)
            for base, dirs, files in walking:
                dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}
                                 and not (Path(base) / d).is_symlink())
                for name in sorted(files):
                    source = Path(base) / name
                    destination = Path("experiments") / label
                    if root.is_dir():
                        destination /= source.relative_to(root)
                    reason = None
                    if source.is_symlink():
                        reason = "symbolic_link_not_followed"
                    elif source.suffix not in TEXT_EXTENSIONS:
                        reason = "weights_cache_or_nontext_artifact"
                    elif source.stat().st_size > 32 * 1024 * 1024:
                        reason = "text_file_over_32_MiB"
                    if reason:
                        manifest["excluded"].append({"source": str(source), "reason": reason})
                        continue
                    raw = source.read_bytes()
                    try:
                        raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        manifest["excluded"].append({"source": str(source), "reason": "non_utf8"})
                        continue
                    if b"\0" in raw:
                        manifest["excluded"].append({"source": str(source), "reason": "binary_content"})
                        continue
                    item = tarfile.TarInfo(destination.as_posix())
                    item.size = len(raw)
                    item.mode = 0o644
                    archive.addfile(item, io.BytesIO(raw))
                    manifest["files"].append({"path": destination.as_posix(),
                        "source": str(source), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
                    record["exported_files"] += 1
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        item = tarfile.TarInfo(f"experiments/manifests/{args.profile}_export.json")
        item.size = len(raw)
        item.mode = 0o644
        archive.addfile(item, io.BytesIO(raw))
    print(json.dumps({"profile": args.profile, "files": len(manifest["files"]),
                      "text_bytes": sum(row["bytes"] for row in manifest["files"]),
                      "excluded": len(manifest["excluded"]), "roots": manifest["roots"]}, indent=2))


if __name__ == "__main__":
    main()
