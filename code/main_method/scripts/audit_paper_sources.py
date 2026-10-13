#!/usr/bin/env python3
"""Compare released source files with a read-only server snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github_ref", help="Fetched GitHub ref to compare, e.g. origin/main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((args.server_snapshot / "server_code_manifest.json").read_text())
    rows = []
    github_rows = []
    for source in sources:
        local = root / source["path"]
        if not local.is_file():
            continue
        raw = local.read_bytes()
        server = (args.server_snapshot / source["path"]).read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        server_text = server.decode("utf-8-sig").replace("\r\n", "\n")
        rows.append({
            "path": source["path"],
            "local_sha256": hashlib.sha256(raw).hexdigest(),
            "server_sha256": source["sha256"],
            "byte_equal": raw == server,
            "normalized_text_equal": text == server_text,
            "python_ast_equal": ast.dump(ast.parse(text)) == ast.dump(ast.parse(server_text)),
        })
        if args.github_ref:
            result = subprocess.run(
                ["git", "show", f"{args.github_ref}:code/main_method/{source['path']}"],
                cwd=root, capture_output=True,
            )
            github_rows.append({
                "path": source["path"],
                "status": "available" if result.returncode == 0 else "not_in_fetched_commit",
                "github_sha256": hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None,
                "github_server_byte_equal": result.stdout == server if result.returncode == 0 else None,
            })
    payload = {
        "paper": "paper/main_v24.pdf",
        "server": "connect.westb.seetacloud.com:36397",
        "server_root": "/root/autodl-tmp/TextResidualSHFMat",
        "checked_date": "2026-10-13",
        "comparisons": rows,
        "new_replication_adapters": [
            "scripts/run_paper_benchmark.py", "scripts/prepare_paper_inputs.py",
            "scripts/summarize_paper_predictions.py", "scripts/audit_paper_sources.py",
        ],
        "runtime_validation": "Not run: restored server lacks torch_geometric and a CUDA GPU",
        "training_runs_started": 0,
        "test_evaluations_started": 0,
    }
    if args.github_ref:
        commit = subprocess.check_output(
            ["git", "rev-parse", args.github_ref], cwd=root, text=True,
        ).strip()
        payload["github_fetched_commit"] = commit
        payload["github_comparisons"] = github_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "compared_files": len(rows),
        "byte_equal": sum(row["byte_equal"] for row in rows),
        "github_available": sum(row["status"] == "available" for row in github_rows),
        "github_server_byte_equal": sum(row["github_server_byte_equal"] is True for row in github_rows),
    }, indent=2))
    if not all(row["python_ast_equal"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
