#!/usr/bin/env python3
"""Validate the mounted task-source layout without running an experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path, id_column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path}: empty file")
    # The server's id_prop.csv files are headerless two-column files. Accept a
    # headed export as well, because both forms occur in public task snapshots.
    if id_column in rows[0]:
        id_index = rows[0].index(id_column)
        rows = rows[1:]
    else:
        id_index = 0
    ids = [str(row[id_index]).strip() for row in rows if row]
    if any(not item for item in ids):
        raise ValueError(f"{path}: empty sample ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample IDs")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/task_dataset_manifest.csv"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    checks = []
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for row in manifest_rows:
        task_root = args.root / row["source_relative_path"]
        id_prop = args.root / row["id_prop_file"]
        description = args.root / row["description_file"]
        ids = read_ids(id_prop, row["id_column"])
        cif_count = sum(1 for item in (task_root / "cif").iterdir() if item.is_file()) - 1
        result = {
            "task": row["task"],
            "id_prop_exists": id_prop.is_file(),
            "description_exists": description.is_file(),
            "id_prop_rows": len(ids),
            "manifest_rows": int(row["total_rows"]),
            "cif_files": cif_count,
            "manifest_cif_files": int(row["cif_files"]),
            "id_prop_sha256_matches": sha256(id_prop) == row["id_prop_sha256"],
            "description_sha256_matches": sha256(description) == row["description_sha256"],
            "ok": (
                len(ids) == int(row["total_rows"])
                and cif_count == int(row["cif_files"])
                and sha256(id_prop) == row["id_prop_sha256"]
                and sha256(description) == row["description_sha256"]
            ),
        }
        checks.append(result)

    payload = {"root": str(args.root), "split_seed": 123, "checks": checks}
    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["ok"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
