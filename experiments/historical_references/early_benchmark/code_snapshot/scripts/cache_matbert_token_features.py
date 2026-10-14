#!/usr/bin/env python3
"""Cache frozen MatBERT token features as memory-mapped NumPy arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_descriptions(
    path: Path,
    id_column: str,
    text_column: str,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    sample_ids: list[str] = []
    texts: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = {id_column, text_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            sample_id = str(row[id_column]).strip()
            text = str(row[text_column]).strip()
            if not sample_id or not text:
                continue
            sample_ids.append(sample_id)
            texts.append(text)
            if limit is not None and len(sample_ids) >= limit:
                break
    if not sample_ids:
        raise ValueError(f"{path} contains no usable descriptions")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("description sample IDs are not unique")
    return sample_ids, texts


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cache_features(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0 or args.max_length <= 2:
        raise ValueError("batch_size must be positive and max_length must exceed 2")
    sample_ids, texts = load_descriptions(
        args.input, args.id_column, args.text_column, args.limit
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "sample_ids.json"
    features_path = args.output_dir / "token_features.npy"
    mask_path = args.output_dir / "attention_mask.npy"
    input_ids_path = args.output_dir / "input_ids.npy"
    metadata_path = args.output_dir / "metadata.json"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    hidden_dim = int(config.hidden_size)
    model_kwargs = {"add_pooling_layer": False} if config.model_type == "bert" else {}
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    expected_shape = (len(sample_ids), args.max_length, hidden_dim)
    completed_rows = 0
    if metadata_path.exists() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            tuple(metadata.get("feature_shape", ())) != expected_shape
            or int(metadata.get("max_length", -1)) != args.max_length
            or str(metadata.get("model_name")) != str(args.model_name)
        ):
            raise ValueError("existing cache metadata does not match this request")
        stored_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if stored_ids != sample_ids:
            raise ValueError("existing token cache sample IDs do not match source CSV")
        completed_rows = int(metadata.get("completed_rows", 0))
        features = np.lib.format.open_memmap(features_path, mode="r+")
        masks = np.lib.format.open_memmap(mask_path, mode="r+")
        input_ids = np.lib.format.open_memmap(input_ids_path, mode="r+")
    else:
        ids_path.write_text(
            json.dumps(sample_ids, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )
        masks = np.lib.format.open_memmap(
            mask_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(sample_ids), args.max_length),
        )
        input_ids = np.lib.format.open_memmap(
            input_ids_path,
            mode="w+",
            dtype=np.int32,
            shape=(len(sample_ids), args.max_length),
        )
        features[:] = 0
        masks[:] = 0
        input_ids[:] = 0

    metadata: dict[str, object] = {
        "status": "running",
        "format": "dense_memmap_v1",
        "source_csv": str(args.input.resolve()),
        "source_sha256": source_sha256(args.input),
        "id_column": args.id_column,
        "text_column": args.text_column,
        "model_name": str(args.model_name),
        "model_checkpoint": str(args.model_name.resolve()),
        "tokenizer_version": transformers.__version__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "num_samples": len(sample_ids),
        "max_length": args.max_length,
        "hidden_dim": hidden_dim,
        "hidden_size": hidden_dim,
        "feature_dtype": "float16",
        "feature_shape": list(expected_shape),
        "mask_shape": [len(sample_ids), args.max_length],
        "input_ids_shape": [len(sample_ids), args.max_length],
        "input_ids_dtype": "int32",
        "attention_mask_dtype": "uint8",
        "padding": "max_length",
        "truncation": True,
        "uses_last_hidden_state": True,
        "forbidden_global_pooling_used": False,
        "special_tokens_excluded": False,
        "completed_rows": completed_rows,
    }
    write_metadata(metadata_path, metadata)

    model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(device)
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for start in range(completed_rows, len(texts), args.batch_size):
            stop = min(start + args.batch_size, len(texts))
            encoded = tokenizer(
                texts[start:stop],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                hidden = model(**model_inputs).last_hidden_state
            features[start:stop] = hidden.detach().cpu().numpy().astype(np.float16)
            masks[start:stop] = encoded["attention_mask"].numpy().astype(np.uint8)
            input_ids[start:stop] = encoded["input_ids"].numpy().astype(np.int32)
            features.flush()
            masks.flush()
            input_ids.flush()
            metadata["completed_rows"] = stop
            write_metadata(metadata_path, metadata)
            if args.progress_every > 0 and (
                stop == len(texts) or stop % args.progress_every < args.batch_size
            ):
                print(f"cached {stop}/{len(texts)} token sequences", flush=True)

    metadata["status"] = "complete"
    metadata["completed_rows"] = len(sample_ids)
    metadata["cache_sha256"] = cache_sha256(
        [ids_path, features_path, mask_path, input_ids_path]
    )
    write_metadata(metadata_path, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model_name",
        type=Path,
        default=Path("/root/autodl-tmp/matbert-base-cased"),
    )
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress_every", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    metadata = cache_features(build_parser().parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
