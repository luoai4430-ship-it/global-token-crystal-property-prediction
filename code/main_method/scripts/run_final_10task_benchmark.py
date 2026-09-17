#!/usr/bin/env python3
"""Ours-only seed-42 benchmark against frozen Original SFTMAT results."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.config import Config as data_config
from data.config import CrystalDataset, JarvisTarget, MPTarget
from data.dataset import get_dataloader
from data.token_sequence import (
    TokenFeatureStore,
    TokenGraphDataset,
    collate_token_graph_samples,
)
from model.net import SFTGNN, SFTGNNMultimodal
from model.true_token_text_residual import TrueTokenTextResidual


SEEDS = (42,)
MODELS = ("ours",)
EPOCHS = 300
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.5
MAX_LENGTH = 128
TRAINING_DTYPE = torch.bfloat16
ORIGINAL_PARAMETER_COUNT = 2_661_154
OURS_PARAMETER_COUNT = 2_800_739
FORMAL_DATA_LOADER_WORKERS = 2
PIN_MEMORY = False
SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

GRAPH_EQUIVALENT_TASK = {
    "MP-FormationEnergy": "MP-Bandgap",
    "MP-ShearModuli": "MP-BulkModuli",
}


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    property_name: str
    target: object
    source_dir: str
    id_column: str
    text_column: str
    unit: str

    @property
    def cif_dir(self) -> Path:
        return Path(self.source_dir) / "cif"

    @property
    def description_csv(self) -> Path:
        return Path(self.source_dir) / "description.csv"


DATA_ROOT = Path("/root/autodl-tmp/dataset1/dataset")
TASKS: dict[str, TaskSpec] = {
    "Jarvis-Bandgap_MBJ": TaskSpec(
        "JARVIS", "Bandgap_MBJ", JarvisTarget.Bandgap_MBJ,
        str(DATA_ROOT / "jarvis/mbj_bandgap"), "Id", "Description", "eV"
    ),
    "Jarvis-Bandgap_OPT": TaskSpec(
        "JARVIS", "Bandgap_OPT", JarvisTarget.Bandgap_OPT,
        str(DATA_ROOT / "jarvis/optb88vdw_bandgap"), "Id", "Description", "eV"
    ),
    "Jarvis-FormationEnergy": TaskSpec(
        "JARVIS", "FormationEnergy", JarvisTarget.FormationEnergy,
        str(DATA_ROOT / "jarvis/formation_energy_peratom"), "Id", "Description", "eV/atom"
    ),
    "Jarvis-TotalEnergy": TaskSpec(
        "JARVIS", "TotalEnergy", JarvisTarget.TotalEnergy,
        str(DATA_ROOT / "jarvis/optb88vdw_total_energy"), "Id", "Description", "eV/atom"
    ),
    "Jarvis-BulkModulusKv": TaskSpec(
        "JARVIS", "BulkModulusKv", JarvisTarget.BulkModulusKv,
        str(DATA_ROOT / "jarvis/bulk_modulus_kv"), "Id", "Description", "GPa"
    ),
    "Jarvis-ShearModulusGv": TaskSpec(
        "JARVIS", "ShearModulusGv", JarvisTarget.ShearModulusGv,
        str(DATA_ROOT / "jarvis/shear_modulus_gv"), "Id", "Description", "GPa"
    ),
    "MP-Bandgap": TaskSpec(
        "MP", "Bandgap", MPTarget.Bandgap,
        str(DATA_ROOT / "mp_2018/MP-bandgap"), "id", "text", "eV"
    ),
    "MP-FormationEnergy": TaskSpec(
        "MP", "FormationEnergy", MPTarget.FormationEnergy,
        str(DATA_ROOT / "mp_2018/MP-formation"), "id", "text", "eV/atom"
    ),
    "MP-BulkModuli": TaskSpec(
        "MP", "BulkModuli", MPTarget.BulkModuli,
        str(DATA_ROOT / "mp_2018_small/bulk"), "Id", "Description", "log10(GPa)"
    ),
    "MP-ShearModuli": TaskSpec(
        "MP", "ShearModuli", MPTarget.ShearModuli,
        str(DATA_ROOT / "mp_2018_small/shear"), "Id", "Description", "log10(GPa)"
    ),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_payload() -> dict[str, object]:
    return {
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "tasks": list(TASKS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "lr": LEARNING_RATE,
        "max_lr": MAX_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "data_loader_workers": FORMAL_DATA_LOADER_WORKERS,
        "prefetch_factor": 1,
        "pin_memory": PIN_MEMORY,
        "torch_multiprocessing_sharing_strategy": SHARING_STRATEGY,
        "drop_last": False,
        "checkpoint_selection": "best Valid MAE only",
        "bandgap_nonnegative_clipping": True,
        "training_dtype": "bfloat16",
        "matbert_max_length": MAX_LENGTH,
        "ours_temperature": TEMPERATURE,
        "ours_loss": "MAE(final,target)+0.2*MAE(graph,target)",
        "original_loss": "MAE(property,target)",
    }


def protocol_sha256() -> str:
    encoded = json.dumps(protocol_payload(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_task(task: str) -> None:
    spec = TASKS[task]
    data_config.dataset_name = (
        CrystalDataset.Jarvis if spec.dataset == "JARVIS" else CrystalDataset.MP
    )
    data_config.target = spec.target
    data_config.batch_size = BATCH_SIZE
    data_config.num_workers = 0
    data_config.use_cif_csv = True
    data_config.cif_dir = str(spec.cif_dir)
    data_config.use_multimodal = False
    data_config.text_emb_path = ""
    data_config.split_seed = 123
    data_config.normalize = True


def graph_cache_names(task: str) -> tuple[Path, list[Path]]:
    configure_task(task)
    raw = PROJECT_ROOT / data_config.raw_data_cache_path()
    prefix = data_config.getDatasetWholeName()
    processed = [
        PROJECT_ROOT / "Dataset/processed" / f"{prefix}-{split}-CGCNN.pt"
        for split in ("train", "valid", "test")
    ]
    return raw, processed


def existing_processed_cache(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.parent.glob(path.stem + "-multimodal-*.pt"))
    preferred = [candidate for candidate in matches if "MatBert_embeddings" in candidate.name]
    source = (preferred or matches)[0] if matches else None
    if source is None or not source.is_file():
        raise FileNotFoundError(f"no structurally equivalent graph cache for {path}")
    return source


def target_by_id(task: str) -> tuple[list[str], dict[str, float]]:
    spec = TASKS[task]
    rows: list[tuple[str, float]] = []
    with (spec.cif_dir / "id_prop.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = float(row[1])
            if task in {"MP-BulkModuli", "MP-ShearModuli"}:
                value = math.log10(value)
            rows.append((str(row[0]).strip(), value))
    ids = [sample_id for sample_id, _ in rows]
    values = dict(rows)
    if len(values) != len(rows):
        raise ValueError(f"{task} id_prop.csv contains duplicate IDs")
    return ids, values


def materialize_retargeted_graph_caches(task: str, scratch_graph: Path) -> dict[str, object] | None:
    """Reuse identical MP structures while replacing only property targets."""
    source_task = GRAPH_EQUIVALENT_TASK.get(task)
    if source_task is None:
        return None
    target_raw, target_processed = graph_cache_names(task)
    if target_raw.is_file() and all(path.is_file() for path in target_processed):
        return None
    source_raw, source_processed = graph_cache_names(source_task)
    if not source_raw.is_file():
        raise FileNotFoundError(source_raw)
    source_processed = [existing_processed_cache(path) for path in source_processed]
    ordered_ids, targets = target_by_id(task)
    source_ids, _ = target_by_id(source_task)
    if ordered_ids != source_ids:
        raise AssertionError(f"{task} and {source_task} do not have identical ordered IDs")

    shuffled_ids = list(ordered_ids)
    random.Random(123).shuffle(shuffled_ids)
    split_lengths: list[int] = []
    split_cids: list[list[str]] = []
    for source in source_processed:
        data, _ = torch.load(source, map_location="cpu", weights_only=False)
        cids = [str(value) for value in data.cid]
        split_lengths.append(len(cids))
        split_cids.append(cids)
        del data
        gc.collect()
    offsets = np.cumsum([0, *split_lengths]).tolist()
    for index, cids in enumerate(split_cids):
        if shuffled_ids[offsets[index]:offsets[index + 1]] != cids:
            raise AssertionError(f"{task} split {index} differs from {source_task}")

    train_values = torch.tensor([targets[cid] for cid in split_cids[0]], dtype=torch.float32)
    mean = float(train_values.mean())
    std = float(train_values.std())
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise FloatingPointError(f"invalid target normalization for {task}")

    raw_target = scratch_graph / target_raw.name
    lightweight_raw = [
        Data(y=torch.tensor(value, dtype=torch.float32), cid=int(sample_id))
        for sample_id, value in ((sample_id, targets[sample_id]) for sample_id in ordered_ids)
    ]
    torch.save(lightweight_raw, raw_target)
    del lightweight_raw
    gc.collect()

    generated = []
    for source, target_path, cids in zip(source_processed, target_processed, split_cids):
        data, slices = torch.load(source, map_location="cpu", weights_only=False)
        data.y = torch.tensor(
            [(targets[cid] - mean) / std for cid in cids], dtype=torch.float32
        )
        generated_path = scratch_graph / target_path.name
        torch.save((data, slices), generated_path)
        generated.append(str(generated_path))
        del data, slices
        gc.collect()
    return {
        "source_task": source_task,
        "target_task": task,
        "ordered_id_match": True,
        "split_id_match": True,
        "target_mean": mean,
        "target_std": std,
        "raw_cache": str(raw_target),
        "processed_caches": generated,
    }


def ensure_graph_cache_paths(task: str, scratch: Path) -> dict[str, object]:
    raw, processed = graph_cache_names(task)
    graph_root_env = os.environ.get("SFTMAT_GRAPH_CACHE_ROOT")
    scratch_graph = (Path(graph_root_env) / task) if graph_root_env else (scratch / "graph")
    scratch_graph.mkdir(parents=True, exist_ok=True)
    retargeted = materialize_retargeted_graph_caches(task, scratch_graph)
    aliases: list[dict[str, str]] = []
    for plain in processed:
        if plain.is_file():
            aliases.append({"path": str(plain), "source": "existing_plain"})
            continue
        matches = sorted(plain.parent.glob(plain.stem + "-multimodal-*.pt"))
        preferred = [path for path in matches if "MatBert_embeddings" in path.name]
        source = (preferred or matches)[0] if matches else None
        if plain.is_symlink():
            plain.unlink()
        if source is not None and source.is_file():
            plain.symlink_to(source.name)
            aliases.append({"path": str(plain), "source": str(source)})
        else:
            target = scratch_graph / plain.name
            plain.symlink_to(target)
            aliases.append({"path": str(plain), "source": str(target)})
    if not raw.is_file():
        if raw.is_symlink():
            raw.unlink()
        target = scratch_graph / raw.name
        raw.symlink_to(target)
        aliases.append({"path": str(raw), "source": str(target)})
    return {"task": task, "aliases": aliases, "retargeted": retargeted}


def load_graph_splits(task: str):
    configure_task(task)
    train_loader, valid_loader, test_loader, mean, std = get_dataloader()
    if train_loader.drop_last is not True:
        # The historical loader is only used to materialize datasets. The
        # benchmark creates its own loaders below with drop_last=False.
        raise AssertionError("unexpected project loader protocol")
    if valid_loader.drop_last or test_loader.drop_last:
        raise AssertionError("project Valid/Test loaders must not drop samples")
    return (
        train_loader.dataset,
        valid_loader.dataset,
        test_loader.dataset,
        float(mean),
        float(std),
    )


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int):
    if workers not in (0, FORMAL_DATA_LOADER_WORKERS):
        raise ValueError("unsupported DataLoader worker count")
    generator = torch.Generator().manual_seed(seed)
    worker_options = (
        {"persistent_workers": True, "prefetch_factor": 1}
        if workers > 0 else {}
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_token_graph_samples,
        generator=generator,
        **worker_options,
    )


def special_token_ids(checkpoint: Path) -> tuple[int, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    values = (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
    if any(value is None for value in values) or len(set(values)) != 3:
        raise ValueError(f"invalid PAD/CLS/SEP IDs: {values}")
    return tuple(int(value) for value in values)


class OriginalSFTMATExact(nn.Module):
    """Input adapter around the project's exact SFTGNNMultimodal class."""

    def __init__(self) -> None:
        super().__init__()
        backbone = SFTGNN(
            num_layers=5,
            in_dim=data_config.getAtomDim(),
            node_feature=128,
            dist_dim=64,
            sh_dim=64,
            use_sh=True,
            use_dist=True,
        )
        self.original = SFTGNNMultimodal(
            sftgnn=backbone,
            crystal_dim=128,
            latent_dim=128,
            text_in_dim=768,
            text_hidden_dim=128,
            text_out_dim=64,
            property_fusion="concat",
        )

    def forward(self, graph, text_tokens, attention_mask, input_ids=None):
        del attention_mask, input_ids
        graph.text_emb = text_tokens[:, 0, :].float()
        output = self.original(graph)
        return {"final_prediction": output["property_pred"]}


def build_model(model_name: str, excluded_ids: tuple[int, int, int]) -> nn.Module:
    if model_name == "original_sftmat":
        model = OriginalSFTMATExact()
        expected = ORIGINAL_PARAMETER_COUNT
    elif model_name == "ours":
        backbone = SFTGNN(
            num_layers=5,
            in_dim=data_config.getAtomDim(),
            node_feature=128,
            dist_dim=64,
            sh_dim=64,
            use_sh=True,
            use_dist=True,
        )
        model = TrueTokenTextResidual(
            backbone,
            control="C1",
            graph_dim=128,
            token_dim=768,
            attention_dim=128,
            residual_hidden_dim=256,
            temperature=TEMPERATURE,
            special_token_ids=excluded_ids,
        )
        expected = OURS_PARAMETER_COUNT
    else:
        raise ValueError(model_name)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise AssertionError(f"{model_name} parameter count {count} != {expected}")
    return model


def physical_outputs(
    model_name: str,
    output: dict[str, torch.Tensor],
    mean: float,
    std: float,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    final = output["final_prediction"].float() * std + mean
    graph = None
    residual = None
    if model_name == "ours":
        graph = output["graph_prediction"].float() * std + mean
        residual = output["residual"].float() * std
    if "Bandgap" in task:
        final = final.clamp_min(0.0)
        if graph is not None:
            graph = graph.clamp_min(0.0)
    return final, graph, residual


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def evaluate(
    model: nn.Module,
    model_name: str,
    dataset: TokenGraphDataset,
    task: str,
    mean: float,
    std: float,
    device: torch.device,
    seed: int,
    workers: int,
    collect: bool,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    loader = make_loader(dataset, EVAL_BATCH_SIZE, False, seed, workers)
    if loader.drop_last:
        raise AssertionError("evaluation loader drops samples")
    model.eval()
    abs_sum = 0.0
    graph_abs_sum = 0.0
    residual_abs_sum = 0.0
    entropy_sum = 0.0
    effective_sum = 0.0
    max_weights: list[float] = []
    special_max = 0.0
    padding_max = 0.0
    count = 0
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=TRAINING_DTYPE,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch.graph,
                    batch.text_tokens,
                    batch.attention_mask,
                    input_ids=batch.input_ids,
                )
            target = batch.graph.y.reshape(-1).float() * std + mean
            final, graph, residual = physical_outputs(
                model_name, output, mean, std, task
            )
            errors = (final - target).abs()
            abs_sum += float(errors.sum().cpu())
            batch_count = int(target.numel())
            count += batch_count
            if graph is not None and residual is not None:
                graph_abs_sum += float((graph - target).abs().sum().cpu())
                residual_abs_sum += float(residual.abs().sum().cpu())
                entropy_sum += float(output["attention_entropy"].sum().cpu())
                effective_sum += float(output["effective_token_count"].sum().cpu())
                max_weights.extend(
                    output["attention_max_weight"].detach().float().cpu().tolist()
                )
                special_max = max(
                    special_max,
                    float(output["special_token_attention_mass"].max().cpu()),
                )
                padding_max = max(
                    padding_max,
                    float(output["padding_attention_mass"].max().cpu()),
                )
            if collect:
                targets = target.cpu().tolist()
                predictions = final.cpu().tolist()
                graph_values = graph.cpu().tolist() if graph is not None else [None] * batch_count
                residual_values = residual.cpu().tolist() if residual is not None else [None] * batch_count
                for index, sample_id in enumerate(batch.sample_ids):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "target": targets[index],
                            "prediction": predictions[index],
                            "absolute_error": abs(predictions[index] - targets[index]),
                            "graph_prediction": graph_values[index],
                            "residual": residual_values[index],
                        }
                    )
    if count != len(dataset):
        raise AssertionError(f"evaluation covered {count}/{len(dataset)} samples")
    metrics = {
        "mae": abs_sum / count,
        "sample_count": count,
        "graph_mae": graph_abs_sum / count if model_name == "ours" else float("nan"),
        "mean_abs_residual": residual_abs_sum / count if model_name == "ours" else float("nan"),
        "attention_entropy": entropy_sum / count if model_name == "ours" else float("nan"),
        "effective_token_count": effective_sum / count if model_name == "ours" else float("nan"),
        "mean_max_attention": float(np.mean(max_weights)) if max_weights else float("nan"),
        "median_max_attention": float(np.median(max_weights)) if max_weights else float("nan"),
        "special_token_attention_max": special_max if model_name == "ours" else float("nan"),
        "padding_attention_max": padding_max if model_name == "ours" else float("nan"),
    }
    frame = pd.DataFrame(records) if collect else None
    return metrics, frame


def preflight_task(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal benchmark preflight requires a BF16 CUDA device")
    seed_everything(SEEDS[0])
    aliases = ensure_graph_cache_paths(args.task, args.scratch)
    train_graphs, valid_graphs, test_graphs, mean, std = load_graph_splits(args.task)
    store = TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    split_graphs = {
        "train": train_graphs,
        "valid": valid_graphs,
        "test": test_graphs,
    }
    datasets = {
        split: TokenGraphDataset(graphs, store, split=split)
        for split, graphs in split_graphs.items()
    }
    if sum(len(dataset) for dataset in datasets.values()) != len(store.sample_ids):
        raise AssertionError("token cache must exactly equal the union of graph splits")
    union_ids = set().union(*(set(dataset.sample_ids) for dataset in datasets.values()))
    if union_ids != set(store.sample_ids):
        raise AssertionError("graph and token-cache ID sets differ")
    if sum(len(dataset.sample_ids) for dataset in datasets.values()) != len(union_ids):
        raise AssertionError("graph splits overlap")

    loader = make_loader(
        datasets["valid"], min(4, len(datasets["valid"])), False, SEEDS[0], 0
    )
    if loader.drop_last:
        raise AssertionError("preflight loader drops samples")
    batch = next(iter(loader))
    if tuple(batch.text_tokens.shape[1:]) != (MAX_LENGTH, 768):
        raise AssertionError(f"invalid token batch shape: {tuple(batch.text_tokens.shape)}")
    excluded_ids = special_token_ids(args.model_checkpoint)
    device = torch.device(args.device)
    batch = batch.to(device)
    model_checks: dict[str, object] = {}
    for model_name in MODELS:
        model = build_model(model_name, excluded_ids).to(device).eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=TRAINING_DTYPE,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch.graph,
                batch.text_tokens,
                batch.attention_mask,
                input_ids=batch.input_ids,
            )
        prediction = output["final_prediction"]
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(f"{model_name} produced non-finite predictions")
        check: dict[str, object] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "prediction_shape": list(prediction.shape),
            "prediction_finite": True,
            "autocast_dtype": "bfloat16",
        }
        if model_name == "ours":
            eligible = output["attention_eligible_mask"]
            q_after = output["q_norm_after_normalization"].float()
            k_after = output["k_norm_after_normalization"].float()[eligible]
            q_deviation = float((q_after - 1.0).abs().max().cpu())
            k_deviation = float((k_after - 1.0).abs().max().cpu())
            special_mass = float(output["special_token_attention_mass"].max().cpu())
            padding_mass = float(output["padding_attention_mass"].max().cpu())
            effective_min = float(output["effective_token_count"].min().cpu())
            if q_deviation > 1e-4 or k_deviation > 1e-4:
                raise AssertionError("cosine q/k normalization is not approximately one")
            if special_mass != 0.0 or padding_mass != 0.0:
                raise AssertionError("special or padding tokens received attention")
            if effective_min <= 1.0:
                raise AssertionError("attention collapsed during preflight")
            check.update(
                {
                    "temperature": TEMPERATURE,
                    "q_norm_after_max_deviation": q_deviation,
                    "k_norm_after_max_deviation": k_deviation,
                    "special_token_attention_max": special_mass,
                    "padding_attention_max": padding_mass,
                    "effective_token_count_min": effective_min,
                }
            )
        model_checks[model_name] = check
        del model
        torch.cuda.empty_cache()

    payload = {
        "status": "passed",
        "task": args.task,
        "dataset": TASKS[args.task].dataset,
        "property": TASKS[args.task].property_name,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "graph_total": sum(len(dataset) for dataset in datasets.values()),
        "token_cache_rows": len(store.sample_ids),
        "token_cache_shape": list(store.features.shape),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "graph_target_mean": mean,
        "graph_target_std": std,
        "drop_last": False,
        "data_loader_workers": FORMAL_DATA_LOADER_WORKERS,
        "pin_memory": PIN_MEMORY,
        "torch_multiprocessing_sharing_strategy": SHARING_STRATEGY,
        "aliases": aliases,
        "model_checks": model_checks,
        "protocol": protocol_payload(),
        "protocol_sha256": protocol_sha256(),
    }
    atomic_json(args.output_root / "preflight" / f"{args.task}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def train_run(args: argparse.Namespace) -> None:
    if args.seed not in SEEDS or args.model_name not in MODELS:
        raise ValueError("run is outside the locked benchmark matrix")
    if args.epochs != EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("epochs and batch size are locked")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal benchmark requires BF16 support")
    seed_everything(args.seed)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("formal run summary already exists")
    train_graphs, valid_graphs, _, mean, std = load_graph_splits(args.task)
    store = TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    train_dataset = TokenGraphDataset(train_graphs, store, split="train")
    valid_dataset = TokenGraphDataset(valid_graphs, store, split="valid")
    train_loader = make_loader(
        train_dataset, BATCH_SIZE, True, args.seed, args.num_workers
    )
    if train_loader.drop_last:
        raise AssertionError("formal training must use drop_last=False")
    device = torch.device(args.device)
    excluded_ids = special_token_ids(args.model_checkpoint)
    model = build_model(args.model_name, excluded_ids).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )
    criterion = nn.L1Loss()
    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=TRAINING_DTYPE,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch.graph,
                    batch.text_tokens,
                    batch.attention_mask,
                    input_ids=batch.input_ids,
                )
                target = batch.graph.y.reshape(-1).float() * std + mean
                final, graph, _ = physical_outputs(
                    args.model_name, output, mean, std, args.task
                )
                loss = criterion(final, target)
                if args.model_name == "ours":
                    loss = loss + 0.2 * criterion(graph, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            n = int(target.numel())
            loss_sum += float(loss.detach().cpu()) * n
            sample_count += n
        if sample_count != len(train_dataset):
            raise AssertionError(f"epoch covered {sample_count}/{len(train_dataset)}")
        valid_metrics, _ = evaluate(
            model, args.model_name, valid_dataset, args.task,
            mean, std, device, args.seed, args.num_workers, False
        )
        valid_mae = float(valid_metrics["mae"])
        if valid_mae < best_mae:
            best_mae = valid_mae
            best_epoch = epoch + 1
            best_state = cpu_state_dict(model)
        print(
            f"[{args.task}:{args.model_name}:seed{args.seed}] "
            f"epoch={epoch + 1}/{EPOCHS} "
            f"loss={loss_sum / sample_count:.8f} "
            f"valid_mae={valid_mae:.8f} best={best_mae:.8f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    valid_metrics, valid_frame = evaluate(
        model, args.model_name, valid_dataset, args.task,
        mean, std, device, args.seed, args.num_workers, True
    )
    if args.model_name == "ours":
        if valid_metrics["special_token_attention_max"] != 0.0:
            raise AssertionError("special tokens received attention")
        if valid_metrics["padding_attention_max"] != 0.0:
            raise AssertionError("padding received attention")
        if valid_metrics["effective_token_count"] <= 1.0:
            raise AssertionError("attention collapsed to one effective token")
    checkpoint = {
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "best_valid_epoch": best_epoch,
        "best_valid_mae": valid_metrics["mae"],
        "model_state_dict": best_state,
        "mean": mean,
        "std": std,
        "parameter_count": parameter_count,
        "protocol_sha256": protocol_sha256(),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "selection_metric": "best Valid MAE only",
        "test_forward_count": 0,
    }
    checkpoint_path = args.output_dir / "best_valid.pt"
    torch.save(checkpoint, checkpoint_path)
    reload_diagnostics = verify_checkpoint_reload(
        model, checkpoint_path, args.model_name, excluded_ids,
        valid_dataset, args.seed, device,
    )
    valid_frame.to_csv(args.output_dir / "valid_predictions.csv", index=False)
    summary = {
        "status": "valid_complete",
        "task": args.task,
        "dataset": TASKS[args.task].dataset,
        "property": TASKS[args.task].property_name,
        "unit": TASKS[args.task].unit,
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": EPOCHS,
        "best_valid_epoch": best_epoch,
        "valid_mae": valid_metrics["mae"],
        "graph_valid_mae": valid_metrics["graph_mae"],
        "mean_abs_residual": valid_metrics["mean_abs_residual"],
        "attention_entropy": valid_metrics["attention_entropy"],
        "effective_token_count": valid_metrics["effective_token_count"],
        "mean_max_attention": valid_metrics["mean_max_attention"],
        "median_max_attention": valid_metrics["median_max_attention"],
        "special_token_attention_max": valid_metrics["special_token_attention_max"],
        "padding_attention_max": valid_metrics["padding_attention_max"],
        "parameter_count": parameter_count,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "training_time_seconds": time.perf_counter() - started,
        "training_autocast_dtype": "bfloat16",
        "drop_last": False,
        "data_loader_workers": args.num_workers,
        "pin_memory": PIN_MEMORY,
        "torch_multiprocessing_sharing_strategy": SHARING_STRATEGY,
        "bandgap_nonnegative_clipping": "Bandgap" in args.task,
        "checkpoint_reload_passed": True,
        **reload_diagnostics,
        "checkpoint_sha256": sha256(checkpoint_path),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def verify_checkpoint_reload(
    reference_model: nn.Module,
    checkpoint_path: Path,
    model_name: str,
    excluded_ids: tuple[int, int, int],
    valid_dataset: TokenGraphDataset,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    """Verify exact serialized state and BF16-equivalent CUDA predictions."""
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = build_model(model_name, excluded_ids).to(device)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    restored_state = {name: value.detach().cpu() for name, value in restored.state_dict().items()}
    state_exact = (
        restored_state.keys() == loaded["model_state_dict"].keys()
        and all(
            torch.equal(restored_state[name], loaded["model_state_dict"][name])
            for name in restored_state
        )
    )
    if not state_exact:
        raise AssertionError("checkpoint reload changed serialized parameter tensors")
    first = next(iter(make_loader(
        valid_dataset, min(4, len(valid_dataset)), False, seed, 0
    ))).to(device)
    reference_model.eval()
    restored.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=TRAINING_DTYPE,
        enabled=device.type == "cuda",
    ):
        reference = reference_model(
            first.graph, first.text_tokens, first.attention_mask, input_ids=first.input_ids
        )["final_prediction"]
        reloaded = restored(
            first.graph, first.text_tokens, first.attention_mask, input_ids=first.input_ids
        )["final_prediction"]
    reload_max_abs = float((reference - reloaded).abs().max().cpu())
    bf16_tolerance = float(4 * torch.finfo(torch.bfloat16).eps)
    if not torch.allclose(
        reference, reloaded, rtol=bf16_tolerance, atol=bf16_tolerance
    ):
        raise AssertionError(
            f"checkpoint reload predictions exceed BF16 tolerance: {reload_max_abs}"
        )
    return {
        "checkpoint_reload_state_exact": True,
        "checkpoint_reload_max_abs": reload_max_abs,
        "checkpoint_reload_rtol": bf16_tolerance,
        "checkpoint_reload_atol": bf16_tolerance,
    }


def recover_valid(args: argparse.Namespace) -> None:
    """Finish artifacts for a 300-epoch run stopped only by the old reload guard."""
    if args.seed not in SEEDS or args.model_name not in MODELS:
        raise ValueError("run is outside the locked benchmark matrix")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("formal run summary already exists")
    checkpoint_path = args.output_dir / "best_valid.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_metadata = {
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
    }
    for name, expected in expected_metadata.items():
        if checkpoint.get(name) != expected:
            raise AssertionError(f"checkpoint metadata mismatch at {name}")
    seed_everything(args.seed)
    train_graphs, valid_graphs, _, mean, std = load_graph_splits(args.task)
    if not math.isclose(float(checkpoint["mean"]), mean, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError("checkpoint target mean changed")
    if not math.isclose(float(checkpoint["std"]), std, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError("checkpoint target std changed")
    store = TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    if store.metadata["cache_sha256"] != checkpoint["token_cache_sha256"]:
        raise AssertionError("regenerated token cache differs from checkpoint cache")
    valid_dataset = TokenGraphDataset(valid_graphs, store, split="valid")
    device = torch.device(args.device)
    excluded_ids = special_token_ids(args.model_checkpoint)
    model = build_model(args.model_name, excluded_ids).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    valid_metrics, valid_frame = evaluate(
        model, args.model_name, valid_dataset, args.task,
        mean, std, device, args.seed, args.num_workers, True,
    )
    if abs(float(checkpoint["best_valid_mae"]) - valid_metrics["mae"]) > 1e-4:
        raise AssertionError("recovered Valid MAE differs from saved best checkpoint")
    if valid_metrics["special_token_attention_max"] != 0.0:
        raise AssertionError("special tokens received attention")
    if valid_metrics["padding_attention_max"] != 0.0:
        raise AssertionError("padding received attention")
    if valid_metrics["effective_token_count"] <= 1.0:
        raise AssertionError("attention collapsed to one effective token")
    reload_diagnostics = verify_checkpoint_reload(
        model, checkpoint_path, args.model_name, excluded_ids,
        valid_dataset, args.seed, device,
    )
    valid_frame.to_csv(args.output_dir / "valid_predictions.csv", index=False)
    summary = {
        "status": "valid_complete",
        "task": args.task,
        "dataset": TASKS[args.task].dataset,
        "property": TASKS[args.task].property_name,
        "unit": TASKS[args.task].unit,
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": EPOCHS,
        "best_valid_epoch": int(checkpoint["best_valid_epoch"]),
        "valid_mae": valid_metrics["mae"],
        "graph_valid_mae": valid_metrics["graph_mae"],
        "mean_abs_residual": valid_metrics["mean_abs_residual"],
        "attention_entropy": valid_metrics["attention_entropy"],
        "effective_token_count": valid_metrics["effective_token_count"],
        "mean_max_attention": valid_metrics["mean_max_attention"],
        "median_max_attention": valid_metrics["median_max_attention"],
        "special_token_attention_max": valid_metrics["special_token_attention_max"],
        "padding_attention_max": valid_metrics["padding_attention_max"],
        "parameter_count": int(checkpoint["parameter_count"]),
        "train_size": len(train_graphs),
        "valid_size": len(valid_dataset),
        "training_time_seconds": args.training_time_seconds,
        "training_autocast_dtype": "bfloat16",
        "drop_last": False,
        "data_loader_workers": args.num_workers,
        "pin_memory": PIN_MEMORY,
        "torch_multiprocessing_sharing_strategy": SHARING_STRATEGY,
        "bandgap_nonnegative_clipping": "Bandgap" in args.task,
        "checkpoint_reload_passed": True,
        **reload_diagnostics,
        "checkpoint_sha256": sha256(checkpoint_path),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
        "recovered_after_reload_guard_failure": True,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def freeze_valid(args: argparse.Namespace) -> None:
    expected = {
        (task, model_name, seed)
        for task in TASKS for model_name in MODELS for seed in SEEDS
    }
    entries = []
    actual = set()
    for path in sorted(args.output_root.glob("runs/*/*/seed_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = (summary["task"], summary["model_name"], int(summary["seed"]))
        actual.add(key)
        checkpoint = path.parent / "best_valid.pt"
        if summary["status"] != "valid_complete" or summary["test_forward_count"] != 0:
            raise AssertionError(f"run is not Valid-only complete: {key}")
        if summary["epochs"] != EPOCHS or summary["drop_last"] is not False:
            raise AssertionError(f"protocol mismatch: {key}")
        if (
            summary["data_loader_workers"] != FORMAL_DATA_LOADER_WORKERS
            or summary["pin_memory"] is not PIN_MEMORY
            or summary["torch_multiprocessing_sharing_strategy"] != SHARING_STRATEGY
        ):
            raise AssertionError(f"DataLoader protocol mismatch: {key}")
        if summary["protocol_sha256"] != protocol_sha256():
            raise AssertionError(f"protocol hash mismatch: {key}")
        if sha256(checkpoint) != summary["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {key}")
        entries.append(
            {
                "task": key[0],
                "model_name": key[1],
                "seed": key[2],
                "best_valid_epoch": summary["best_valid_epoch"],
                "valid_mae": summary["valid_mae"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "token_cache_sha256": summary["token_cache_sha256"],
            }
        )
    expected_count = len(expected)
    if actual != expected or len(entries) != expected_count:
        raise AssertionError(
            f"cannot freeze: expected {expected_count} exact runs, got {len(entries)}; "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    if list(args.output_root.glob("runs/*/*/seed_*/test_summary.json")):
        raise AssertionError("Test output exists before Valid freeze")
    manifest = {
        "status": "valid_selection_frozen",
        "protocol": protocol_payload(),
        "protocol_sha256": protocol_sha256(),
        "entry_count": len(entries),
        "entries": entries,
        "test_forward_count": 0,
    }
    atomic_json(args.output_root / "valid_freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def frozen_entry(output_root: Path, task: str, model_name: str, seed: int):
    manifest_path = output_root / "valid_freeze_manifest.json"
    if not manifest_path.is_file():
        raise PermissionError("Test is forbidden before all Valid selections are frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "valid_selection_frozen":
        raise PermissionError("invalid freeze manifest")
    matches = [
        entry for entry in manifest["entries"]
        if entry["task"] == task
        and entry["model_name"] == model_name
        and int(entry["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError("frozen run entry is missing or duplicated")
    return matches[0]


def test_once(args: argparse.Namespace) -> None:
    run_dir = args.output_root / "runs" / args.task / args.model_name / f"seed_{args.seed}"
    result_path = run_dir / "test_summary.json"
    marker_path = run_dir / "test_forward_started.json"
    if result_path.exists() or marker_path.exists():
        raise PermissionError("Test has already started for this frozen checkpoint")
    entry = frozen_entry(args.output_root, args.task, args.model_name, args.seed)
    checkpoint_path = Path(entry["checkpoint"])
    if sha256(checkpoint_path) != entry["checkpoint_sha256"]:
        raise AssertionError("frozen checkpoint changed before Test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != protocol_sha256():
        raise AssertionError("checkpoint protocol changed")
    _, _, test_graphs, mean, std = load_graph_splits(args.task)
    store = TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    if store.metadata["cache_sha256"] != entry["token_cache_sha256"]:
        raise AssertionError("regenerated token cache differs from frozen Valid cache")
    test_dataset = TokenGraphDataset(test_graphs, store, split="test")
    device = torch.device(args.device)
    excluded_ids = special_token_ids(args.model_checkpoint)
    model = build_model(args.model_name, excluded_ids).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    atomic_json(
        marker_path,
        {
            "status": "test_forward_started",
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "allowed_forward_count": 1,
        },
    )
    metrics, frame = evaluate(
        model, args.model_name, test_dataset, args.task,
        mean, std, device, args.seed, args.num_workers, True
    )
    frame.to_csv(run_dir / "test_predictions.csv", index=False)
    result = {
        "status": "test_complete",
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "best_valid_epoch": entry["best_valid_epoch"],
        "valid_mae": entry["valid_mae"],
        "test_mae": metrics["mae"],
        "test_size": len(test_dataset),
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "token_cache_sha256": store.metadata["cache_sha256"],
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 1,
        "selected_by": "Valid only",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def summarize(args: argparse.Namespace) -> None:
    ours_rows = []
    expected = {
        (task, model_name, seed)
        for task in TASKS for model_name in MODELS for seed in SEEDS
    }
    actual = set()
    for summary_path in sorted(args.output_root.glob("runs/*/*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test = json.loads((summary_path.parent / "test_summary.json").read_text(encoding="utf-8"))
        key = (summary["task"], summary["model_name"], int(summary["seed"]))
        actual.add(key)
        if test["test_forward_count"] != 1 or test["selected_by"] != "Valid only":
            raise AssertionError(f"invalid Test protocol: {key}")
        ours_rows.append(
            {
                **{name: summary[name] for name in (
                    "task", "dataset", "property", "unit", "model_name", "seed",
                    "epochs", "best_valid_epoch", "valid_mae", "parameter_count",
                    "train_size", "valid_size", "training_time_seconds",
                    "training_autocast_dtype", "drop_last",
                    "data_loader_workers", "pin_memory",
                    "torch_multiprocessing_sharing_strategy",
                    "bandgap_nonnegative_clipping", "checkpoint_sha256",
                    "token_cache_sha256", "protocol_sha256"
                )},
                "test_mae": test["test_mae"],
                "test_size": test["test_size"],
                "test_forward_count": test["test_forward_count"],
            }
        )
    if actual != expected or len(ours_rows) != len(expected):
        raise AssertionError("final summary requires exactly ten Ours seed-42 runs")
    ours_frame = pd.DataFrame(ours_rows).sort_values(["task"])
    baseline = pd.read_csv(args.baseline_csv)
    if set(baseline.task) != set(TASKS) or len(baseline) != len(TASKS):
        raise AssertionError("frozen Original SFTMAT baseline must contain ten exact tasks")
    if set(baseline.baseline_seed.astype(int)) != {42}:
        raise AssertionError("frozen Original SFTMAT baseline is not seed 42")
    if not bool((baseline.frozen_test_mae.astype(float) > 0).all()):
        raise AssertionError("frozen baseline contains invalid Test MAE")
    baseline_copy = args.output_root / "original_sftmat_frozen_baseline.csv"
    shutil.copy2(args.baseline_csv, baseline_copy)

    combined_rows = []
    for row in baseline.itertuples(index=False):
        combined_rows.append(
            {
                "task": row.task, "dataset": row.dataset,
                "property": row.property, "unit": row.unit,
                "model_name": "original_sftmat", "seed": 42,
                "epochs": 300, "best_valid_epoch": float("nan"),
                "valid_mae": row.valid_mae, "test_mae": row.frozen_test_mae,
                "parameter_count": ORIGINAL_PARAMETER_COUNT,
                "source": "frozen paper result",
                "test_forward_count_in_this_benchmark": 0,
            }
        )
    for row in ours_frame.to_dict(orient="records"):
        combined_rows.append(
            {
                **row,
                "source": "new formal Ours run",
                "test_forward_count_in_this_benchmark": 1,
            }
        )
    pd.DataFrame(combined_rows).sort_values(["task", "model_name"]).to_csv(
        args.output_root / "final_10task_per_seed.csv", index=False
    )
    valid_rows = []
    test_rows = []
    main_rows = []
    relative_test: dict[str, float] = {}
    relative_valid: dict[str, float] = {}
    seed_wins: dict[str, int] = {}
    for task, spec in TASKS.items():
        ours = ours_frame[ours_frame.task == task].iloc[0]
        base = baseline[baseline.task == task].iloc[0]
        base_valid = float(base.valid_mae) if pd.notna(base.valid_mae) else float("nan")
        base_test = float(base.frozen_test_mae)
        ours_valid = float(ours.valid_mae)
        ours_test = float(ours.test_mae)
        if math.isfinite(base_valid):
            relative_valid[task] = (base_valid - ours_valid) / base_valid * 100
        test_gain = (base_test - ours_test) / base_test * 100
        relative_test[task] = test_gain
        wins = int(ours_test < base_test)
        seed_wins[task] = wins
        for model_name, valid_mae in (
            ("original_sftmat", base_valid), ("ours", ours_valid)
        ):
            valid_rows.append(
                {
                    "task": task, "dataset": spec.dataset,
                    "property": spec.property_name, "unit": spec.unit,
                    "model_name": model_name, "seed": 42,
                    "valid_mae": valid_mae,
                    "source": "frozen paper result" if model_name == "original_sftmat" else "new formal run",
                }
            )
        for model_name, test_mae in (
            ("original_sftmat", base_test), ("ours", ours_test)
        ):
            test_rows.append(
                {
                    "task": task, "dataset": spec.dataset,
                    "property": spec.property_name, "unit": spec.unit,
                    "model_name": model_name, "seed": 42,
                    "test_mae": test_mae,
                    "source": "frozen paper result" if model_name == "original_sftmat" else "new formal run",
                }
            )
        main_rows.append(
            {
                "Dataset": spec.dataset,
                "Property": spec.property_name,
                "Unit": spec.unit,
                "Original SFTMAT Test MAE": base_test,
                "Ours Test MAE": ours_test,
                "Improvement percent": test_gain,
                "Improved Seeds": f"{wins}/1",
                "Winner": "Ours" if test_gain > 0 else "Original SFTMAT",
            }
        )
    pd.DataFrame(valid_rows).to_csv(
        args.output_root / "final_10task_valid_summary.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(
        args.output_root / "final_10task_test_summary.csv", index=False
    )
    pd.DataFrame(main_rows).to_csv(
        args.output_root / "final_10task_main_table.csv", index=False
    )
    jarvis_tasks = [task for task, spec in TASKS.items() if spec.dataset == "JARVIS"]
    mp_tasks = [task for task, spec in TASKS.items() if spec.dataset == "MP"]
    mean = lambda values: float(np.mean(list(values)))
    wins_total = sum(relative_test[task] > 0 for task in TASKS)
    wins_jarvis = sum(relative_test[task] > 0 for task in jarvis_tasks)
    wins_mp = sum(relative_test[task] > 0 for task in mp_tasks)
    macro = mean(relative_test.values())
    jarvis_macro = mean(relative_test[task] for task in jarvis_tasks)
    mp_macro = mean(relative_test[task] for task in mp_tasks)
    degraded = [task for task in TASKS if relative_test[task] < -1.0]
    if wins_total >= 8 and jarvis_macro > 0 and mp_macro > 0 and len(degraded) <= 1:
        status = "STRONG_SFTMAT_IMPROVEMENT"
    elif wins_total >= 6 and jarvis_macro > 0 and mp_macro > 0:
        status = "PARTIAL_SFTMAT_IMPROVEMENT"
    else:
        status = "NO_SFTMAT_IMPROVEMENT"
    statistics = {
        "status": status,
        "comparison_scope": "Ours seed42 versus frozen Original SFTMAT seed42 paper results",
        "baseline_csv": str(baseline_copy),
        "baseline_csv_sha256": sha256(baseline_copy),
        "test_relative_improvement_percent_by_task": relative_test,
        "valid_relative_improvement_percent_by_task": relative_valid,
        "improved_seeds_by_task": seed_wins,
        "ten_task_wins": wins_total,
        "jarvis_wins": wins_jarvis,
        "mp_wins": wins_mp,
        "ten_task_macro_relative_improvement_percent": macro,
        "jarvis_macro_relative_improvement_percent": jarvis_macro,
        "mp_macro_relative_improvement_percent": mp_macro,
        "tasks_improved_in_seed42": [task for task in TASKS if seed_wins[task] == 1],
        "tasks_degraded_by_more_than_1_percent": degraded,
        "worst_task": min(relative_test, key=relative_test.get),
        "worst_task_improvement_percent": min(relative_test.values()),
        "new_ours_test_forward_count": len(TASKS),
    }
    atomic_json(args.output_root / "final_macro_statistics.json", statistics)
    selection = {
        "status": status,
        "scope": "one formal Ours run per task at seed42; Original SFTMAT reused",
        "baseline_retrained": False,
        "new_ours_run_count": len(TASKS),
        "selected_by": "Valid only",
        "models_frozen_before_test": True,
        "checkpoints_frozen_before_test": True,
        "hyperparameters_frozen_before_test": True,
        "test_forward_count_per_run": 1,
        "test_used_for_retraining_or_selection": False,
        "checks": {
            "at_least_8_of_10_wins": wins_total >= 8,
            "at_least_6_of_10_wins": wins_total >= 6,
            "jarvis_macro_positive": jarvis_macro > 0,
            "mp_macro_positive": mp_macro > 0,
            "at_most_1_task_degraded_gt_1pct": len(degraded) <= 1,
        },
    }
    atomic_json(args.output_root / "final_selection.json", selection)
    replace = status == "STRONG_SFTMAT_IMPROVEMENT"
    report = f"""# Final 10-Task Benchmark Report

## Original SFTMAT

Original SFTMAT is the complete graph-text `SFTGNNMultimodal` model, not a
graph-only model. It concatenates a 128-dimensional SFTGNN crystal embedding
with a 64-dimensional projection of the frozen 768-dimensional MatBERT CLS
feature and predicts with the original `192 -> 256 -> 1` property head.

## Main result

- Final status: `{status}`
- Ours wins: **{wins_total}/10** tasks.
- JARVIS wins: **{wins_jarvis}/6** tasks.
- MP wins: **{wins_mp}/4** tasks.
- Ten-task macro relative improvement: **{macro:.6f}%**.
- JARVIS macro relative improvement: **{jarvis_macro:.6f}%**.
- MP macro relative improvement: **{mp_macro:.6f}%**.
- Worst degradation: **{statistics['worst_task']}** at **{statistics['worst_task_improvement_percent']:.6f}%**.
- Ours may replace Original SFTMAT as the paper main model: **{'Yes' if replace else 'No'}**.

## Scope and provenance

Only Ours was newly trained, once per task with seed 42. Original SFTMAT was
not retrained: its ten frozen paper results were reused. The MBJ comparison uses
the retained Original SFTMAT seed-42 row (Test MAE 0.25798455), while the
paper-reported six-seed MBJ mean (0.26355210) is retained in the baseline CSV
as a provenance note. Nine non-MBJ frozen baselines survive as aggregate
paper-table rows with recorded raw-log paths; the raw logs are not present in
this clone.

All task-level aggregation averages relative improvement rather than raw MAE.
Every checkpoint was selected by Valid only, frozen before Test, and evaluated
on Test exactly once. Test was not used to retrain, tune, remove seeds, or
select a model.
"""
    (args.output_root / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(statistics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--task", choices=TASKS, required=True)
    preflight.add_argument("--token_cache", type=Path, required=True)
    preflight.add_argument("--scratch", type=Path, required=True)
    preflight.add_argument("--output_root", type=Path, required=True)
    preflight.add_argument("--model_checkpoint", type=Path, required=True)
    preflight.add_argument("--device", default="cuda")
    train = sub.add_parser("train")
    train.add_argument("--task", choices=TASKS, required=True)
    train.add_argument("--model_name", choices=MODELS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--token_cache", type=Path, required=True)
    train.add_argument("--output_dir", type=Path, required=True)
    train.add_argument("--model_checkpoint", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument(
        "--num_workers", type=int,
        choices=(FORMAL_DATA_LOADER_WORKERS,), default=FORMAL_DATA_LOADER_WORKERS
    )
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    recover = sub.add_parser("recover-valid")
    recover.add_argument("--task", choices=TASKS, required=True)
    recover.add_argument("--model_name", choices=MODELS, required=True)
    recover.add_argument("--seed", type=int, choices=SEEDS, required=True)
    recover.add_argument("--token_cache", type=Path, required=True)
    recover.add_argument("--output_dir", type=Path, required=True)
    recover.add_argument("--model_checkpoint", type=Path, required=True)
    recover.add_argument("--device", default="cuda")
    recover.add_argument(
        "--num_workers", type=int,
        choices=(FORMAL_DATA_LOADER_WORKERS,), default=FORMAL_DATA_LOADER_WORKERS
    )
    recover.add_argument("--training_time_seconds", type=float, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output_root", type=Path, required=True)
    test = sub.add_parser("test")
    test.add_argument("--task", choices=TASKS, required=True)
    test.add_argument("--model_name", choices=MODELS, required=True)
    test.add_argument("--seed", type=int, choices=SEEDS, required=True)
    test.add_argument("--token_cache", type=Path, required=True)
    test.add_argument("--output_root", type=Path, required=True)
    test.add_argument("--model_checkpoint", type=Path, required=True)
    test.add_argument("--device", default="cuda")
    test.add_argument(
        "--num_workers", type=int,
        choices=(FORMAL_DATA_LOADER_WORKERS,), default=FORMAL_DATA_LOADER_WORKERS
    )
    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--output_root", type=Path, required=True)
    summarize_parser.add_argument("--baseline_csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        preflight_task(args)
    elif args.command == "train":
        train_run(args)
    elif args.command == "recover-valid":
        recover_valid(args)
    elif args.command == "freeze":
        freeze_valid(args)
    elif args.command == "test":
        test_once(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
