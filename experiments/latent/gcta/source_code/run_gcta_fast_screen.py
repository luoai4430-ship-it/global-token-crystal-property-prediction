#!/usr/bin/env python3
"""Validation-only GCTA fast screen on two band-gap tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_seed43_benchmark as base  # noqa: E402


SEED = 42
TASKS = ("Jarvis-Bandgap_MBJ", "MP-Bandgap")
VARIANTS = ("B1_NaiveConcat", "P2_GCTA")
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.5
MAX_LENGTH = 128
TRAINING_DTYPE = torch.bfloat16
WORKERS = 2
GRADIENT_ACTIVE_THRESHOLD = 1e-8
ATTENTION_CHANGE_THRESHOLD = 1e-4
ZERO_CONTROL_DEFINITION = "qc=0; q=normalize(qg); prediction_head_cls=correct"


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
        "stage": "GCTA_fast_screen",
        "seed": SEED,
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "epochs_max": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "learning_rate": LEARNING_RATE,
        "max_learning_rate": MAX_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "single final-head raw-unit MAE",
        "checkpoint_selection": "minimum validation MAE only",
        "split_seed": 123,
        "target_normalization": "task train-only mean/std from graph cache",
        "training_dtype": "bfloat16",
        "temperature": TEMPERATURE,
        "bandgap_nonnegative_clipping": True,
        "query_controls": ["normal", "query_cls_shuffled", "query_cls_zero"],
        "query_control_selection_use": False,
        "test_forward_count": 0,
        "forbidden": [
            "dual-model prediction ensemble",
            "prediction averaging",
            "router",
            "sample-wise alpha",
            "property-specific gate",
            "dataset-specific gate",
            "tau search",
            "loss-weight search",
            "TAGIF tuning",
            "test evaluation",
        ],
        "mechanism_thresholds": {
            "wqc_gradient_norm_min": GRADIENT_ACTIVE_THRESHOLD,
            "mean_attention_l1_vs_b1_min": ATTENTION_CHANGE_THRESHOLD,
        },
    }


def protocol_sha256() -> str:
    payload = json.dumps(protocol_payload(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GCTAFamily(nn.Module):
    """Naive concat baseline and Global-conditioned token attention."""

    def __init__(self, variant: str, special_token_ids: tuple[int, int, int]) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown GCTA variant: {variant}")
        self.variant = variant
        self.temperature = TEMPERATURE
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self.sftgnn = base.SFTGNN(
            num_layers=5,
            in_dim=base.data_config.getAtomDim(),
            node_feature=128,
            dist_dim=64,
            sh_dim=64,
            use_sh=True,
            use_dist=True,
        )
        self.query_graph_projection = nn.Linear(128, 128, bias=False)
        self.key_projection = nn.Linear(768, 128, bias=False)
        self.value_projection = nn.Linear(768, 128, bias=False)
        self.token_norm = nn.LayerNorm(128)
        self.cls_projection = nn.Linear(768, 128)
        self.cls_norm = nn.LayerNorm(128)
        self.naive_projection = nn.Linear(384, 128)
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        if variant == "P2_GCTA":
            self.query_cls_projection = nn.Linear(128, 128, bias=False)
            nn.init.zeros_(self.query_cls_projection.weight)

    def _eligible_mask(
        self, attention_mask: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mask = attention_mask.bool()
        special_mask = torch.zeros_like(raw_mask)
        for token_id in self.special_token_ids:
            special_mask |= input_ids.eq(token_id)
        eligible = raw_mask & ~special_mask
        if not bool(eligible.any(dim=1).all()):
            raise ValueError("every sample must contain a non-special token")
        return eligible, special_mask

    def _pool_tokens(
        self,
        graph_embedding: torch.Tensor,
        cls_embedding: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eligible, special_mask = self._eligible_mask(attention_mask, input_ids)
        qg = self.query_graph_projection(graph_embedding)
        qc = torch.zeros_like(qg)
        if self.variant == "P2_GCTA":
            qc = self.query_cls_projection(cls_embedding)
        query = F.normalize((qg + qc).unsqueeze(1), p=2.0, dim=-1, eps=1e-12)
        keys = F.normalize(
            self.key_projection(text_tokens.float()), p=2.0, dim=-1, eps=1e-12
        )
        values = self.value_projection(text_tokens.float())
        scores = (query * keys).sum(dim=-1) / self.temperature
        scores = scores.masked_fill(~eligible, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        return self.token_norm(pooled), attention, eligible, special_mask, qg, qc

    def forward(
        self,
        graph,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        query_cls_override: torch.Tensor | None = None,
        query_cls_zero: bool = False,
    ) -> dict[str, torch.Tensor]:
        if text_tokens.ndim != 3 or tuple(text_tokens.shape[-1:]) != (768,):
            raise ValueError(f"invalid token tensor: {tuple(text_tokens.shape)}")
        graph_embedding = self.sftgnn.forward_embedding(graph)
        cls_source = text_tokens[:, 0, :].float()
        cls_embedding = self.cls_norm(self.cls_projection(cls_source))
        query_cls_source = cls_source
        if query_cls_override is not None:
            if query_cls_override.shape != cls_source.shape:
                raise ValueError("query CLS override shape does not match the batch")
            query_cls_source = query_cls_override.float()
        query_cls_embedding = self.cls_norm(self.cls_projection(query_cls_source))
        if query_cls_zero:
            # Zero after the biased CLS projection and affine normalization.
            query_cls_embedding = torch.zeros_like(query_cls_embedding)
        token_embedding, attention, eligible, special_mask, qg, qc = self._pool_tokens(
            graph_embedding, query_cls_embedding, text_tokens, attention_mask, input_ids
        )
        fused = F.silu(
            self.naive_projection(
                torch.cat((graph_embedding, cls_embedding, token_embedding), dim=-1)
            )
        )

        prediction = self.prediction_head(fused).squeeze(-1)
        probabilities = attention.clamp_min(torch.finfo(attention.dtype).tiny)
        entropy = -(attention * probabilities.log()).sum(dim=-1)
        raw_mask = attention_mask.bool()
        return {
            "final_prediction": prediction,
            "graph_embedding": graph_embedding,
            "token_embedding": token_embedding,
            "cls_embedding": cls_embedding,
            "query_cls_embedding": query_cls_embedding,
            "qg": qg,
            "qc": qc,
            "attention_weights": attention,
            "attention_entropy": entropy,
            "effective_token_count": entropy.exp(),
            "attention_max_weight": attention.max(dim=-1).values,
            "attention_eligible_mask": eligible,
            "special_token_attention_mass": (
                attention * special_mask.to(attention.dtype)
            ).sum(dim=-1),
            "padding_attention_mass": (
                attention * (~raw_mask).to(attention.dtype)
            ).sum(dim=-1),
        }


def build_model(variant: str, excluded_ids: tuple[int, int, int]) -> GCTAFamily:
    return GCTAFamily(variant, excluded_ids)


def physical_prediction(
    output: dict[str, torch.Tensor], mean: float, std: float, task: str
) -> torch.Tensor:
    prediction = output["final_prediction"].float() * std + mean
    if "Bandgap" in task:
        prediction = prediction.clamp_min(0.0)
    return prediction


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def finite_or_none(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "max_abs": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "max_abs": float(np.max(np.abs(array))),
    }


def shuffled_cls_mapping(sample_ids: list[str], seed: int) -> dict[str, str]:
    if len(sample_ids) < 2:
        raise ValueError("shuffled CLS control requires at least two samples")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(1, len(sample_ids)))
    sources = sample_ids[offset:] + sample_ids[:offset]
    mapping = dict(zip(sample_ids, sources))
    if any(target == source for target, source in mapping.items()):
        raise AssertionError("shuffled CLS mapping contains a fixed point")
    return mapping


def cls_override_for_batch(
    sample_ids: list[str],
    mapping: dict[str, str],
    store: base.TokenFeatureStore,
    device: torch.device,
) -> torch.Tensor:
    cls_rows = [store.fetch(mapping[sample_id])[0][0].float() for sample_id in sample_ids]
    return torch.stack(cls_rows, dim=0).to(device, non_blocking=True)


def evaluate(
    model: GCTAFamily,
    variant: str,
    dataset: base.TokenGraphDataset,
    store: base.TokenFeatureStore,
    task: str,
    mean: float,
    std: float,
    device: torch.device,
    seed: int,
    collect: bool,
    query_mode: str = "normal",
    cls_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    loader = base.make_loader(dataset, EVAL_BATCH_SIZE, False, seed, WORKERS)
    model.eval()
    abs_sum = 0.0
    count = 0
    entropy_values: list[float] = []
    effective_values: list[float] = []
    max_attention_values: list[float] = []
    qc_qg_ratios: list[float] = []
    qc_qg_cosines: list[float] = []
    special_max = 0.0
    padding_max = 0.0
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            query_cls_override = None
            query_cls_zero = False
            if query_mode == "query_cls_shuffled":
                if cls_mapping is None:
                    raise ValueError("query shuffled CLS control requires a mapping")
                query_cls_override = cls_override_for_batch(
                    batch.sample_ids, cls_mapping, store, device
                )
            elif query_mode == "query_cls_zero":
                query_cls_zero = True
            elif query_mode != "normal":
                raise ValueError(f"unknown query mode: {query_mode}")
            with torch.autocast(
                device_type=device.type,
                dtype=TRAINING_DTYPE,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch.graph,
                    batch.text_tokens,
                    batch.attention_mask,
                    batch.input_ids,
                    query_cls_override=query_cls_override,
                    query_cls_zero=query_cls_zero,
                )
            if query_cls_zero and bool(torch.count_nonzero(output["qc"])):
                raise AssertionError("query_cls_zero must remove the entire qc term")
            target = batch.graph.y.reshape(-1).float() * std + mean
            prediction = physical_prediction(output, mean, std, task)
            errors = (prediction - target).abs()
            abs_sum += float(errors.sum().cpu())
            count += int(target.numel())
            entropy_values.extend(output["attention_entropy"].float().cpu().tolist())
            effective_values.extend(
                output["effective_token_count"].float().cpu().tolist()
            )
            max_attention_values.extend(
                output["attention_max_weight"].float().cpu().tolist()
            )
            special_max = max(
                special_max,
                float(output["special_token_attention_mass"].max().cpu()),
            )
            padding_max = max(
                padding_max,
                float(output["padding_attention_mass"].max().cpu()),
            )
            qg = output["qg"].float()
            qc = output["qc"].float()
            if variant == "P2_GCTA":
                qc_qg_ratios.extend(
                    (qc.norm(dim=-1) / qg.norm(dim=-1).clamp_min(1e-12)).cpu().tolist()
                )
                qc_qg_cosines.extend(
                    F.cosine_similarity(qc, qg, dim=-1, eps=1e-12).cpu().tolist()
                )
            if collect:
                targets = target.cpu().tolist()
                predictions = prediction.cpu().tolist()
                attentions = output["attention_weights"].float().cpu().tolist()
                for index, sample_id in enumerate(batch.sample_ids):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "target": targets[index],
                            "prediction": predictions[index],
                            "absolute_error": abs(
                                predictions[index] - targets[index]
                            ),
                            "query_mode": query_mode,
                            "attention_json": json.dumps(attentions[index]),
                        }
                    )
    if count != len(dataset):
        raise AssertionError(f"evaluation covered {count}/{len(dataset)} samples")
    ratio_stats = finite_or_none(qc_qg_ratios)
    cosine_stats = finite_or_none(qc_qg_cosines)
    metrics: dict[str, object] = {
        "mae": abs_sum / count,
        "sample_count": count,
        "attention_entropy_mean": float(np.mean(entropy_values)),
        "effective_token_count_mean": float(np.mean(effective_values)),
        "median_max_attention": float(np.median(max_attention_values)),
        "mean_max_attention": float(np.mean(max_attention_values)),
        "special_token_attention_max": special_max,
        "padding_attention_max": padding_max,
        "qc_qg_ratio_mean": ratio_stats["mean"],
        "qc_qg_ratio_median": ratio_stats["median"],
        "qc_qg_ratio_std": ratio_stats["std"],
        "cos_qc_qg_mean": cosine_stats["mean"],
        "cos_qc_qg_median": cosine_stats["median"],
        "cos_qc_qg_std": cosine_stats["std"],
        "query_mode": query_mode,
    }
    return metrics, pd.DataFrame(records) if collect else None


def module_gradient_norm(module: nn.Module | None) -> float | None:
    if module is None:
        return None
    total = 0.0
    found = False
    for parameter in module.parameters():
        if parameter.grad is not None:
            found = True
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total) if found else 0.0


def validation_gradient_diagnostics(
    model: GCTAFamily,
    dataset: base.TokenGraphDataset,
    task: str,
    mean: float,
    std: float,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    model.zero_grad(set_to_none=True)
    loader = base.make_loader(dataset, EVAL_BATCH_SIZE, False, SEED, WORKERS)
    sample_count = len(dataset)
    covered = 0
    for batch in loader:
        batch = batch.to(device)
        output = model(
            batch.graph,
            batch.text_tokens.float(),
            batch.attention_mask,
            batch.input_ids,
        )
        target = batch.graph.y.reshape(-1).float() * std + mean
        prediction = physical_prediction(output, mean, std, task)
        loss = (prediction - target).abs().sum() / sample_count
        loss.backward()
        covered += int(target.numel())
    if covered != sample_count:
        raise AssertionError("gradient diagnostic did not cover validation set")
    result = {
        "gradient_norm_Wc": module_gradient_norm(
            getattr(model, "cls_projection", None)
        ),
        "gradient_norm_Wqc": module_gradient_norm(
            getattr(model, "query_cls_projection", None)
        ),
        "gradient_definition": "L2 norm of full-validation raw-MAE gradient",
    }
    model.zero_grad(set_to_none=True)
    return result


def verify_serialized_state(
    checkpoint_path: Path,
    variant: str,
    excluded_ids: tuple[int, int, int],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = build_model(variant, excluded_ids)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name, value in restored.state_dict().items():
        if not torch.equal(value.cpu(), checkpoint["model_state_dict"][name]):
            raise AssertionError(f"checkpoint tensor changed after reload: {name}")


def train_run(args: argparse.Namespace) -> None:
    if args.task not in TASKS or args.variant not in VARIANTS or args.seed != SEED:
        raise ValueError("run is outside the locked GCTA matrix")
    if args.epochs != EPOCHS or args.patience != PATIENCE:
        raise ValueError("epoch and patience settings are locked")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GCTA screen requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("completed run already exists")
    seed_everything(SEED)
    started = time.perf_counter()
    train_graphs, valid_graphs, _, mean, std = base.load_graph_splits(args.task)
    store = base.TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    train_dataset = base.TokenGraphDataset(train_graphs, store, split="train")
    valid_dataset = base.TokenGraphDataset(valid_graphs, store, split="valid")
    train_loader = base.make_loader(
        train_dataset, BATCH_SIZE, True, SEED, args.num_workers
    )
    if train_loader.drop_last:
        raise AssertionError("GCTA training must not drop samples")
    device = torch.device(args.device)
    excluded_ids = base.special_token_ids(args.model_checkpoint)
    model = build_model(args.variant, excluded_ids).to(device)
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
    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    stopped_epoch = EPOCHS
    for epoch in range(1, EPOCHS + 1):
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
                    batch.input_ids,
                )
                target = batch.graph.y.reshape(-1).float() * std + mean
                prediction = physical_prediction(output, mean, std, args.task)
                loss = F.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite GCTA training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_count = int(target.numel())
            loss_sum += float(loss.detach().cpu()) * batch_count
            sample_count += batch_count
        if sample_count != len(train_dataset):
            raise AssertionError(
                f"epoch covered {sample_count}/{len(train_dataset)} samples"
            )
        valid_metrics, _ = evaluate(
            model,
            args.variant,
            valid_dataset,
            store,
            args.task,
            mean,
            std,
            device,
            SEED,
            False,
        )
        valid_mae = float(valid_metrics["mae"])
        if valid_mae < best_mae:
            best_mae = valid_mae
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"[{args.task}:{args.variant}:seed{SEED}] "
            f"epoch={epoch}/{EPOCHS} loss={loss_sum / sample_count:.8f} "
            f"valid_mae={valid_mae:.8f} best={best_mae:.8f} "
            f"best_epoch={best_epoch} stale={epochs_without_improvement}/{PATIENCE}",
            flush=True,
        )
        if epochs_without_improvement >= PATIENCE:
            stopped_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    valid_metrics, valid_frame = evaluate(
        model,
        args.variant,
        valid_dataset,
        store,
        args.task,
        mean,
        std,
        device,
        SEED,
        True,
    )
    if abs(float(valid_metrics["mae"]) - best_mae) > 5e-4:
        raise AssertionError("reloaded best-state validation MAE changed")
    if valid_metrics["special_token_attention_max"] != 0.0:
        raise AssertionError("special tokens received attention")
    if valid_metrics["padding_attention_max"] != 0.0:
        raise AssertionError("padding tokens received attention")
    if float(valid_metrics["effective_token_count_mean"]) <= 1.0:
        raise AssertionError("TrueToken attention collapsed to one token")
    gradient_metrics = validation_gradient_diagnostics(
        model, valid_dataset, args.task, mean, std, device
    )

    checkpoint = {
        "task": args.task,
        "variant": args.variant,
        "seed": SEED,
        "best_valid_epoch": best_epoch,
        "best_valid_mae": valid_metrics["mae"],
        "model_state_dict": best_state,
        "mean": mean,
        "std": std,
        "parameter_count": parameter_count,
        "protocol_sha256": protocol_sha256(),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "selection_metric": "minimum validation MAE only",
        "test_forward_count": 0,
    }
    checkpoint_path = args.output_dir / "best_valid.pt"
    torch.save(checkpoint, checkpoint_path)
    verify_serialized_state(checkpoint_path, args.variant, excluded_ids)
    valid_frame.to_csv(args.output_dir / "valid_predictions.csv", index=False)

    query_controls = None
    if args.variant == "P2_GCTA":
        mapping = shuffled_cls_mapping(valid_dataset.sample_ids, SEED + 1009)
        mapping_payload = "\n".join(
            f"{key}\t{mapping[key]}" for key in sorted(mapping)
        ).encode("utf-8")
        shuffled_metrics, _ = evaluate(
            model,
            args.variant,
            valid_dataset,
            store,
            args.task,
            mean,
            std,
            device,
            SEED,
            False,
            query_mode="query_cls_shuffled",
            cls_mapping=mapping,
        )
        zero_metrics, _ = evaluate(
            model,
            args.variant,
            valid_dataset,
            store,
            args.task,
            mean,
            std,
            device,
            SEED,
            False,
            query_mode="query_cls_zero",
        )
        query_controls = {
            "zero_control_definition": ZERO_CONTROL_DEFINITION,
            "normal_mae": valid_metrics["mae"],
            "query_cls_shuffled_mae": shuffled_metrics["mae"],
            "query_cls_zero_mae": zero_metrics["mae"],
            "normal_better_than_shuffled": bool(
                float(valid_metrics["mae"]) < float(shuffled_metrics["mae"])
            ),
            "normal_better_than_zero": bool(
                float(valid_metrics["mae"]) < float(zero_metrics["mae"])
            ),
            "mapping_kind": "seeded cyclic derangement",
            "mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
            "selection_use": False,
            "test_forward_count": 0,
        }
        atomic_json(args.output_dir / "query_controls.json", query_controls)

    summary = {
        "status": "valid_complete",
        "task": args.task,
        "dataset": base.TASKS[args.task].dataset,
        "property": base.TASKS[args.task].property_name,
        "unit": base.TASKS[args.task].unit,
        "variant": args.variant,
        "seed": SEED,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "stopped_epoch": stopped_epoch,
        "early_stopped": stopped_epoch < EPOCHS,
        "best_valid_epoch": best_epoch,
        "valid_mae": valid_metrics["mae"],
        "parameter_count": parameter_count,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "target_mean": mean,
        "target_std": std,
        "training_time_seconds": time.perf_counter() - started,
        **{
            key: valid_metrics[key]
            for key in (
                "attention_entropy_mean",
                "effective_token_count_mean",
                "median_max_attention",
                "mean_max_attention",
                "special_token_attention_max",
                "padding_attention_max",
                "qc_qg_ratio_mean",
                "qc_qg_ratio_median",
                "qc_qg_ratio_std",
                "cos_qc_qg_mean",
                "cos_qc_qg_median",
                "cos_qc_qg_std",
            )
        },
        **gradient_metrics,
        "query_controls": query_controls,
        "temperature": TEMPERATURE,
        "training_autocast_dtype": "bfloat16",
        "drop_last": False,
        "data_loader_workers": args.num_workers,
        "checkpoint_reload_state_exact": True,
        "checkpoint_sha256": sha256(checkpoint_path),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
        "training_runs": 1,
        "tau_search": 0,
        "loss_weight_search": 0,
        "checkpoint_reselection": 0,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def preflight(args: argparse.Namespace) -> None:
    if args.task not in TASKS:
        raise ValueError("task is outside the GCTA screen")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GCTA preflight requires BF16 CUDA")
    aliases = base.ensure_graph_cache_paths(args.task, args.scratch)
    train_graphs, valid_graphs, _, mean, std = base.load_graph_splits(args.task)
    store = base.TokenFeatureStore(args.token_cache, expected_length=MAX_LENGTH)
    train_dataset = base.TokenGraphDataset(train_graphs, store, split="train")
    valid_dataset = base.TokenGraphDataset(valid_graphs, store, split="valid")
    if set(train_dataset.sample_ids) & set(valid_dataset.sample_ids):
        raise AssertionError("train and validation IDs overlap")
    loader = base.make_loader(valid_dataset, min(4, len(valid_dataset)), False, SEED, 0)
    batch = next(iter(loader))
    excluded_ids = base.special_token_ids(args.model_checkpoint)
    device = torch.device(args.device)
    batch = batch.to(device)
    checks: dict[str, object] = {}
    initial_states: dict[str, dict[str, torch.Tensor]] = {}
    for variant in VARIANTS:
        seed_everything(SEED)
        model = build_model(variant, excluded_ids).to(device).eval()
        if sum(isinstance(module, base.SFTGNN) for module in model.modules()) != 1:
            raise AssertionError(f"{variant} does not contain exactly one SFTGNN")
        if sum(name == "prediction_head" for name, _ in model.named_modules()) != 1:
            raise AssertionError(f"{variant} does not contain exactly one final head")
        forbidden_names = [
            name
            for name, _ in model.named_modules()
            if not name.startswith("sftgnn.")
            and any(term in name.lower() for term in ("router", "gate", "alpha"))
        ]
        if forbidden_names:
            raise AssertionError(f"forbidden module names: {forbidden_names}")
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=TRAINING_DTYPE,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch.graph.clone(),
                batch.text_tokens.clone(),
                batch.attention_mask.clone(),
                batch.input_ids.clone(),
            )
        if not bool(torch.isfinite(output["final_prediction"]).all()):
            raise FloatingPointError(f"{variant} produced non-finite predictions")
        special_mass = float(output["special_token_attention_mass"].max().cpu())
        padding_mass = float(output["padding_attention_mass"].max().cpu())
        effective_min = float(output["effective_token_count"].min().cpu())
        if special_mass != 0.0 or padding_mass != 0.0 or effective_min <= 1.0:
            raise AssertionError(f"{variant} failed attention-mask preflight")
        if variant == "P2_GCTA":
            wqc = model.query_cls_projection.weight.detach()
            if float(wqc.abs().max().cpu()) != 0.0:
                raise AssertionError("P2 Wqc must be zero-initialized")
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        initial_states[variant] = state
        checks[variant] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "prediction_shape": list(output["final_prediction"].shape),
            "single_sftgnn": True,
            "single_prediction_head": True,
            "temperature": TEMPERATURE,
            "special_token_attention_max": special_mass,
            "padding_attention_max": padding_mass,
            "effective_token_count_min": effective_min,
            "wqc_zero_initialized": variant == "P2_GCTA",
            "forbidden_modules_absent": True,
        }
        del model
        torch.cuda.empty_cache()

    shared_prefixes = (
        "sftgnn.",
        "query_graph_projection.",
        "key_projection.",
        "value_projection.",
        "token_norm.",
        "cls_projection.",
        "cls_norm.",
        "naive_projection.",
        "prediction_head.",
    )
    reference = initial_states["B1_NaiveConcat"]
    shared_equal = True
    for variant in VARIANTS[1:]:
        for name, value in reference.items():
            if name.startswith(shared_prefixes):
                shared_equal &= name in initial_states[variant] and torch.equal(
                    value, initial_states[variant][name]
                )
    if not shared_equal:
        raise AssertionError("shared modules do not have seed-aligned initialization")
    b1_count = int(checks["B1_NaiveConcat"]["parameter_count"])
    p1_count = int(checks["P2_GCTA"]["parameter_count"])
    payload = {
        "status": "passed",
        "task": args.task,
        "split_sizes": {"train": len(train_dataset), "valid": len(valid_dataset)},
        "target_mean": mean,
        "target_std": std,
        "token_cache_rows": len(store.sample_ids),
        "token_cache_shape": list(store.features.shape),
        "token_cache_sha256": store.metadata["cache_sha256"],
        "shared_initialization_equal": shared_equal,
        "p2_added_params_vs_b1_pct": 100.0 * (p1_count - b1_count) / b1_count,
        "model_checks": checks,
        "aliases": aliases,
        "protocol": protocol_payload(),
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
    }
    atomic_json(args.output_root / "preflight" / f"{args.task}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def relative_macro_gain(frame: pd.DataFrame, baseline: str, candidate: str) -> float:
    pivot = frame.pivot(index="task", columns="variant", values="valid_mae")
    return 100.0 * float((1.0 - pivot[candidate] / pivot[baseline]).mean())


def corrected_query_controls(output_root: Path, row: dict) -> dict:
    """Repair only inference controls from a frozen checkpoint, preserving summaries."""
    directory = output_root / "runs" / row["task"] / "P2_GCTA" / "seed_42"
    sidecar = directory / "query_controls_corrected.json"
    controls = row["query_controls"]
    if sidecar.exists():
        controls = json.loads(sidecar.read_text(encoding="utf-8"))
        if controls.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise AssertionError("corrected controls refer to a different checkpoint")
    if controls.get("zero_control_definition") == ZERO_CONTROL_DEFINITION:
        return controls

    import run_final_10task_queue as cache_common

    repair_root = output_root / "control_repair"
    cache_common.OUTPUT_ROOT = repair_root
    cache_common.SCRATCH_ROOT = Path("/dev/shm") / output_root.name / "control_repair"
    cache_common.STATE_PATH = repair_root / "cache_state.json"
    cache_common.LOG_ROOT = repair_root / "logs"
    cache_common.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cache_common.SCRIPT = Path(__file__).resolve()
    cache_common.PYTHON = Path(sys.executable)
    model_checkpoint = cache_common.MODEL_CHECKPOINT
    frozen = {name: sha256(directory / name) for name in (
        "summary.json", "best_valid.pt", "valid_predictions.csv", "query_controls.json"
    )}
    print(f"Repairing query controls only: {row['task']}", flush=True)
    scratch, token_cache = cache_common.prepare_task(row["task"], "valid")
    try:
        store = base.TokenFeatureStore(token_cache, expected_length=MAX_LENGTH)
        if store.metadata["cache_sha256"] != row["token_cache_sha256"]:
            raise AssertionError("control repair token cache differs from training cache")
        _, graphs, _, mean, std = base.load_graph_splits(row["task"])
        dataset = base.TokenGraphDataset(graphs, store, split="valid")
        if len(dataset) != row["valid_size"]:
            raise AssertionError("control repair validation size differs")
        checkpoint = torch.load(directory / "best_valid.pt", map_location="cpu", weights_only=False)
        if sha256(directory / "best_valid.pt") != row["checkpoint_sha256"]:
            raise AssertionError("selected checkpoint changed")
        device = torch.device("cuda")
        model = build_model("P2_GCTA", base.special_token_ids(model_checkpoint)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        mapping = shuffled_cls_mapping(dataset.sample_ids, SEED + 1009)
        mapping_payload = "\n".join(f"{key}\t{mapping[key]}" for key in sorted(mapping)).encode("utf-8")
        if hashlib.sha256(mapping_payload).hexdigest() != row["query_controls"]["mapping_sha256"]:
            raise AssertionError("control repair sample mapping differs")
        corrected = dict(row["query_controls"])
        for mode in ("normal", "query_cls_shuffled", "query_cls_zero"):
            metrics, frame = evaluate(
                model, "P2_GCTA", dataset, store, row["task"], mean, std,
                device, SEED, True, query_mode=mode, cls_mapping=mapping,
            )
            if mode == "normal" and abs(metrics["mae"] - row["valid_mae"]) > 5e-4:
                raise AssertionError("control repair normal MAE differs from the frozen result")
            corrected[f"{mode}_mae"] = metrics["mae"]
            if mode == "query_cls_zero":
                assert metrics["qc_qg_ratio_mean"] == 0.0
                corrected["zero_qc_qg_ratio_mean"] = metrics["qc_qg_ratio_mean"]
            frame.to_csv(directory / f"{mode}_control_corrected.csv", index=False)
        corrected.update(
            zero_control_definition=ZERO_CONTROL_DEFINITION,
            checkpoint_sha256=row["checkpoint_sha256"],
            normal_better_than_shuffled=corrected["normal_mae"] < corrected["query_cls_shuffled_mae"],
            normal_better_than_zero=corrected["normal_mae"] < corrected["query_cls_zero_mae"],
            test_forward_count=0,
            training_runs=0,
            checkpoint_reselection=0,
        )
        for name, digest in frozen.items():
            if sha256(directory / name) != digest:
                raise AssertionError(f"control repair changed frozen artifact: {name}")
        atomic_json(sidecar, corrected)
        del model, dataset, store, graphs, checkpoint
        torch.cuda.empty_cache()
        return corrected
    finally:
        if not scratch.resolve().is_relative_to(cache_common.SCRATCH_ROOT.resolve()):
            raise AssertionError("control scratch cleanup escaped its root")
        shutil.rmtree(scratch)


def summarize(args: argparse.Namespace) -> None:
    summary_paths = sorted(args.output_root.glob("runs/*/*/seed_42/summary.json"))
    if len(summary_paths) != len(TASKS) * len(VARIANTS):
        raise AssertionError(f"expected four validation summaries, found {len(summary_paths)}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    if any(int(row["test_forward_count"]) != 0 for row in summaries):
        raise AssertionError("a Test forward was recorded")
    keys = {(row["task"], row["variant"]) for row in summaries}
    if keys != {(task, variant) for task in TASKS for variant in VARIANTS}:
        raise AssertionError("summary matrix is incomplete or duplicated")
    rows = []
    diagnostic_rows = []
    control_rows = []
    for row in summaries:
        rows.append(
            {
                "task": row["task"],
                "dataset": row["dataset"],
                "variant": row["variant"],
                "seed": row["seed"],
                "valid_mae": row["valid_mae"],
                "best_valid_epoch": row["best_valid_epoch"],
                "stopped_epoch": row["stopped_epoch"],
                "early_stopped": row["early_stopped"],
                "parameter_count": row["parameter_count"],
                "training_time_seconds": row["training_time_seconds"],
                "test_forward_count": 0,
            }
        )
        diagnostic_rows.append(
            {
                "task": row["task"],
                "variant": row["variant"],
                "valid_mae": row["valid_mae"],
                "qc_qg_ratio_mean": row["qc_qg_ratio_mean"],
                "qc_qg_ratio_median": row["qc_qg_ratio_median"],
                "qc_qg_ratio_std": row["qc_qg_ratio_std"],
                "cos_qc_qg_mean": row["cos_qc_qg_mean"],
                "cos_qc_qg_median": row["cos_qc_qg_median"],
                "cos_qc_qg_std": row["cos_qc_qg_std"],
                "gradient_norm_Wc": row["gradient_norm_Wc"],
                "gradient_norm_Wqc": row["gradient_norm_Wqc"],
                "attention_entropy": row["attention_entropy_mean"],
                "effective_token_count": row["effective_token_count_mean"],
                "median_max_attention": row["median_max_attention"],
                "mean_attention_l1_vs_b1": None,
                "test_forward_count": 0,
            }
        )
        if row["variant"] == "P2_GCTA":
            controls = corrected_query_controls(args.output_root, row)
            for control in ("normal", "query_cls_shuffled", "query_cls_zero"):
                control_rows.append(
                    {
                        "task": row["task"],
                        "control": control,
                        "valid_mae": controls[f"{control}_mae"],
                        "selected_checkpoint_unchanged": True,
                        "graph_unchanged": True,
                        "prediction_head_cls_correct": True,
                        "zero_control_definition": controls["zero_control_definition"],
                        "test_forward_count": 0,
                    }
                )

    result_frame = pd.DataFrame(rows).sort_values(["task", "variant"])
    diagnostic_frame = pd.DataFrame(diagnostic_rows).sort_values(["task", "variant"])
    control_frame = pd.DataFrame(control_rows).sort_values(["task", "control"])

    def attention_l1(task: str) -> float:
        b1_path = args.output_root / "runs" / task / "B1_NaiveConcat" / "seed_42" / "valid_predictions.csv"
        p2_path = args.output_root / "runs" / task / "P2_GCTA" / "seed_42" / "valid_predictions.csv"
        b1 = pd.read_csv(b1_path)
        p2 = pd.read_csv(p2_path)
        merged = b1[["sample_id", "attention_json"]].merge(
            p2[["sample_id", "attention_json"]],
            on="sample_id",
            suffixes=("_b1", "_p2"),
            validate="one_to_one",
        )
        if len(merged) != len(b1) or len(merged) != len(p2):
            raise AssertionError(f"attention rows are not sample-aligned for {task}")
        distances = []
        for row in merged.itertuples(index=False):
            left = np.asarray(json.loads(row.attention_json_b1), dtype=np.float64)
            right = np.asarray(json.loads(row.attention_json_p2), dtype=np.float64)
            distances.append(float(np.abs(left - right).mean()))
        return float(np.mean(distances))

    for task in TASKS:
        mask = (diagnostic_frame["task"] == task) & (diagnostic_frame["variant"] == "P2_GCTA")
        diagnostic_frame.loc[mask, "mean_attention_l1_vs_b1"] = attention_l1(task)

    pivot = result_frame.pivot(index="task", columns="variant", values="valid_mae")
    gain_b1 = relative_macro_gain(result_frame, "B1_NaiveConcat", "P2_GCTA")
    wins_b1 = int((pivot["P2_GCTA"] < pivot["B1_NaiveConcat"]).sum())
    p1_diag = diagnostic_frame[diagnostic_frame["variant"] == "P2_GCTA"]
    no_task_degradation = bool(
        ((pivot["P2_GCTA"] / pivot["B1_NaiveConcat"] - 1.0) <= 0.005).all()
    )
    query_active = bool(
        (p1_diag["qc_qg_ratio_mean"].astype(float) > 0.0).all()
        and (p1_diag["gradient_norm_Wqc"].astype(float) > GRADIENT_ACTIVE_THRESHOLD).all()
    )
    attention_changed = bool(
        (p1_diag["mean_attention_l1_vs_b1"].astype(float) > ATTENTION_CHANGE_THRESHOLD).all()
    )
    control_pivot = control_frame.pivot(index="task", columns="control", values="valid_mae")
    query_semantic_both_tasks = bool(
        (
            (control_pivot["normal"] < control_pivot["query_cls_shuffled"])
            & (control_pivot["normal"] < control_pivot["query_cls_zero"])
        ).all()
    )
    query_semantic_macro_signal = bool(
        control_pivot["normal"].mean() < control_pivot["query_cls_shuffled"].mean()
        and control_pivot["normal"].mean() < control_pivot["query_cls_zero"].mean()
    )
    parameter_counts = {
        variant: int(result_frame[result_frame["variant"] == variant]["parameter_count"].iloc[0])
        for variant in VARIANTS
    }
    added_pct = 100.0 * (
        parameter_counts["P2_GCTA"] - parameter_counts["B1_NaiveConcat"]
    ) / parameter_counts["B1_NaiveConcat"]
    parameter_gate = added_pct < 5.0
    success = bool(
        wins_b1 == 2
        and gain_b1 >= 0.5
        and no_task_degradation
        and query_active
        and query_semantic_both_tasks
        and attention_changed
        and parameter_gate
    )
    weak = bool(
        wins_b1 >= 1 and gain_b1 > 0.0 and query_semantic_macro_signal
    )
    decision = (
        "GLOBAL_CONDITIONED_TOKEN_GAIN"
        if success
        else "WEAK_GLOBAL_CONDITIONING_SIGNAL"
        if weak
        else "NO_GLOBAL_CONDITIONING_GAIN"
    )
    result_frame["p2_macro_gain_vs_b1_pct"] = gain_b1
    result_frame["decision"] = decision
    result_frame.to_csv(args.output_root / "gcta_fast_screen.csv", index=False)
    diagnostic_frame.to_csv(args.output_root / "gcta_attention_diagnostics.csv", index=False)
    control_frame.to_csv(args.output_root / "gcta_query_controls.csv", index=False)
    parameter_payload = {
        "parameter_counts": parameter_counts,
        "p2_added_params_vs_b1": parameter_counts["P2_GCTA"] - parameter_counts["B1_NaiveConcat"],
        "p2_added_params_vs_b1_pct": added_pct,
        "under_5_percent": parameter_gate,
        "protocol_sha256": protocol_sha256(),
        "test_forward_count": 0,
    }
    atomic_json(args.output_root / "gcta_parameter_count.json", parameter_payload)

    jarvis = pivot.loc["Jarvis-Bandgap_MBJ"]
    mp = pivot.loc["MP-Bandgap"]
    report_lines = [
        "# GCTA Fast Screen",
        "",
        f"Decision: **{decision}**",
        "",
        "## Validation MAE",
        "",
        "| Task | B1 Naive concat | P2 GCTA |",
        "|---|---:|---:|",
    ]
    for task in TASKS:
        values = pivot.loc[task]
        report_lines.append(
            f"| {task} | {values['B1_NaiveConcat']:.8f} | {values['P2_GCTA']:.8f} |"
        )
    report_lines.extend(
        [
            "",
            f"- P2 wins versus B1: {wins_b1}/2; two-task macro gain: {gain_b1:.3f}%.",
            f"- No task degrades by more than 0.5%: {no_task_degradation}.",
            f"- Query CLS controls favor the correct CLS on both tasks: {query_semantic_both_tasks}.",
            f"- Wqc branch is active: {query_active}.",
            f"- GCTA attention changes relative to B1: {attention_changed}.",
            f"- Added parameters versus B1: {added_pct:.3f}% (<5%: {parameter_gate}).",
            "",
            "## Required questions",
            "",
            f"1. GCTA exceeds naive concatenation: {'Yes' if wins_b1 == 2 and gain_b1 > 0 else 'No'}.",
            "2. Correct query CLS beats shuffled and zero controls on both tasks: "
            + str(query_semantic_both_tasks) + ".",
            "3. Mean ||qc||/||qg|| by task: "
            + ", ".join(
                f"{row.task}={float(row.qc_qg_ratio_mean):.4f}"
                for row in p1_diag.itertuples()
            )
            + ".",
            "4. Mean L1 distance between GCTA and B1 attention by task: "
            + ", ".join(
                f"{row.task}={float(row.mean_attention_l1_vs_b1):.6f}"
                for row in p1_diag.itertuples()
            )
            + ".",
            "5. Worth extending to four tasks: "
            + ("Yes." if decision == "GLOBAL_CONDITIONED_TOKEN_GAIN" else "Only as a weak-signal confirmation." if decision == "WEAK_GLOBAL_CONDITIONING_SIGNAL" else "No."),
            "",
            "## Protocol integrity",
            "",
            "- Validation-only checkpoint selection and reporting.",
            "- Query controls use the selected P2 checkpoint without retraining.",
            "- No rank, temperature, loss-weight, gate, router, or ensemble search.",
            "- `test_forward_count = 0` for every run and control.",
        ]
    )
    (args.output_root / "gcta_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    selection = {
        "decision": decision,
        "wins_vs_b1": wins_b1,
        "macro_gain_vs_b1_pct": gain_b1,
        "no_task_degradation_gt_0p5pct": no_task_degradation,
        "query_semantic_both_tasks": query_semantic_both_tasks,
        "query_active": query_active,
        "attention_changed": attention_changed,
        "parameter_gate": parameter_gate,
        "test_forward_count": 0,
    }
    atomic_json(args.output_root / "gcta_selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--task", choices=TASKS, required=True)
    preflight_parser.add_argument("--token_cache", type=Path, required=True)
    preflight_parser.add_argument("--scratch", type=Path, required=True)
    preflight_parser.add_argument("--output_root", type=Path, required=True)
    preflight_parser.add_argument("--model_checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--device", default="cuda")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--task", choices=TASKS, required=True)
    train_parser.add_argument("--variant", choices=VARIANTS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--token_cache", type=Path, required=True)
    train_parser.add_argument("--output_dir", type=Path, required=True)
    train_parser.add_argument("--model_checkpoint", type=Path, required=True)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--num_workers", type=int, default=WORKERS)
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--patience", type=int, default=PATIENCE)

    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--output_root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "train":
        train_run(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
