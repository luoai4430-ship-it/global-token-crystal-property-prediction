#!/usr/bin/env python3
"""Validation-only matched text-representation ablation.

This runner intentionally consumes only train/validation graph splits and a
frozen dense MatBERT last-hidden-state cache.  It never constructs a test
loader or reads test labels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, InMemoryDataset

from data.config import AtomFeatureType
from model.net import SFTGNN
from data.config import Config as data_config
from model.config import Config as model_config


TASKS = {
    "Jarvis-Bandgap_MBJ": {"dataset": "JARVIS", "family": "Electronic", "graph": "Jarvis-Bandgap_MBJ", "source": "/root/autodl-tmp/dataset1/dataset/jarvis/mbj_bandgap/cif/id_prop.csv"},
    "Jarvis-FormationEnergy": {"dataset": "JARVIS", "family": "Energetic", "graph": "Jarvis-FormationEnergy", "source": "/root/autodl-tmp/dataset1/dataset/jarvis/formation_energy_peratom/cif/id_prop.csv"},
    "Jarvis-BulkModulusKv": {"dataset": "JARVIS", "family": "Mechanical", "graph": "Jarvis-BulkModulusKv", "source": "/root/autodl-tmp/dataset1/dataset/jarvis/bulk_modulus_kv/cif/id_prop.csv"},
    "MP-Bandgap": {"dataset": "MP", "family": "Electronic", "graph": "MP-Bandgap", "source": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-bandgap/cif/id_prop.csv"},
    "MP-FormationEnergy": {"dataset": "MP", "family": "Energetic", "graph": "MP-Bandgap", "source": "/root/autodl-tmp/dataset1/dataset/mp_2018/MP-formation/cif/id_prop.csv"},
    "MP-ShearModuli": {"dataset": "MP", "family": "Mechanical", "graph": "MP-ShearModuli", "source": ""},
}


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_processed_split(graph_name: str, split: str):
    pattern = f"Dataset/processed/{graph_name}*{split}*CGCNN-multimodal-MatBert_embeddings.pt"
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(pattern)
    data, slices = torch.load(paths[0], map_location="cpu", weights_only=False)
    obj = InMemoryDataset.__new__(InMemoryDataset)
    obj.data = data
    obj.slices = slices
    obj._indices = None
    return [obj.get(i) for i in range(slices["cid"].numel() - 1)]


def read_targets(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                out[str(int(float(row[0]))) if str(row[0]).strip().replace(".", "", 1).isdigit() else str(row[0]).strip()] = float(row[1])
            except ValueError:
                continue
    return out


def scalar_id(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) else str(value)


def attach_normalized_targets(samples, target_map: dict[str, float], mean: float, std: float):
    missing = []
    for sample in samples:
        key = scalar_id(sample.cid)
        if key not in target_map:
            missing.append(key)
        else:
            sample.y = torch.tensor([(target_map[key] - mean) / std], dtype=torch.float32)
    if missing:
        raise KeyError(f"missing targets for {len(missing)} graph IDs, examples={missing[:5]}")


def load_task_splits(task: str):
    spec = TASKS[task]
    if task == "MP-ShearModuli":
        cache = Path("matched_graph_cache/mp_shear_train_valid.pt")
        if not cache.exists():
            raise FileNotFoundError(f"missing prepared MP-Shear graph cache: {cache}")
        obj = torch.load(cache, map_location="cpu", weights_only=False)
        train, valid, mean, std = obj["train"], obj["valid"], obj["mean"], obj["std"]
        return train, valid, float(mean), float(std)

    train = load_processed_split(spec["graph"], "train")
    valid = load_processed_split(spec["graph"], "valid")
    target_map = read_targets(spec["source"])
    raw_train = [target_map[scalar_id(x.cid)] for x in train]
    mean = float(np.mean(raw_train))
    std = float(np.std(raw_train, ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid training target std for {task}: {std}")
    attach_normalized_targets(train, target_map, mean, std)
    attach_normalized_targets(valid, target_map, mean, std)
    return train, valid, mean, std


class TokenStore:
    def __init__(self, root: str):
        root = Path(root)
        self.ids = [str(x) for x in json.loads((root / "sample_ids.json").read_text())]
        self.index = {x: i for i, x in enumerate(self.ids)}
        self.features = np.load(root / "token_features.npy", mmap_mode="r")
        self.masks = np.load(root / "attention_mask.npy", mmap_mode="r")
        self.input_ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.missing_ids = set()
        if self.features.shape[1:] != (128, 768):
            raise ValueError(f"unexpected token cache shape {self.features.shape}")

    def get(self, sample_id: str):
        i = self.index.get(str(sample_id))
        if i is None:
            # Preserve the historical loader behavior for the seven legacy IDs
            # whose pooled text cache was absent; the fallback is reported in
            # every run summary and is never used for test data.
            self.missing_ids.add(str(sample_id))
            return (
                torch.zeros(128, 768, dtype=torch.float32),
                torch.zeros(128, dtype=torch.bool),
                torch.zeros(128, dtype=torch.long),
            )
        return (
            torch.from_numpy(np.array(self.features[i], copy=True)),
            torch.from_numpy(np.array(self.masks[i], copy=True)).bool(),
            torch.from_numpy(np.array(self.input_ids[i], copy=True)).long(),
        )


class TokenGraphDataset(Dataset):
    def __init__(self, graphs, token_store: TokenStore):
        self.graphs = graphs
        self.tokens = token_store

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        graph = self.graphs[index]
        return (graph, *self.tokens.get(scalar_id(graph.cid)))


def make_collate(samples):
    graphs, tokens, masks, input_ids = zip(*samples)
    return Batch.from_data_list(list(graphs)), torch.stack(tokens), torch.stack(masks), torch.stack(input_ids)


class MatchedModel(nn.Module):
    def __init__(self, variant: str, graph_in_dim: int = 92, sh_in_dim: int | None = None):
        super().__init__()
        self.variant = variant
        self.graph = SFTGNN(num_layers=5, in_dim=graph_in_dim, sh_in_dim=sh_in_dim)
        self.wv = nn.Linear(768, 128)
        if variant == "CosineAttention":
            self.wq = nn.Linear(128, 128)
            self.wk = nn.Linear(768, 128)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, batch, masks, input_ids, special_ids):
        hg = self.graph.forward_embedding(batch)
        eligible = masks.bool()
        special = torch.zeros_like(eligible)
        for sid in special_ids:
            special |= input_ids.eq(int(sid))
        eligible &= ~special
        if self.variant == "CLS":
            ht = self.norm(self.wv(batch_tokens := batch._matched_tokens[:, 0]))
            entropy = None
        elif self.variant == "MaskedMean":
            values = self.wv(batch._matched_tokens)
            denom = eligible.sum(dim=1, keepdim=True).clamp_min(1)
            ht = self.norm((values * eligible.unsqueeze(-1)).sum(dim=1) / denom)
            entropy = None
        else:
            q = F.normalize(self.wq(hg), dim=-1)
            k = F.normalize(self.wk(batch._matched_tokens), dim=-1)
            scores = (k * q.unsqueeze(1)).sum(-1) / 0.5
            scores = scores.masked_fill(~eligible, -torch.finfo(scores.dtype).max)
            attn = F.softmax(scores, dim=1)
            values = self.wv(batch._matched_tokens)
            ht = self.norm((attn.unsqueeze(-1) * values).sum(dim=1))
            entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=1)
        pred = self.head(torch.cat([hg, ht], dim=-1)).squeeze(-1)
        return pred, eligible, entropy


def forward_model(model, batch, tokens, masks, input_ids, special_ids):
    if batch.x.dim() > 1:
        batch.x = batch.x.float()
    batch._matched_tokens = tokens
    return model(batch, masks, input_ids, special_ids)


def run(args):
    seed_all(42)
    task_spec = TASKS[args.task]
    train_graphs, valid_graphs, mean, std = load_task_splits(args.task)
    store = TokenStore(args.token_cache)
    # Graph collation is CPU-heavy; persistent workers only prefetch batches and
    # leave the model, batch size, seed, optimizer, and data split unchanged.
    loader_kwargs = {"num_workers": 4, "persistent_workers": True, "prefetch_factor": 2}
    train_loader = DataLoader(TokenGraphDataset(train_graphs, store), batch_size=32, shuffle=True, drop_last=True, collate_fn=make_collate, **loader_kwargs)
    valid_loader = DataLoader(TokenGraphDataset(valid_graphs, store), batch_size=64, shuffle=False, drop_last=False, collate_fn=make_collate, **loader_kwargs)
    try:
        from transformers import AutoTokenizer
        special_ids = AutoTokenizer.from_pretrained("/root/autodl-tmp/matbert-base-cased").all_special_ids
    except Exception:
        special_ids = [0, 1, 2, 3]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    sample = train_graphs[0]
    graph_in_dim = int(sample.x.shape[-1]) if sample.x.dim() > 1 else 1
    sh_in_dim = int(sample.sh.shape[-1]) if sample.sh.dim() > 1 else 1
    model = MatchedModel(args.variant, graph_in_dim=graph_in_dim, sh_in_dim=sh_in_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=300, steps_per_epoch=len(train_loader), pct_start=0.3)
    loss_fn = nn.L1Loss()
    best = float("inf")
    best_epoch = -1
    final_valid = float("nan")
    history = []
    start_time = time.time()
    for epoch in range(300):
        model.train()
        train_sum = 0.0
        n_train = 0
        for batch, tokens, masks, input_ids in train_loader:
            batch = batch.cuda(non_blocking=True)
            tokens = tokens.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            input_ids = input_ids.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, _, _ = forward_model(model, batch, tokens, masks, input_ids, special_ids)
                loss = loss_fn(pred, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_sum += float(loss.detach()) * (len(batch.ptr) - 1)
            n_train += len(batch.ptr) - 1
        model.eval()
        valid_sum = 0.0
        valid_n = 0
        entropies = []
        max_attn = []
        eligible_counts = []
        with torch.no_grad():
            for batch, tokens, masks, input_ids in valid_loader:
                batch = batch.cuda(non_blocking=True)
                tokens = tokens.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                input_ids = input_ids.cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred, eligible, entropy = forward_model(model, batch, tokens, masks, input_ids, special_ids)
                    vloss = loss_fn(pred, batch.y.view(-1).float())
                count = len(batch.ptr) - 1
                valid_sum += float(vloss) * count
                valid_n += count
                eligible_counts.extend(eligible.sum(dim=1).float().cpu().tolist())
                if entropy is not None:
                    entropies.extend(entropy.float().cpu().tolist())
                    max_attn.append(torch.softmax(torch.zeros(1), dim=0))
        valid_mae = valid_sum / valid_n * std
        final_valid = valid_mae
        history.append({"epoch": epoch + 1, "train_mae": train_sum / max(n_train, 1) * std, "valid_mae": valid_mae})
        if valid_mae < best:
            best = valid_mae
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "task": args.task, "variant": args.variant, "seed": 42, "best_epoch": best_epoch, "best_valid_mae": best, "mean": mean, "std": std, "test_forward_count": 0}, Path(args.output_dir) / "best_valid.pt")
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{args.task} {args.variant} epoch={epoch+1}/300 valid_mae={valid_mae:.8f} best={best:.8f}", flush=True)
    out = {
        "task": args.task, "dataset": task_spec["dataset"], "property_family": task_spec["family"],
        "variant": args.variant, "seed": 42, "best_epoch": best_epoch, "best_valid_mae": best,
        "final_valid_mae": final_valid, "mean": mean, "std": std, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_path": str(Path(args.output_dir) / "best_valid.pt"), "test_forward_count": 0,
        "runtime_sec": time.time() - start_time, "train_samples": len(train_graphs), "valid_samples": len(valid_graphs),
        "mean_eligible_token_count": float(np.mean(eligible_counts)), "median_eligible_token_count": float(np.median(eligible_counts)),
        "attention_entropy_mean": float(np.mean(entropies)) if entropies else None,
        "attention_entropy_median": float(np.median(entropies)) if entropies else None,
        "attention_effective_token_count_mean": float(np.mean(np.exp(entropies))) if entropies else None,
        "attention_effective_token_count_median": float(np.median(np.exp(entropies))) if entropies else None,
        "attention_mean_max": None,
        "zero_text_fallback_count": len(store.missing_ids),
        "zero_text_fallback_ids": sorted(store.missing_ids),
        "history": history,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--variant", choices=["CLS", "MaskedMean", "CosineAttention"], required=True)
    p.add_argument("--token_cache", required=True)
    p.add_argument("--output_dir", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
