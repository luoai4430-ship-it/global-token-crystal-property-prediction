#!/usr/bin/env python3
"""Prepare train/validation-only graph and description inputs for the matched ablation."""
from __future__ import annotations

import csv
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset

from data.graph import atoms2graph
from jarvis.core.atoms import Atoms


BASE = Path('/root/autodl-tmp')
ROOT = Path('/root/autodl-tmp/TextResidualSHFMat')
OUT = ROOT / 'matched_ablation_inputs'
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / 'matched_graph_cache').mkdir(parents=True, exist_ok=True)
MISSING = {}

TASKS = {
    'Jarvis-Bandgap_MBJ': (ROOT / 'Dataset/processed/Jarvis-Bandgap_MBJ', BASE / 'dataset1/dataset/jarvis/mbj_bandgap/description.csv'),
    'Jarvis-FormationEnergy': (ROOT / 'Dataset/processed/Jarvis-FormationEnergy', BASE / 'dataset1/dataset/jarvis/formation_energy_peratom/description.csv'),
    'Jarvis-BulkModulusKv': (ROOT / 'Dataset/processed/Jarvis-BulkModulusKv', BASE / 'dataset1/dataset/jarvis/bulk_modulus_kv/description.csv'),
    'MP-Bandgap': (ROOT / 'Dataset/processed/MP-Bandgap', BASE / 'dataset1/dataset/mp_2018/MP-bandgap/description.csv'),
    'MP-FormationEnergy': (ROOT / 'Dataset/processed/MP-Bandgap', BASE / 'dataset1/dataset/mp_2018/MP-formation/description.csv'),
}


def processed_ids(prefix: Path):
    ids = []
    for split in ('train', 'valid'):
        paths = glob.glob(str(prefix) + f'*{split}*CGCNN-multimodal-MatBert_embeddings.pt')
        if not paths:
            raise FileNotFoundError(f'no processed {split} file for {prefix}')
        data, slices = torch.load(paths[0], map_location='cpu', weights_only=False)
        cid_start, cid_end = slices['cid'][0].item(), slices['cid'][-1].item()
        values = data.cid[cid_start:cid_end]
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        ids.extend([str(int(x)) for x in values])
    return set(ids)


def filter_description(source: Path, ids: set[str], out_name: str, fallbacks=()):
    with source.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
        fields = handle  # keep the context alive only for readability
    if not rows:
        raise ValueError(source)
    id_col = 'Id' if 'Id' in rows[0] else 'id'
    text_col = 'Description' if 'Description' in rows[0] else 'text'
    by_id = {str(row[id_col]).strip(): row for row in rows if str(row[text_col]).strip()}
    missing = ids - set(by_id)
    for fallback in fallbacks:
        if not missing:
            break
        with Path(fallback).open(newline='', encoding='utf-8-sig') as handle:
            fallback_rows = list(csv.DictReader(handle))
        for row in fallback_rows:
            key = str(row[id_col]).strip()
            if key in missing and str(row[text_col]).strip():
                by_id[key] = row
        missing = ids - set(by_id)
    if missing:
        MISSING[out_name] = sorted(missing, key=lambda x: int(x) if x.isdigit() else x)
        print(f'WARNING: {source} missing {len(missing)} IDs; runner will use zero-token fallback: {sorted(missing)[:10]}')
    kept = [by_id[key] for key in sorted(ids & set(by_id), key=lambda x: int(x) if x.isdigit() else x)]
    out = OUT / out_name
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept)
    print(f'{out}: {len(kept)} rows, id={id_col}, text={text_col}')
    return out, id_col, text_col


def prepare_shear():
    source = BASE / 'dataset1/dataset/mp_2018_small/shear/description.csv'
    cif_root = BASE / 'dataset1/dataset/mp_2018_small/shear/cif'
    with source.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    id_col = 'Id' if 'Id' in rows[0] else 'id'
    prop_col = 'prop' if 'prop' in rows[0] else 'shear modulus'
    random.Random(123).shuffle(rows)
    n_train = int(0.8 * len(rows))
    n_valid = int(0.1 * len(rows))
    train_rows, valid_rows = rows[:n_train], rows[n_train:n_train + n_valid]
    raw_train = [float(row[prop_col]) for row in train_rows]
    mean = float(np.mean(np.log10(np.maximum(raw_train, 1e-30))))
    std = float(np.std(np.log10(np.maximum(raw_train, 1e-30)), ddof=1))

    def make_graph(row):
        sid = str(row[id_col]).strip()
        cif = cif_root / f'{sid}.cif'
        atoms = Atoms.from_cif(str(cif), use_cif2cell=False)
        graph = atoms2graph(atoms.to_dict())
        target = math.log10(max(float(row[prop_col]), 1e-30))
        graph.cid = int(float(sid))
        graph.y = torch.tensor([(target - mean) / std], dtype=torch.float32)
        return graph

    train = [make_graph(row) for row in train_rows]
    valid = [make_graph(row) for row in valid_rows]
    cache = ROOT / 'matched_graph_cache/mp_shear_train_valid.pt'
    torch.save({'train': train, 'valid': valid, 'mean': mean, 'std': std, 'split_seed': 123, 'test_built': False}, cache)
    ids = {str(row[id_col]).strip() for row in train_rows + valid_rows}
    filter_description(source, ids, 'MP-ShearModuli.csv')
    print(f'{cache}: train={len(train)} valid={len(valid)} mean={mean} std={std}')


def main():
    shared_mp_ids = processed_ids(TASKS['MP-Bandgap'][0])
    for task, (prefix, source) in TASKS.items():
        ids = processed_ids(prefix)
        # MP-FormationEnergy uses the exact MP-Bandgap graph split.
        fallbacks = ()
        if task == 'Jarvis-Bandgap_MBJ':
            fallbacks = (
                BASE / 'dataset1/dataset/jarvis/formation_energy_peratom/description.csv',
                BASE / 'dataset1/dataset/jarvis/bulk_modulus_kv/description.csv',
            )
        filter_description(source, ids, f'{task}.csv', fallbacks=fallbacks)
    prepare_shear()
    (OUT / 'missing_text_ids.json').write_text(json.dumps(MISSING, indent=2), encoding='utf-8')
    print(f'missing-text manifest: {OUT / "missing_text_ids.json"}')


if __name__ == '__main__':
    main()
