"""CPU regression check for the inference-only query-zero correction."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedGraph(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_embedding(self, graph):
        return graph


def family(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GCTAFamily")
    namespace = {
        "torch": torch, "nn": nn, "F": F,
        "VARIANTS": ("B1_NaiveConcat", "P2_GCTA"), "TEMPERATURE": 0.5,
        "base": SimpleNamespace(SFTGNN=FixedGraph, data_config=SimpleNamespace(getAtomDim=lambda: 92)),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["GCTAFamily"]("P2_GCTA", (0, 1, 2))


def main():
    root = Path("/root/autodl-tmp/TextResidualSHFMat/scripts")
    before = family(root / "run_gcta_fast_screen.pre_query_zero_fix.py")
    after = family(root / "run_gcta_fast_screen.py")
    checkpoint = torch.load(
        "/root/gcta_fast_screen_seed42_20260902/runs/Jarvis-Bandgap_MBJ/P2_GCTA/seed_42/best_valid.pt",
        map_location="cpu", weights_only=False,
    )
    state = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.startswith("sftgnn.")}
    before.load_state_dict(state, strict=True)
    after.load_state_dict(state, strict=True)
    before.eval()
    after.eval()
    torch.manual_seed(2718)
    graph = torch.randn(3, 128)
    tokens = torch.randn(3, 8, 768)
    ids = torch.arange(3, 11).expand(3, -1)
    mask = torch.ones(3, 8, dtype=torch.long)
    with torch.no_grad():
        for kwargs in ({}, {"query_cls_override": tokens.flip(0)[:, 0, :]}):
            left = before(graph, tokens, mask, ids, **kwargs)
            right = after(graph, tokens, mask, ids, **kwargs)
            assert all(torch.equal(left[k], right[k]) for k in left), "nonzero-control path changed"
        normal = after(graph, tokens, mask, ids)
        old_zero = before(graph, tokens, mask, ids, query_cls_zero=True)
        zero = after(graph, tokens, mask, ids, query_cls_zero=True)
        assert torch.count_nonzero(zero["qc"]) == 0
        assert torch.equal(zero["cls_embedding"], normal["cls_embedding"])
        keys = F.normalize(after.key_projection(tokens), dim=-1)
        query = F.normalize(zero["qg"], dim=-1).unsqueeze(1)
        expected = ((keys * query).sum(-1) / .5).softmax(-1)
        assert torch.equal(zero["attention_weights"], expected)
    print(json.dumps({
        "normal_and_shuffled_paths_exactly_unchanged": True,
        "prediction_head_cls_unchanged": True,
        "zero_qc_max_abs": float(zero["qc"].abs().max()),
        "attention_equals_graph_only_query": True,
        "old_raw_zero_qc_l2": float(old_zero["qc"][0].norm()),
        "training_runs": 0,
        "test_forward_count": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
