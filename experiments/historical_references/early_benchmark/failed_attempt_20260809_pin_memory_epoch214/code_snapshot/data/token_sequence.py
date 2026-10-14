from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def scalar_sample_id(value) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"sample ID tensor is not scalar: {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"sample ID collection is not scalar: {value}")
        value = value[0]
    return str(value)


def graph_dataset_ids(dataset) -> list[str]:
    return [scalar_sample_id(dataset[index].cid) for index in range(len(dataset))]


class TokenFeatureStore:
    """Read a strict dense MatBERT token cache without changing its token axis."""

    def __init__(self, directory: Path, expected_length: int = 128):
        self.directory = Path(directory)
        self.metadata = json.loads(
            (self.directory / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("status") != "complete":
            raise ValueError(f"token cache is incomplete: {self.directory}")
        if not self.metadata.get("uses_last_hidden_state"):
            raise ValueError("cache was not declared as outputs.last_hidden_state")
        if self.metadata.get("forbidden_global_pooling_used") is not False:
            raise ValueError("cache provenance does not reject global pooling")

        self.sample_ids = [
            str(value)
            for value in json.loads(
                (self.directory / "sample_ids.json").read_text(encoding="utf-8")
            )
        ]
        self.id_to_index = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }
        if len(self.id_to_index) != len(self.sample_ids):
            raise ValueError("token cache contains duplicate sample IDs")

        self.features = np.load(
            self.directory / "token_features.npy", mmap_mode="r"
        )
        self.attention_masks = np.load(
            self.directory / "attention_mask.npy", mmap_mode="r"
        )
        self.input_ids = np.load(self.directory / "input_ids.npy", mmap_mode="r")
        expected_features = (len(self.sample_ids), expected_length, 768)
        expected_sequence = (len(self.sample_ids), expected_length)
        if self.features.shape != expected_features:
            raise ValueError(
                f"expected token cache {expected_features}, got {self.features.shape}"
            )
        if self.attention_masks.shape != expected_sequence:
            raise ValueError("attention-mask cache shape is invalid")
        if self.input_ids.shape != expected_sequence:
            raise ValueError("input-ID cache shape is invalid")
        if self.features.dtype != np.float16:
            raise ValueError(f"token cache must be float16, got {self.features.dtype}")

    @property
    def sequence_length(self) -> int:
        return int(self.features.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.features.shape[2])

    def require_ids(self, sample_ids: Sequence[str], split: str) -> None:
        missing = [str(value) for value in sample_ids if str(value) not in self.id_to_index]
        if missing:
            raise ValueError(
                f"token cache is missing {len(missing)} {split} IDs, e.g. {missing[:5]}"
            )

    def fetch(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.id_to_index.get(str(sample_id))
        if index is None:
            raise KeyError(f"sample {sample_id} is absent from token cache")
        tokens = torch.from_numpy(np.array(self.features[index], copy=True))
        mask = torch.from_numpy(
            np.array(self.attention_masks[index], copy=True)
        ).bool()
        input_ids = torch.from_numpy(
            np.array(self.input_ids[index], copy=True)
        ).long()
        if tokens.ndim != 2 or tokens.shape != (self.sequence_length, self.hidden_size):
            raise AssertionError("single-sample token features must be [L,768]")
        if mask.ndim != 1 or input_ids.ndim != 1:
            raise AssertionError("single-sample masks and input IDs must be [L]")
        if not bool(mask.any()):
            raise ValueError(f"sample {sample_id} has no valid tokens")
        return tokens, mask, input_ids


@dataclass
class TokenGraphSample:
    graph: object
    sample_id: str
    text_sample_id: str
    text_tokens: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor


@dataclass
class TokenGraphBatch:
    graph: Batch
    sample_ids: list[str]
    text_sample_ids: list[str]
    text_tokens: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor

    def to(self, device: torch.device) -> "TokenGraphBatch":
        self.graph = self.graph.to(device)
        self.text_tokens = self.text_tokens.to(device, non_blocking=True)
        self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        self.input_ids = self.input_ids.to(device, non_blocking=True)
        return self


class TokenGraphDataset(Dataset):
    def __init__(
        self,
        graph_dataset,
        token_store: TokenFeatureStore,
        text_mapping: Mapping[str, str] | None = None,
        split: str = "dataset",
    ):
        self.graph_dataset = graph_dataset
        self.token_store = token_store
        self.sample_ids = graph_dataset_ids(graph_dataset)
        token_store.require_ids(self.sample_ids, split)
        self.text_mapping = (
            {str(key): str(value) for key, value in text_mapping.items()}
            if text_mapping is not None
            else None
        )
        if self.text_mapping is not None:
            if set(self.text_mapping) != set(self.sample_ids):
                raise ValueError("text mapping must exactly cover the split")
            mapped = [self.text_mapping[sample_id] for sample_id in self.sample_ids]
            if any(source == target for source, target in zip(self.sample_ids, mapped)):
                raise ValueError("shuffled token mapping contains a fixed point")
            if len(set(mapped)) != len(mapped):
                raise ValueError("shuffled token mapping is not one-to-one")
            token_store.require_ids(mapped, f"{split} shuffled")

    def __len__(self) -> int:
        return len(self.graph_dataset)

    def __getitem__(self, index: int) -> TokenGraphSample:
        graph = self.graph_dataset[index].clone()
        sample_id = self.sample_ids[index]
        text_sample_id = (
            self.text_mapping[sample_id] if self.text_mapping is not None else sample_id
        )
        tokens, mask, input_ids = self.token_store.fetch(text_sample_id)
        return TokenGraphSample(
            graph=graph,
            sample_id=sample_id,
            text_sample_id=text_sample_id,
            text_tokens=tokens,
            attention_mask=mask,
            input_ids=input_ids,
        )


def collate_token_graph_samples(samples: Sequence[TokenGraphSample]) -> TokenGraphBatch:
    if not samples:
        raise ValueError("cannot collate an empty token-graph batch")
    tokens = torch.stack([sample.text_tokens for sample in samples], dim=0)
    masks = torch.stack([sample.attention_mask for sample in samples], dim=0)
    input_ids = torch.stack([sample.input_ids for sample in samples], dim=0)
    if tokens.ndim != 3:
        raise AssertionError(f"batch token features must be [B,L,768], got {tokens.shape}")
    if masks.ndim != 2:
        raise AssertionError(f"batch attention mask must be [B,L], got {masks.shape}")
    if tokens.shape[:2] != masks.shape or input_ids.shape != masks.shape:
        raise AssertionError("batched token features, masks and input IDs do not align")
    if tokens.shape[1] <= 1:
        raise AssertionError("batched token sequence length must exceed one")
    return TokenGraphBatch(
        graph=Batch.from_data_list([sample.graph for sample in samples]),
        sample_ids=[sample.sample_id for sample in samples],
        text_sample_ids=[sample.text_sample_id for sample in samples],
        text_tokens=tokens,
        attention_mask=masks,
        input_ids=input_ids,
    )
