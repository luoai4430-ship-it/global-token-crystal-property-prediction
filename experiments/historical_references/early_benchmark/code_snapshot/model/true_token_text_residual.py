from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import SFTGNN


CONTROL_TO_MODE = {
    "A": "cls",
    "B": "masked_mean",
    "C": "true_token",
    "D": "true_token",
    "C0": "true_token",
    "C1": "cosine_true_token",
    "D1": "cosine_true_token",
}


class TrueTokenTextResidual(nn.Module):
    """Graph prediction plus a correction from a strict MatBERT token sequence."""

    def __init__(
        self,
        sftgnn: SFTGNN,
        control: str,
        graph_dim: int = 128,
        token_dim: int = 768,
        attention_dim: int = 128,
        residual_hidden_dim: int = 256,
        temperature: float = 1.0,
        special_token_ids: Sequence[int] = (0, 2, 3),
    ) -> None:
        super().__init__()
        if control not in CONTROL_TO_MODE:
            raise ValueError(f"unknown control: {control}")
        self.sftgnn = sftgnn
        self.control = control
        self.mode = CONTROL_TO_MODE[control]
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self.query_projection = nn.Linear(graph_dim, attention_dim, bias=False)
        self.key_projection = nn.Linear(token_dim, attention_dim, bias=False)
        self.value_projection = nn.Linear(token_dim, attention_dim, bias=False)
        self.text_norm = nn.LayerNorm(attention_dim)
        self.graph_head = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim + attention_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(residual_hidden_dim, 1),
        )
        self.scale = attention_dim ** -0.5

    @staticmethod
    def _validate_tokens(
        text_tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if text_tokens.ndim != 3:
            raise ValueError(
                f"text_tokens must be [B,L,768], got {tuple(text_tokens.shape)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must be [B,L], got {tuple(attention_mask.shape)}"
            )
        if text_tokens.shape[:2] != attention_mask.shape:
            raise ValueError("text token and attention-mask shapes do not align")
        if text_tokens.shape[1] <= 1:
            raise ValueError("text token sequence length must be greater than one")
        mask = attention_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must have at least one valid text token")
        return mask

    def _attention_mask(
        self,
        mask: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        special_mask = torch.zeros_like(mask)
        if input_ids is not None:
            if input_ids.shape != mask.shape:
                raise ValueError("input IDs and attention-mask shapes do not align")
            for token_id in self.special_token_ids:
                special_mask |= input_ids.eq(token_id)
        if self.mode != "cosine_true_token":
            return mask, special_mask
        if input_ids is None:
            raise ValueError("cosine true-token attention requires input IDs")
        eligible = mask & ~special_mask
        if not bool(eligible.any(dim=1).all()):
            raise ValueError("every sample must contain a non-special attention token")
        return eligible, special_mask

    def _pool_text(
        self,
        graph_embedding: torch.Tensor,
        text_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        values = self.value_projection(text_tokens.float())
        batch_size, sequence_length = mask.shape
        diagnostics = {
            "attention_logits": values.new_full(
                (batch_size, sequence_length), float("nan")
            ),
            "q_norm_before_normalization": values.new_full(
                (batch_size,), float("nan")
            ),
            "q_norm_after_normalization": values.new_full(
                (batch_size,), float("nan")
            ),
            "k_norm_before_normalization": values.new_full(
                (batch_size, sequence_length), float("nan")
            ),
            "k_norm_after_normalization": values.new_full(
                (batch_size, sequence_length), float("nan")
            ),
        }
        if self.mode == "cls":
            if not bool(mask[:, 0].all()):
                raise ValueError("CLS position must be valid for every sample")
            attention = values.new_zeros(mask.shape)
            attention[:, 0] = 1.0
        elif self.mode == "masked_mean":
            attention = mask.to(values.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0)
        elif self.mode == "true_token":
            query = self.query_projection(graph_embedding).unsqueeze(1)
            keys = self.key_projection(text_tokens.float())
            scores = (query * keys).sum(dim=-1) * self.scale
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
        else:
            query = self.query_projection(graph_embedding).unsqueeze(1)
            keys = self.key_projection(text_tokens.float())
            diagnostics["q_norm_before_normalization"] = query.norm(dim=-1).squeeze(1)
            diagnostics["k_norm_before_normalization"] = keys.norm(dim=-1)
            normalized_query = F.normalize(query, p=2.0, dim=-1, eps=1e-12)
            normalized_keys = F.normalize(keys, p=2.0, dim=-1, eps=1e-12)
            diagnostics["q_norm_after_normalization"] = normalized_query.norm(
                dim=-1
            ).squeeze(1)
            diagnostics["k_norm_after_normalization"] = normalized_keys.norm(
                dim=-1
            )
            scores = (normalized_query * normalized_keys).sum(dim=-1)
            scores = scores / self.temperature
            diagnostics["attention_logits"] = scores
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        return self.text_norm(pooled), attention, diagnostics

    def forward(
        self,
        graph,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        zero_text: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw_mask = self._validate_tokens(text_tokens, attention_mask)
        mask, special_mask = self._attention_mask(raw_mask, input_ids)
        graph_embedding = self.sftgnn.forward_embedding(graph)
        graph_prediction = self.graph_head(graph_embedding).squeeze(-1)
        pooled_text, attention, diagnostics = self._pool_text(
            graph_embedding, text_tokens, mask
        )
        if zero_text:
            pooled_text = torch.zeros_like(pooled_text)
        residual = self.residual_head(
            torch.cat((graph_embedding, pooled_text), dim=-1)
        ).squeeze(-1)
        final_prediction = graph_prediction + residual
        probabilities = attention.clamp_min(torch.finfo(attention.dtype).tiny)
        entropy = -(attention * probabilities.log()).sum(dim=-1)
        valid_token_count = mask.sum(dim=-1)
        entropy_denominator = valid_token_count.to(entropy.dtype).log()
        normalized_entropy = torch.where(
            valid_token_count > 1,
            entropy / entropy_denominator.clamp_min(torch.finfo(entropy.dtype).eps),
            torch.zeros_like(entropy),
        )
        output = {
            "graph_embedding": graph_embedding,
            "text_embedding": pooled_text,
            "graph_prediction": graph_prediction,
            "residual": residual,
            "final_prediction": final_prediction,
            "attention_weights": attention,
            "attention_eligible_mask": mask,
            "special_token_mask": special_mask,
            "attention_entropy": entropy,
            "normalized_attention_entropy": normalized_entropy,
            "attention_max_weight": attention.max(dim=-1).values,
            "valid_token_count": valid_token_count,
            "effective_token_count": entropy.exp(),
            "special_token_attention_mass": (
                attention * special_mask.to(attention.dtype)
            ).sum(dim=-1),
            "padding_attention_mass": (
                attention * (~raw_mask).to(attention.dtype)
            ).sum(dim=-1),
        }
        output.update(diagnostics)
        return output
