from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyMLP(nn.Module):
    def __init__(self, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(sers_dim + tran_dim + meta_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is not None:
            sers = sers * modality_mask[:, 0:1]
            tran = tran * modality_mask[:, 1:2]
            meta = meta * modality_mask[:, 2:3]
        logits = self.backbone(torch.cat([sers, tran, meta], dim=-1))
        return {"logits": logits, "aux_loss": logits.new_zeros(())}


class LateMLP(nn.Module):
    def __init__(self, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.sers_enc = MLPBlock(sers_dim, d_model * 2, d_model, dropout)
        self.tran_enc = MLPBlock(tran_dim, d_model * 2, d_model, dropout)
        self.meta_enc = MLPBlock(meta_dim, d_model * 2, d_model, dropout)
        self.head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is None:
            modality_mask = torch.ones(sers.size(0), 3, device=sers.device)
        hs = self.sers_enc(sers) * modality_mask[:, 0:1]
        ht = self.tran_enc(tran) * modality_mask[:, 1:2]
        hm = self.meta_enc(meta) * modality_mask[:, 2:3]
        logits = self.head(torch.cat([hs, ht, hm], dim=-1))
        return {"logits": logits, "aux_loss": logits.new_zeros(())}


class GatedFusion(nn.Module):
    def __init__(self, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.sers_enc = MLPBlock(sers_dim, d_model * 2, d_model, dropout)
        self.tran_enc = MLPBlock(tran_dim, d_model * 2, d_model, dropout)
        self.meta_enc = MLPBlock(meta_dim, d_model * 2, d_model, dropout)
        self.gate = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3))
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is None:
            modality_mask = torch.ones(sers.size(0), 3, device=sers.device)
        hs = self.sers_enc(sers) * modality_mask[:, 0:1]
        ht = self.tran_enc(tran) * modality_mask[:, 1:2]
        hm = self.meta_enc(meta) * modality_mask[:, 2:3]
        stack = torch.stack([hs, ht, hm], dim=1)
        gate_logits = self.gate(torch.cat([hs, ht, hm], dim=-1)).masked_fill(modality_mask <= 0, -1e9)
        gate = torch.softmax(gate_logits, dim=-1)
        fused = torch.sum(stack * gate.unsqueeze(-1), dim=1)
        context = torch.cat([hs, ht, hm], dim=-1).mean(dim=-1, keepdim=True).expand_as(fused)
        logits = self.head(torch.cat([fused, context], dim=-1))
        return {"logits": logits, "aux_loss": logits.new_zeros(())}


class TensorFusion(nn.Module):
    def __init__(self, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.sers_enc = MLPBlock(sers_dim, d_model * 2, d_model, dropout)
        self.tran_enc = MLPBlock(tran_dim, d_model * 2, d_model, dropout)
        self.meta_enc = MLPBlock(meta_dim, d_model * 2, d_model, dropout)
        pair_dim = d_model * d_model
        self.pair_proj = nn.Sequential(nn.Linear(pair_dim * 3, d_model * 2), nn.LayerNorm(d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, 2))

    @staticmethod
    def _outer_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,bj->bij", a, b).reshape(a.size(0), -1)

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is None:
            modality_mask = torch.ones(sers.size(0), 3, device=sers.device)
        hs = self.sers_enc(sers) * modality_mask[:, 0:1]
        ht = self.tran_enc(tran) * modality_mask[:, 1:2]
        hm = self.meta_enc(meta) * modality_mask[:, 2:3]
        pair_feat = self.pair_proj(torch.cat([self._outer_flat(hs, ht), self._outer_flat(hs, hm), self._outer_flat(ht, hm)], dim=-1))
        logits = self.head(torch.cat([hs, ht, hm, pair_feat + hs * ht * hm], dim=-1))
        return {"logits": logits, "aux_loss": logits.new_zeros(())}


class CrossAttentionFusion(nn.Module):
    def __init__(self, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.sers_enc = MLPBlock(sers_dim, d_model * 2, d_model, dropout)
        self.tran_enc = MLPBlock(tran_dim, d_model * 2, d_model, dropout)
        self.meta_enc = MLPBlock(meta_dim, d_model * 2, d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is None:
            modality_mask = torch.ones(sers.size(0), 3, device=sers.device)
        stack = torch.stack([self.sers_enc(sers), self.tran_enc(tran), self.meta_enc(meta)], dim=1)
        stack = stack * modality_mask.unsqueeze(-1)
        attn_out, _ = self.attn(stack, stack, stack)
        fused = self.norm(stack + attn_out).reshape(sers.size(0), -1)
        logits = self.head(fused)
        return {"logits": logits, "aux_loss": logits.new_zeros(())}


def build_deep_baseline(name: str, sers_dim: int, tran_dim: int, meta_dim: int, d_model: int = 128, dropout: float = 0.15) -> nn.Module:
    if name == "early_mlp":
        return EarlyMLP(sers_dim, tran_dim, meta_dim, d_model, dropout)
    if name == "late_mlp":
        return LateMLP(sers_dim, tran_dim, meta_dim, d_model, dropout)
    if name == "gated_fusion":
        return GatedFusion(sers_dim, tran_dim, meta_dim, d_model, dropout)
    if name == "tensor_fusion":
        return TensorFusion(sers_dim, tran_dim, meta_dim, d_model=min(d_model, 96), dropout=dropout)
    if name == "cross_attention_fusion":
        return CrossAttentionFusion(sers_dim, tran_dim, meta_dim, d_model, dropout)
    raise ValueError(f"Unknown deep baseline: {name}")
