from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES = ["sers", "transcriptome", "metabolome", "consensus", "conflict", "fusion"]


def _as_index_tensor(groups: Sequence[Sequence[int]]) -> List[torch.Tensor]:
    return [torch.tensor(g, dtype=torch.long) for g in groups if len(g) > 0]


def masked_mean(stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (stack * weights).sum(dim=1) / denom


class GroupTokenizer(nn.Module):
    def __init__(self, groups: Sequence[Sequence[int]], d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.groups = _as_index_tensor(groups)
        self.proj = nn.ModuleList([nn.Linear(len(g), d_model) for g in self.groups])
        self.pos_embed = nn.Parameter(torch.randn(1, len(self.groups), d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, idx in enumerate(self.groups):
            tokens.append(self.proj[i](x.index_select(1, idx.to(x.device))))
        tokens = torch.stack(tokens, dim=1)
        return self.dropout(self.norm(tokens + self.pos_embed[:, : tokens.size(1)]))


class TokenEncoder(nn.Module):
    def __init__(self, groups: Sequence[Sequence[int]], d_model: int, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.tokenizer = GroupTokenizer(groups, d_model, dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class RCCRMoESemanticReliability(nn.Module):
    def __init__(
        self,
        group_spec,
        d_model: int = 128,
        num_classes: int = 2,
        dropout: float = 0.15,
        use_consensus_token: bool = True,
        use_conflict_token: bool = True,
        use_shared_private: bool = True,
        use_moe_routing: bool = True,
        use_refinement_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_consensus_token = use_consensus_token
        self.use_conflict_token = use_conflict_token
        self.use_shared_private = use_shared_private
        self.use_moe_routing = use_moe_routing
        self.use_refinement_head = use_refinement_head

        self.sers_enc = TokenEncoder(group_spec.sers_groups, d_model, dropout=dropout)
        self.tran_enc = TokenEncoder(group_spec.tran_groups, d_model, dropout=dropout)
        self.meta_enc = TokenEncoder(group_spec.meta_groups, d_model, dropout=dropout)

        self.modality_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)
        rel_hidden = max(32, d_model // 2)
        self.reliability_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, rel_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(rel_hidden, 1))
            for _ in range(3)
        ])
        # Residual calibration strengths: initialized near zero so the model starts close to S0.
        self.rel_consensus_alpha = nn.Parameter(torch.tensor(-4.0))
        self.rel_conflict_alpha = nn.Parameter(torch.tensor(-4.0))
        self.rel_gate_alpha = nn.Parameter(torch.tensor(-4.0))
        self.fuser = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(3)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        self.sers_shared = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.tran_shared = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.meta_shared = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.sers_private = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.tran_private = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.meta_private = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())

        self.consensus_token = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.conflict_token = nn.Sequential(nn.Linear(d_model * 9, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

        self.expert_sers = nn.Linear(d_model, num_classes)
        self.expert_tran = nn.Linear(d_model, num_classes)
        self.expert_meta = nn.Linear(d_model, num_classes)
        self.expert_consensus = nn.Linear(d_model, num_classes)
        self.expert_conflict = nn.Linear(d_model, num_classes)
        self.expert_fusion = nn.Linear(d_model, num_classes)
        self.global_fusion_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

        self.gate = nn.Sequential(nn.Linear(d_model * 6, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 6))
        self.reliability_calibrator = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 6))
        self.refinement_head = (
            nn.Sequential(nn.Linear(d_model * 7, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, num_classes))
            if use_refinement_head
            else None
        )

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(self, sers: torch.Tensor, tran: torch.Tensor, meta: torch.Tensor, modality_mask: torch.Tensor | None = None):
        if modality_mask is None:
            modality_mask = torch.ones(sers.size(0), 3, device=sers.device)
        modality_mask = modality_mask.float()

        tok_s = self.sers_enc(sers) * modality_mask[:, 0].view(-1, 1, 1)
        tok_t = self.tran_enc(tran) * modality_mask[:, 1].view(-1, 1, 1)
        tok_m = self.meta_enc(meta) * modality_mask[:, 2].view(-1, 1, 1)

        pool_s = self._pool(tok_s)
        pool_t = self._pool(tok_t)
        pool_m = self._pool(tok_m)

        if self.use_shared_private:
            shared_s = self.sers_shared(pool_s)
            shared_t = self.tran_shared(pool_t)
            shared_m = self.meta_shared(pool_m)
            private_s = self.sers_private(pool_s)
            private_t = self.tran_private(pool_t)
            private_m = self.meta_private(pool_m)
        else:
            shared_s, shared_t, shared_m = pool_s, pool_t, pool_m
            private_s = torch.zeros_like(pool_s)
            private_t = torch.zeros_like(pool_t)
            private_m = torch.zeros_like(pool_m)

        shared_stack = torch.stack([shared_s, shared_t, shared_m], dim=1)
        reliability_logits = torch.cat([
            self.reliability_heads[0](shared_s.detach()),
            self.reliability_heads[1](shared_t.detach()),
            self.reliability_heads[2](shared_m.detach()),
        ], dim=1)
        reliability = torch.sigmoid(reliability_logits) * modality_mask
        reliability_route = reliability.detach()
        rel_weights = reliability_route / reliability_route.sum(dim=1, keepdim=True).clamp_min(1e-6)
        plain_consensus = masked_mean(shared_stack, modality_mask)
        reliable_consensus = (shared_stack * rel_weights.unsqueeze(-1)).sum(dim=1)
        alpha_c = torch.sigmoid(self.rel_consensus_alpha)
        consensus_src = plain_consensus + alpha_c * (reliable_consensus - plain_consensus)
        consensus_seed = self.consensus_token(torch.cat([shared_s, shared_t, shared_m], dim=-1)) if self.use_consensus_token else torch.zeros_like(consensus_src)

        # Reliability-conditioned disagreement: representation distance is emphasized only when both modalities are trusted.
        rs, rt, rm = reliability[:, 0:1], reliability[:, 1:2], reliability[:, 2:3]
        alpha_f = torch.sigmoid(self.rel_conflict_alpha)
        sf_st = 1.0 + alpha_f * ((rs * rt) - 0.5)
        sf_sm = 1.0 + alpha_f * ((rs * rm) - 0.5)
        sf_tm = 1.0 + alpha_f * ((rt * rm) - 0.5)
        pair_st = torch.abs(shared_s - shared_t) * sf_st
        pair_sm = torch.abs(shared_s - shared_m) * sf_sm
        pair_tm = torch.abs(shared_t - shared_m) * sf_tm
        conflict_input = torch.cat(
            [
                pair_st,
                pair_sm,
                pair_tm,
                (shared_s * shared_t) * sf_st,
                (shared_s * shared_m) * sf_sm,
                (shared_t * shared_m) * sf_tm,
                torch.abs(shared_stack - consensus_src.unsqueeze(1)).reshape(shared_s.size(0), -1),
            ],
            dim=-1,
        )
        conflict_seed = self.conflict_token(conflict_input) if self.use_conflict_token else torch.zeros_like(consensus_src)

        token_stream = [
            tok_s + self.modality_embed[0].view(1, 1, -1),
            tok_t + self.modality_embed[1].view(1, 1, -1),
            tok_m + self.modality_embed[2].view(1, 1, -1),
        ]
        if self.use_consensus_token:
            token_stream.append(consensus_seed.unsqueeze(1))
        if self.use_conflict_token:
            token_stream.append(conflict_seed.unsqueeze(1))
        fused_tokens = torch.cat(token_stream, dim=1)
        for block in self.fuser:
            fused_tokens = block(fused_tokens)
        fused_tokens = self.norm(fused_tokens)
        fused_pool = fused_tokens.mean(dim=1)

        if self.use_consensus_token and self.use_conflict_token:
            consensus_pool = fused_tokens[:, -2]
            conflict_pool = fused_tokens[:, -1]
        elif self.use_consensus_token:
            consensus_pool = fused_tokens[:, -1]
            conflict_pool = torch.zeros_like(consensus_pool)
        elif self.use_conflict_token:
            consensus_pool = torch.zeros_like(conflict_seed)
            conflict_pool = fused_tokens[:, -1]
        else:
            consensus_pool = torch.zeros_like(consensus_src)
            conflict_pool = torch.zeros_like(consensus_src)

        expert_logits = torch.stack(
            [
                self.expert_sers(pool_s),
                self.expert_tran(pool_t),
                self.expert_meta(pool_m),
                self.expert_consensus(consensus_pool),
                self.expert_conflict(conflict_pool),
                self.expert_fusion(fused_pool),
            ],
            dim=1,
        )
        global_logits = self.global_fusion_head(fused_pool)

        mod_probs = F.softmax(expert_logits[:, :3, :], dim=-1)
        def _js(p, q):
            m = 0.5 * (p + q)
            return 0.5 * ((p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1) + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=-1))
        js_st = _js(mod_probs[:, 0], mod_probs[:, 1])
        js_sm = _js(mod_probs[:, 0], mod_probs[:, 2])
        js_tm = _js(mod_probs[:, 1], mod_probs[:, 2])
        js_pair = torch.stack([js_st, js_sm, js_tm], dim=-1)
        gate_input = torch.cat([shared_s, shared_t, shared_m, consensus_pool, conflict_pool, fused_pool], dim=-1)
        base_gate_logits = self.gate(gate_input)
        alpha_g = torch.sigmoid(self.rel_gate_alpha)
        calibration_features = torch.cat([reliability_route, js_pair], dim=-1)
        gate_logits = base_gate_logits + alpha_g * self.reliability_calibrator(calibration_features)
        gate_logits = gate_logits.clone()
        gate_logits[:, :3] = gate_logits[:, :3] + alpha_g * torch.log(reliability_route.clamp_min(1e-4))
        gate_mask = torch.cat([modality_mask, torch.ones_like(modality_mask)], dim=1)
        gate_logits = gate_logits.masked_fill(gate_mask <= 0, -1e9)
        gate = F.softmax(gate_logits, dim=-1) if self.use_moe_routing else gate_mask / gate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        role_logits = torch.sum(gate.unsqueeze(-1) * expert_logits, dim=1)

        base_logits = 0.5 * role_logits + 0.5 * global_logits
        refinement_logits = torch.zeros_like(base_logits) if self.refinement_head is None else self.refinement_head(torch.cat([gate_input, fused_pool], dim=-1))
        logits = base_logits + refinement_logits

        pairwise_consensus = torch.stack(
            [
                F.cosine_similarity(shared_s, shared_t, dim=-1),
                F.cosine_similarity(shared_s, shared_m, dim=-1),
                F.cosine_similarity(shared_t, shared_m, dim=-1),
            ],
            dim=-1,
        )
        pairwise_conflict = torch.stack(
            [
                torch.norm(shared_s - shared_t, dim=-1),
                torch.norm(shared_s - shared_m, dim=-1),
                torch.norm(shared_t - shared_m, dim=-1),
            ],
            dim=-1,
        )

        return {
            "logits": logits,
            "aux_loss": logits.new_zeros(()),
            "gate": gate,
            "gate_logits": gate_logits,
            "expert_logits": expert_logits,
            "role_logits": role_logits,
            "global_logits": global_logits,
            "refinement_logits": refinement_logits,
            "shared_sers": shared_s,
            "shared_tran": shared_t,
            "shared_meta": shared_m,
            "private_sers": private_s,
            "private_tran": private_t,
            "private_meta": private_m,
            "consensus_token": consensus_pool,
            "conflict_token": conflict_pool,
            "consensus_src": consensus_src,
            "fused_pool": fused_pool,
            "pairwise_consensus": pairwise_consensus,
            "pairwise_conflict": pairwise_conflict,
            "reliability": reliability,
            "reliability_logits": reliability_logits,
            "pairwise_js": js_pair,
            "rel_alphas": torch.stack([alpha_c, alpha_f, alpha_g]),
            "group_tokens": {"sers": tok_s, "transcriptome": tok_t, "metabolome": tok_m},
        }


def build_rccr_moe(group_spec, d_model: int = 128, dropout: float = 0.15, **kwargs) -> RCCRMoESemanticReliability:
    return RCCRMoESemanticReliability(group_spec, d_model=d_model, dropout=dropout, **kwargs)
