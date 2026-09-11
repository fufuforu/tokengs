from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate_layout import reduce_aligned_candidates
from .types import CandidateLayout, InstanceDecode


class SceneObjectQueryDecoder(nn.Module):
    def __init__(self, slot_dim: int = 512, embedding_dim: int = 64, num_queries: int = 100,
                 num_layers: int = 2, num_heads: int = 8, mlp_hidden: int = 2048) -> None:
        super().__init__()
        from tokengs.models.instance_group_head import _ObjectQueryDecoderLayer
        self.object_queries = nn.Parameter(0.02 * torch.randn(num_queries, slot_dim))
        self.layers = nn.ModuleList([_ObjectQueryDecoderLayer(slot_dim, num_heads, mlp_hidden) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(slot_dim)
        self.readout = nn.Linear(slot_dim, embedding_dim)

    def forward(self, scene_ins: torch.Tensor) -> torch.Tensor:
        queries = self.object_queries.unsqueeze(0).expand(scene_ins.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, scene_ins)
        return F.normalize(self.readout(self.norm(queries)), dim=-1, eps=1e-6)


class SceneGlobalInstanceHead(nn.Module):
    def __init__(self, slot_dim: int = 512, embedding_dim: int = 64, num_queries: int = 100,
                 max_candidates: int = 16, query_layers: int = 2, query_heads: int = 8,
                 temperature_init: float = 10.0) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.embedding_dim = int(embedding_dim)
        self.num_queries = int(num_queries)
        self.candidate_embedding_readout = nn.Linear(slot_dim, max_candidates * embedding_dim)
        self.candidate_objectness_readout = nn.Linear(slot_dim, max_candidates)
        self.query_decoder = SceneObjectQueryDecoder(slot_dim, embedding_dim, num_queries, query_layers, query_heads, slot_dim * 4)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature_init)).log())
        with torch.no_grad():
            nn.init.trunc_normal_(self.candidate_embedding_readout.weight, std=0.01)
            nn.init.zeros_(self.candidate_embedding_readout.bias)
            nn.init.trunc_normal_(self.candidate_objectness_readout.weight, std=0.01)
            nn.init.constant_(self.candidate_objectness_readout.bias, -2.0)

    def forward(self, scene_ins: torch.Tensor, layout: CandidateLayout, *, gate_gradient_scale: float) -> InstanceDecode:
        b, p, d = scene_ins.shape
        if p != 2048 or d != 512:
            raise ValueError(f"scene_ins must be [B,2048,512], got {tuple(scene_ins.shape)}")
        full_e = self.candidate_embedding_readout(scene_ins).view(b, p, self.max_candidates, self.embedding_dim)
        full_o = self.candidate_objectness_readout(scene_ins).view(b, p, self.max_candidates, 1)
        e = reduce_aligned_candidates(full_e, layout, gate_gradient_scale=gate_gradient_scale, normalize=True)
        objectness = reduce_aligned_candidates(full_o, layout, gate_gradient_scale=gate_gradient_scale, normalize=False)
        queries = self.query_decoder(scene_ins)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        sim = temperature * torch.einsum("bnd,bqd->bnq", e, queries)
        log_query_given_fg = F.log_softmax(sim, dim=-1)
        fg_logits = log_query_given_fg + F.logsigmoid(objectness)
        void_logit = F.logsigmoid(-objectness)
        assignment_logits = torch.cat((fg_logits, void_logit), dim=-1)
        assignment_probabilities = F.softmax(assignment_logits, dim=-1)
        if not torch.isfinite(assignment_probabilities).all() or not torch.allclose(
            assignment_probabilities.sum(-1), torch.ones_like(assignment_probabilities[..., 0]), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("invalid instance assignment probabilities")
        return InstanceDecode(
            gaussian_embeddings=e,
            gaussian_objectness_logits=objectness,
            object_queries=queries,
            assignment_logits=assignment_logits,
            assignment_probabilities=assignment_probabilities,
            query_features=queries,
        )
