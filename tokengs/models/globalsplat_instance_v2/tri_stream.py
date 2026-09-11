from __future__ import annotations

import copy
from functools import lru_cache

import torch
import torch.nn as nn

from .dependency import GlobalSplatSymbols
from .types import TriStreamOutput


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.detach() + float(scale) * (x - x.detach())


class TriStreamAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 512,
                 instance_layerscale: float = 1e-3,
                 reconstruction_layerscale: float = 1e-4) -> None:
        super().__init__()
        self.ln_gi = nn.LayerNorm(dim)
        self.ln_ti = nn.LayerNorm(dim)
        self.ln_ii = nn.LayerNorm(dim)
        self.linear_i_in = nn.Linear(3 * dim, hidden_dim)
        self.linear_i_out = nn.Linear(hidden_dim, dim)
        self.ls_i = nn.Parameter(torch.full((dim,), float(instance_layerscale)))
        self.ln_gr = nn.LayerNorm(dim)
        self.ln_tr = nn.LayerNorm(dim)
        self.ln_ir = nn.LayerNorm(dim)
        self.linear_r_in = nn.Linear(3 * dim, hidden_dim)
        self.linear_r_out = nn.Linear(hidden_dim, 2 * dim)
        self.ls_g = nn.Parameter(torch.full((dim,), float(reconstruction_layerscale)))
        self.ls_t = nn.Parameter(torch.full((dim,), float(reconstruction_layerscale)))
        with torch.no_grad():
            for module in (self.linear_i_out, self.linear_r_out):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, geo: torch.Tensor, tex: torch.Tensor, ins: torch.Tensor,
                *, recon_to_instance_grad_scale: float,
                instance_to_reconstruction_gate: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_for_i = grad_scale(geo, recon_to_instance_grad_scale)
        t_for_i = grad_scale(tex, recon_to_instance_grad_scale)
        h_i = torch.cat((self.ln_gi(g_for_i), self.ln_ti(t_for_i), self.ln_ii(ins)), dim=-1)
        ins_out = ins + self.ls_i * self.linear_i_out(torch.nn.functional.gelu(self.linear_i_in(h_i)))
        h_r = torch.cat((self.ln_gr(geo), self.ln_tr(tex), self.ln_ir(ins_out)), dim=-1)
        delta = self.linear_r_out(torch.nn.functional.gelu(self.linear_r_in(h_r)))
        d_g, d_t = delta.chunk(2, dim=-1)
        gate = float(instance_to_reconstruction_gate)
        return (
            geo + gate * self.ls_g * d_g,
            tex + gate * self.ls_t * d_t,
            ins_out,
        )


@lru_cache(maxsize=4)
def _tri_class(symbols_class: type[nn.Module]):
    class TriStreamSlotEncoder(symbols_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.slot_to_ins = copy.deepcopy(self.slot_to_geo)
            self.ins_rounds = copy.deepcopy(self.geo_rounds)
            self.tri_adapters = nn.ModuleList([TriStreamAdapter(self.D_lat, self.D_lat) for _ in range(self.rounds)])

        def forward_joint(self, x: torch.Tensor, state: torch.Tensor, *,
                          recon_to_instance_grad_scale: float,
                          instance_to_reconstruction_gate: float,
                          camera_motion_tokens: torch.Tensor | None = None,
                          mem_drop_p: float = 0.0) -> TriStreamOutput:
            batch_size = x.shape[0]
            slots0, cls_size, reg_start, reg_end = self._build_slots(batch_size, state)
            geo = self.slot_to_geo(slots0)
            tex = self.slot_to_tex(slots0)
            # The instance-to-reconstruction scale is exactly zero during
            # steps 1--50.  Keep the instance forward identical, but prevent
            # its loss from reaching shared slot initialization parameters
            # (scene tokens, slot registers, auxiliary queries, and scale
            # embedding) through the fresh instance stream.
            instance_slots0 = slots0 if float(instance_to_reconstruction_gate) > 0.0 else slots0.detach()
            ins = self.slot_to_ins(instance_slots0)
            mem = self._build_memory(x, camera_motion_tokens, mem_drop_p)
            mask = self._make_reg_mask(geo.size(1), reg_start, reg_end, geo.device, geo.dtype)
            # During the initial instance warm-up, the instance stream must
            # learn from its own fresh parameters without sending gradients
            # into the shared token memory.  Detaching only this input keeps
            # the forward values identical; after the return gate opens, the
            # normal shared-memory path is restored.
            instance_mem = mem if float(instance_to_reconstruction_gate) > 0.0 else mem.detach()
            for round_idx in range(self.rounds):
                geo1 = self.geo_rounds[round_idx](geo, mem, reg_mask=mask)
                tex1 = self.tex_rounds[round_idx](tex, mem, reg_mask=mask)
                ins1 = self.ins_rounds[round_idx](ins, instance_mem, reg_mask=mask)
                geo2, tex2 = self.pair_adapters[round_idx](geo1, tex1)
                geo, tex, ins = self.tri_adapters[round_idx](
                    geo2, tex2, ins1,
                    recon_to_instance_grad_scale=recon_to_instance_grad_scale,
                    instance_to_reconstruction_gate=instance_to_reconstruction_gate,
                )
            start = self.num_aux_queries
            end = start + cls_size
            scale = end
            aux = (geo[:, :self.num_aux_queries] + tex[:, :self.num_aux_queries] + ins[:, :self.num_aux_queries]) / 3.0
            return TriStreamOutput(
                scene_geo=geo[:, start:end].contiguous(),
                scene_tex=tex[:, start:end].contiguous(),
                scene_ins=ins[:, start:end].contiguous(),
                aux=aux.contiguous(),
                scale_token=((geo[:, scale:scale + 1] + tex[:, scale:scale + 1]) / 2.0).contiguous(),
            )

    TriStreamSlotEncoder.__name__ = "TriStreamSlotEncoder"
    return TriStreamSlotEncoder


def build_tri_stream_slot_encoder(symbols: GlobalSplatSymbols,
                                  pretrained_dual_state: dict[str, torch.Tensor],
                                  *,
                                  initialize_instance_from_geometry: bool = True) -> nn.Module:
    cls = _tri_class(symbols.DualStreamSlotEncoder)
    encoder = cls(
        dim_latent=512, dim_token=768, heads=8, rounds=4,
        slot_calib_layers_per_round=2, qkv_bias=True, mlp_ratio=4.0,
        readout_ln=True, attn_init_values=1e-3, mlp_init_values=1e-3,
        num_aux_queries=4, mem_reg_amount=4, slot_reg_amount=4,
        use_camera_diff_as_input=False,
    )
    base_keys = set(encoder.state_dict())
    provided = {key: value for key, value in pretrained_dual_state.items() if key in base_keys}
    # The caller supplies official slot_encoder keys.  Load the original dual
    # stream before replacing the copied instance stream.
    instance_prefixes = ("slot_to_ins.", "ins_rounds.", "tri_adapters.")
    expected_base = [key for key in base_keys if not key.startswith(instance_prefixes)]
    missing = [key for key in expected_base if key not in provided]
    unexpected = [key for key in provided if key not in expected_base]
    if missing or unexpected:
        raise RuntimeError(f"tri-stream base restore mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    encoder.load_state_dict(provided, strict=False)
    if initialize_instance_from_geometry:
        encoder.slot_to_ins = copy.deepcopy(encoder.slot_to_geo)
        encoder.ins_rounds = copy.deepcopy(encoder.geo_rounds)
    # When the short Phase-J recipe requests fresh instance initialization,
    # the constructor-created slot_to_ins/ins_rounds remain independent
    # random parameters.  They are still registered and trainable, while the
    # geometry/appearance streams retain the strictly restored Phase-R state.
    for left, right in zip(encoder.slot_to_ins.parameters(), encoder.slot_to_geo.parameters()):
        if left is right or left.data_ptr() == right.data_ptr():
            raise RuntimeError("instance stream must not share Parameter storage with geometry stream")
    return encoder
