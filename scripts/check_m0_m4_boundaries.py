"""One-batch gradient-boundary check for the m0 vs m4 configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


class _LocalAccelerator:
    is_main_process = True


_FULL3 = (
    "workspace/semantic_v6_absolute_units_recon_full3/model_best.safetensors"
)


def _main() -> None:
    ckpt = load_file(_FULL3, device="cpu")
    out = {}
    for name in (
        "semantic_v6_absolute_units_true_shared_joint_m4",
        "semantic_v6_absolute_units_true_shared_head_only_m0",
    ):
        opt = config_defaults[name]
        opt.num_workers = 0
        loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
        data = next(iter(loader))
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        model = model_registry[opt.model_type](opt).cuda().train()
        abs_state = {
            k.split(".", 1)[1]: v
            for k, v in ckpt.items()
            if k.startswith("absolute_gs_head.")
        }
        model.absolute_gs_head.load_state_dict(abs_state, strict=True)
        torch.manual_seed((int(opt.seed) + 987654321) % (2**31))
        model.tsh_instance_head.reset_parameters_fresh()
        model.teacher_lambda_eff = 1.0
        model.tsh_instance_loss_weight_eff = 1.0
        model.tsh_unit_grad_eff = float(
            opt.tsh_unit_gradient_multiplier_max
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(data)
        unit_params = [
            p for n, p in model.named_parameters()
            if n.startswith(
                (
                    "absolute_gs_head.tok_norm",
                    "absolute_gs_head.tok_proj",
                    "absolute_gs_head.unit_queries",
                    "absolute_gs_head.unit_readout",
                )
            )
        ]
        decoder_params = [
            p for n, p in model.named_parameters()
            if n.startswith(
                (
                    "absolute_gs_head.center_mlp",
                    "absolute_gs_head.slot_emb",
                    "absolute_gs_head.gs_decoder",
                )
            )
        ]
        head_params = [
            p for n, p in model.named_parameters()
            if n.startswith("tsh_instance_head.")
        ]
        inst = (
            o["loss_instance_group"]
            * float(opt.tsh_lambda_instance)
        )
        gu = torch.autograd.grad(
            inst, unit_params, retain_graph=True, allow_unused=True
        )
        gd = torch.autograd.grad(
            inst, decoder_params, retain_graph=True, allow_unused=True
        )
        gh = torch.autograd.grad(
            inst, head_params, retain_graph=False, allow_unused=True
        )
        norm = lambda gs: float(
            sum(
                (g.double().norm().item() ** 2)
                for g in gs
                if g is not None
            )
            ** 0.5
        )
        out[name] = {
            "multiplier_max": float(
                opt.tsh_unit_gradient_multiplier_max
            ),
            "instance_only_unit_grad": norm(gu),
            "instance_only_decoder_grad": norm(gd),
            "instance_only_head_grad": norm(gh),
            "instance_loss": float(o["loss_instance_group"].detach()),
        }
        del model
    Path("workspace/m0_m4_boundary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    assert out[
        "semantic_v6_absolute_units_true_shared_head_only_m0"
    ]["instance_only_unit_grad"] == 0.0
    assert out[
        "semantic_v6_absolute_units_true_shared_joint_m4"
    ]["instance_only_unit_grad"] > 0.0
    for key in out:
        assert out[key]["instance_only_decoder_grad"] == 0.0
        assert out[key]["instance_only_head_grad"] > 0.0


if __name__ == "__main__":
    _main()
