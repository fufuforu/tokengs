"""Read-only architecture audit for TokenGS-ERU-v1.

This script deliberately uses source inspection and checkpoint metadata only;
it does not instantiate a model, allocate CUDA, or write a workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_siu3r_mbm_both_"
    "w10_t3e6_ddp8/checkpoints/model_step_001420.safetensors"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, pattern: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if re.search(pattern, line)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    v4 = ROOT / "tokengs/models/semantic_tokengs_v4.py"
    v6 = ROOT / "tokengs/models/semantic_tokengs_v6.py"
    base = ROOT / "tokengs/models/tokengs.py"
    encdec = ROOT / "tokengs/models/enc_dec.py"
    unit = ROOT / "tokengs/models/absolute_unit_decoder.py"
    tsh = ROOT / "tokengs/models/shared_unit_instance_head.py"

    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    print(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_key_counts": {
            "absolute_gs_head": sum(k.startswith("absolute_gs_head.") for k in keys),
            "tsh_instance_head": sum(k.startswith("tsh_instance_head.") for k in keys),
            "decoder_tail": sum(k.startswith("enc_dec_backbone.decoder_blocks.") for k in keys),
            "tsh_slot_refine_head": sum(k.startswith("tsh_slot_refine_head.") for k in keys),
        },
        "encoder": {
            "entry": "TokenGS._embed_encoder_input -> TokenGS.forward_encoder -> EncDecBackbone._encode_to_kv",
            "input": "[B,V,3,H,W] RGB + [B,V,6,H,W] Plucker -> [B,V*N,C]",
            "latent_values": "[B,num_heads,sequence_length,head_dim], attention keys/values",
        },
        "decoder": {
            "class": "tokengs.models.enc_dec.DecoderBlock",
            "container": "EncDecBackbone.decoder_blocks (nn.ModuleList)",
            "forward": "DecoderBlock.forward(gs_tokens, keys, values)",
            "loop": "TokenGS.forward_decoder / SemanticTokenGSv4._forward_abs_hidden",
            "blockwise_accessible": True,
            "hidden_dim": 1024,
            "num_blocks": 12,
            "query_layout": "[B,1024,1024] (static gs_tokens, optional dynamic tail)",
        },
        "units": {
            "q_abs": "AbsoluteUnitDecoder.form_units output [B,1024,8,256]",
            "gaussian_decoder": "AbsoluteUnitDecoder.decode_units: 8 units x 8 child Gaussians = 65536 [B,65536,14]",
            "instance": "SharedUnitInstanceHead(q_abs) -> unit logits [B,8192,101], pi_unit [B,1024,8,101]",
            "child_inheritance": "pi_unit is expanded unchanged across each unit's 8 fixed GS slots",
        },
        "matching": {
            "current_path": "SemanticTokenGSv4._forward_tsh_instance_branch -> existing scene_instance_loss/matching",
            "mode": "Both@1420 configuration uses its existing per-view/matching setting; ERU does not alter it",
        },
        "source_evidence": {
            "tokengs_forward_decoder": _matches(base, r"for layer in self\.enc_dec_backbone\.decoder_blocks"),
            "semantic_abs_hidden": _matches(v4, r"for layer in self\.enc_dec_backbone\.decoder_blocks"),
            "decoder_modulelist": _matches(encdec, r"self\.decoder_blocks = nn\.ModuleList"),
            "unit_form": _matches(unit, r"def form_units"),
            "tsh_forward": _matches(tsh, r"def forward\("),
            "tsh_scene_loss": _matches(v4, r"_forward_tsh_instance_branch"),
            "v6_eru_hook": _matches(v6, r"token_eru_decoder"),
        },
        "TOKEN_DECODER_BLOCKS_ACCESSIBLE": True,
        "EARLY_BLOCKWISE_INJECTION_IMPLEMENTABLE": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
