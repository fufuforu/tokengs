"""Offline feature-headroom probe (mechanism M3: DINO scale/depth/resolution).

Read-only, no training.  Frozen wide7l@8000 + DINO 8-local-unit pipeline,
LSM 40 held-out scenes.  For each candidate per-unit feature F (built from
different DINOv2 layers / resolutions) it reports:

  - same/diff cosine gap of F under GT instance membership;
  - GT-prototype oracle AP50: per GT instance prototype = mean of F over its
    units, hard cosine assignment, rendered through the frozen geometry and
    scored with the LSM protocol (same convention as the 0.534 oracle).

Variants:
  full_embedding : the actual 0.324 unit embedding (unit_feat+patch+DINO)
  last_r252      : DINOv2 block-12 @ 252 (== current usage, oracle ~0.534)
  mid_r252       : DINOv2 block-8 @ 252
  mid_last_r252  : concat(block-8, block-12) @ 252
  last_r336      : DINOv2 block-12 @ 336 (24x24 patches, finer boundaries)

Answer: does the GT-prototype oracle move when the feature changes?  If it
stays ~0.53 for every variant, the feature is not the bottleneck and the
M2/M1 trained-head probes are not worth running.

Usage:
    python scripts/probe_feature_headroom.py \
        --workspace workspace/probe_feature_headroom --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models.instance_group_head import (  # noqa: E402
    TokenLocalUnitGrouping,
    _project_dense_features,
)
from tokengs.options import config_defaults  # noqa: E402

from dino_dense_feature_oracle import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    _rescale_intrinsics,
    load_dinov2,
)
from diagnose_3d_seeded_prototypes import _render_and_eval  # noqa: E402
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
)
from train_seedness_head import _cheap_unit_forward_patch, _load_model  # noqa: E402


def _dino_grid_multi(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    target: int,
    n_blocks: int,
) -> list[torch.Tensor]:
    """DINO patch grids [B,V,D,Hf,Wf] per requested last-n block outputs."""
    b, v, c, h, w = images.shape
    x = F.interpolate(
        images.reshape(b * v, c, h, w).to(device),
        size=(target, target),
        mode="bilinear",
        align_corners=False,
    )
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    with torch.no_grad():
        outs = model.get_intermediate_layers(
            x, n=n_blocks, return_class_token=False
        )
    grids = []
    for tokens in outs:
        dim = tokens.shape[-1]
        hf = wf = int(round(tokens.shape[1] ** 0.5))
        feats = tokens.reshape(b * v, hf, wf, dim).permute(0, 3, 1, 2)
        feats = F.normalize(feats, dim=1)
        grids.append(feats.reshape(b, v, dim, hf, wf))
    return grids


def _unit_feature(
    a4: np.ndarray,
    means_t: np.ndarray,
    token_count: int,
    p: int,
    k: int,
    grid: torch.Tensor,
    cam2world: torch.Tensor,
    intrinsics: torch.Tensor,
    hw: tuple[int, int],
) -> np.ndarray:
    """Project dense grid onto GS centers -> per-unit mean -> normalized."""
    with torch.no_grad():
        fused, _ = _project_dense_features(
            torch.as_tensor(means_t.reshape(1, -1, 3)).cuda(),
            grid,
            cam2world,
            _rescale_intrinsics(intrinsics, (256, 256), hw),
            hw,
        )
    gs = fused[0].cpu().numpy()
    mass = a4.sum(axis=1).clip(min=1e-6)  # [T,K]
    u = np.einsum("tpk,tpd->tkd", a4, gs.reshape(token_count, p, -1))
    u = (u / mass[:, :, None]).reshape(-1, gs.shape[-1])
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-8)
    return u.astype(np.float32)


def _same_diff_stats(
    feat: np.ndarray,
    dom_inst: np.ndarray,
    fg: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    fg_ids = np.unique(dom_inst[fg])
    fg_u = np.where(fg)[0]
    same, diff = [], []
    for gid in fg_ids.tolist():
        members = np.where(dom_inst == gid)[0]
        if members.size < 2:
            continue
        for _ in range(min(100, members.size)):
            i, j = rng.choice(members, 2, replace=False)
            same.append(float(feat[i] @ feat[j]))
    if fg_u.size >= 2:
        for _ in range(4000):
            i, j = rng.choice(fg_u, 2, replace=False)
            if dom_inst[i] != dom_inst[j]:
                diff.append(float(feat[i] @ feat[j]))
    s = float(np.mean(same)) if same else np.nan
    d = float(np.mean(diff)) if diff else np.nan
    return {"same": s, "diff": d, "gap": (s - d) if s == s and d == d else np.nan}


def _gt_prototype_labels(
    feat: np.ndarray,
    dom_inst: np.ndarray,
    fg: np.ndarray,
) -> np.ndarray:
    """GT-instance mean prototypes -> hard cosine assignment -> labels."""
    fg_ids = np.unique(dom_inst[fg])
    proto = np.zeros((fg_ids.size, feat.shape[1]), dtype=np.float64)
    for gi, gid in enumerate(fg_ids.tolist()):
        members = np.where(dom_inst == gid)[0]
        proto[gi] = feat[members].mean(axis=0)
    proto = proto / np.maximum(
        np.linalg.norm(proto, axis=-1, keepdims=True), 1e-8
    )
    labels = np.full(feat.shape[0], -1, dtype=np.int64)
    if fg_ids.size:
        sim = feat[fg] @ proto.T
        labels[fg] = np.argmax(sim, axis=1)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=(
            "workspace/semantic_v6_unit_shaping_img_dino_train_3000/"
            "checkpoints/model_step_003000.safetensors"
        ),
    )
    parser.add_argument(
        "--workspace", default="workspace/probe_feature_headroom"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_unit_shaping_img_dino_train"]
    opt.data_mode = (("scannet_lsm_instance_eval", 1),)
    opt.dataset_kwargs = {
        "lsm_manifest_path": str(
            ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
        )
    }
    opt.workspace = str(out_dir)
    opt.experiment_name = out_dir.name
    manifest_audit = _audit_lsm_manifest(
        str(ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json")
    )

    TokenLocalUnitGrouping._forward_unit_embedding = _cheap_unit_forward_patch
    model = _load_model(opt, args.resume, patched=True)
    model.eval().cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(
        getattr(opt, "instance_branch_void_fg_share", 0.5)
    )
    dinov2 = load_dinov2(torch.device("cuda"))

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    variant_names = [
        "full_embedding",
        "last_r252",
        "mid_r252",
        "mid_last_r252",
        "last_r336",
    ]
    per_scene: dict[str, dict] = {}
    t_start = time.time()
    for i, data in enumerate(test_loader):
        if args.max_scenes > 0 and i >= args.max_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = str(data["scene_name"][0])
        with torch.no_grad():
            model(data, compute_quality_metrics=False)
        buf = br._seed_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)
        means_t = means_np.reshape(token_count, p, 3).cpu().numpy()
        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        inst_onehot = np.zeros(
            (n_gs, int(gs_inst.max()) + 1), dtype=np.float64
        )
        inst_onehot[np.arange(n_gs), gs_inst] = 1.0
        u_mass = a4.sum(axis=1).clip(min=1e-6)
        u_inst = np.einsum(
            "tpk,tpm->tkm", a4,
            inst_onehot.reshape(token_count, p, -1),
        ).reshape(u_count, -1)
        dom_inst = u_inst.argmax(axis=1)
        fg = (
            (u_inst.max(axis=1) / u_mass.reshape(-1).clip(min=1e-6) > 0.5)
            & (dom_inst != 0)
        )
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        cam2world = buf["cam_to_world_input"].float()
        intrin_in = buf["intrinsics_input"].float()

        feats: dict[str, np.ndarray] = {}
        e_np = buf["e"][0].cpu().numpy().astype(np.float32)
        feats["full_embedding"] = e_np / np.maximum(
            np.linalg.norm(e_np, axis=-1, keepdims=True), 1e-8
        )
        grids12 = _dino_grid_multi(
            dinov2, data["images_input"], torch.device("cuda"),
            target=252, n_blocks=12,
        )
        assert len(grids12) == 12, f"expected 12 blocks, got {len(grids12)}"
        g_last = grids12[-1]
        g_mid = grids12[7]
        hw252 = (252, 252)
        feats["last_r252"] = _unit_feature(
            a4, means_t, token_count, p, k, g_last, cam2world, intrin_in, hw252
        )
        feats["mid_r252"] = _unit_feature(
            a4, means_t, token_count, p, k, g_mid, cam2world, intrin_in, hw252
        )
        cat = torch.cat([g_mid, g_last], dim=2)  # [B,V,2D,18,18]
        feats["mid_last_r252"] = _unit_feature(
            a4, means_t, token_count, p, k, cat, cam2world, intrin_in, hw252
        )
        g336 = _dino_grid_multi(
            dinov2, data["images_input"], torch.device("cuda"),
            target=336, n_blocks=1,
        )[0]
        feats["last_r336"] = _unit_feature(
            a4, means_t, token_count, p, k, g336, cam2world, intrin_in,
            (336, 336),
        )

        entry: dict = {}
        for vname in variant_names:
            feat = feats[vname]
            stats = _same_diff_stats(feat, dom_inst, fg, rng)
            labels = _gt_prototype_labels(feat, dom_inst, fg)
            fg_share_u = fg.astype(np.float64)
            ap = _render_and_eval(
                labels, fg_share_u, a4, gaussians, cam_view, intrinsics,
                renderer, render_scale, void_fg_share, gs_inst, gt_maps,
                scene_name,
            )
            entry[vname] = {
                "same": stats["same"],
                "diff": stats["diff"],
                "gap": stats["gap"],
                "ap50": ap["ap50"],
                "ap25": ap["ap25"],
                "ap": ap["ap"],
                "num_clusters": ap["num_clusters"],
            }
        per_scene[scene_name] = entry
        elapsed = time.time() - t_start
        print(
            f"[probe] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _mean_sub(prefix: str, sub: str, scenes: dict) -> float:
        vals = []
        for s in scenes:
            d = scenes[s].get(prefix)
            if isinstance(d, dict) and sub in d and d[sub] == d[sub]:
                vals.append(d[sub])
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for vname in variant_names:
        summary[vname] = {
            "same": _mean_sub(vname, "same", per_scene),
            "diff": _mean_sub(vname, "diff", per_scene),
            "gap": _mean_sub(vname, "gap", per_scene),
            "ap50": _mean_sub(vname, "ap50", per_scene),
            "ap25": _mean_sub(vname, "ap25", per_scene),
            "ap": _mean_sub(vname, "ap", per_scene),
        }
    payload = {
        "checkpoint": args.resume,
        "protocol": manifest_audit,
        "reference": {
            "pipeline_ap50": 0.324,
            "gt_prototype_oracle_ap50": 0.534,
        },
        "variants": summary,
    }
    (out_dir / "feature_headroom.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "feature_headroom_per_scene.json").write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[probe] summary:")
    print(
        f"{'variant':16s} {'same':>6s} {'diff':>6s} {'gap':>6s} "
        f"{'AP50':>6s} {'AP25':>6s}"
    )
    for vname in variant_names:
        s = summary[vname]
        print(
            f"{vname:16s} {s['same']:6.3f} {s['diff']:6.3f} "
            f"{s['gap']:6.3f} {s['ap50']:6.3f} {s['ap25']:6.3f}"
        )


if __name__ == "__main__":
    main()
