"""Seedness-head diagnostic: can a supervised one-shot proposal be learned?

Freezes the 0.324 pipeline (wide7l@8000 + DINO 8-local-units) and trains a
small per-unit seedness head only.  Supervision: each unit's target is a 3D
gaussian centrality w.r.t. its GT-instance center (built from the 15-view
pseudo-GT unit distribution p_u and unit 3D centers) -- the most central
unit of an instance scores ~1, background/mixed units score 0.

At inference the head scores all units once (one-shot), greedy 3D NMS turns
score peaks into seeds, and we measure:
  - seed recall: GT instances with >=1 seed / one-seed-per-instance ratio;
  - seed purity: dominant-instance mass of accepted seeds;
  - end-to-end AP50: learned seeds -> prototype mean-shift -> render -> LSM
    AP, compared against the 3D-FPS floor (0.226) and Agglomerative (0.324).

No changes to the main model; the head lives in this script.

Usage:
    python scripts/train_seedness_head.py \
        --workspace workspace/seedness_head_diag --gpu 0 \
        --num_steps 2000 --ckpt_freq 500
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
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader  # noqa: E402
from tokengs.models import model_registry  # noqa: E402
from tokengs.models.instance_group_head import (  # noqa: E402
    TokenLocalUnitGrouping,
    _project_dense_features,
)
from tokengs.options import config_defaults  # noqa: E402
from tokengs.utils.instance_ap import (  # noqa: E402
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

from ablate_unit_clustering import _gs_level_stats, _max_iou_recall  # noqa: E402
from diagnose_3d_seeded_prototypes import (  # noqa: E402
    _prototype_iteration,
    _render_and_eval,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
)


class SeednessHead(nn.Module):
    def __init__(self, feat_dim: int = 256, pos_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + pos_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, e: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([e, pos], dim=-1)).squeeze(-1)


def _cheap_unit_forward_patch(self, *args, **kwargs):
    """Stand-in for _forward_unit_embedding: embeddings + pseudo p_u only.

    Avoids the expensive agglomerative clustering / render (eval path) and
    InfoNCE (train path); stores what the seedness diagnostic needs.
    """
    (
        a,
        unit_feat,
        unit_center,
        means,
        gaussians,
        data,
        opt_inner,
        training,
        model_input,
        batch_size,
        token_count,
        n_gs,
        p,
        dense_features,
    ) = args
    del training
    k = self.units_per_token
    u_count = token_count * k
    unit_feat_flat = unit_feat.reshape(batch_size, u_count, self.gs_feat_dim)
    center = unit_center.reshape(batch_size, u_count, 3).float()
    scene_center = center.mean(dim=1, keepdim=True)
    scene_scale = (
        (center - scene_center).square().mean(dim=(1, 2), keepdim=True)
        .sqrt()
        .clamp_min(1e-3)
    )
    pos_norm = (center - scene_center) / scene_scale
    unit_dino = (
        self._dino_unit_features(
            a, means, data, opt_inner, batch_size, token_count, n_gs, p
        )
        if self.dino_unit
        else None
    )
    if self.identity_encoder is not None:
        e = self.identity_encoder(unit_feat_flat, pos_norm)
    else:
        e = F.normalize(unit_feat_flat.float(), dim=-1)
    parts = [e]
    if self.unit_image and dense_features is not None:
        B_, VP, C_ = dense_features.shape
        V_ = int(getattr(opt_inner, "num_input_views", 8))
        P_ = VP // V_
        Hf = Wf = int(round(P_**0.5))
        dense_img = (
            dense_features.reshape(B_, V_, P_, C_)
            .permute(0, 1, 3, 2)
            .reshape(B_, V_, C_, Hf, Wf)
        )
        fused, _ = _project_dense_features(
            means.reshape(batch_size, n_gs, 3),
            dense_img,
            data["cam_to_world_input"],
            data["intrinsics_input"],
            tuple(opt_inner.img_size),
        )
        img_gs = self.unit_image_net(fused)
        img_gs_t = img_gs.reshape(batch_size, token_count, p, -1)
        unit_img = torch.einsum("btpk,btpd->btkd", a, img_gs_t)
        unit_img = unit_img.reshape(batch_size, u_count, -1)
        parts.append(F.normalize(unit_img.float(), dim=-1))
    if unit_dino is not None:
        parts.append(unit_dino)
    e = F.normalize(torch.cat(parts, dim=-1), dim=-1) if len(parts) > 1 else e

    gs_gt, gs_conf, gs_nviews = self._pseudo_gs_labels(
        means.reshape(batch_size, n_gs, 3), data, opt_inner
    )
    gt0 = gs_gt.view(batch_size, token_count, p)
    high_conf = (
        (gs_conf >= self.pseudo_conf)
        & (gs_nviews >= self.pseudo_min_views)
    ).view(batch_size, token_count, p)
    ids_present = torch.unique(gt0)
    id_map = {int(iid): idx for idx, iid in enumerate(ids_present.tolist())}
    m = len(ids_present)
    onehot = torch.zeros(
        (batch_size, token_count, p, m),
        device=means.device,
        dtype=a.dtype,
    )
    for iid, idx in id_map.items():
        onehot[..., idx] = (gt0 == iid).to(a.dtype) * high_conf.to(a.dtype)
    h = torch.einsum("btpk,btpi->btki", a, onehot)
    mass = h.sum(-1, keepdim=True).clamp_min(1e-6)
    p_u = (h / mass).reshape(batch_size, u_count, m)
    bg_idx = id_map.get(0, -1)
    self._seed_buffers = {
        "e": e.detach().float(),
        "pos_norm": pos_norm.detach().float(),
        "p_u": p_u.detach().float(),
        "a": a.detach().float(),
        "means": means.detach().float(),
        "gaussians": gaussians.detach().float(),
        "cam_view": model_input.decoder.cam_view.detach().float(),
        "intrinsics": model_input.decoder.intrinsics.detach().float(),
        "cam_to_world_input": data["cam_to_world_input"].detach().float(),
        "intrinsics_input": data["intrinsics_input"].detach().float(),
        "bg_idx": bg_idx,
    }
    return {
        "loss_instance_group": torch.zeros((), device=e.device),
    }


def _gaussian_centrality_targets(
    pos: np.ndarray, p_u: np.ndarray, bg_idx: int
) -> np.ndarray:
    """Per-unit 3D gaussian centrality to its GT-instance center, [0,1]."""
    u_count, m = p_u.shape
    dom = p_u.argmax(axis=1)
    mass = p_u.max(axis=1)
    fg = (mass > 0.5) & (dom != bg_idx)
    target = np.zeros(u_count, dtype=np.float64)
    inst_ids = np.unique(dom[fg])
    for iid in inst_ids.tolist():
        members = np.where(fg & (dom == iid))[0]
        w = p_u[members, iid]
        w = w / w.sum()
        c = (pos[members] * w[:, None]).sum(axis=0)
        d2 = ((pos[members] - c) ** 2).sum(axis=1)
        sigma = float(np.sqrt((d2 * w).sum()).clip(min=0.05))
        d2_members = ((pos[members] - c) ** 2).sum(axis=1)
        target[members] = np.exp(-d2_members / (2.0 * sigma * sigma))
    return target


def _nms_seeds(
    score: np.ndarray,
    pos: np.ndarray,
    radius: float,
    threshold: float,
) -> np.ndarray:
    """Greedy 3D NMS over per-unit scores -> seed indices."""
    order = np.argsort(-score)
    accepted = []
    for uu in order.tolist():
        if score[uu] < threshold:
            continue
        if any(np.linalg.norm(pos[uu] - pos[j]) < radius for j in accepted):
            continue
        accepted.append(uu)
    return np.asarray(accepted, dtype=np.int64)


def _seed_metrics(
    seeds: np.ndarray,
    dom_inst: np.ndarray,
    fg: np.ndarray,
    p_u: np.ndarray,
    bg_idx: int,
) -> dict:
    fg_ids = np.unique(dom_inst[fg])
    if fg_ids.size == 0:
        return {
            "seed_recall": np.nan, "one_seed_frac": np.nan,
            "seeds_per_instance": np.nan, "seed_purity": np.nan,
            "num_seeds": 0,
        }
    per_inst = {int(i): 0 for i in fg_ids.tolist()}
    purities = []
    for s in seeds.tolist():
        if fg[s] and dom_inst[s] != bg_idx:
            per_inst[int(dom_inst[s])] += 1
            purities.append(float(p_u[s, int(dom_inst[s])]))
    counts = np.asarray([per_inst[int(i)] for i in fg_ids.tolist()])
    seed_purity = float(np.mean(purities)) if purities else np.nan
    return {
        "seed_recall": float((counts >= 1).mean()),
        "one_seed_frac": float((counts == 1).mean()),
        "seeds_per_instance": float(counts.mean()),
        "seed_purity": seed_purity,
        "num_seeds": int(seeds.size),
    }


def _load_model(opt, resume: str, patched: bool):
    from safetensors.torch import load_file

    if patched:
        TokenLocalUnitGrouping._forward_unit_embedding = _cheap_unit_forward_patch
    model = model_registry[opt.model_type](opt)
    ck = load_file(resume, device="cpu")
    torch.nn.Module.load_state_dict(model, ck, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in ck):
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            bck = load_file(backbone_path, device="cpu")
            prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in bck.items()
                if (key.startswith(prefixes) or key == "gs_tokens")
                and key in native
                and native[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
    return model


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
        "--workspace", default="workspace/seedness_head_diag"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--ckpt_freq", type=int, default=500)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_eval_scenes", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(42)
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = config_defaults["semantic_v6_unit_shaping_img_dino_train"]
    opt.workspace = str(out_dir)
    opt.experiment_name = out_dir.name
    _ = _LocalAccelerator()
    train_loader, _, _, _ = get_multi_dataloader(opt, _LocalAccelerator())
    model = _load_model(opt, args.resume, patched=True)
    model.eval().cuda()
    br = model.instance_branch
    for p in model.parameters():
        p.requires_grad_(False)

    head = SeednessHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"[seedness] trainable head params: "
        f"{sum(p.numel() for p in head.parameters())}"
    )

    t_start = time.time()
    step = 0
    while step < args.num_steps:
        for data in train_loader:
            if step >= args.num_steps:
                break
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            with torch.no_grad():
                model(data, compute_quality_metrics=False)
                buf = br._seed_buffers
                e = buf["e"][0]
                pos = buf["pos_norm"][0]
                p_u = buf["p_u"][0]
                bg_idx = buf["bg_idx"]
            target_np = _gaussian_centrality_targets(
                pos.cpu().numpy(), p_u.cpu().numpy(), bg_idx
            )
            target = torch.from_numpy(target_np).float().cuda()
            logit = head(e, pos)
            pos_w = float((target < 0.5).sum().clamp_min(1)) / float(
                (target >= 0.5).sum().clamp_min(1)
            )
            loss = F.binary_cross_entropy_with_logits(
                logit, target, pos_weight=torch.tensor(pos_w, device=logit.device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.print_freq == 0 or step == args.num_steps:
                with torch.no_grad():
                    pred = torch.sigmoid(logit)
                    mean_t = float(target.mean())
                    mean_p = float(pred.mean())
                    fg_acc = float(
                        ((pred > 0.5) == (target > 0.5))[target > 0.5].float().mean()
                    ) if (target > 0.5).any() else float("nan")
                print(
                    f"[seedness] step={step}/{args.num_steps} loss={float(loss):.4f} "
                    f"tgt_mean={mean_t:.4f} pred_mean={mean_p:.4f} "
                    f"fg_acc={fg_acc:.3f} elapsed={time.time() - t_start:.0f}s",
                    flush=True,
                )
            if step % args.ckpt_freq == 0 or step == args.num_steps:
                (out_dir / "checkpoints").mkdir(exist_ok=True)
                torch.save(
                    head.state_dict(),
                    out_dir / "checkpoints" / f"seedness_{step:06d}.pt",
                )
    print("[seedness] training done")

    # ---------------- eval on LSM 40 scenes ----------------
    eval_opt = config_defaults["semantic_v6_unit_shaping_img_dino_train"]
    eval_opt.data_mode = (("scannet_lsm_instance_eval", 1),)
    eval_opt.dataset_kwargs = {
        "lsm_manifest_path": str(
            ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json"
        )
    }
    eval_opt.workspace = str(out_dir)
    eval_opt.experiment_name = out_dir.name
    manifest_audit = _audit_lsm_manifest(
        str(ROOT / "data/scannet_prompt/lsm_instance_eval_manifest.json")
    )
    eval_model = _load_model(eval_opt, args.resume, patched=True)
    eval_model.eval().cuda()
    ebr = eval_model.instance_branch
    renderer = ebr.renderer
    render_scale = float(getattr(eval_opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(
        getattr(eval_opt, "instance_branch_void_fg_share", 0.5)
    )
    _, test_loader, _, _ = get_multi_dataloader(
        eval_opt, _LocalAccelerator()
    )

    radii = [0.2, 0.4, 0.6, 0.8, 1.0]
    threshs = [0.2, 0.4, 0.6]
    per_scene: dict[str, dict] = {}
    for i, data in enumerate(test_loader):
        if args.max_eval_scenes > 0 and i >= args.max_eval_scenes:
            break
        data = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in data.items()
        }
        scene_name = str(data["scene_name"][0])
        with torch.no_grad():
            eval_model(data, compute_quality_metrics=False)
        buf = ebr._seed_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = ebr.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)
        means_t = means_np.reshape(token_count, p, 3).cpu().numpy()
        e_np = buf["e"][0].cpu().numpy()
        pos_np = buf["pos_norm"][0].cpu().numpy()
        p_u_np = buf["p_u"][0].cpu().numpy()
        bg_idx = buf["bg_idx"]
        u_mass = a4.sum(axis=1).clip(min=1e-6)
        unit_center_world = (
            np.einsum("tpk,tpx->tkx", a4, means_t) / u_mass[:, :, None]
        ).reshape(u_count, 3)
        dom_inst = p_u_np.argmax(axis=1)
        fg = (p_u_np.max(axis=1) > 0.5) & (dom_inst != bg_idx)
        with torch.no_grad():
            score = torch.sigmoid(
                head(
                    torch.from_numpy(e_np).float().cuda(),
                    torch.from_numpy(pos_np).float().cuda(),
                )
            ).cpu().numpy()
        gs_inst, _, _ = ebr._pseudo_gs_labels(means_np, data, eval_opt)
        gs_inst = gs_inst[0].cpu().numpy()
        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gt_maps = data["instance_label_output"][0].cpu().numpy()
        entry: dict = {}
        for r in radii:
            for t in threshs:
                seeds = _nms_seeds(score, unit_center_world, r, t)
                vname = f"r{r:.1f}_t{t:.1f}"
                entry[vname + "_seed"] = _seed_metrics(
                    seeds, dom_inst, fg, p_u_np, bg_idx
                )
                if seeds.size == 0:
                    entry[vname] = {"ap50": 0.0, "num_clusters": 0}
                    continue
                labels = _prototype_iteration(
                    e_np, seeds, fg, 3
                )
                fg_share_u = fg.astype(np.float64)
                entry[vname] = _render_and_eval(
                    labels, fg_share_u, a4, gaussians, cam_view, intrinsics,
                    renderer, render_scale, void_fg_share, gs_inst, gt_maps,
                    scene_name,
                )
        per_scene[scene_name] = entry
        print(
            f"[seedness-eval] {scene_name} done "
            f"({i + 1}/{min(args.max_eval_scenes or 40, 40)})",
            flush=True,
        )

    def _mean_sub(prefix: str, sub: str, scenes: dict) -> float:
        vals = []
        for s in scenes:
            d = scenes[s].get(prefix)
            if isinstance(d, dict) and sub in d and d[sub] == d[sub]:
                vals.append(d[sub])
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_seed(prefix: str, sub: str, scenes: dict) -> float:
        vals = []
        for s in scenes:
            d = scenes[s].get(prefix + "_seed")
            if d is not None and sub in d and d[sub] == d[sub]:
                vals.append(d[sub])
        return float(np.mean(vals)) if vals else float("nan")

    summary = {}
    for r in radii:
        for t in threshs:
            vname = f"r{r:.1f}_t{t:.1f}"
            summary[vname] = {
                "ap50": _mean_sub(vname, "ap50", per_scene),
                "seed_recall": _mean_seed(vname, "seed_recall", per_scene),
                "one_seed_frac": _mean_seed(vname, "one_seed_frac", per_scene),
                "seeds_per_instance": _mean_seed(
                    vname, "seeds_per_instance", per_scene
                ),
                "seed_purity": _mean_seed(vname, "seed_purity", per_scene),
            }
    payload = {
        "checkpoint": args.resume,
        "num_train_steps": args.num_steps,
        "protocol": manifest_audit,
        "reference": {
            "agglomerative_ap50": 0.324,
            "fps3d_mean_shift_ap50": 0.226,
            "gt_prototype_oracle_ap50": 0.534,
        },
        "variants": summary,
    }
    (out_dir / "seedness_diag.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "seedness_per_scene.json").write_text(
        json.dumps(per_scene, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[seedness] summary (radius x threshold):")
    for vname, vals in summary.items():
        print(
            f"  {vname:14s} AP50={vals['ap50']:.3f} "
            f"recall={vals['seed_recall']:.3f} "
            f"one_seed={vals['one_seed_frac']:.3f} "
            f"seeds/inst={vals['seeds_per_instance']:.2f} "
            f"purity={vals['seed_purity']:.3f}"
        )


if __name__ == "__main__":
    main()
