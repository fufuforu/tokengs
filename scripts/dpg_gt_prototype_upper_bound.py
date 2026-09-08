"""Offline upper bound: GT prototypes built from the TRAINING pseudo-label
distributions (conf-filtered p_u) vs the clean-dominant-instance prototypes
used by the 0.534 oracle.

The DPG learner is supervised to match P_gt = p_u-weighted unit-embedding
means (conf-filtered 15-view pseudo labels).  If those P_gt are noisy, the
learnable prototypes can never exceed the AP of hard cosine assignment to
P_gt itself.  This script measures that ceiling on LSM 40 scenes and
compares it with the clean-label GT-prototype oracle (0.534).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch._dynamo.config.disable = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.instance_group_head import (
    TokenLocalUnitGrouping,
    _project_dense_features,
)
from tokengs.options import config_defaults
from tokengs.utils.instance_ap import (
    gt_masks_from_instance_map,
    instance_ap,
    masks_from_group_probs,
)

from ablate_unit_clustering import _gs_level_stats, _max_iou_recall  # noqa: E402
from dino_dense_feature_oracle import (  # noqa: E402
    _dino_frame_hw,
    _rescale_intrinsics,
    dinov2_patch_features,
    load_dinov2,
)
from eval_instance_lsm_protocol import (  # noqa: E402
    _LocalAccelerator,
    _audit_lsm_manifest,
    _load_checkpoint_arch,
)


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
        "--workspace", default="workspace/dpg_gt_proto_upper"
    )
    parser.add_argument("--label", default="dpg_gt_proto_upper")
    parser.add_argument("--model_type", default="semantic_tokengs_v6")
    parser.add_argument("--num_groups", type=int, default=64)
    parser.add_argument("--lsm_manifest", default=(
        "data/scannet_prompt/lsm_instance_eval_manifest.json"
    ))
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=15)
    args = parser.parse_args()

    if args.num_input_views != 8 or args.num_views != 15:
        raise ValueError("LSM protocol requires 8 context + 7 target views")
    manifest_audit = _audit_lsm_manifest(str(ROOT / args.lsm_manifest))

    opt = config_defaults["eval_scannet_lsm_instance"]
    opt.model_type = args.model_type
    opt.resume = args.resume
    opt.workspace = args.workspace
    opt.experiment_name = Path(args.workspace).name
    opt.instance_group_num_groups = int(args.num_groups)
    opt.num_input_views = int(args.num_input_views)
    opt.num_views = int(args.num_views)
    opt.dataset_kwargs = {
        **dict(opt.dataset_kwargs or {}),
        "lsm_manifest_path": str(ROOT / args.lsm_manifest),
    }
    _load_checkpoint_arch(args, opt)
    # Reproduce the DPG training supervision: conf-filtered p_u.
    opt.instance_branch_pseudo_conf = 0.6
    opt.instance_branch_pseudo_min_views = 3
    opt.instance_branch_pseudo_unit_min_mass = 0.25
    out_dir = Path(args.workspace)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dinov2 = load_dinov2(device)

    _orig_forward = TokenLocalUnitGrouping._forward_unit_embedding

    def _patched_forward(
        self, a, unit_feat, unit_center, means, gaussians, data, opt_inner,
        training, model_input, batch_size, token_count, n_gs, p,
        dense_features=None,
    ):
        out = _orig_forward(
            self, a, unit_feat, unit_center, means, gaussians, data,
            opt_inner, training, model_input, batch_size, token_count, n_gs,
            p, dense_features,
        )
        if not training:
            self._ub_buffers = {
                "a": a.detach().float(),
                "means": means.detach().float(),
                "gaussians": gaussians.detach().float(),
                "cam_view": model_input.decoder.cam_view.detach().float(),
                "intrinsics": model_input.decoder.intrinsics.detach().float(),
                "cam_to_world_input": data["cam_to_world_input"].detach().float(),
                "intrinsics_input": data["intrinsics_input"].detach().float(),
            }
        return out

    TokenLocalUnitGrouping._forward_unit_embedding = _patched_forward

    from safetensors.torch import load_file

    model = model_registry[opt.model_type](opt)
    resume_ckpt = load_file(args.resume, device="cpu")
    torch.nn.Module.load_state_dict(model, resume_ckpt, strict=False)
    if not any(key.startswith("enc_dec_backbone.") for key in resume_ckpt):
        backbone_path = str(getattr(opt, "backbone_resume", "") or "")
        if backbone_path and Path(backbone_path).is_file():
            backbone_ckpt = load_file(backbone_path, device="cpu")
            frozen_prefixes = (
                "enc_dec_backbone.",
                "patch_embed.",
                "patch_plucker_embed.",
                "activation_head.",
                "anchor_pos_encoder.",
            )
            native_state = torch.nn.Module.state_dict(model)
            loadable = {
                key: value
                for key, value in backbone_ckpt.items()
                if (key.startswith(frozen_prefixes) or key == "gs_tokens")
                and key in native_state
                and native_state[key].shape == value.shape
            }
            torch.nn.Module.load_state_dict(model, loadable, strict=False)
            print(
                f"[dpg-ub] frozen-backbone eval: loaded "
                f"{len(loadable)} backbone keys from {backbone_path}"
            )
    model.eval()
    model = model.cuda()
    br = model.instance_branch
    renderer = br.renderer
    render_scale = float(getattr(opt, "instance_group_render_scale", 1.0))
    void_fg_share = float(getattr(opt, "instance_branch_void_fg_share", 0.5))
    min_pred_pixels = 1
    min_gt_pixels = 1
    max_pred = 100

    _, test_loader, _, _ = get_multi_dataloader(opt, _LocalAccelerator())

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
        with torch.inference_mode():
            model(data, compute_quality_metrics=False)
        buf = br._ub_buffers
        batch_size = 1
        token_count = buf["a"].shape[1]
        k = br.units_per_token
        p = buf["a"].shape[2]
        u_count = token_count * k
        n_gs = token_count * p
        a4 = buf["a"][0].reshape(token_count, p, k).cpu().numpy()
        means_np = buf["means"].reshape(batch_size, n_gs, 3)

        dino_grid = dinov2_patch_features(
            dinov2, data["images_input"], device
        )
        with torch.inference_mode():
            fused, _ = _project_dense_features(
                means_np,
                dino_grid,
                buf["cam_to_world_input"],
                _rescale_intrinsics(
                    buf["intrinsics_input"],
                    tuple(opt.img_size),
                    _dino_frame_hw(),
                ),
                _dino_frame_hw(),
            )
        gs_dino = fused[0].cpu().numpy()
        unit_dino = np.einsum(
            "tpk,tpd->tkd", a4, gs_dino.reshape(token_count, p, -1)
        )
        mass = np.einsum("tpk->tk", a4).clip(min=1e-6)
        unit_dino = (unit_dino / mass[:, :, None]).reshape(u_count, -1)
        unit_dino = unit_dino / np.maximum(
            np.linalg.norm(unit_dino, axis=-1, keepdims=True), 1e-8
        )

        gs_inst, _, _ = br._pseudo_gs_labels(means_np, data, opt)
        gs_inst = gs_inst[0].cpu().numpy()
        # p_u with the SAME conf filtering as DPG training.
        gs_gt3 = gs_inst.reshape(token_count, p)
        gt0 = torch.from_numpy(gs_gt3).long()
        high_conf = torch.from_numpy(
            np.ones_like(gs_gt3, dtype=bool)
        )  # will recompute below
        # Recompute conf-filtered p_u exactly like _forward_dpg:
        p_u = br.last_unit_pu[0].cpu().numpy()
        bg_idx = int(br.last_unit_bg_idx)
        fg_ids = [
            i for i in range(p_u.shape[-1]) if i != bg_idx
        ] if bg_idx >= 0 else list(range(p_u.shape[-1]))
        fg_ids = [i for i in fg_ids if p_u[:, i].sum() > 1e-3]

        # Variant 1: training-supervision GT prototypes (p_u-weighted mean).
        p_fg = p_u[:, fg_ids]
        denom = p_fg.sum(axis=0, keepdims=True).clip(min=1e-6)
        P_gt_pu = (unit_dino.T @ p_fg) / denom  # [D,F]
        P_gt_pu = P_gt_pu / np.maximum(
            np.linalg.norm(P_gt_pu, axis=0, keepdims=True), 1e-8
        )
        P_gt_pu = P_gt_pu.T  # [F,D]
        # Variant 2: clean dominant-instance prototypes (oracle style).
        dom = p_u.argmax(axis=1)
        fg_mask = (p_u.max(axis=1) > 0.5) & (dom != 0) & (
            np.isin(dom, fg_ids)
        )
        clean_ids = np.unique(dom[fg_mask])
        P_clean = np.stack(
            [unit_dino[fg_mask & (dom == gid)].mean(axis=0)
             for gid in clean_ids]
        )
        P_clean = P_clean / np.maximum(
            np.linalg.norm(P_clean, axis=-1, keepdims=True), 1e-8
        )

        gaussians = buf["gaussians"].float()
        cam_view = buf["cam_view"].float()
        intrinsics = buf["intrinsics"].float()
        gt_maps = data["instance_label_output"][0].cpu().numpy()

        scene_res = {}
        for vname, P, fg_ids_used in (
            ("train_pu_proto", P_gt_pu, fg_ids),
            ("clean_dom_proto", P_clean, clean_ids.tolist()),
        ):
            sim = unit_dino @ P.T  # [U,F]
            labels = np.full(u_count, -1, dtype=np.int64)
            fg_units = np.where(p_u[:, fg_ids_used].sum(axis=1) > 0)[0]
            labels[fg_units] = np.argmax(sim[fg_units], axis=1)
            fg_share_u = (
                1.0 - p_u[:, bg_idx] if bg_idx >= 0
                else np.ones(u_count)
            )
            cluster_list = []
            for c in np.unique(labels):
                if c < 0:
                    continue
                sel = labels == c
                if fg_share_u[sel].mean() >= void_fg_share:
                    cluster_list.append(int(c))
            used = {c: j for j, c in enumerate(cluster_list)}
            num = len(cluster_list)
            onehot = np.zeros((u_count, num + 1), dtype=np.float32)
            for uu in range(u_count):
                c = int(labels[uu])
                onehot[uu, used[c] if c in used else num] = 1.0
            unit_probs_t = onehot.reshape(token_count, k, num + 1)
            group_probs = np.einsum(
                "tpk,tkl->tpl", a4, unit_probs_t
            ).reshape(n_gs, num + 1)
            group_probs_t = torch.from_numpy(group_probs).unsqueeze(0).cuda()
            with torch.inference_mode():
                render = renderer.render_feature_channels(
                    gaussians, group_probs_t, cam_view,
                    intrinsics=intrinsics, opacity_scale=render_scale,
                )
            rendered_channels = (
                render["images_pred"].cpu().numpy()[0]
                / (render["alphas_pred"].cpu().numpy()[0] + 1e-5)
            )
            rendered_probs = rendered_channels / np.maximum(
                rendered_channels.sum(axis=1, keepdims=True), 1e-6
            )
            view_count = rendered_probs.shape[0]
            pred_masks, pred_scores, pred_image_ids = [], [], []
            gt_masks, gt_image_ids = [], []
            for v in range(view_count):
                image_id = f"{scene_name}:b0"
                masks, scores = masks_from_group_probs(
                    rendered_probs[v],
                    void_channel=num,
                    min_mask_area=min_pred_pixels,
                )
                masks = masks[:max_pred]
                scores = scores[:max_pred]
                gts = gt_masks_from_instance_map(
                    gt_maps[v], min_mask_area=min_gt_pixels
                )
                pred_masks.extend(masks)
                pred_scores.extend(scores)
                pred_image_ids.extend([image_id] * len(masks))
                gt_masks.extend(gts)
                gt_image_ids.extend([image_id] * len(gts))
            results = instance_ap(
                pred_masks, pred_scores, gt_masks,
                thresholds=(0.25, 0.5, 0.75),
                vectorized=True,
                pred_image_ids=pred_image_ids,
                gt_image_ids=gt_image_ids,
            )
            coco_ap = instance_ap(
                pred_masks, pred_scores, gt_masks,
                thresholds=tuple(t / 100 for t in range(50, 100, 5)),
                vectorized=True,
                pred_image_ids=pred_image_ids,
                gt_image_ids=gt_image_ids,
            )
            recall = _max_iou_recall(pred_masks, gt_masks)
            gs_stats = _gs_level_stats(group_probs, gs_inst, num)
            scene_res[vname] = {
                "ap25": results["ap_25"],
                "ap50": results["ap_50"],
                "ap": coco_ap["ap_mean"],
                "num_pred": len(pred_masks),
                "num_gt": len(gt_masks),
                "num_clusters": num,
                "recall_05": recall[0.5],
                **gs_stats,
            }
        per_scene[scene_name] = scene_res
        elapsed = time.time() - t_start
        print(
            f"[dpg-ub] {scene_name} done "
            f"({i + 1}/{min(args.max_scenes or 40, 40)}) "
            f"pu={scene_res['train_pu_proto']['ap50']:.3f} "
            f"clean={scene_res['clean_dom_proto']['ap50']:.3f} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

    def _mean(key: str, vname: str) -> float:
        vals = [per_scene[s][vname][key] for s in per_scene]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "num_scenes": len(per_scene),
        "train_pu_proto_ap50": _mean("ap50", "train_pu_proto"),
        "train_pu_proto_ap25": _mean("ap25", "train_pu_proto"),
        "clean_dom_proto_ap50": _mean("ap50", "clean_dom_proto"),
        "clean_dom_proto_ap25": _mean("ap25", "clean_dom_proto"),
        "train_pu_proto_pred": _mean("num_pred", "train_pu_proto"),
        "clean_dom_proto_pred": _mean("num_pred", "clean_dom_proto"),
        "reference": {
            "dino_agglomerative_ap50": 0.324,
            "dino_gt_prototype_oracle_ap50": 0.534,
        },
    }
    (out_dir / "dpg_gt_proto_upper.json").write_text(
        json.dumps(
            {"summary": summary, "per_scene": per_scene},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n[dpg-ub] summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
