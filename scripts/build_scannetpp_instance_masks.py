"""Generate per-frame ScanNet++ instance masks by projecting the 3D mesh.

For each scene:
  1. Load the aligned mesh (vertices + faces), per-vertex segment indices
     (``scans/segments.json``) and the segment -> objectId annotations
     (``scans/segments_anno.json``); build compact per-vertex instance ids.
  2. Convert every annotated face into one Gaussian (centroid, size from the
     triangle, rotation from the face normal, opacity 1, colour = instance id)
     and rasterize it with gsplat into each dslr frame.
  3. Frames are fisheye (nerfstudio ``OPENCV_FISHEYE``); images are
     undistorted with cv2 and the pinhole camera is used for rendering, so
     the RGB and the mask are pixel-aligned and the training loader can feed
     pinhole intrinsics to the (frozen) pinhole renderer.

Outputs (per scene): ``<out_root>/<scene>/{images,masks}/<stem>.png`` plus a
``manifest.json`` with c2w, pinhole K, and the compact instance id map.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat.rendering import rasterization


SCANNETPP_ROOT = Path("/datasets2/scannetpp/data")
DEFAULT_OUT_ROOT = Path("/space0/mawb/tokengs/data/scannetpp_processed")


def load_mesh(scene_dir: Path):
    ply = PlyData.read(str(scene_dir / "scans" / "mesh_aligned_0.05.ply"))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32)
    faces = np.stack(
        [np.asarray(f, dtype=np.int64) for f in ply["face"]["vertex_indices"]]
    )
    return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(faces)


def build_instance_ids(scene_dir: Path, num_vertices: int):
    seg = json.loads((scene_dir / "scans" / "segments.json").read_text())
    seg_indices = np.asarray(seg["segIndices"], dtype=np.int64)
    if len(seg_indices) != num_vertices:
        raise RuntimeError(
            f"segIndices len {len(seg_indices)} != vertices {num_vertices}"
        )
    anno = json.loads((scene_dir / "scans" / "segments_anno.json").read_text())
    segment_to_object = {}
    for group in anno["segGroups"]:
        object_id = int(group["objectId"])
        for segment in group["segments"]:
            segment_to_object[int(segment)] = object_id
    object_ids = sorted(
        {object_id for object_id in segment_to_object.values()}
    )
    object_to_compact = {
        object_id: index + 1 for index, object_id in enumerate(object_ids)
    }
    vertex_object = np.zeros(num_vertices, dtype=np.int64)
    for vertex_index, segment in enumerate(seg_indices):
        object_id = segment_to_object.get(int(segment), 0)
        vertex_object[vertex_index] = object_to_compact.get(object_id, 0)
    id_map = {
        str(compact): {"objectId": object_id}
        for object_id, compact in object_to_compact.items()
    }
    return vertex_object, id_map


def faces_to_gaussians(
    xyz: torch.Tensor,
    faces: torch.Tensor,
    vertex_ids: torch.Tensor,
    device: torch.device,
):
    face_xyz = xyz[faces]  # [F,3,3]
    face_id = torch.mode(vertex_ids[faces], dim=1).values
    keep = face_id > 0
    face_xyz = face_xyz[keep]
    face_id = face_id[keep]
    means = face_xyz.mean(dim=1)
    e0 = face_xyz[:, 1] - face_xyz[:, 0]
    e1 = face_xyz[:, 2] - face_xyz[:, 0]
    normal = torch.cross(e0, e1)
    normal_len = normal.norm(dim=-1).clamp_min(1e-8)
    normal = normal / normal_len[:, None]
    edge_len = torch.stack(
        [
            (face_xyz[:, 1] - face_xyz[:, 2]).norm(dim=-1),
            (face_xyz[:, 0] - face_xyz[:, 2]).norm(dim=-1),
            (face_xyz[:, 0] - face_xyz[:, 1]).norm(dim=-1),
        ],
        dim=-1,
    ).max(dim=-1).values
    scale = (edge_len * 0.5).clamp_min(1e-3)
    scales = torch.stack([scale, scale, scale * 0.08], dim=-1)
    b1 = torch.nn.functional.normalize(e0, dim=-1)
    b2 = torch.cross(normal, b1)
    rot = torch.stack([b1, b2, normal], dim=-1)  # [F,3,3] columns
    quats = rotmat_to_quat(rot)
    opacity = torch.ones_like(scale)
    colors = face_id.float().unsqueeze(-1).repeat(1, 3)
    return (
        means.to(device),
        scales.to(device),
        quats.to(device),
        opacity.to(device),
        colors.to(device),
    )


def rotmat_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [N,3,3] -> quaternions [N,4] (w,x,y,z)."""
    r00, r01, r02 = rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2]
    r10, r11, r12 = rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2]
    r20, r21, r22 = rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2]
    trace = r00 + r11 + r22
    w = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 0.5
    x = (r21 - r12) / (4.0 * w + 1e-8)
    y = (r02 - r20) / (4.0 * w + 1e-8)
    z = (r10 - r01) / (4.0 * w + 1e-8)
    return torch.stack([w, x, y, z], dim=-1)


def render_mask(
    means, scales, quats, opacity, colors, w2c, K, width, height, device
):
    viewmats = w2c.float().unsqueeze(0).unsqueeze(0).to(device)
    Ks = K.float().unsqueeze(0).unsqueeze(0).to(device)
    backgrounds = torch.zeros(1, 1, 3, dtype=torch.float32, device=device)
    rendered, _, _ = rasterization(
        means=means.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacity.unsqueeze(0),
        colors=colors.unsqueeze(0),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=0.05,
        far_plane=200.0,
        backgrounds=backgrounds,
        render_mode="RGB",
        packed=False,
    )
    mask = rendered[0, 0, :, :, 0].detach().cpu().numpy()  # [H,W] instance id
    return np.clip(np.rint(mask), 0, 65534).astype(np.uint16)


def process_scene(scene_id: str, out_root: Path, render_scale: float, device):
    scene_dir = SCANNETPP_ROOT / scene_id
    transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    if not transforms_path.is_file():
        print(f"[skip] {scene_id}: no nerfstudio transforms")
        return False
    out_scene = out_root / scene_id
    image_dir = out_scene / "images"
    mask_dir = out_scene / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_scene / "manifest.json"
    if manifest_path.exists():
        print(f"[skip] {scene_id}: already processed")
        return False

    xyz, rgb, faces = load_mesh(scene_dir)
    vertex_ids_np, id_map = build_instance_ids(scene_dir, xyz.shape[0])
    vertex_ids = torch.from_numpy(vertex_ids_np)
    means, scales, quats, opacity, colors = faces_to_gaussians(
        xyz, faces, vertex_ids, device
    )
    del xyz, faces, vertex_ids
    torch.cuda.empty_cache()
    print(
        f"[{scene_id}] gaussians={means.shape[0]} "
        f"instances={len(id_map)}"
    )

    transforms = json.loads(transforms_path.read_text())
    fx, fy, cx, cy = (
        float(transforms["fl_x"]),
        float(transforms["fl_y"]),
        float(transforms["cx"]),
        float(transforms["cy"]),
    )
    width, height = int(transforms["w"]), int(transforms["h"])
    k1, k2, k3, k4 = (
        float(transforms["k1"]),
        float(transforms["k2"]),
        float(transforms["k3"]),
        float(transforms["k4"]),
    )
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([[k1, k2, k3, k4]], dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )

    out_w = max(1, int(round(width * render_scale)))
    out_h = max(1, int(round(height * render_scale)))
    K_render = new_K * render_scale
    K_render[2, 2] = 1.0

    manifest_frames = []
    for frame_index, frame in enumerate(transforms["frames"]):
        stem = Path(frame["file_path"]).stem
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        image_path = (
            scene_dir / "dslr" / "resized_undistorted_images" / frame["file_path"]
        )
        if not image_path.is_file():
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        if render_scale != 1.0:
            undistorted = cv2.resize(
                undistorted,
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )
        Image.fromarray(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)).save(
            str(image_dir / f"{stem}.jpg"), quality=95
        )
        mask = render_mask(
            means,
            scales,
            quats,
            opacity,
            colors,
            torch.from_numpy(w2c),
            torch.from_numpy(K_render),
            out_w,
            out_h,
            device,
        )
        Image.fromarray(mask, mode="I;16").save(str(mask_dir / f"{stem}.png"))
        manifest_frames.append(
            {
                "stem": stem,
                "image": f"images/{stem}.jpg",
                "mask": f"masks/{stem}.png",
                "c2w": c2w.tolist(),
                "K": new_K.tolist(),
                "K_render": K_render.tolist(),
                "width": out_w,
                "height": out_h,
            }
        )
        if (frame_index + 1) % 20 == 0:
            print(f"[{scene_id}] {frame_index + 1}/{len(transforms['frames'])}")

    manifest = {
        "scene_id": scene_id,
        "id_map": id_map,
        "num_frames": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[{scene_id}] done: {len(manifest_frames)} frames")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--scenes-file", default="")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.scene_id:
        scene_ids = [args.scene_id]
    elif args.scenes_file:
        scene_ids = [
            line.strip()
            for line in Path(args.scenes_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        scene_ids = sorted(
            path.name
            for path in SCANNETPP_ROOT.iterdir()
            if path.is_dir() and (path / "dslr" / "nerfstudio").is_dir()
        )
    if args.max_scenes > 0:
        scene_ids = scene_ids[args.start_index : args.start_index + args.max_scenes]
    else:
        scene_ids = scene_ids[args.start_index :]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for scene_id in scene_ids:
        try:
            if process_scene(scene_id, out_root, args.render_scale, device):
                done += 1
        except Exception as exc:
            print(f"[error] {scene_id}: {type(exc).__name__}: {exc}")
    print(f"[build-scannetpp] finished {done}/{len(scene_ids)} scenes")


if __name__ == "__main__":
    main()
