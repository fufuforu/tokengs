# SPDX-License-Identifier: Apache-2.0
#
# DL3DV-style RE10K torch loader for TokenGS.
#
# Main RGB/camera data:
#   /datasets2/mengxl/re10k/train/*.torch
#
# Optional semantic pkl:
#   /space/daiy/datasets/train/<scene_id>/scene_data_complete.pkl
#
# This class follows the same style as TokenGS native dl3dv.py:
#   __len__()
#   load_video_reader(idx)
#   count_frames(idx)
#   get_data(idx, data_fields, frame_indices, ...)
#
# It returns:
#   output_dict["__key__"]
#   output_dict[DF_IMAGE_RGB]              # [T, 3, H, W], float, [0, 1]
#   output_dict[DF_CAMERA_C2W_TRANSFORM]   # [T, 4, 4]
#   output_dict[DF_CAMERA_INTRINSICS]      # [T, 4], pixel-space fx fy cx cy
#
# Optional:
#   output_dict["semantics"]

import argparse
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
import json
from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_IMAGE_RGB,
)


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_resolution(resolution):
    """
    resolution:
        None / "original"
        "448x256" means W=448, H=256
        "256x256"
        [H, W]
    return:
        None or (H, W)
    """
    if resolution is None:
        return None

    if isinstance(resolution, str):
        if resolution.lower() in ["none", "original"]:
            return None

        if "x" in resolution:
            w, h = resolution.lower().split("x")
            return int(h), int(w)

        if resolution.isdigit():
            s = int(resolution)
            return s, s

    if isinstance(resolution, (tuple, list)):
        assert len(resolution) == 2
        return int(resolution[0]), int(resolution[1])

    raise ValueError(f"Unsupported resolution: {resolution}")


def image_to_pil(x: Any) -> Image.Image:
    """
    RE10K torch chunks usually store image as encoded jpg/png bytes in UInt8 Tensor.
    """
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, bytes):
        return Image.open(BytesIO(x)).convert("RGB")

    if isinstance(x, bytearray):
        return Image.open(BytesIO(bytes(x))).convert("RGB")

    if torch.is_tensor(x):
        x = x.detach().cpu()

        # encoded image bytes
        if x.dtype == torch.uint8 and x.ndim == 1:
            return Image.open(BytesIO(x.numpy().tobytes())).convert("RGB")

        # already image tensor
        arr = x.numpy()
        return array_to_pil(arr)

    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8 and x.ndim == 1:
            return Image.open(BytesIO(x.tobytes())).convert("RGB")

        return array_to_pil(x)

    raise TypeError(f"Unsupported image type: {type(x)}")


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")

    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr).convert("RGB")

    raise ValueError(f"Unsupported image array shape: {arr.shape}")


def pil_to_tensor(img: Image.Image, resolution=None):
    """
    Args:
        img: PIL image
        resolution: None or (H, W)

    Returns:
        tensor: [3, H, W], float, [0, 1]
        original_size: (W_old, H_old)
        output_size: (W_new, H_new)
    """
    img = img.convert("RGB")
    w_old, h_old = img.size

    if resolution is not None:
        h_new, w_new = resolution
        img = img.resize((w_new, h_new), Image.BILINEAR)
    else:
        w_new, h_new = w_old, h_old

    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0

    return tensor, (w_old, h_old), (w_new, h_new)


def cameras18_to_c2w_and_intrinsics(cameras: torch.Tensor, original_sizes, output_sizes):
    """
    RE10K torch camera convention:
        cameras[:, :4]  = fx, fy, cx, cy, usually normalized
        cameras[:, 6:]  = W2C 3x4 matrix flatten

    Output follows DL3DV-style TokenGS:
        c2w:        [T, 4, 4]
        intrinsics: [T, 4], pixel-space fx fy cx cy
    """
    cameras = cameras.float()

    if cameras.ndim != 2 or cameras.shape[1] < 18:
        raise ValueError(f"Expected cameras [T, >=18], got {tuple(cameras.shape)}")

    T = cameras.shape[0]

    intr = cameras[:, :4].clone()
    fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]

    w_old = torch.tensor([s[0] for s in original_sizes], dtype=torch.float32)
    h_old = torch.tensor([s[1] for s in original_sizes], dtype=torch.float32)
    w_new = torch.tensor([s[0] for s in output_sizes], dtype=torch.float32)
    h_new = torch.tensor([s[1] for s in output_sizes], dtype=torch.float32)

    # 如果小于 4，基本可以认为是归一化内参。
    # RE10K / MVSplat 风格一般是 normalized fx fy cx cy。
    if intr.abs().max() <= 4.0:
        fx = fx * w_new
        cx = cx * w_new
        fy = fy * h_new
        cy = cy * h_new
    else:
        # 如果已经是 pixel 内参，则根据 resize 比例缩放。
        sx = w_new / w_old
        sy = h_new / h_old

        fx = fx * sx
        cx = cx * sx
        fy = fy * sy
        cy = cy * sy

    intrinsics = torch.stack([fx, fy, cx, cy], dim=-1).float()

    w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
    w2c[:, :3, :4] = cameras[:, 6:18].reshape(T, 3, 4)

    c2w = torch.linalg.inv(w2c).float()

    return c2w, intrinsics


def polygons_to_mask(polys, height: int, width: int):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if polys is None:
        return np.zeros((height, width), dtype=np.uint8)

    for poly in polys:
        arr = np.asarray(poly, dtype=np.float32)

        if arr.size < 6:
            continue

        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)

        # normalized polygon coordinates
        if arr.max() <= 2.0:
            arr[:, 0] *= width - 1
            arr[:, 1] *= height - 1

        xy = [tuple(p) for p in arr.tolist()]
        draw.polygon(xy, outline=1, fill=1)

    return np.asarray(mask, dtype=np.uint8)


class RE10KTorchDL3DVStyle:
    def __init__(
        self,
        root_path,
        split="train",
        semantic_root=None,
        resolution="448x256",
        include_semantics=False,
        semantic_mask_resolution=(224, 224),
        scene_in_chunk=0,
        index_file=None,
        max_scenes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.root_path = Path(root_path)
        self.split = split
        self.semantic_root = Path(semantic_root) if semantic_root is not None else None
        self.resolution = parse_resolution(resolution)
        self.include_semantics = include_semantics
        self.semantic_mask_resolution = tuple(semantic_mask_resolution)
        self.scene_in_chunk = int(scene_in_chunk)

        self.index_file = (
            Path(index_file)
            if index_file is not None
            else None
        )
        self.max_scenes = (
            None if max_scenes is None else int(max_scenes)
        )

        self.root_split = self.resolve_root_split(self.root_path, self.split)

        self.sample_list = self.load_or_build_scene_index()

        if len(self.sample_list) == 0:
            raise RuntimeError(
                f"No valid RE10K scenes found under {self.root_split}"
            )

        self.is_static = True

        self._cached_chunk_path = None
        self._cached_chunk = None

        print(
            f"[RE10KTorchDL3DVStyle] "
            f"num indexed scenes={len(self.sample_list)}"
        )
        print(
            f"[RE10KTorchDL3DVStyle] "
            f"index_file={self.index_file}"
        )

    @staticmethod
    def resolve_root_split(root_path: Path, split: str):
        root_path = Path(root_path)

        if (root_path / split).exists():
            return root_path / split

        if root_path.exists() and len(list(root_path.glob("*.torch"))) > 0:
            return root_path

        raise FileNotFoundError(
            f"Cannot find torch split directory. "
            f"root_path={root_path}, split={split}"
        )

    def load_or_build_scene_index(self):
        """
        一个 sample 对应一个 RE10K scene，而不是一个 .torch chunk。

        索引结构：
            {
                "chunk_path": ".../000000.torch",
                "local_idx": 3,
                "scene_key": "...",
                "num_frames": 117,
            }
        """

        if (
            self.index_file is not None
            and self.index_file.exists()
        ):
            print(
                "[RE10KTorchDL3DVStyle] "
                f"loading scene index: {self.index_file}"
            )

            with self.index_file.open(
                "r",
                encoding="utf-8",
            ) as f:
                sample_list = json.load(f)

            if self.max_scenes is not None:
                sample_list = sample_list[:self.max_scenes]

            return sample_list

        sample_list = self.build_sample_list(
            max_scenes=self.max_scenes,
        )

        if self.index_file is not None:
            self.index_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = self.index_file.with_suffix(
                self.index_file.suffix + ".tmp"
            )

            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    sample_list,
                    f,
                    ensure_ascii=False,
                )

            temporary_path.replace(self.index_file)

            print(
                "[RE10KTorchDL3DVStyle] "
                f"saved scene index: {self.index_file}"
            )

        return sample_list

    def build_sample_list(self, max_scenes=None):
        chunk_paths = sorted(self.root_split.glob("*.torch"))

        if len(chunk_paths) == 0:
            raise RuntimeError(f"No .torch files found under {self.root_split}")

        sample_list = []

        print(f"[RE10KTorchDL3DVStyle] scanning {len(chunk_paths)} torch chunks...")

        for chunk_path in chunk_paths:
            chunk = torch_load_cpu(chunk_path)

            if not isinstance(chunk, list):
                raise TypeError(f"Expected chunk to be list, got {type(chunk)} from {chunk_path}")

            for local_idx, example in enumerate(chunk):
                if not isinstance(example, dict):
                    continue

                if "key" not in example or "images" not in example or "cameras" not in example:
                    continue

                scene_key = str(example["key"])
                num_frames = len(example["images"])

                sample_list.append(
                    {
                        "chunk_path": str(chunk_path),
                        "local_idx": int(local_idx),
                        "scene_key": scene_key,
                        "num_frames": int(num_frames),
                    }
                )

                if max_scenes is not None and len(sample_list) >= max_scenes:
                    break

            if max_scenes is not None and len(sample_list) >= max_scenes:
                break

        if len(sample_list) == 0:
            raise RuntimeError(f"No valid RE10K examples found under {self.root_split}")

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def load_chunk(self, chunk_path: Path):
        chunk_path = Path(chunk_path)

        if self._cached_chunk_path == str(chunk_path) and self._cached_chunk is not None:
            return self._cached_chunk

        chunk = torch_load_cpu(chunk_path)

        self._cached_chunk_path = str(chunk_path)
        self._cached_chunk = chunk

        return chunk

    def load_video_reader(self, idx):
        """
        sample_list[idx] 对应一个具体 scene：

            chunk_path + local_idx
        """

        sample = self.sample_list[idx]

        chunk_path = Path(sample["chunk_path"])
        local_idx = int(sample["local_idx"])
        expected_key = str(sample["scene_key"])

        chunk = self.load_chunk(chunk_path)

        if not isinstance(chunk, list):
            raise TypeError(
                f"Expected chunk to be list, "
                f"got {type(chunk)} from {chunk_path}"
            )

        if not (0 <= local_idx < len(chunk)):
            raise IndexError(
                f"local_idx={local_idx} out of range for "
                f"{chunk_path}, chunk size={len(chunk)}"
            )

        example = chunk[local_idx]

        if not isinstance(example, dict):
            raise TypeError(
                f"Expected example dict, got {type(example)} "
                f"from {chunk_path}, local_idx={local_idx}"
            )

        required_keys = {"key", "images", "cameras"}
        missing_keys = required_keys - set(example.keys())

        if missing_keys:
            raise KeyError(
                f"Missing keys {missing_keys} in "
                f"{chunk_path}, local_idx={local_idx}"
            )

        clip_name = str(example["key"])
        video_length = len(example["images"])

        if clip_name != expected_key:
            raise RuntimeError(
                "RE10K index mismatch: "
                f"expected={expected_key}, actual={clip_name}, "
                f"chunk={chunk_path}, local_idx={local_idx}"
            )

        return example, clip_name, video_length

    def count_cameras(self, video_idx: int) -> int:
        return 1

    def count_frames(self, idx):
        return int(self.sample_list[idx]["num_frames"])

    def get_semantic_pkl_path(self, scene_key: str):
        if self.semantic_root is None:
            return None

        scene_names = [scene_key, Path(scene_key).name]
        scene_names = list(dict.fromkeys(scene_names))

        if self.split == "train":
            split_candidates = ["train", "val", "test"]
        elif self.split in ["val", "valid", "validation"]:
            split_candidates = ["val", "test", "train"]
        elif self.split == "test":
            split_candidates = ["test", "val", "train"]
        else:
            split_candidates = [self.split, "train", "val", "test"]

        for sp in split_candidates:
            for sn in scene_names:
                p = self.semantic_root / sp / sn / "scene_data_complete.pkl"
                if p.exists():
                    return p

        return None

    def load_semantics(self, scene_key: str, frame_indices: List[int]):
        """
        scene_data_complete.pkl only contains:
            polygons
            embeddings
        """
        Hm, Wm = self.semantic_mask_resolution

        empty = {
            "frame_indices": frame_indices,
            "masks": [torch.empty(0, Hm, Wm, dtype=torch.bool) for _ in frame_indices],
            "features": [torch.empty(0, 0, dtype=torch.float32) for _ in frame_indices],
            "object_ids": [[] for _ in frame_indices],
            "pkl_path": None,
        }

        pkl_path = self.get_semantic_pkl_path(scene_key)

        if pkl_path is None:
            return empty

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        polygons = data.get("polygons", {})
        embeddings = data.get("embeddings", {})

        all_masks = []
        all_features = []
        all_object_ids = []

        for frame_idx in frame_indices:
            frame_keys = [frame_idx, str(frame_idx)]

            frame_polygons = {}
            frame_embeddings = {}

            for k in frame_keys:
                if isinstance(polygons, dict) and k in polygons:
                    frame_polygons = polygons[k]
                    break

            for k in frame_keys:
                if isinstance(embeddings, dict) and k in embeddings:
                    frame_embeddings = embeddings[k]
                    break

            obj_ids = sorted(
                list(set(frame_polygons.keys()) & set(frame_embeddings.keys())),
                key=lambda x: str(x),
            )

            masks_this = []
            feats_this = []

            for obj_id in obj_ids:
                mask_np = polygons_to_mask(frame_polygons[obj_id], Hm, Wm)
                masks_this.append(torch.from_numpy(mask_np).bool())

                feat = frame_embeddings[obj_id]
                if torch.is_tensor(feat):
                    feat_tensor = feat.detach().cpu().float()
                else:
                    feat_tensor = torch.tensor(np.asarray(feat), dtype=torch.float32)

                feats_this.append(feat_tensor)

            if len(masks_this) > 0:
                masks_this = torch.stack(masks_this, dim=0)
                feats_this = torch.stack(feats_this, dim=0)
            else:
                masks_this = torch.empty(0, Hm, Wm, dtype=torch.bool)
                feats_this = torch.empty(0, 0, dtype=torch.float32)

            all_masks.append(masks_this)
            all_features.append(feats_this)
            all_object_ids.append(obj_ids)

        return {
            "frame_indices": frame_indices,
            "masks": all_masks,
            "features": all_features,
            "object_ids": all_object_ids,
            "pkl_path": str(pkl_path),
        }

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = "opencv",
    ):
        assert camera_convention == "opencv"

        example, clip_name, total_frames = self.load_video_reader(idx)

        if frame_indices is None:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(x) for x in frame_indices]

        frame_indices = [x for x in frame_indices if 0 <= x < total_frames]

        if len(frame_indices) == 0:
            raise RuntimeError(f"No valid frame_indices for scene {clip_name}")
        # import pdb;pdb.set_trace()
        # Load images.
        img_seq = []
        original_sizes = []
        output_sizes = []

        for frame_idx in frame_indices:
            img = image_to_pil(example["images"][frame_idx])
            img_tensor, original_size, output_size = pil_to_tensor(img, self.resolution)

            img_seq.append(img_tensor)
            original_sizes.append(original_size)
            output_sizes.append(output_size)

        img_seq = torch.stack(img_seq, dim=0).contiguous()

        # Load cameras.
        cameras_all = example["cameras"]

        if not torch.is_tensor(cameras_all):
            cameras_all = torch.tensor(np.asarray(cameras_all), dtype=torch.float32)

        cameras = cameras_all[frame_indices].float()

        c2w, intrinsics = cameras18_to_c2w_and_intrinsics(
            cameras,
            original_sizes=original_sizes,
            output_sizes=output_sizes,
        )

        output_dict = {}
        output_dict["__key__"] = clip_name

        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                output_dict[data_field] = img_seq

            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                output_dict[data_field] = c2w

            elif data_field == DF_CAMERA_INTRINSICS:
                output_dict[data_field] = intrinsics

        if self.include_semantics:
            output_dict["semantics"] = self.load_semantics(clip_name, frame_indices)

        return output_dict


def inspect_dataset(root_path, split="train", semantic_root=None, max_scenes=5):
    dataset = RE10KTorchDL3DVStyle(
        root_path=root_path,
        split=split,
        semantic_root=semantic_root,
        resolution="original",
        include_semantics=False,
        max_scenes=max_scenes,
    )

    print("\n[INSPECT]")
    print("num scenes:", len(dataset))

    for idx in range(min(max_scenes, len(dataset))):
        example, clip_name, total_frames = dataset.load_video_reader(idx)

        print("-" * 80)
        print("idx:", idx)
        print("clip_name:", clip_name)
        print("total_frames:", total_frames)
        print("example keys:", list(example.keys()))

        print("cameras:", type(example["cameras"]), getattr(example["cameras"], "shape", None))
        print("num images:", len(example["images"]))

        img = image_to_pil(example["images"][0])
        print("first image size:", img.size)

        sem_path = dataset.get_semantic_pkl_path(clip_name)
        print("semantic pkl:", sem_path)


def test_load(root_path, split, semantic_root, resolution, include_semantics, num_frames):
    dataset = RE10KTorchDL3DVStyle(
        root_path=root_path,
        split=split,
        semantic_root=semantic_root,
        resolution=resolution,
        include_semantics=include_semantics,
        max_scenes=20,
    )

    frame_indices = list(range(min(num_frames, dataset.count_frames(0))))

    output = dataset.get_data(
        idx=0,
        data_fields=[
            DF_IMAGE_RGB,
            DF_CAMERA_C2W_TRANSFORM,
            DF_CAMERA_INTRINSICS,
        ],
        frame_indices=frame_indices,
    )

    print("\n[TEST LOAD]")
    print("__key__:", output["__key__"])
    print("image:", tuple(output[DF_IMAGE_RGB].shape), output[DF_IMAGE_RGB].dtype)
    print("image range:", float(output[DF_IMAGE_RGB].min()), float(output[DF_IMAGE_RGB].max()))
    print("c2w:", tuple(output[DF_CAMERA_C2W_TRANSFORM].shape), output[DF_CAMERA_C2W_TRANSFORM].dtype)
    print("intrinsics:", tuple(output[DF_CAMERA_INTRINSICS].shape), output[DF_CAMERA_INTRINSICS].dtype)
    print("first intrinsics:", output[DF_CAMERA_INTRINSICS][0].tolist())

    if include_semantics:
        sem = output["semantics"]
        print("semantic pkl:", sem["pkl_path"])
        print("num semantic frames:", len(sem["masks"]))

        if len(sem["masks"]) > 0:
            print("first masks:", tuple(sem["masks"][0].shape))
            print("first features:", tuple(sem["features"][0].shape))
            print("first object ids:", sem["object_ids"][0][:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--semantic-root", type=str, default=None)
    parser.add_argument("--resolution", type=str, default="448x256")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--test-load", action="store_true")
    parser.add_argument("--include-semantics", action="store_true")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=5)
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(
            root_path=args.root_path,
            split=args.split,
            semantic_root=args.semantic_root,
            max_scenes=args.max_scenes,
        )
        return

    if args.test_load:
        test_load(
            root_path=args.root_path,
            split=args.split,
            semantic_root=args.semantic_root,
            resolution=args.resolution,
            include_semantics=args.include_semantics,
            num_frames=args.num_frames,
        )
        return

    print("Use --inspect or --test-load")


if __name__ == "__main__":
    main()
