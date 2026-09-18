#!/usr/bin/env python3
import inspect
import json
import os
import pathlib
import sys
import time

import torch


def main() -> int:
    started = time.time()
    import gsplat
    from gsplat.rendering import rasterization

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda", requires_grad=True)
    scales = torch.tensor([[0.3, 0.3, 0.3]], device="cuda", requires_grad=True)
    opacities = torch.tensor([0.9], device="cuda", requires_grad=True)
    colors = torch.tensor([[0.2, 0.4, 0.8]], device="cuda", requires_grad=True)
    viewmats = torch.eye(4, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    Ks = torch.tensor(
        [[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    renders, alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width=16,
        height=16,
        render_mode="RGB+D",
        packed=True,
    )
    torch.cuda.synchronize()
    output_finite = bool(torch.isfinite(renders).all().item() and torch.isfinite(alphas).all().item())
    loss = renders.sum() + alphas.sum()
    loss.backward()
    torch.cuda.synchronize()
    backward_finite = all(
        x.grad is not None and bool(torch.isfinite(x.grad).all().item())
        for x in (means, quats, scales, opacities, colors)
    )
    module_path = pathlib.Path(gsplat.__file__).resolve()

    imports = {}
    try:
        import src.pipeline as official_pipeline
        from src.config import load_typed_root_config
        from src.evaluator import Evaluator
        from src.models.model import SIU3RModel

        imports = {
            "pipeline": official_pipeline.__file__,
            "model": SIU3RModel.__module__,
            "evaluator": Evaluator.__module__,
            "config_loader": load_typed_root_config.__module__,
            "pipeline_import_valid": True,
        }
    except Exception as exc:
        imports = {
            "pipeline_import_valid": False,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
        }

    result = {
        "python": sys.executable,
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gsplat_module_path": str(module_path),
        "gsplat_version": getattr(gsplat, "__version__", None),
        "gsplat_extension_paths": [
            str(pathlib.Path(p).resolve())
            for p in sys.path
            if p and "venv_gpu_v4" in str(pathlib.Path(p).resolve())
        ],
        "gsplat_cuda_kernel_valid": True,
        "gsplat_output_finite": output_finite,
        "gsplat_backward_valid": backward_finite,
        "renders_shape": list(renders.shape),
        "alphas_shape": list(alphas.shape),
        "meta_keys": sorted(meta.keys()),
        "rasterization_signature": str(inspect.signature(rasterization)),
        "official_imports": imports,
        "runtime_seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if output_finite and backward_finite and imports.get("pipeline_import_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
