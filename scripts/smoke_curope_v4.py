#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

import torch


def main() -> int:
    started = time.time()
    import curope

    module_path = pathlib.Path(curope.__file__).resolve()
    tokens = torch.randn((1, 4, 2, 16), device="cuda", dtype=torch.float32)
    positions = torch.tensor([[[0, 0], [1, 2], [3, 5], [7, 11]]], device="cuda", dtype=torch.int64)
    before = tokens.clone()
    curope.rope_2d(tokens, positions, 100.0, 1.0)
    torch.cuda.synchronize()
    forward_finite = bool(torch.isfinite(tokens).all().item())
    forward_changed = bool(not torch.equal(before, tokens))

    grad = torch.ones_like(tokens)
    curope.rope_2d(grad, positions, 100.0, -1.0)
    torch.cuda.synchronize()
    backward_finite = bool(torch.isfinite(grad).all().item())

    result = {
        "python": sys.executable,
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "curope_module_path": str(module_path),
        "curope_extension_exists": module_path.suffix == ".so" and module_path.is_file(),
        "curope_cuda_kernel_valid": forward_finite and forward_changed,
        "curope_output_finite": forward_finite,
        "curope_backward_valid": backward_finite,
        "tokens_shape": list(tokens.shape),
        "positions_shape": list(positions.shape),
        "runtime_seconds": time.time() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["curope_cuda_kernel_valid"] and result["curope_backward_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
