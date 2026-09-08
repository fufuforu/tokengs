"""240-only stand-in for the ``fused_ssim_cuda`` extension.

The real ``fused-ssim`` extension is fetched from GitHub, which is not
reachable from this cluster (only the tuna/aliyun PyPI mirrors are).  The
recon-only LSM evaluation never calls fused-ssim (quality metrics use
lpips / skimage), so this shim only needs to satisfy the import performed
by ``tokengs.rendering.fused_ssim`` when CUDA is available.  If any
training path accidentally reaches these functions it fails loudly instead
of silently degrading.
"""

def fusedssim(*_args, **_kwargs):
    raise NotImplementedError(
        "fused_ssim_cuda is a 240-only import shim; fused-ssim training "
        "losses are not supported in the recon-only eval environment."
    )


def fusedssim_backward(*_args, **_kwargs):
    raise NotImplementedError(
        "fused_ssim_cuda is a 240-only import shim; fused-ssim training "
        "losses are not supported in the recon-only eval environment."
    )
