# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from gsplat.rendering import rasterization


class DeferredBP(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_w2c, test_intr, 
               W, H, near_plane, far_plane, backgrounds):
        rgbd, alpha, info = rasterization(
            means=xyz, 
            quats=rotation, 
            scales=scale, 
            opacities=opacity, 
            colors=feature,
            viewmats=test_w2c, 
            Ks=test_intr, 
            width=W, 
            height=H, 
            near_plane=near_plane, 
            far_plane=far_plane,
            backgrounds=backgrounds,
            render_mode="RGB+ED", 
            packed=False,   # important for correct means2d shape
        ) # (1, H, W, 3) 
        image, depth = rgbd[..., :3], rgbd[..., 3:]
        return image, alpha, depth, info['means2d']     # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_w2cs, test_intr,
                W, H, near_plane, far_plane, backgrounds):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds)
        ctx.W = W
        ctx.H = H
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        N = xyz.shape[1]
        with torch.no_grad():
            B, V = test_intr.shape[:2]
            images = torch.zeros(B, V, H, W, 3).to(xyz.device)
            alphas = torch.zeros(B, V, H, W, 1).to(xyz.device)
            depths = torch.zeros(B, V, H, W, 1).to(xyz.device)
            means2ds = torch.zeros(B, V, N, 2).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth, means2d = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1]
                    )
                    images[ib, iv:iv+1] = image
                    alphas[ib, iv:iv+1] = alpha
                    depths[ib, iv:iv+1] = depth
                    means2ds[ib, iv:iv+1] = means2d
        images = images.requires_grad_()
        alphas = alphas.requires_grad_()
        depths = depths.requires_grad_()
        means2ds = means2ds.requires_grad_()
        return images, alphas, depths, means2ds

    @staticmethod
    def backward(ctx, images_grad, alphas_grad, depths_grad, means2ds_grad):
        xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V = test_intr.shape[:2]
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth, means2d = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1]
                    )
                    render_split = torch.cat([image.reshape(-1), alpha.reshape(-1), depth.reshape(-1), means2d.reshape(-1)], dim=-1)
                    grad_split = torch.cat([images_grad[ib, iv:iv+1].reshape(-1), alphas_grad[ib, iv:iv+1].reshape(-1), depths_grad[ib, iv:iv+1].reshape(-1), means2ds_grad[ib, iv:iv+1].reshape(-1)], dim=-1) 
                    render_split.backward(grad_split)

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None

class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt
        
    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None):
        B, V = cam_view.shape[:2]
        # pos, opacity, scale, rotation, shs
        means3D = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3:4].contiguous().float().squeeze(-1)
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()
        rgbs = gaussians[..., 11:].contiguous().float() # [N, 3]

        viewmat = cam_view.float().transpose(3, 2)  # [B, V, 4, 4]
        Ks = torch.tensor([[[[view_intrinsic[0],0.,view_intrinsic[2]],[0.,view_intrinsic[1],view_intrinsic[3]],[0., 0., 1.]] for view_intrinsic in batch_intrinsic] for batch_intrinsic in intrinsics], dtype=means3D.dtype, device=means3D.device)
        backgrounds = bg_color[None, None].repeat(B, V, 1).to(means3D.device, means3D.dtype) if bg_color is not None else torch.ones(B, V, 3, dtype=means3D.dtype, device=means3D.device)

        H, W = self.opt.img_size
        near_plane, far_plane = self.opt.znear, self.opt.zfar
            
        if self.opt.deferred_bp:
            return self.render_deferred(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane)
        else:
            return self.render_standard(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane)

    def render_prompt_scores(self, gaussians, gaussian_scores, cam_view, intrinsics=None):
        """Render per-Gaussian probabilities through the unchanged RGB rasterizer.

        ``gaussian_scores`` is [B, Q, N]. Each query is copied to RGB feature
        channels in a temporary tensor; the source Gaussian geometry, opacity,
        and RGB tensor are never modified. The returned composite is therefore
        the regular alpha-composited score, with a separately returned alpha.
        """
        if gaussian_scores.ndim != 3:
            raise ValueError("gaussian_scores must have shape [B,Q,N]")
        if gaussians.ndim != 3 or gaussians.shape[0] != gaussian_scores.shape[0]:
            raise ValueError("gaussians and gaussian_scores batch dimensions disagree")
        if gaussians.shape[1] != gaussian_scores.shape[2]:
            raise ValueError("gaussians and gaussian_scores Gaussian counts disagree")
        if getattr(self.opt, "deferred_bp", False):
            raise ValueError("Prompt score rendering requires deferred_bp=False")

        probabilities = []
        alphas = []
        zero_background = torch.zeros(3, dtype=gaussians.dtype, device=gaussians.device)
        for query_index in range(gaussian_scores.shape[1]):
            score_rgb = gaussian_scores[:, query_index, :, None].expand(-1, -1, 3)
            # Keep geometry and opacity exactly as supplied. RGB is replaced only
            # in this temporary feature tensor used by the existing rasterizer.
            score_gaussians = torch.cat((gaussians[..., :11], score_rgb), dim=-1)
            rendered = self.render(
                score_gaussians,
                cam_view,
                bg_color=zero_background,
                intrinsics=intrinsics,
            )
            probabilities.append(rendered["images_pred"][:, :, :1])
            alphas.append(rendered["alphas_pred"])
        return {
            "rendered_prompt_probability": torch.stack(probabilities, dim=1),
            "rendered_alpha": torch.stack(alphas, dim=1),
        }

    def render_feature_channels(
        self,
        gaussians,
        features,
        cam_view,
        intrinsics=None,
        opacity_scale: float = 1.0,
    ):
        """Alpha-composite arbitrary per-Gaussian channels via gsplat N-D support.

        ``features`` is [B, N, D]. Geometry and opacity come from the unchanged
        Gaussian tensor; every feature channel is composited independently so
        the output is a D-channel map [B, V, D, H, W] plus alpha. Requires
        deferred_bp=False (same constraint as prompt score rendering).
        """
        if getattr(self.opt, "deferred_bp", False):
            raise ValueError(
                "Feature channel rendering requires deferred_bp=False"
            )
        if features.ndim != 3 or features.shape[1] != gaussians.shape[1]:
            raise ValueError(
                f"features must have shape [B,N,D], got {tuple(features.shape)} "
                f"with {gaussians.shape[1]} Gaussians"
            )
        B, V = cam_view.shape[:2]
        means3D = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3:4].contiguous().float().squeeze(-1)
        if opacity_scale != 1.0:
            # Sharpened opacity: the front-most Gaussian becomes effectively
            # opaque, so the alpha-composited per-pixel group channels are
            # dominated by one Gaussian instead of a uniform mix of many.
            # This keeps the rendered per-pixel group probabilities sharp
            # (the token-level assignment is already ~0.92 confident) and
            # makes both the training gradients and the eval confidence
            # meaningful instead of near-uniform.
            opacity = 1.0 - (1.0 - opacity).clamp(0.0, 1.0).pow(
                float(opacity_scale)
            )
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()
        colors = features.contiguous().float()
        feature_dim = colors.shape[-1]

        viewmat = cam_view.float().transpose(3, 2)  # [B, V, 4, 4]
        Ks = torch.tensor(
            [
                [
                    [
                        [view_intrinsic[0], 0.0, view_intrinsic[2]],
                        [0.0, view_intrinsic[1], view_intrinsic[3]],
                        [0.0, 0.0, 1.0],
                    ]
                    for view_intrinsic in batch_intrinsic
                ]
                for batch_intrinsic in intrinsics
            ],
            dtype=means3D.dtype,
            device=means3D.device,
        )
        backgrounds = torch.zeros(
            B, V, feature_dim, dtype=colors.dtype, device=colors.device
        )
        H, W = self.opt.img_size
        near_plane, far_plane = self.opt.znear, self.opt.zfar

        images, alphas = [], []
        for b in range(B):
            rendered_image_all, rendered_alpha_all, _ = rasterization(
                means=means3D[b],
                quats=rotations[b],
                scales=scales[b],
                opacities=opacity[b],
                colors=colors[b],
                viewmats=viewmat[b],
                Ks=Ks[b],
                width=W,
                height=H,
                near_plane=near_plane,
                far_plane=far_plane,
                backgrounds=backgrounds[b],
                render_mode="RGB",
                packed=False,
            )
            for rendered_image, rendered_alpha in zip(
                rendered_image_all, rendered_alpha_all
            ):
                images.append(rendered_image.permute(2, 0, 1))  # [D, H, W]
                alphas.append(rendered_alpha.permute(2, 0, 1))  # [1, H, W]
        images = torch.stack(images).view(B, V, feature_dim, H, W)
        alphas = torch.stack(alphas).view(B, V, 1, H, W)
        return {"images_pred": images, "alphas_pred": alphas}

    def render_token_features(
        self,
        gaussians: torch.Tensor,
        token_features: torch.Tensor,
        local_codes: torch.Tensor,
        local_basis: torch.Tensor,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
        local_residual_scale: float = 0.1,
        feature_chunk_size: int = 32,
        render_scale: float = 0.5,
        detach_geometry: bool = True,
    ) -> dict:
        """Render a token-compressed semantic field without materializing N x C.

        Args:
            gaussians: [B, N_gaussian, 14]
            token_features: [B, N_token, C]
            local_codes: [B, N_token, G, R]
            local_basis: [R, C]
        """
        if gaussians.ndim != 3:
            raise ValueError(
                f"gaussians must be [B,N,14], got {tuple(gaussians.shape)}"
            )
        if token_features.ndim != 3:
            raise ValueError(
                "token_features must be [B,N_token,C], "
                f"got {tuple(token_features.shape)}"
            )
        if local_codes.ndim != 4:
            raise ValueError(
                "local_codes must be [B,N_token,G,R], "
                f"got {tuple(local_codes.shape)}"
            )
        if local_basis.ndim != 2:
            raise ValueError(
                f"local_basis must be [R,C], got {tuple(local_basis.shape)}"
            )

        B, N_gaussian, _ = gaussians.shape
        B_token, N_token, feature_dim = token_features.shape
        B_code, N_token_code, gs_per_token, rank = local_codes.shape

        if B != B_token or B != B_code:
            raise ValueError(
                f"Batch mismatch: gaussians={B}, tokens={B_token}, codes={B_code}"
            )
        if N_token != N_token_code:
            raise ValueError(
                f"Token count mismatch: {N_token} vs {N_token_code}"
            )
        if N_gaussian != N_token * gs_per_token:
            raise ValueError(
                f"N_gaussian={N_gaussian} != "
                f"N_token({N_token}) * G({gs_per_token})"
            )
        if local_basis.shape != (rank, feature_dim):
            raise ValueError(
                "Basis shape mismatch: expected "
                f"({rank}, {feature_dim}), got {tuple(local_basis.shape)}"
            )

        means3D = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3:4].contiguous().float().squeeze(-1)
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()

        if detach_geometry:
            means3D = means3D.detach()
            opacity = opacity.detach()
            scales = scales.detach()
            rotations = rotations.detach()

        viewmat = cam_view.float().transpose(3, 2)

        H_rgb, W_rgb = self.opt.img_size
        H_feature = max(1, int(round(H_rgb * float(render_scale))))
        W_feature = max(1, int(round(W_rgb * float(render_scale))))

        scale_x = W_feature / float(W_rgb)
        scale_y = H_feature / float(H_rgb)
        intrinsics_scaled = intrinsics.float().clone()
        intrinsics_scaled[..., 0] *= scale_x
        intrinsics_scaled[..., 1] *= scale_y
        intrinsics_scaled[..., 2] *= scale_x
        intrinsics_scaled[..., 3] *= scale_y

        B_camera, V = intrinsics_scaled.shape[:2]
        if B_camera != B:
            raise ValueError(
                f"camera batch mismatch: {B_camera} vs {B}"
            )

        Ks = torch.zeros(
            B,
            V,
            3,
            3,
            dtype=means3D.dtype,
            device=means3D.device,
        )
        Ks[..., 0, 0] = intrinsics_scaled[..., 0]
        Ks[..., 1, 1] = intrinsics_scaled[..., 1]
        Ks[..., 0, 2] = intrinsics_scaled[..., 2]
        Ks[..., 1, 2] = intrinsics_scaled[..., 3]
        Ks[..., 2, 2] = 1.0

        near_plane = float(self.opt.znear)
        far_plane = float(self.opt.zfar)

        batch_features = []
        batch_alphas = []

        for b in range(B):
            rendered_chunks = []
            alpha_b = None

            token_features_b = token_features[b].float()
            local_codes_b = local_codes[b].float()
            local_basis_float = local_basis.float()

            for start in range(0, feature_dim, int(feature_chunk_size)):
                end = min(start + int(feature_chunk_size), feature_dim)

                token_chunk = token_features_b[:, start:end]
                residual_chunk = torch.einsum(
                    "tgr,rc->tgc",
                    local_codes_b,
                    local_basis_float[:, start:end],
                )
                gaussian_chunk = (
                    token_chunk[:, None, :]
                    + float(local_residual_scale) * residual_chunk
                ).reshape(N_gaussian, end - start).contiguous()

                backgrounds = torch.zeros(
                    V,
                    end - start,
                    dtype=gaussian_chunk.dtype,
                    device=gaussian_chunk.device,
                )

                rendered_chunk, rendered_alpha, _ = rasterization(
                    means=means3D[b],
                    quats=rotations[b],
                    scales=scales[b],
                    opacities=opacity[b],
                    colors=gaussian_chunk,
                    viewmats=viewmat[b],
                    Ks=Ks[b],
                    width=W_feature,
                    height=H_feature,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    packed=False,
                    backgrounds=backgrounds,
                    render_mode="RGB",
                )

                rendered_chunk = rendered_chunk.permute(0, 3, 1, 2)
                rendered_chunks.append(rendered_chunk)

                if alpha_b is None:
                    alpha_b = rendered_alpha.permute(0, 3, 1, 2)

            batch_features.append(torch.cat(rendered_chunks, dim=1))
            batch_alphas.append(alpha_b)

        return {
            "semantic_features_pred": torch.stack(batch_features, dim=0),
            "semantic_alphas_pred": torch.stack(batch_alphas, dim=0),
        }


    def render_deferred(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane):
        images, alphas, depths, means2ds = DeferredBP.apply(means3D, rgbs, scales, rotations, opacity, viewmat, Ks, W, H, near_plane, far_plane, backgrounds)
        return {
            "images_pred": images.permute(0,1,4,2,3), # [B, V, 3, H, W]
            "alphas_pred": alphas.permute(0,1,4,2,3), # [B, V, 1, H, W]
            "depths_pred": depths.permute(0,1,4,2,3), # [B, V, 1, H, W]
            "means2d_pred": means2ds, # [B, V, N, 2]
        }
                
                
    def render_standard(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane):
        # gaussians: [B, N, 14]
        # cam_pos: [B, V, 3]
        B, V = Ks.shape[:2]

        # loop of loop...
        images, alphas, depths, means2ds = [], [], [], []
        for b in range(B):
            rendered_image_all, rendered_alpha_all, info = rasterization(
                means=means3D[b],
                quats=rotations[b],
                scales=scales[b],
                opacities=opacity[b],
                colors=rgbs[b],
                viewmats=viewmat[b],
                Ks=Ks[b],
                width=W,
                height=H,
                near_plane=near_plane,
                far_plane=far_plane,
                packed=False,
                backgrounds=backgrounds[b],
                render_mode="RGB+ED",
            )
            for rendered_image, rendered_alpha, means2d in zip(rendered_image_all, rendered_alpha_all, info['means2d']):
                depths.append(rendered_image[...,3:].permute(2, 0, 1))
                rendered_image = rendered_image[...,:3].permute(2, 0, 1)
                images.append(rendered_image)
                alphas.append(rendered_alpha.permute(2, 0, 1))
                means2ds.append(means2d) # [N, 2]
        images, alphas, depths, means2ds = torch.stack(images), torch.stack(alphas), torch.stack(depths), torch.stack(means2ds)
        images, alphas, depths, means2ds = images.view(B, V, *images.shape[1:]), alphas.view(B, V, *alphas.shape[1:]), depths.view(B, V, *depths.shape[1:]), means2ds.view(B, V, *means2ds.shape[1:])

        return {
            "images_pred": images, # [B, V, 3, H, W]
            "alphas_pred": alphas, # [B, V, 1, H, W]
            "depths_pred": depths, # [B, V, 1, H, W]
            "means2d_pred": means2ds, # [B, V, N, 2]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = torch.logit(opacity.clamp(1e-8, 1 - 1e-8))  # inverse sigmoid
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
