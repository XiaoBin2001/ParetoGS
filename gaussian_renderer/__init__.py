"""
Renderer wrapper around the CUDA Gaussian rasterizer.

This module hides the low-level rasterizer argument layout from training code.
The main path renders normal 3DGS images and keeps gradients when training needs
them. This wrapper requests RGB rendering; inverse-depth supervision is removed.
"""
from __future__ import annotations

import torch
import math
from utils.rasterizer_backend import GaussianRasterizationSettings, GaussianRasterizer
from arguments import GroupParams
from scene.cameras import Camera, MiniCam
from scene.gaussian_model import GaussianModel as GaussianModel3DGS

RenderCamera = Camera | MiniCam

def _build_screenspace_points(
    pc: GaussianModel3DGS,
    track_gradients: bool,
    gaussian_count: int | None = None,
) -> torch.Tensor:
    """
        Create the 2D placeholder tensor passed to the rasterizer.

        The CUDA backward pass writes gradients into this tensor. Channels 0-1
        are the standard screen-space gradients used by 3DGS; channels 2-3 store
        absolute gradients used by ABSGS and ImprovedGS.
    """
    if gaussian_count is None:
        gaussian_count = int(pc.get_xyz.shape[0])
    screenspace_points = torch.zeros(
        (int(gaussian_count), 4),
        dtype=pc.get_xyz.dtype,
        requires_grad=track_gradients,
        device=pc.get_xyz.device,
    )
    if track_gradients:
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass
    return screenspace_points


def _build_3dgs_raster_settings(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float,
    inference_only: bool,
    pixel_weights: torch.Tensor | None,
) -> GaussianRasterizationSettings:
    """
        Build rasterizer settings for the main 3DGS render call.

        These settings choose image size, camera transforms, active SH degree,
        debug flags, and optional pixel weights used for edge-aware scoring.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        inference_only=inference_only,
        compute_invdepth=False,
        pixel_weights=pixel_weights,
    )


def render_3dgs(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    track_gradients: bool = True,
    inference_only: bool = False,
    pixel_weights: torch.Tensor | None = None,
    gaussian_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
        Render one camera view with the standard 3DGS parameter tensors.

        The function always passes DC color, higher-order SH, scale, rotation,
        and opacity explicitly. During training it returns gradients and
        visibility data for densification; during inference it returns only the
        rendered image to reduce work.
    """
    if inference_only:
        track_gradients = False
    if inference_only and pixel_weights is not None:
        raise ValueError("pixel_weights is not supported when inference_only=True.")

    if gaussian_indices is None:
        means3d = pc.get_xyz
        features_dc = pc.get_features_dc
        features_rest = pc.get_features_rest
        opacities = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
    else:
        if gaussian_indices.dtype != torch.long or gaussian_indices.ndim != 1:
            raise ValueError("gaussian_indices must be a 1D int64 tensor.")
        if gaussian_indices.device != pc.get_xyz.device:
            gaussian_indices = gaussian_indices.to(device=pc.get_xyz.device)
        means3d = torch.index_select(pc.get_xyz, 0, gaussian_indices)
        features_dc = torch.index_select(pc.get_features_dc, 0, gaussian_indices)
        features_rest = torch.index_select(pc.get_features_rest, 0, gaussian_indices)
        opacities = torch.index_select(pc.get_opacity, 0, gaussian_indices)
        scales = torch.index_select(pc.get_scaling, 0, gaussian_indices)
        rotations = torch.index_select(pc.get_rotation, 0, gaussian_indices)

    screenspace_points = _build_screenspace_points(pc, track_gradients, int(means3d.shape[0]))
    if pixel_weights is not None:
        pixel_weights = pixel_weights.to(device=pc.get_xyz.device)
        if pixel_weights.dtype == torch.uint8:
            pixel_weights = pixel_weights.float().div_(255.0)
        else:
            pixel_weights = pixel_weights.to(dtype=torch.float32)

    raster_settings = _build_3dgs_raster_settings(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier,
        inference_only,
        pixel_weights,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image, accum_weights = rasterizer(
        means3D=means3d,
        means2D=screenspace_points,
        dc=features_dc,
        shs=features_rest,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )

    if inference_only:
        return {
            "render": rendered_image,
        }

    render_pkg = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,
    }
    if accum_weights is not None and accum_weights.numel() > 0:
        render_pkg["accum_weights"] = accum_weights
    return render_pkg


def render(
    viewpoint_camera: RenderCamera,
    pc: GaussianModel3DGS,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    track_gradients: bool = True,
    inference_only: bool = False,
    pixel_weights: torch.Tensor | None = None,
    gaussian_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
        Common render entry point used by training and evaluation code.

        It currently forwards directly to `render_3dgs`, which keeps call sites
        stable if more render variants are added later.
    """
    return render_3dgs(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        track_gradients=track_gradients,
        inference_only=inference_only,
        pixel_weights=pixel_weights,
        gaussian_indices=gaussian_indices,
    )
