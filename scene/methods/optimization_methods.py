"""
Optimization stage for one training iteration.

This module turns one sampled camera into gradients. It renders the view, masks
invalid pixels if needed, computes image reconstruction loss, adds optional
regularization, and calls backward so later stages can update parameters and
Gaussian structure.
"""
from typing import Any

import torch
from fused_ssim import fused_ssim

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.loss_utils import l1_loss
from scene.methods.regularization_methods import apply_regularization_method
from scene.training_context import TrainingContext
from scene.training_runtime import TrainingLoopState


def _enable_debug_if_needed(context: TrainingContext, iteration: int) -> None:
    """
        Enable renderer debug mode one iteration before the requested debug step.

        The rasterizer uses the flag during the next render call, so the switch
        is flipped just before the target iteration is rendered.
    """
    debug_from = int(getattr(context.runtime_args, "debug_from", -1))
    if (iteration - 1) == debug_from:
        context.pipe.debug = True


def _select_training_background(context: TrainingContext, loop_state: TrainingLoopState) -> torch.Tensor:
    """
        Choose the background color for this render.

        Random background is a training augmentation. Otherwise the dataset's
        fixed white or black background is reused from loop state.
    """
    return torch.rand((3), device="cuda") if context.opt.random_background else loop_state["background"]


def _compute_reconstruction_loss(image: torch.Tensor, gt_image: torch.Tensor, lambda_dssim: float) -> torch.Tensor:
    """
        Combine pixel L1 loss and fused SSIM into the standard 3DGS image loss.
    """
    l1_value = l1_loss(image, gt_image)
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    return (1.0 - float(lambda_dssim)) * l1_value + float(lambda_dssim) * (1.0 - ssim_value)


def run_3dgs_optimization_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
    viewpoint_cam: Any,
) -> dict[str, Any]:
    """
        Render one camera and create gradients for this iteration.

        The returned `render_state` keeps only the render outputs needed by later
        stages, such as visibility, radii, and view-space gradient tensors.
    """
    pipe = context.pipe
    _enable_debug_if_needed(context, iteration)

    bg = _select_training_background(context, loop_state)
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, track_gradients=True)
    image = render_pkg["render"]

    # Match the loss to valid image pixels only when the dataset provides an alpha mask.
    alpha_mask = viewpoint_cam.alpha_mask_cuda()
    if alpha_mask is not None:
        image = image * alpha_mask

    gt_image = viewpoint_cam.original_image_cuda()
    loss = _compute_reconstruction_loss(image, gt_image, float(context.opt.lambda_dssim))

    loss = apply_regularization_method(context, gaussians, iteration, loss)
    loss.backward()
    return {
        "loss": loss,
        "render_state": {
            "bg": bg,
            "viewpoint_camera": viewpoint_cam,
            "viewspace_point_tensor": render_pkg["viewspace_points"],
            "visibility_filter": render_pkg["visibility_filter"],
            "radii": render_pkg["radii"],
        },
    }
