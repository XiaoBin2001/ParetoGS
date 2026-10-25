"""
Small stages called from the main training loop.

These helpers keep the loop in `train.py` readable. They update progress output,
run scheduled evaluation and saving, adjust learning-rate/SH schedules, and
perform optimizer steps according to the active training method.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.loss_utils import l1_loss
from scene.training_context import TrainingContext
from scene.training_runtime import TrainingLoopState, save_training_checkpoint, synchronized_timestamp, training_report


def _resolve_dense_optimizer_type(opt: Any, model_label: str = "3DGS") -> str:
    """
        Validate and return the configured optimizer type for dense Gaussian tensors.
    """
    configured_optimizer_type = opt.optimizer_type
    if configured_optimizer_type not in {"default", "ours_adam"}:
        raise ValueError(
            "{} currently supports only optimizer_type in {{default, ours_adam}}; got {}.".format(
                str(model_label).upper(),
                configured_optimizer_type,
            )
        )
    return configured_optimizer_type


def update_training_progress_bar(
    progress_bar: Any,
    iteration: int,
    ema_loss_for_log: float,
    loss_value: float,
    num_gaussians: int,
) -> float:
    """
        Update the tqdm postfix with smoothed loss and current Gaussian count.
    """
    ema_loss_for_log = 0.4 * loss_value + 0.6 * ema_loss_for_log
    if iteration % 10 == 0:
        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}", "N_GS": f"{num_gaussians}"})
        progress_bar.update(10)
    return ema_loss_for_log


def run_log_and_save_stage(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loop_state: TrainingLoopState,
    iteration: int,
) -> float:
    """
        Run scheduled evaluation, model saving, and checkpoint saving.

        Return reporting time, excluded from training time.
        PLY and checkpoint writes remain part of training time.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene during log/save stage.")

    runtime_args = context.runtime_args
    dataset = context.dataset
    pipe = context.pipe
    report_time_seconds = 0.0
    if iteration in loop_state["testing_iterations"]:
        report_start = synchronized_timestamp()
        training_report(
            iteration,
            l1_loss,
            scene,
            render,
            (pipe, loop_state["background"], 1.0, False, False, None),
            loop_state["report_lpips_model"],
            bool(getattr(runtime_args, "report_lpips_test", False)),
            bool(getattr(runtime_args, "report_lpips_train", False)),
            loop_state["log_path"],
            iteration == min(loop_state["testing_iterations"]),
        )
        report_time_seconds += synchronized_timestamp() - report_start
    if iteration in loop_state["saving_iterations"]:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        save_start = synchronized_timestamp()
        scene.save(iteration)
        save_seconds = synchronized_timestamp() - save_start
        context.runtime_state["ply_save_seconds"] = (
            float(context.runtime_state.get("ply_save_seconds", 0.0)) + save_seconds
        )
    context.runtime_state["report_time_seconds"] = (
        float(context.runtime_state.get("report_time_seconds", 0.0)) + report_time_seconds
    )
    if iteration in loop_state["checkpoint_iterations"]:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        save_training_checkpoint(
            os.path.join(scene.model_path, "chkpnt{}.pth".format(iteration)),
            gaussians,
            iteration,
            context.runtime_state,
            context.opt,
            loop_state,
        )
    return report_time_seconds


def _validate_dense_optimizer_type(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    model_label: str = "3DGS",
) -> None:
    """
        Ensure the model optimizer type matches the current config.
    """
    configured_optimizer_type = _resolve_dense_optimizer_type(context.opt, model_label)
    if str(getattr(gaussians, "optimizer_type", "default")).lower() != configured_optimizer_type:
        raise ValueError(
            "GaussianModel{} optimizer_type={} does not match current config {}.".format(
                str(model_label).upper(),
                getattr(gaussians, "optimizer_type", "default"),
                configured_optimizer_type,
            )
        )


def update_3dgs_training_schedule(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> None:
    """
        Update learning rate and SH degree before rendering the current iteration.
    """
    _validate_dense_optimizer_type(context, gaussians, "3dgs")
    gaussians.update_learning_rate(int(iteration))
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()


def should_run_parameter_update(opt: Any, iteration: int, method: str, use_mu: bool) -> bool:
    """
        Decide whether this iteration should step trainable parameters.

        ImprovedGS/GNS can use MU, which skips dense optimizer steps later in
        training to reduce update frequency. Other methods step every iteration
        until the final one.
    """
    if iteration >= int(opt.iterations):
        return False
    if method not in ("improvedgs", "gns") or not use_mu:
        return True

    mu_start_iter = int(opt.mu_start_iter)
    if mu_start_iter < 0:
        mu_start_iter = 20_000
    mu_interval = max(int(opt.mu_interval), 1)
    if int(opt.mu_interval) <= 1:
        mu_interval = 5

    mu_second_start_iter = int(opt.mu_second_start_iter)
    mu_second_interval = max(int(opt.mu_second_interval), 1)
    if mu_second_start_iter <= mu_start_iter:
        mu_second_start_iter = mu_start_iter + 1

    return (
        iteration < mu_start_iter
        or (mu_start_iter <= iteration < mu_second_start_iter and iteration % mu_interval == 0)
        or (iteration >= mu_second_start_iter and iteration % mu_second_interval == 0)
    )


def _step_dense_optimizer(
    gaussians: GaussianModel3DGS,
    optimizer_type: str,
    step_size: int,
    iteration: int,
) -> None:
    """
        Step either the custom GaussianAdam optimizer or the default Adam optimizer.
    """
    if optimizer_type == "ours_adam":
        gaussians.optimizer.step(step_size, iteration)
    else:
        gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def run_3dgs_parameter_update_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    radii: torch.Tensor,
) -> None:
    """
        Apply optimizer updates for one iteration.

        Dense Gaussian tensors follow the configured method update schedule.
    """
    method = str(context.method_config["training_method"]).lower()
    use_mu = bool(context.method_config.get("use_mu", False))
    if not should_run_parameter_update(context.opt, iteration, method, use_mu):
        return

    optimizer_type = _resolve_dense_optimizer_type(context.opt, "3dgs")
    _step_dense_optimizer(gaussians, optimizer_type, radii.shape[0], iteration)
