"""
Pruning stage for methods that remove low-value Gaussians.

RAP performs opacity reset first and prunes after a delay. GNS uses a
regularized pruning window: it raises opacity learning rate, applies threshold
pruning during the window, then enforces the final budget at the end.
"""
import os
from typing import Any

import torch
from tqdm import tqdm

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.experiment_utils import append_log_lines
from scene.training_context import TrainingContext


def _gns_protected_mask(
    gaussians: GaussianModel3DGS,
    state: dict[str, Any] | None,
) -> torch.Tensor:
    protected_ids = None if state is None else state.get("gns_protected_source_ids")
    if protected_ids is None:
        return torch.zeros_like(gaussians._source_ids, dtype=torch.bool)
    return torch.isin(
        gaussians._source_ids,
        protected_ids.to(device=gaussians._source_ids.device),
    )


def _record_gns_prune_event(
    gaussians: GaussianModel3DGS,
    prune_mask: torch.Tensor,
    state: dict[str, Any] | None,
    iteration: int | None,
) -> None:
    """Record GNS deletion order without changing the pruning decision."""
    if state is None or not bool(state.get("record_gns_prune_trace", False)):
        return
    source_ids = gaussians._source_ids
    if source_ids.ndim != 1 or int(source_ids.shape[0]) != int(prune_mask.shape[0]):
        raise ValueError("GNS prune tracing requires one source ID per Gaussian.")
    removed_indices = torch.nonzero(prune_mask, as_tuple=False).flatten()
    if removed_indices.numel() == 0:
        return
    removed_opacity = torch.index_select(
        gaussians.get_opacity.detach().flatten(), 0, removed_indices
    )
    least_to_most = torch.argsort(removed_opacity, descending=False, stable=True)
    removed_ids = torch.index_select(source_ids, 0, removed_indices)[least_to_most]
    state.setdefault("gns_prune_trace_deleted_ids", []).append(
        removed_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
    )
    state.setdefault("gns_prune_trace_events", []).append({
        "iteration": int(iteration or 0),
        "removed_count": int(removed_ids.shape[0]),
        "remaining_count": int(source_ids.shape[0] - removed_ids.shape[0]),
    })


def _save_gns_prune_trace(
    gaussians: GaussianModel3DGS,
    state: dict[str, Any],
    iteration: int,
) -> None:
    if not bool(state.get("record_gns_prune_trace", False)):
        return
    deleted_parts = state.get("gns_prune_trace_deleted_ids", [])
    deleted_ids = (
        torch.cat(deleted_parts).to(dtype=torch.long).contiguous()
        if deleted_parts else torch.empty(0, dtype=torch.long)
    )
    survivor_ids = gaussians._source_ids.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()
    nested_source_ids = torch.cat((survivor_ids, deleted_ids.flip(0))).contiguous()
    initial_count = int(state.get("gns_prune_trace_initial_count", nested_source_ids.shape[0]))
    if int(nested_source_ids.shape[0]) != initial_count:
        raise ValueError("GNS prune trace does not cover the initial Gaussian set.")
    if int(torch.unique(nested_source_ids).numel()) != initial_count:
        raise ValueError("GNS prune trace contains duplicate source IDs.")
    scene = state.get("scene")
    if scene is None:
        raise ValueError("GNS prune trace cannot be saved without a scene output path.")
    torch.save({
        "format": "paretogs_gns_prune_trace",
        "version": 1,
        "initial_count": initial_count,
        "final_count": int(survivor_ids.shape[0]),
        "finished_iteration": int(iteration),
        "nested_source_ids": nested_source_ids,
        "events": state.get("gns_prune_trace_events", []),
    }, os.path.join(scene.model_path, "gns_prune_trace.pt"))


def _should_run_rap_opacity_reset(dataset: Any, opt: Any, iteration: int) -> bool:
    """
        Check whether RAP should reset opacity on this iteration.

        RAP resets only inside the densification window and only on the normal
        opacity-reset interval.
    """
    densify_from_iter = int(opt.densify_from_iter)
    densify_until_iter = int(opt.densify_until_iter)
    if not (densify_from_iter < iteration < densify_until_iter):
        return False
    del dataset
    return iteration % int(opt.opacity_reset_interval) == 0


def _schedule_next_rap_prune_if_needed(state: dict[str, Any], opt: Any, iteration: int) -> None:
    """
        After an opacity reset, schedule RAP's delayed pruning step.

        The delayed step removes a configured low-opacity percentage after the
        model has had a short time to recover from the reset.
    """
    if float(opt.rap_prune_ratio) > 0 and int(state["rap_reset_count"]) < int(opt.rap_rounds):
        state["rap_trigger_iterations"].add(int(iteration) + int(opt.rap_prune_offset))
    state["rap_reset_count"] += 1


def resolve_regularized_prune_target_budget(
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
) -> int | None:
    """
        Resolve and cache the final Gaussian count used by GNS pruning.

        `num` mode uses a fixed count. `rate` mode converts a keep percentage
        into a rounded count based on the model size at the start of pruning.
    """
    cached_budget = state.get("reg_prune_target_budget")
    if cached_budget is not None:
        return int(cached_budget)

    budget_mode = str(opt.final_budget_mode).lower()
    if budget_mode == "off":
        return None
    if budget_mode == "num":
        target_budget = int(opt.final_budget)
    elif budget_mode == "rate":
        reference_count = state.get("reg_prune_budget_reference_count")
        if reference_count is None:
            reference_count = int(gaussians.get_opacity.shape[0])
            state["reg_prune_budget_reference_count"] = reference_count
        final_rate = min(max(float(opt.final_rate), 0.0), 1.0)
        target_budget = int(round((int(reference_count) * final_rate) / 10000.0) * 10000.0)
        target_budget = max(target_budget, 1)
    else:
        raise ValueError("Unsupported final_budget_mode: {}".format(opt.final_budget_mode))

    target_budget = min(max(int(target_budget), 1), int(gaussians.get_opacity.shape[0]))
    state["reg_prune_target_budget"] = target_budget
    return target_budget


def apply_gns_threshold_prune(
    gaussians: GaussianModel3DGS,
    min_opacity: float,
    final_budget: int | None = None,
    state: dict[str, Any] | None = None,
    iteration: int | None = None,
) -> None:
    """
        Remove Gaussians below the opacity threshold without crossing the budget.

        If threshold pruning would remove too many Gaussians, the function falls
        back to exact budget pruning.
    """
    opacity = gaussians.get_opacity
    if opacity.numel() == 0:
        return
    opacity_flat = opacity.detach().flatten()
    prune_mask = (opacity_flat < float(min_opacity)).flatten()
    prune_mask &= ~_gns_protected_mask(gaussians, state)
    if final_budget is not None:
        current_count = int(opacity_flat.shape[0])
        target_budget = min(max(int(final_budget), 1), current_count)
        if current_count - int(prune_mask.sum().item()) < target_budget:
            apply_gns_final_prune(gaussians, target_budget, state, iteration)
            return
    if prune_mask.any():
        _record_gns_prune_event(gaussians, prune_mask, state, iteration)
        gaussians.prune_points(prune_mask)


def apply_gns_final_prune(
    gaussians: GaussianModel3DGS,
    final_budget: int,
    state: dict[str, Any] | None = None,
    iteration: int | None = None,
) -> None:
    """
        Keep the top-opacity Gaussians until the exact final budget is reached.
    """
    importance = gaussians.get_opacity.detach().flatten()
    if importance.numel() <= int(final_budget):
        return
    protected_mask = _gns_protected_mask(gaussians, state)
    protected_count = int(protected_mask.sum().item())
    if protected_count > int(final_budget):
        raise ValueError(
            "GNS protects {} Gaussians, exceeding final budget {}.".format(
                protected_count, int(final_budget)
            )
        )
    prune_mask = ~protected_mask
    remaining_slots = int(final_budget) - protected_count
    if remaining_slots > 0:
        candidate_indices = torch.nonzero(~protected_mask, as_tuple=False).flatten()
        randomized_importance = torch.index_select(
            importance, 0, candidate_indices
        ) + torch.rand_like(candidate_indices, dtype=importance.dtype) * 1e-6
        keep_candidates = torch.topk(
            randomized_importance,
            min(remaining_slots, int(candidate_indices.numel())),
            largest=True,
            sorted=False,
        ).indices
        prune_mask[candidate_indices[keep_candidates]] = False
    _record_gns_prune_event(gaussians, prune_mask, state, iteration)
    gaussians.prune_points(prune_mask)


def finish_gns_pruning(
    iteration: int,
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
) -> None:
    """
        Mark GNS pruning as complete and restore the opacity learning rate.

        The completion message is written both to tqdm and to the experiment log
        so batch runs can inspect when pruning stopped.
    """
    if state.get("gns_lr_scaled", False):
        gaussians.update_opacity_lr(1.0 / float(opt.gns_opacity_lr_scale))
        state["gns_lr_scaled"] = False
    state["gns_finished"] = True
    state["reg_prune_finished"] = True
    _save_gns_prune_trace(gaussians, state, iteration)
    if iteration < int(opt.reg_prune_until_iter):
        setattr(opt, "reg_prune_until_iter", int(iteration))
    message = "[GNS] Regularized pruning finished at iter {}".format(iteration)
    tqdm.write("\n" + message)
    scene = state.get("scene")
    if scene is not None:
        append_log_lines(os.path.join(scene.model_path, "log.txt"), [message, ""])


def apply_gns_pruning(
    iteration: int,
    gaussians: GaussianModel3DGS,
    opt: Any,
    state: dict[str, Any],
) -> None:
    """
        Run GNS pruning logic for the current iteration.

        The first pruning iteration scales the opacity learning rate. During the
        pruning window it performs periodic threshold pruning, and on the final
        iteration it enforces the exact target budget.
    """
    if bool(state.get("gns_finished", False)):
        return

    reg_start = int(opt.reg_prune_from_iter)
    reg_end = int(opt.reg_prune_until_iter)
    if int(iteration) == reg_start and not state["gns_lr_scaled"]:
        resolve_regularized_prune_target_budget(gaussians, opt, state)
        if bool(state.get("record_gns_prune_trace", False)):
            state["gns_prune_trace_initial_count"] = int(gaussians.get_xyz.shape[0])
            state["gns_prune_trace_deleted_ids"] = []
            state["gns_prune_trace_events"] = []
        gaussians.update_opacity_lr(float(opt.gns_opacity_lr_scale))
        state["gns_lr_scaled"] = True

    if reg_start <= int(iteration) < reg_end:
        if int(iteration) % 10 == 0:
            final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
            apply_gns_threshold_prune(
                gaussians,
                float(opt.min_opacity),
                final_budget=final_budget,
                state=state,
                iteration=iteration,
            )
            if final_budget is not None and int(gaussians.get_xyz.shape[0]) <= int(final_budget):
                finish_gns_pruning(iteration, gaussians, opt, state)
        return

    if int(iteration) == reg_end:
        final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
        if final_budget is not None:
            apply_gns_final_prune(gaussians, final_budget, state, iteration)
        finish_gns_pruning(iteration, gaussians, opt, state)


def run_3dgs_pruning_method(context: TrainingContext, gaussians: GaussianModel3DGS, iteration: int) -> None:
    """
        Dispatch RAP and GNS pruning actions for one training iteration.
    """
    opt = context.opt
    state = context.runtime_state
    method = str(context.method_config["training_method"]).lower()
    use_rap = bool(context.method_config.get("use_rap", False))

    if int(iteration) == 300 and method in ("improvedgs", "gns") and use_rap:
        gaussians.only_prune(0.02)

    if use_rap:
        rap_trigger_iterations = state.get("rap_trigger_iterations", set())
        if int(iteration) in rap_trigger_iterations:
            gaussians.only_prune(float(opt.rap_prune_ratio), percent=True)
            rap_trigger_iterations.remove(int(iteration))
        if _should_run_rap_opacity_reset(context.dataset, opt, iteration):
            gaussians.reset_opacity(float(opt.improvedgs_reset_min_opacity))
            _schedule_next_rap_prune_if_needed(state, opt, iteration)

    if method == "gns":
        apply_gns_pruning(iteration, gaussians, opt, state)
