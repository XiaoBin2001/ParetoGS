"""Resume GNS pruning at the baseline's save-before-structure checkpoint boundary."""
from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch


_STATE_FIELDS = (
    "record_gns_prune_trace", "gns_lr_scaled", "gns_finished", "reg_prune_finished",
    "reg_prune_target_budget", "reg_prune_budget_reference_count", "opacity_min",
    "gns_prune_trace_initial_count", "gns_prune_trace_deleted_ids",
    "gns_prune_trace_events", "gns_protected_source_ids",
)
_CONFIG_FIELDS = (
    "reg_prune_from_iter", "reg_prune_until_iter", "gns_opacity_reg",
    "gns_opacity_lr_scale", "final_budget", "final_budget_mode", "final_rate",
)


def _camera_keys(views: list[Any]) -> list[tuple[str, int]]:
    return [(str(view.image_name), int(view.uid)) for view in views]


def capture_pruning_state(gaussians: Any, state: dict, opt: Any, loop_state: dict | None) -> dict | None:
    if opt is None or str(getattr(opt, "training_method", "")).lower() != "gns":
        return None
    saved = {key: copy.deepcopy(state[key]) for key in _STATE_FIELDS if key in state}
    if torch.is_tensor(saved.get("gns_protected_source_ids")):
        saved["gns_protected_source_ids"] = saved["gns_protected_source_ids"].cpu()
    result = {
        "version": 1, "phase": "before_structure", "state": saved,
        "config": {key: getattr(opt, key) for key in _CONFIG_FIELDS if hasattr(opt, key)},
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(gaussians.get_xyz.device) if gaussians.get_xyz.is_cuda else None,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
    }
    if loop_state is not None:
        result["training_viewpoints"] = _camera_keys(loop_state["training_viewpoints"])
        result["viewpoint_stack"] = _camera_keys(loop_state["viewpoint_stack"])
    return result


def restore_pruning_state(gaussians: Any, state: dict, opt: Any, loop_state: dict, iteration: int) -> None:
    payload = state.pop("gns_resume_state", None)
    if payload is None:
        if (
            str(getattr(opt, "training_method", "")).lower() == "gns"
            and state.get("record_gns_prune_trace")
            and int(opt.reg_prune_from_iter) <= iteration <= int(opt.reg_prune_until_iter)
        ):
            raise ValueError("This legacy checkpoint lacks GNS trace state; resume before the pruning window.")
        return
    if payload.get("version") != 1 or payload.get("phase") != "before_structure":
        raise ValueError("Unsupported GNS checkpoint boundary.")
    if str(getattr(opt, "training_method", "")).lower() != "gns":
        raise ValueError("A GNS checkpoint must be rebased before changing training method.")
    config = payload["config"]
    for key in ("reg_prune_from_iter", "final_budget", "final_budget_mode", "final_rate"):
        if key in config and getattr(opt, key) != config[key]:
            raise ValueError("GNS resume changed {}; start a rebased stage instead.".format(key))
    state.update(payload["state"])
    protected = state.get("gns_protected_source_ids")
    if torch.is_tensor(protected):
        state["gns_protected_source_ids"] = protected.to(gaussians.get_xyz.device)
    for key, value in config.items():
        setattr(opt, key, value)
    if "training_viewpoints" in payload:
        current = {key: view for key, view in zip(
            _camera_keys(loop_state["training_viewpoints"]), loop_state["training_viewpoints"]
        )}
        saved_keys = [tuple(key) for key in payload["training_viewpoints"]]
        if set(current) != set(saved_keys):
            raise ValueError("GNS checkpoint training views do not match this run.")
        loop_state["training_viewpoints"] = [current[key] for key in saved_keys]
        loop_state["viewpoint_stack"] = [current[tuple(key)] for key in payload["viewpoint_stack"]]
    torch.set_rng_state(payload["rng_cpu"].cpu())
    if payload["rng_cuda"] is not None and gaussians.get_xyz.is_cuda:
        torch.cuda.set_rng_state(payload["rng_cuda"].cpu(), gaussians.get_xyz.device)
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    # A saved iteration has completed its optimizer step but not yet its pruning.
    from scene.methods.pruning_methods import apply_gns_pruning

    apply_gns_pruning(iteration, gaussians, opt, state)


def reset_pruning_state_for_rebase(payload: dict) -> None:
    saved = payload.pop("gns_state", None)
    if saved is None or not saved["state"].get("gns_lr_scaled"):
        return
    scale = float(saved["config"]["gns_opacity_lr_scale"])
    optimizer = payload["model_params"][12]
    for group in optimizer["param_groups"]:
        if group.get("name") == "opacity":
            group["lr"] /= scale
