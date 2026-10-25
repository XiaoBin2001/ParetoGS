"""
Shared training context.

The training loop is split into optimization, logging, densification, pruning,
and scheduling modules. This file defines the context object they all receive,
plus the runtime-state cache used for method-specific temporary data.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import torch
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from scene.methods.densification_methods import prepare_edge_maps
from scene.methods.training_config import build_training_method_config


class RuntimeState(TypedDict, total=False):
    """
        Optional scratch fields shared across training stages.

        Values here are created once and updated over many iterations, such as
        edge maps for ImprovedGS and pruning state for RAP or GNS.
    """
    edge_maps: list[torch.Tensor]
    edge_camera_pool: list[Any]
    edge_map_pool: list[torch.Tensor]
    opacity_min: Optional[float]
    rap_reset_count: int
    rap_trigger_iterations: set[int]
    gns_lr_scaled: bool
    gns_finished: bool
    reg_prune_finished: bool
    reg_prune_target_budget: Optional[int]
    reg_prune_budget_reference_count: Optional[int]
    record_gns_prune_trace: bool
    gns_prune_trace_initial_count: int
    gns_prune_trace_deleted_ids: list[torch.Tensor]
    gns_prune_trace_events: list[dict[str, int]]
    gns_protected_source_ids: torch.Tensor
    gns_resume_state: dict[str, Any] | None
    scene: Optional[Scene]
    pipe: Any
    train_cameras: list[Any]


@dataclass(slots=True)
class TrainingContext:
    """
        Bundle long-lived objects needed by training stages.

        Passing this object keeps stage function signatures short while still
        making dataset, optimizer options, renderer options, scene, and runtime
        state explicit.
    """
    dataset: Any
    opt: Any
    pipe: Any
    runtime_args: Any
    scene: Optional[Scene] = None
    method_config: dict[str, Any] = field(default_factory=dict)
    runtime_state: RuntimeState = field(default_factory=dict)


def initialize_runtime_state(
    scene: Scene,
    gaussians: GaussianModel3DGS,
    opt: Any,
    method_config: dict[str, Any],
) -> RuntimeState:
    """
        Create method-specific runtime state after the scene and model exist.

        ImprovedGS/GNS precompute edge maps for edge-aware scoring. Shared fields
        are initialized here for the later training stages.
    """
    method = str(method_config.get("training_method", "3dgs")).lower()
    runtime_state: RuntimeState = {
        "edge_maps": [],
        "edge_camera_pool": [],
        "edge_map_pool": [],
        "opacity_min": None,
        "rap_reset_count": 0,
        "rap_trigger_iterations": set(),
        "gns_lr_scaled": False,
        "gns_finished": False,
        "reg_prune_finished": False,
        "reg_prune_target_budget": None,
        "reg_prune_budget_reference_count": None,
        "scene": None,
        "pipe": None,
        "train_cameras": scene.getTrainCameras().copy(),
    }
    if method in ("improvedgs", "gns"):
        train_cameras = runtime_state["train_cameras"]
        edge_maps = prepare_edge_maps(train_cameras, opt) if bool(method_config.get("use_eas", True)) else []
        runtime_state["edge_maps"] = edge_maps
        runtime_state["edge_camera_pool"] = train_cameras.copy()
        runtime_state["edge_map_pool"] = edge_maps.copy()
    return runtime_state


def build_training_context(dataset: Any, opt: Any, pipe: Any, runtime_args: Any) -> TrainingContext:
    """
        Build the context skeleton before scene creation.

        Method settings are parsed early and written back to `opt`, so later
        modules can read both normal options and resolved method switches from
        the same object.
    """
    method_config = build_training_method_config(opt)

    # Store resolved method settings on opt so later stage calls do not repeat parsing.
    for key, value in method_config.items():
        setattr(opt, key, value)
    return TrainingContext(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        runtime_args=runtime_args,
        method_config=method_config,
        runtime_state={},
    )


def attach_scene_and_gaussians_to_context(
    context: TrainingContext,
    scene: Scene,
    gaussians: GaussianModel3DGS,
) -> TrainingContext:
    """
        Attach scene-dependent objects and initialize the runtime-state cache.
    """
    context.scene = scene
    context.runtime_state = initialize_runtime_state(scene, gaussians, context.opt, context.method_config)
    return context
