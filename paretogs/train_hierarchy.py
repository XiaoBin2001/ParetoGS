"""Train and package a ParetoGS hierarchy through descending GNS stages."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
from plyfile import PlyData, PlyElement

from utils.artifact_utils import load_torch_asset
from paretogs.adapters import (
    ADAPTER_TENSOR_KEYS,
    NESTED_ADAPTER_BUNDLE_FORMAT,
    quantize_adapter_payload,
    tensor_storage_bytes,
)
from paretogs.prefix import save_nested_order
from paretogs.checkpoints import (
    RAW_PARAMETER_INDICES, PARAMETER_NAMES,
    _normalize_model_params, _load_checkpoint, _decode_source_ids, _membership_indices,
    load_anchor_state, write_anchor_state,
)


def _resolve_ratio_budgets(full_count: int, ratios: list[float]) -> list[int]:
    """All ratios refer to the actual trained Full count, including 100%."""
    ratios = sorted(set(float(value) for value in ratios))
    if not ratios or ratios[0] <= 0 or ratios[-1] != 1.0:
        raise ValueError("Anchor ratios must be in (0, 1] and include 1.")
    budgets = [round(full_count * ratio) for ratio in ratios]
    if budgets[0] <= 0 or len(set(budgets)) != len(budgets):
        raise ValueError("Ratios produce zero or duplicate integer budgets.")
    return budgets


def _run(command: list[str], repo_root: str) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def _rebase_checkpoint(source_path: str, output_path: str, iteration: int) -> None:
    payload = load_torch_asset(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_params" not in payload:
        raise ValueError("Unsupported checkpoint: {}".format(source_path))
    payload = dict(payload)
    payload["model_params"] = _normalize_model_params(payload["model_params"], source_path)
    from scene.methods.pruning_checkpoint import reset_pruning_state_for_rebase

    reset_pruning_state_for_rebase(payload)
    payload["iteration"] = int(iteration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)


def _train_stage(
    repo_root: str,
    source_path: str,
    model_path: str,
    checkpoint_path: str,
    training_method: str,
    source_budget: int,
    budget: int,
    end_iteration: int,
    reg_prune_from_iter: int,
    reg_prune_until_iter: int,
    data_device: str = "cpu",
) -> None:
    command = [
        sys.executable,
        "train.py",
        "--source_path", source_path,
        "--model_path", model_path,
        "--data_device", data_device,
        "--iterations", str(end_iteration),
        "--position_lr_max_steps", "30000",
        "--save_iterations", str(end_iteration) if training_method == "improvedgs" else "0",
        "--checkpoint_iterations", str(end_iteration),
        "--report_lpips_test", "false",
        "--report_lpips_train", "false",
        "--record_gns_prune_trace", "true",
        "--empty_cache_interval", "0",
        "--training_method", training_method,
        "--use_las", "true",
        "--use_rap", "true",
        "--use_eas", "true",
        "--use_mu", "true",
        "--budget", str(source_budget),
        "--final_budget_mode", "num",
        "--final_budget", str(budget),
        "--reg_prune_from_iter", str(reg_prune_from_iter),
        "--reg_prune_until_iter", str(reg_prune_until_iter),
        "--start_checkpoint_dir", os.path.dirname(checkpoint_path),
        "--start_checkpoint_file", os.path.basename(checkpoint_path),
    ]
    _run(command, repo_root)


def _load_prune_trace_shell(
    path: str,
    lower_ids: torch.Tensor,
    upper_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one upper-to-lower GNS trace and return its low-to-high shell IDs."""
    trace = load_torch_asset(path, map_location="cpu")
    if not isinstance(trace, dict) or trace.get("format") != "paretogs_gns_prune_trace":
        raise ValueError("Unsupported GNS prune trace: {}".format(path))
    if int(trace.get("version", 0)) != 1:
        raise ValueError("Unsupported GNS prune trace version: {}".format(path))
    lower = int(lower_ids.shape[0])
    upper = int(upper_ids.shape[0])
    nested_ids = trace.get("nested_source_ids")
    if not torch.is_tensor(nested_ids):
        raise ValueError("GNS prune trace has no nested source IDs: {}".format(path))
    nested_ids = nested_ids.to(dtype=torch.long, device="cpu").flatten().contiguous()
    if (
        int(trace.get("final_count", 0)) != lower
        or int(trace.get("initial_count", 0)) != upper
        or int(nested_ids.shape[0]) != upper
    ):
        raise ValueError(
            "GNS prune trace counts do not match interval {}:{}: {}".format(
                lower, upper, path
            )
        )
    if not torch.equal(torch.sort(nested_ids[:lower]).values, torch.sort(lower_ids).values):
        raise ValueError("GNS prune trace survivors do not match its lower anchor.")
    if not torch.equal(torch.sort(nested_ids).values, torch.sort(upper_ids).values):
        raise ValueError("GNS prune trace initial set does not match its upper anchor.")
    metadata = {
        "strategy": "reverse_gns_prune_trace",
        "finished_iteration": int(trace.get("finished_iteration", 0)),
        "event_count": len(trace.get("events", [])),
    }
    return nested_ids[lower:], metadata


def _checkpoint_gaussian_count(checkpoint_path: str) -> int:
    _payload, model = _load_checkpoint(checkpoint_path)
    ids = _decode_source_ids(model)
    return int(ids.shape[0])


def _build_order_and_adapters(
    full_checkpoint: str,
    specialist_checkpoints: dict[int, str],
    budgets: list[int],
    precision: str,
    prune_trace_paths: dict[int, str] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_state = load_anchor_state(full_checkpoint)
    base_model = base_state["parameters"]
    base_ids = base_state["source_ids"]
    full_budget = int(base_ids.shape[0])
    if budgets[-1] != full_budget:
        raise ValueError("Largest budget {} does not match full checkpoint count {}.".format(
            budgets[-1], full_budget
        ))

    membership: dict[int, torch.Tensor] = {}
    specialist_models: dict[int, dict[str, Any]] = {}
    for budget in budgets[:-1]:
        state = load_anchor_state(specialist_checkpoints[budget])
        ids = state["source_ids"]
        if int(ids.shape[0]) != budget:
            raise ValueError("Specialist {} has {} Gaussians.".format(budget, int(ids.shape[0])))
        membership[budget] = _membership_indices(base_ids, ids)
        specialist_models[budget] = state

    prune_trace_paths = {
        int(budget): os.path.abspath(path)
        for budget, path in (prune_trace_paths or {}).items()
    }
    unexpected_trace_budgets = set(prune_trace_paths) - set(budgets[:-1])
    if unexpected_trace_budgets:
        raise ValueError(
            "Prune traces target unknown lower anchors: {}.".format(
                sorted(unexpected_trace_budgets)
            )
        )

    previous_budget = budgets[0]
    previous = membership[previous_budget]
    order_parts = [previous]
    shell_order_strategies: dict[str, dict[str, Any]] = {}
    for upper_budget in budgets[1:]:
        current = (
            membership[upper_budget]
            if upper_budget != full_budget else
            torch.arange(full_budget, dtype=torch.long)
        )
        current_mask = torch.zeros(full_budget, dtype=torch.bool)
        current_mask[current] = True
        if previous.numel() and not bool(current_mask[previous].all().item()):
            raise ValueError(
                "Top-down specialists are not nested at budget {}.".format(upper_budget)
            )
        previous_mask = torch.zeros(full_budget, dtype=torch.bool)
        previous_mask[previous] = True
        shell = current[~previous_mask[current]]
        trace_path = prune_trace_paths.get(previous_budget)
        if trace_path:
            lower_ids = base_ids[previous]
            upper_ids = base_ids[current]
            trace_shell_ids, trace_metadata = _load_prune_trace_shell(
                trace_path, lower_ids, upper_ids
            )
            traced_shell = _membership_indices(base_ids, trace_shell_ids)
            if not torch.equal(torch.sort(traced_shell).values, torch.sort(shell).values):
                raise ValueError(
                    "GNS prune trace shell does not match interval {}:{}.".format(
                        previous_budget, upper_budget
                    )
                )
            shell = traced_shell
            shell_order_strategies["{}:{}".format(previous_budget, upper_budget)] = (
                trace_metadata
            )
        order_parts.append(shell)
        previous = current
        previous_budget = upper_budget
    order = torch.cat(order_parts).contiguous()
    if int(torch.unique(order).numel()) != full_budget:
        raise ValueError("Constructed nested order is not a permutation.")

    adapters: dict[int, dict[str, Any]] = {}
    for budget in budgets[:-1]:
        state = specialist_models[budget]
        model = state["parameters"]
        specialist_ids = state["source_ids"]
        sorted_ids, sorted_rows = torch.sort(specialist_ids)
        prefix_ids = base_ids[order[:budget]]
        positions = torch.searchsorted(sorted_ids, prefix_ids)
        if not torch.equal(sorted_ids[positions], prefix_ids):
            raise ValueError("Nested prefix and specialist membership differ at {}.".format(budget))
        rows = sorted_rows[positions]
        payload: dict[str, Any] = {
            "mode": "opacity_scale3_dc_xyz_rotation_shrest",
            "budget": budget,
        }
        for parameter_index in RAW_PARAMETER_INDICES:
            name = PARAMETER_NAMES[parameter_index]
            payload[name] = (
                torch.index_select(model[name], 0, rows)
                - torch.index_select(base_model[name], 0, order[:budget])
            ).contiguous()
        for name in ADAPTER_TENSOR_KEYS:
            payload.setdefault(name, None)
        adapters[budget] = quantize_adapter_payload(payload, precision)

    bundle = {
        "format": NESTED_ADAPTER_BUNDLE_FORMAT,
        "version": 1,
        "method": "top-down nested GNS parameter deltas",
        "base_gaussian_count": full_budget,
        "budgets": budgets,
        "storage_precision": adapters[budgets[0]]["storage_precision"],
        "shell_order_strategies": shell_order_strategies,
        "adapters": adapters,
    }
    bundle["tensor_storage_bytes"] = tensor_storage_bytes(bundle)
    return order, bundle


def _package_full_model(
    full_model_path: str,
    output_model_path: str,
    full_iteration: int,
    order: torch.Tensor,
) -> None:
    os.makedirs(output_model_path, exist_ok=False)
    for name in ("training_parameters.json", "cameras.json", "input.ply"):
        source = os.path.join(full_model_path, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_model_path, name))
    source_ply = os.path.join(
        full_model_path,
        "point_cloud",
        "iteration_{}".format(full_iteration),
        "point_cloud.ply",
    )
    target_ply = os.path.join(
        output_model_path,
        "point_cloud",
        "iteration_{}".format(full_iteration),
        "point_cloud.ply",
    )
    if not os.path.isfile(source_ply):
        raise FileNotFoundError(source_ply)
    os.makedirs(os.path.dirname(target_ply), exist_ok=True)
    ply = PlyData.read(source_ply)
    vertex_data = ply["vertex"].data
    if len(vertex_data) != int(order.shape[0]):
        raise ValueError("PLY/checkpoint Gaussian counts do not match.")
    ordered_vertex = vertex_data[order.numpy()]
    ordered_element = PlyElement.describe(ordered_vertex, "vertex")
    PlyData(
        [ordered_element],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(target_ply)
    parameters_path = os.path.join(output_model_path, "training_parameters.json")
    with open(parameters_path, "r", encoding="utf-8") as input_file:
        parameters = json.load(input_file)
    parameters["model_path"] = output_model_path
    with open(parameters_path, "w", encoding="utf-8") as output_file:
        json.dump(parameters, output_file, indent=2)


def run_pipeline(
    source_path: str,
    output_root: str,
    budgets: list[int],
    full_checkpoint: str | None,
    full_model_path: str | None,
    source_budget: int,
    iteration: int,
    full_iteration: int,
    rebase_iteration: int,
    post_prune_iterations: int,
    precision: str,
    run_postprocess: bool,
    data_device: str = "cpu",
) -> dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_path = os.path.abspath(source_path)
    output_root = os.path.abspath(output_root)
    full_checkpoint = os.path.abspath(full_checkpoint) if full_checkpoint else None
    full_model_path = os.path.abspath(full_model_path) if full_model_path else None
    budgets = sorted(set(int(value) for value in budgets))
    nominal_budgets = list(budgets)
    source_budget = int(source_budget)
    iteration = int(iteration)
    full_iteration = int(full_iteration)
    post_prune_iterations = int(post_prune_iterations)
    reg_prune_from_iter = int(rebase_iteration) + 1
    reg_prune_until_iter = int(iteration) - post_prune_iterations
    if len(budgets) < 2 or budgets[0] <= 0:
        raise ValueError("At least two positive budgets are required.")
    if source_budget < budgets[-1]:
        raise ValueError("source_budget must be at least the largest target budget.")
    if post_prune_iterations < 0:
        raise ValueError("post_prune_iterations must be non-negative.")
    if full_iteration < iteration:
        raise ValueError("full_iteration must be at least the descendant iteration.")
    if reg_prune_until_iter < reg_prune_from_iter:
        raise ValueError("The pruning interval is empty; adjust iteration/rebase settings.")
    if not full_checkpoint or not full_model_path:
        raise ValueError("full_checkpoint and full_model_path are required.")
    if budgets[-1] != _checkpoint_gaussian_count(full_checkpoint):
        raise ValueError("Largest anchor must equal the actual Full checkpoint count.")
    if os.path.exists(output_root):
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output_root))
    os.makedirs(os.path.join(output_root, "checkpoints"), exist_ok=False)

    pipeline_started = time.perf_counter()
    stages = []
    specialist_checkpoints: dict[int, str] = {}
    specialist_states: dict[int, str] = {}
    owned_checkpoints: set[str] = set()
    retired_files: list[str] = []

    def save_progress(previous: str | None, rebased: str | None = None) -> None:
        """Commit replacement states before retiring this run's consumed files."""
        retiring = [rebased] if rebased is not None else []
        if previous in owned_checkpoints:
            retiring.append(previous)
            for stage in stages:
                if stage["checkpoint"] == previous:
                    stage["checkpoint"] = None
        retired_files.extend(retiring)
        progress = {
            "full_anchor_state": full_checkpoint,
            "anchor_states": {**specialist_checkpoints, **specialist_states},
            "latest_checkpoint": stages[-1]["checkpoint"],
            "stages": stages,
            "retired_files": retired_files,
        }
        path = os.path.join(output_root, "pipeline_progress.json")
        with open(path + ".tmp", "w", encoding="utf-8") as output_file:
            json.dump(progress, output_file, indent=2)
        os.replace(path + ".tmp", path)
        for path in retiring:
            os.remove(path)
            owned_checkpoints.discard(path)

    previous_checkpoint = full_checkpoint
    pending_budgets = budgets[:-1]
    for descending_index, budget in enumerate(reversed(pending_budgets), start=1):
        stage_start_checkpoint = previous_checkpoint
        rebased_checkpoint = os.path.join(
            output_root, "checkpoints", "rebased_for_{}.pth".format(budget)
        )
        _rebase_checkpoint(stage_start_checkpoint, rebased_checkpoint, rebase_iteration)
        model_path = os.path.join(
            output_root, "stage_{:02d}_{}".format(descending_index, budget)
        )
        started = time.perf_counter()
        _train_stage(
            repo_root,
            source_path,
            model_path,
            rebased_checkpoint,
            "gns",
            source_budget,
            budget,
            iteration,
            reg_prune_from_iter,
            reg_prune_until_iter,
            data_device,
        )
        previous_checkpoint = os.path.join(model_path, "chkpnt{}.pth".format(iteration))
        specialist_checkpoints[budget] = previous_checkpoint
        state_path = os.path.join(model_path, "anchor_state.pt")
        saved_count = write_anchor_state(previous_checkpoint, state_path)
        if saved_count != budget:
            raise ValueError("Stage {} saved {} Gaussians; retaining recovery checkpoints.".format(
                budget, saved_count
            ))
        specialist_states[budget] = state_path
        owned_checkpoints.add(previous_checkpoint)
        stages.append({
            "budget": budget,
            "iteration": iteration,
            "training_method": "gns",
            "model_path": model_path,
            "start_checkpoint": stage_start_checkpoint,
            "checkpoint": previous_checkpoint,
            "anchor_state": state_path,
            "wall_seconds": time.perf_counter() - started,
        })
        save_progress(stage_start_checkpoint, rebased_checkpoint)

    prune_trace_paths = {}
    for lower_budget, checkpoint in specialist_checkpoints.items():
        trace_path = os.path.join(os.path.dirname(checkpoint), "gns_prune_trace.pt")
        if os.path.isfile(trace_path):
            prune_trace_paths[lower_budget] = trace_path
    order, bundle = _build_order_and_adapters(
        full_checkpoint,
        {**specialist_checkpoints, **specialist_states},
        budgets,
        precision,
        prune_trace_paths,
    )
    bundle["source_budget"] = source_budget
    bundle["nominal_budgets"] = nominal_budgets
    bundle["anchor_ratios"] = [value / budgets[-1] for value in budgets]
    bundle["descendant_iteration"] = iteration
    bundle["full_iteration"] = full_iteration
    bundle["post_prune_iterations"] = post_prune_iterations
    checkpoint_order_path = os.path.join(output_root, "checkpoint_nested_order.pt")
    save_nested_order(
        checkpoint_order_path,
        order,
        strategy="top_down_gns_checkpoint_order",
    )
    model_path = os.path.join(output_root, "model")
    _package_full_model(full_model_path, model_path, full_iteration, order)
    save_nested_order(
        os.path.join(model_path, "nested_order.pt"),
        torch.arange(int(order.shape[0]), dtype=torch.long),
        strategy="top_down_gns_physically_ordered",
    )
    full_bundle_path = os.path.join(model_path, "nested_adapters_{}_full.pt".format(precision))
    torch.save(bundle, full_bundle_path)
    bundle_path = full_bundle_path
    selected_storage_bytes = int(bundle["tensor_storage_bytes"])
    save_progress(None)

    construction_wall_seconds = time.perf_counter() - pipeline_started
    evaluation_wall_seconds = 0.0
    if run_postprocess:
        evaluation_started = time.perf_counter()
        _run([
            sys.executable,
            "postprocess.py",
            "--model_path", model_path,
            "--iteration", str(full_iteration),
            "--include_train", "false",
            "--skip_test", "false",
            "--no_save_test_images", "true",
            "--fps_warmup_rounds", "0",
            "--fps_measure_rounds", "0",
            "--prefix_budgets", *[str(value) for value in budgets],
            "--prefix_adapter_path", bundle_path,
        ], repo_root)
        evaluation_wall_seconds = time.perf_counter() - evaluation_started

    result = {
        "method": "top-down nested GNS with quantized density-conditioned deltas",
        "source_path": source_path,
        "budgets": budgets,
        "nominal_budgets": nominal_budgets,
        "anchor_ratios": [value / budgets[-1] for value in budgets],
        "source_budget": source_budget,
        "descendant_iteration": iteration,
        "full_iteration": full_iteration,
        "rebase_iteration": int(rebase_iteration),
        "reg_prune_from_iter": reg_prune_from_iter,
        "reg_prune_until_iter": reg_prune_until_iter,
        "post_prune_iterations": post_prune_iterations,
        "full_checkpoint": full_checkpoint if os.path.isfile(full_checkpoint) else None,
        "full_anchor_state": full_checkpoint,
        "full_model_path": full_model_path,
        "stages": stages,
        "anchor_states": {**specialist_checkpoints, **specialist_states},
        "latest_checkpoint": previous_checkpoint,
        "retired_files": retired_files,
        "model_path": model_path,
        "checkpoint_nested_order": checkpoint_order_path,
        "adapter_bundle": bundle_path,
        "adapter_tensor_storage_bytes": selected_storage_bytes,
        "trajectory_ordered_shells": sorted(bundle["shell_order_strategies"]),
        "construction_wall_seconds": construction_wall_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "total_wall_seconds": time.perf_counter() - pipeline_started,
    }
    with open(os.path.join(output_root, "pipeline_summary.json"), "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_root", required=True)
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--budgets", nargs="+", type=int)
    budget_group.add_argument("--budget_ratios", nargs="+", type=float)
    parser.add_argument("--source_budget", type=int)
    parser.add_argument("--full_checkpoint", required=True)
    parser.add_argument("--full_model_path", required=True)
    parser.add_argument(
        "--iteration",
        type=int,
        default=25000,
        help=(
            "End iteration for each pruned descendant stage. The defaults "
            "implement 3K pruning + 7K fine-tuning from iteration 14999."
        ),
    )
    parser.add_argument(
        "--full_iteration",
        type=int,
        default=30000,
        help=(
            "Full-model training/PLY iteration (default: 30000); this is "
            "independent of the 3K+7K descendant schedule."
        ),
    )
    parser.add_argument("--rebase_iteration", type=int, default=14999)
    parser.add_argument(
        "--post_prune_iterations",
        type=int,
        default=7000,
        help="Fine-tuning iterations after GNS pruning (default: 7000).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp16",
    )
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--run_postprocess", action="store_true")
    args = parser.parse_args()
    full_count = _checkpoint_gaussian_count(args.full_checkpoint)
    resolved_budgets = (
        _resolve_ratio_budgets(full_count, args.budget_ratios)
        if args.budget_ratios is not None else args.budgets
    )
    resolved_source_budget = args.source_budget or full_count
    summary = run_pipeline(
        args.source_path,
        args.output_root,
        resolved_budgets,
        args.full_checkpoint,
        args.full_model_path,
        resolved_source_budget,
        args.iteration,
        args.full_iteration,
        args.rebase_iteration,
        args.post_prune_iterations,
        args.precision,
        args.run_postprocess,
        args.data_device,
    )
    print(json.dumps(summary, indent=2))
