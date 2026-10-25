"""
Runtime helpers used by the training script.

This module contains code that is not tied to one training method: output folder
setup, checkpoint I/O, argument parsing, shared loop state, camera sampling, and
periodic evaluation reports.
"""
import os
import json
from pathlib import Path
import random
import time
import uuid
from argparse import ArgumentParser, Namespace
from typing import Any, Optional, TypedDict

import torch
import numpy as np
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from utils.experiment_utils import (
    append_log_lines, archive_existing_output, write_training_parameters,
    resolve_start_checkpoint_path, validate_gns_start,
)
from utils.loss_utils import psnr
from utils.loss_utils import ssim
from scene.training_context import TrainingContext


class TrainingLoopState(TypedDict):
    """
        State reused by many training-loop stages.

        Keeping these values in one typed dictionary makes call sites shorter
        while still showing which runtime objects are expected.
    """
    background: torch.Tensor
    log_path: str
    report_lpips_model: Optional[LPIPS]
    training_viewpoints: list[Any]
    viewpoint_stack: list[Any]
    testing_iterations: list[int]
    saving_iterations: list[int]
    checkpoint_iterations: list[int]


def synchronized_timestamp() -> float:
    """
        Return a timer value after pending CUDA work has finished.

        CUDA kernels are asynchronous, so synchronization is needed before timing
        sections that include GPU work.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _ensure_model_path(args: Any) -> None:
    """
        Fill `args.model_path` with a unique output folder when it is empty.
    """
    if args.model_path:
        return
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/data/experiments/", unique_str[0:10], "training")


def _build_training_background(dataset: Any) -> torch.Tensor:
    """
        Build the fixed white or black CUDA background tensor for training renders.
    """
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def _build_lpips_model_preserving_rng() -> LPIPS:
    """
        Create LPIPS without advancing the training RNG streams.
    """
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model = LPIPS("vgg").to("cuda").eval()
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.random.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return model


def prepare_output_and_logger(args: Any, runtime_args: Optional[Any] = None) -> None:
    """
        Prepare the experiment folder before training starts.

        Existing output is archived unless this run resumes in place. The current
        arguments are then written so the run can be reproduced later.
    """
    _ensure_model_path(args)

    should_archive_existing = True
    if runtime_args is not None:
        should_archive_existing = not is_inplace_checkpoint_resume(runtime_args)
    if should_archive_existing:
        archived_path = archive_existing_output(args.model_path)
        if archived_path is not None:
            print("Archived existing output to {}".format(archived_path))

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    if should_archive_existing:
        write_training_parameters(args.model_path, runtime_args if runtime_args is not None else args)
    else:
        print("[checkpoint] Keeping the original training_parameters.json.")


def build_training_loop_state(
    context: TrainingContext,
    report_lpips_test: bool,
    report_lpips_train: bool,
) -> TrainingLoopState:
    """
        Build shared loop state after scene creation.

        This attaches scene-related objects into `context.runtime_state`, creates
        the background tensor, prepares evaluation settings, and initializes the
        shuffled training-camera pool.
    """
    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene before building loop state.")

    dataset = context.dataset
    opt = context.opt
    runtime_args = context.runtime_args
    background = _build_training_background(dataset)
    runtime_state = context.runtime_state
    runtime_state["scene"] = scene
    runtime_state["pipe"] = context.pipe
    needs_lpips_model = (
        (report_lpips_test and bool(scene.getTestCameras()))
        or (report_lpips_train and bool(scene.getTrainCameras()))
    )

    training_viewpoints = scene.getTrainCameras().copy()
    holdout_stride = int(
        getattr(runtime_args, "training_view_holdout_stride", 0)
    )
    holdout_offset = int(
        getattr(runtime_args, "training_view_holdout_offset", 0)
    )
    if holdout_stride > 1:
        if not 0 <= holdout_offset < holdout_stride:
            raise ValueError(
                "training_view_holdout_offset must lie in [0, stride)."
            )
        training_viewpoints = [
            view
            for index, view in enumerate(training_viewpoints)
            if index % holdout_stride != holdout_offset
        ]
        if not training_viewpoints:
            raise ValueError("Training-view holdout removed every training camera.")

    return {
        "background": background,
        "log_path": os.path.join(scene.model_path, "log.txt"),
        "report_lpips_model": _build_lpips_model_preserving_rng() if needs_lpips_model else None,
        "training_viewpoints": training_viewpoints,
        "viewpoint_stack": training_viewpoints.copy(),
        "testing_iterations": list(runtime_args.test_iterations),
        "saving_iterations": list(runtime_args.save_iterations),
        "checkpoint_iterations": list(runtime_args.checkpoint_iterations),
    }


def sample_training_viewpoint(
    scene: Scene,
    viewpoint_stack: list[Any],
    training_viewpoints: list[Any] | None = None,
) -> Any:
    """
        Draw one random training camera without replacement.

        When the local pool becomes empty, it is refilled from the scene so each
    pass sees all training cameras in a random order.
    """
    if not viewpoint_stack:
        viewpoint_stack.extend(
            (training_viewpoints or scene.getTrainCameras()).copy()
        )
    random_index = torch.randint(0, len(viewpoint_stack), (1,)).item()
    return viewpoint_stack.pop(random_index)


def parse_training_arguments(argv: list[str]) -> tuple[Any, Any, Any, Namespace]:
    """
        Build the training CLI parser, register all option groups, and parse argv.
    """
    from arguments import ModelParams, OptimizationParams, PipelineParams, parse_bool_arg

    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000 * (i + 1) for i in range(1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[],
                        help="Model export iterations; 0 disables all PLY saves, including the final export.")
    parser.add_argument("--quiet", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--progress_bar_width", type=int, default=90)
    parser.add_argument("--report_lpips_test", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--report_lpips_train", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--resume_training", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--start_checkpoint_dir", type=str, default="")
    parser.add_argument("--start_checkpoint_file", type=str, default="")
    parser.add_argument("--record_gns_prune_trace", default=False, nargs="?", const=True, type=parse_bool_arg)
    parser.add_argument("--empty_cache_interval", type=int, default=200)
    parser.add_argument("--training_view_holdout_stride", type=int, default=0)
    parser.add_argument("--training_view_holdout_offset", type=int, default=0)
    args = parser.parse_args(argv)
    if is_inplace_checkpoint_resume(args):
        path = Path(args.model_path) / "training_parameters.json"
        if not path.is_file():
            raise FileNotFoundError("Saved training parameters missing: {}".format(path))
        saved = json.loads(path.read_text())
        if os.path.realpath(saved["model_path"]) != os.path.realpath(args.model_path):
            raise ValueError("Saved model_path does not match resume output")
        controls = {key: getattr(args, key) for key in ("model_path", "resume_training", "start_checkpoint_dir", "start_checkpoint_file")}
        vars(args).update({key: value for key, value in saved.items() if hasattr(args, key)})
        vars(args).update(controls)
        print("[checkpoint] Using original training parameters from {}.".format(path))
    validate_gns_start(args.model_path, args)
    return model_params, optimization_params, pipeline_params, args


def save_training_checkpoint(
    checkpoint_path: str,
    gaussians: GaussianModel3DGS,
    iteration: int,
    runtime_state: dict[str, Any] | None = None,
    opt: Any = None,
    loop_state: dict[str, Any] | None = None,
) -> None:
    """
        Save Gaussian model state and iteration in the current checkpoint format.
    """
    payload = {
        "model_params": gaussians.capture(),
        "iteration": int(iteration),
    }
    if runtime_state is not None:
        from scene.methods.pruning_checkpoint import capture_pruning_state

        pruning_state = capture_pruning_state(gaussians, runtime_state, opt, loop_state)
        if pruning_state is not None:
            payload["gns_state"] = pruning_state
    temporary = checkpoint_path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    # Timing is written after the model so this node includes checkpoint I/O.
    if runtime_state is not None and "training_clock_start" in runtime_state:
        from utils.artifact_utils import file_sha256
        from utils.experiment_utils import atomic_json
        timing = {
            "iteration": int(iteration), "checkpoint_sha256": file_sha256(checkpoint_path),
            "elapsed_seconds": (float(runtime_state.get("elapsed_prefix_seconds", 0.0))
                                + synchronized_timestamp() - runtime_state["training_clock_start"]
                                - float(runtime_state.get("report_time_seconds", 0.0))),
            "ply_save_seconds": float(runtime_state.get("ply_save_seconds", 0.0)),
            "definition": "loop_excluding_reporting_including_ply_and_checkpoint",
        }
        atomic_json(Path(checkpoint_path).with_suffix(".timing.json"), timing)


def load_training_checkpoint(checkpoint_path: str, runtime_state: dict | None = None) -> tuple[Any, int]:
    """
        Load checkpoint payload and validate that it matches the supported model type.

        The loader accepts both the current dictionary format and the older
        two-item tuple format, then returns model parameters plus start iteration.
    """
    # Trusted training checkpoints include NumPy and RNG state beyond tensors.
    payload = torch.load(checkpoint_path, weights_only=False)
    model_params: Any
    iteration: int

    if isinstance(payload, dict):
        if "model_params" not in payload:
            raise ValueError("This file is not a resumable training checkpoint: {}".format(checkpoint_path))
        if runtime_state is not None:
            runtime_state["gns_resume_state"] = payload.get("gns_state")
        model_params = payload.get("model_params")
        iteration = int(payload.get("iteration", 0))
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        model_params = payload[0]
        iteration = int(payload[1])
    else:
        raise ValueError("Unsupported checkpoint payload format: {}".format(type(payload).__name__))

    checkpoint_model_type = ""
    if isinstance(model_params, tuple):
        scaling_index: Optional[int] = None
        if len(model_params) >= 15:
            scaling_index = 5
        elif len(model_params) in {12, 13, 14}:
            scaling_index = 4
        if scaling_index is not None and scaling_index < len(model_params):
            scaling_tensor = model_params[scaling_index]
            if isinstance(scaling_tensor, torch.Tensor) and scaling_tensor.ndim >= 2:
                scaling_dim = int(scaling_tensor.shape[-1])
                if scaling_dim == 2:
                    checkpoint_model_type = "2dgs"
                elif scaling_dim == 3:
                    checkpoint_model_type = "3dgs"

    if checkpoint_model_type != "" and checkpoint_model_type != "3dgs":
        print(
            "Checkpoint type mismatch. The open-source version only supports `3dgs`, but checkpoint `{}` belongs to `{}`. Training stopped.".format(
                checkpoint_path,
                checkpoint_model_type,
            )
        )
        raise SystemExit(1)

    return model_params, iteration


def is_inplace_checkpoint_resume(runtime_args: Any) -> bool:
    """
        Check whether resume should reuse the current output folder.

        An empty checkpoint directory means the checkpoint is expected inside
        `model_path`, so archiving that folder would remove the resume source.
    """
    model_path = str(getattr(runtime_args, "model_path", "") or "")
    if not model_path:
        return False
    checkpoint = resolve_start_checkpoint_path(model_path, runtime_args)
    return checkpoint is not None and os.path.commonpath(
        [os.path.realpath(model_path), checkpoint]) == os.path.realpath(model_path)


def load_checkpoint_timing(checkpoint_path: str) -> dict:
    from utils.artifact_utils import file_sha256
    path = Path(checkpoint_path).with_suffix(".timing.json")
    if not path.is_file():
        print("[checkpoint] No timing prefix; accumulated time starts at this resume.")
        return {}
    value = json.loads(path.read_text())
    if value["checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError("Checkpoint timing does not match model bytes: {}".format(path))
    return value


def _evaluate_report_view(
    viewpoint: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    l1_loss_fn: Any,
    lpips_model: Optional[LPIPS],
    report_lpips: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
        Render one validation view and compute all requested image metrics.
    """
    image = render_func(viewpoint, scene.gaussians, *render_args)["render"]
    image = torch.clamp(image, 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image_cuda(), 0.0, 1.0)
    image_batch = image.unsqueeze(0)
    gt_image_batch = gt_image.unsqueeze(0)
    l1_value = l1_loss_fn(image, gt_image).mean().double()
    psnr_value = psnr(image, gt_image).mean().double()
    ssim_value = ssim(image_batch, gt_image_batch).mean().double()
    lpips_value = None
    if lpips_model is not None and report_lpips:
        lpips_value = lpips_model(image_batch, gt_image_batch).mean().double()
    return l1_value, psnr_value, ssim_value, lpips_value


def training_report(
    iteration: int,
    l1_loss_fn: Any,
    scene: Scene,
    render_func: Any,
    render_args: tuple[Any, ...],
    lpips_model: Optional[LPIPS],
    report_lpips_test: bool,
    report_lpips_train: bool,
    log_path: Optional[str],
    is_first_report_iteration: bool,
) -> None:
    """
        Evaluate train and test cameras and write a compact report.

        The function averages metrics over each split, prints them through tqdm,
        and mirrors the same lines into the experiment log file.
    """
    torch.cuda.empty_cache()
    any_split_reports_lpips = report_lpips_test or report_lpips_train
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras(), "report_lpips": report_lpips_test},
        {"name": "train", "cameras": scene.getTrainCameras(), "report_lpips": report_lpips_train},
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0 if config["report_lpips"] and lpips_model is not None else None
            for viewpoint in config["cameras"]:
                l1_value, psnr_value, ssim_value, lpips_value = _evaluate_report_view(
                    viewpoint,
                    scene,
                    render_func,
                    render_args,
                    l1_loss_fn,
                    lpips_model,
                    config["report_lpips"],
                )
                l1_test += l1_value
                psnr_test += psnr_value
                ssim_test += ssim_value
                if lpips_test is not None and lpips_value is not None:
                    lpips_test += lpips_value
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            ssim_test /= len(config["cameras"])
            if lpips_test is not None:
                lpips_test /= len(config["cameras"])
            split_label = "Test " if config["name"] == "test" else "Train"
            message = "[{}] {}: [L1 {:.5f}] [PSNR {:.3f}] [SSIM {:.5f}]".format(
                iteration,
                split_label,
                float(l1_test),
                float(psnr_test),
                float(ssim_test),
            )
            if config["report_lpips"] and lpips_test is not None:
                message += " [LPIPS {:.5f}]".format(float(lpips_test))
            elif any_split_reports_lpips:
                message += " [LPIPS -------]"
            message += " [N_GS {}]".format(scene.gaussians.get_xyz.shape[0])
            log_lines = [message]
            if config["name"] == "train":
                log_lines.append("")
                tqdm.write(message)
            else:
                terminal_message = message if is_first_report_iteration else "\n" + message
                tqdm.write(terminal_message)
            if log_path is not None:
                append_log_lines(log_path, log_lines)
    torch.cuda.empty_cache()
