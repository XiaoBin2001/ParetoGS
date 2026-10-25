"""Evaluate one explicit PLY or compact anchor state on a saved scene."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from paretogs.checkpoints import load_anchor_state
from paretogs.scene import load_model_parameters
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.evaluation.render_metric import build_background, evaluate_split


_STATE_FIELDS = {
    "xyz_delta": "_xyz",
    "dc_delta": "_features_dc",
    "rest_delta": "_features_rest",
    "scale_delta": "_scaling",
    "rotation_delta": "_rotation",
    "opacity_delta": "_opacity",
}


def _load_anchor_gaussians(path: str, sh_degree: int) -> GaussianModel:
    state = load_anchor_state(path)
    gaussians = GaussianModel(sh_degree)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    for state_name, model_name in _STATE_FIELDS.items():
        value = state["parameters"][state_name].detach().to("cuda").contiguous()
        value.requires_grad_(False)
        setattr(gaussians, model_name, value)
    gaussians._source_ids = state["source_ids"].to("cuda").contiguous()
    return gaussians


def evaluate_model(
    model_path: str,
    iteration: int,
    output_path: str,
    artifact_root: str,
    ply_path: str = "",
    anchor_state_path: str = "",
    save_images: bool = False,
    ground_truth_root: str = "",
    save_ground_truth: bool = True,
    include_per_view: bool = False,
    source_path: str = "",
    view_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Render one explicit model while reusing the backbone camera/data settings."""
    from lpipsPyTorch.modules.lpips import LPIPS

    if bool(ply_path) == bool(anchor_state_path):
        raise ValueError("Choose exactly one of ply_path or anchor_state_path.")
    model_path = os.path.abspath(model_path)
    artifact_root = os.path.abspath(artifact_root)
    dataset, pipeline = load_model_parameters(model_path)
    if source_path:
        dataset.source_path = os.path.abspath(source_path)
    if ply_path:
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(os.path.abspath(ply_path))
        source = os.path.abspath(ply_path)
        source_type = "ply"
    else:
        gaussians = _load_anchor_gaussians(os.path.abspath(anchor_state_path), dataset.sh_degree)
        source = os.path.abspath(anchor_state_path)
        source_type = "anchor_state"
    scene = Scene(
        dataset, gaussians, load_iteration=iteration, shuffle=False,
        load_gaussians=False,
    )
    views = scene.getTestCameras()
    if view_indices:
        if any(index < 0 or index >= len(views) for index in view_indices):
            raise ValueError('View index is outside the test split.')
        views = [views[index] for index in view_indices]
    if not views:
        raise ValueError("The test split contains no views.")
    metrics = evaluate_split(
        dataset=dataset,
        split_name="test",
        views=views,
        iteration=iteration,
        gaussians=gaussians,
        pipeline=pipeline,
        background=build_background(dataset),
        lpips_model=LPIPS("vgg").to("cuda").eval(),
        save_images=save_images,
        fps_warmup_rounds=0,
        fps_measure_rounds=0,
        training_time_minutes=None,
        return_per_view=include_per_view,
        output_root=artifact_root,
        ground_truth_root=os.path.abspath(ground_truth_root) if ground_truth_root else None,
        save_ground_truth=save_images and save_ground_truth,
    )
    result = {
        "model_path": model_path,
        "iteration": int(iteration),
        "source_type": source_type,
        "source": source,
        "actual_budget": int(gaussians.get_xyz.shape[0]),
        "evaluation_split": "test",
        "artifact_root": artifact_root,
        "saved_images": bool(save_images),
        "metrics": metrics,
    }
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact_root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ply_path")
    source.add_argument("--anchor_state_path")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--ground_truth_root", default="")
    parser.add_argument("--skip_ground_truth", action="store_true")
    parser.add_argument("--include_per_view", action="store_true")
    parser.add_argument("--source_path", default="")
    parser.add_argument("--view_indices", nargs="+", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_model(
        model_path=args.model_path,
        iteration=args.iteration,
        output_path=args.output,
        artifact_root=args.artifact_root,
        ply_path=args.ply_path or "",
        anchor_state_path=args.anchor_state_path or "",
        save_images=args.save_images,
        ground_truth_root=args.ground_truth_root,
        save_ground_truth=not args.skip_ground_truth,
        include_per_view=args.include_per_view,
        source_path=args.source_path,
        view_indices=args.view_indices,
    ), indent=2))


if __name__ == "__main__":
    main()
