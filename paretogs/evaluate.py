"""Evaluate ParetoGS at arbitrary budgets between trained anchors."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from utils.evaluation.render_metric import build_background, evaluate_split, write_opacity_report
from paretogs.decoder import BudgetDecoder
from paretogs.scene import load_scene


def evaluate_interpolation(
    model_path: str,
    iteration: int,
    adapter_path: str,
    budgets: list[int],
    output_path: str,
    curve_path: str = "",
    active_anchors: list[int] | None = None,
    evaluation_split: str = "test",
    view_indices: list[int] | None = None,
    include_per_view: bool = False,
    runtime: Any = None,
    artifact_root: str = "",
    save_image_budgets: list[int] | None = None,
    fps_warmup_rounds: int = 0,
    fps_measure_rounds: int = 0,
    training_time_minutes: float | None = None,
    opacity_root: str = "",
    variant: str = "full",
) -> dict[str, Any]:
    """Evaluate the same decoder used for deployment, export and video."""
    from lpipsPyTorch.modules.lpips import LPIPS

    if runtime is None:
        model_path = os.fspath(model_path)
        dataset, pipeline, scene, gaussians = load_scene(model_path, iteration)
        decoder = BudgetDecoder.from_assets(
            gaussians, adapter_path, os.path.join(dataset.model_path, "nested_order.pt"),
            curve_path, active_anchors,
        )
    else:
        decoder = runtime
        dataset, pipeline, scene = runtime.dataset, runtime.pipeline, runtime.scene
    gaussian_count = decoder.gaussian_count
    anchors, curve = decoder.anchors, decoder.curve
    bundle_anchors = sorted(set(int(value) for value in decoder.bundle["budgets"]))
    budgets = sorted(set(int(value) for value in budgets))
    if not budgets or any(not anchors[0] <= value <= gaussian_count for value in budgets):
        raise ValueError("Requested budgets must lie inside the trained range.")
    if evaluation_split not in {"train", "test"}:
        raise ValueError("evaluation_split must be train or test.")
    views = scene.getTestCameras() if evaluation_split == "test" else scene.getTrainCameras()
    if view_indices:
        if any(index < 0 or index >= len(views) for index in view_indices):
            raise ValueError("View index is outside the selected split.")
        views = [views[index] for index in view_indices]
    if not views:
        raise ValueError("The requested evaluation split contains no views.")
    lpips_model = LPIPS("vgg").to("cuda").eval()
    background = build_background(dataset)
    artifact_root = os.path.abspath(artifact_root) if artifact_root else ""
    image_budgets = set(int(value) for value in (save_image_budgets or []))
    unexpected_image_budgets = image_budgets - set(budgets)
    if unexpected_image_budgets:
        raise ValueError(
            "Image budgets were not requested for evaluation: {}.".format(
                sorted(unexpected_image_budgets)
            )
        )
    shared_gt_root = os.path.join(artifact_root, "gt") if artifact_root else None
    ground_truth_written = False
    results = {}
    for budget_index, budget in enumerate(budgets, 1):
        print(
            "Evaluating budget {}/{}: K={}".format(budget_index, len(budgets), budget),
            flush=True,
        )
        proxy, metadata = decoder.decode(budget, variant=variant)
        if opacity_root:
            opacity_path = os.path.join(opacity_root, 'budget_{}'.format(budget))
            os.makedirs(opacity_path, exist_ok=True)
            write_opacity_report(opacity_path, iteration, proxy)
        is_anchor = budget in anchors
        save_images = budget in image_budgets
        budget_root = (
            os.path.join(artifact_root, "budget_{}".format(budget))
            if artifact_root else None
        )
        metrics = evaluate_split(
            dataset=dataset, split_name=evaluation_split, views=views,
            iteration=iteration, gaussians=proxy, pipeline=pipeline,
            background=background, lpips_model=lpips_model,
            save_images=save_images, fps_warmup_rounds=fps_warmup_rounds, fps_measure_rounds=fps_measure_rounds,
            training_time_minutes=training_time_minutes,
            output_tag=("serialized_anchor_" if is_anchor else "smooth_interpolation_") + str(budget),
            return_per_view=include_per_view,
            output_root=budget_root,
            ground_truth_root=shared_gt_root,
            save_ground_truth=save_images and not ground_truth_written,
        )
        ground_truth_written = ground_truth_written or save_images
        if is_anchor:
            results[str(budget)] = {"mode": "trained_anchor", "metrics": metrics}
        else:
            lower, upper = metadata["interval"]
            results[str(budget)] = {
                "lower_anchor": lower, "upper_anchor": upper,
                **metadata, "metrics": metrics,
            }
        del proxy
    result = {
        "model_path": os.path.abspath(model_path), "iteration": int(iteration),
        "adapter_path": os.path.abspath(os.fspath(adapter_path)),
        "curve_path": curve.get("path") if curve else None,
        "variant": variant, "evaluation_split": evaluation_split,
        "view_indices": list(view_indices or []),
        "includes_per_view_metrics": bool(include_per_view),
        "artifact_root": artifact_root or None,
        "saved_image_budgets": sorted(image_budgets),
        "bundle_anchors": bundle_anchors, "trained_anchors": anchors,
        "requested_budgets": budgets,
        "untrained_budgets": [budget for budget in budgets if budget not in anchors],
        "interpolation": (
            "trained shell order + calibrated interval powers"
            if curve else "trained shell order + t^4 common state + t^2 shell fade-in"
        ),
        "results": results,
    }
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--budgets", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curve_path", default="")
    parser.add_argument("--active_anchors", nargs="+", type=int)
    parser.add_argument("--evaluation_split", choices=("train", "test"), default="test")
    parser.add_argument("--view_indices", nargs="*", type=int, default=[])
    parser.add_argument("--include_per_view", action="store_true")
    parser.add_argument("--artifact_root", default="")
    parser.add_argument("--save_image_budgets", nargs="*", type=int, default=[])
    args = parser.parse_args()
    print(json.dumps(evaluate_interpolation(
        args.model_path,
        args.iteration,
        args.adapter_path,
        args.budgets,
        args.output,
        args.curve_path,
        args.active_anchors,
        args.evaluation_split,
        args.view_indices,
        args.include_per_view,
        artifact_root=args.artifact_root,
        save_image_budgets=args.save_image_budgets,
    ), indent=2))
