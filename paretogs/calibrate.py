"""Calibrate ParetoGS interpolation powers on training views only."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch

from gaussian_renderer import render
from scene.methods.optimization_methods import _compute_reconstruction_loss
from utils.evaluation.render_metric import build_background
from paretogs.decoder import (
    INTERPOLATION_CURVE_FORMAT,
    _band_opacity_permutation,
    _interpolated_payload,
    _interval_key,
    _resolve_active_anchors,
)
from paretogs.adapters import build_adapted_prefix, load_adapter_bundle
from paretogs.prefix import load_nested_order
from paretogs.scene import load_scene


def _calibration_views(views: list[Any], count: int) -> list[Any]:
    count = min(int(count), len(views))
    if count <= 0:
        raise ValueError("view_count must be positive.")
    return [views[int((index + 0.5) * len(views) / count)] for index in range(count)]


def _interval_budgets(lower: int, upper: int, progresses: list[float]) -> list[int]:
    budgets = []
    for progress in progresses:
        if not 0.0 < float(progress) < 1.0:
            raise ValueError("Calibration progresses must lie strictly inside (0, 1).")
        budget = int(round(lower + float(progress) * (upper - lower)))
        budget = min(max(budget, lower + 1), upper - 1)
        budgets.append(budget)
    return sorted(set(budgets))


def calibrate_interpolation(
    model_path: str,
    iteration: int,
    adapter_path: str,
    output_path: str,
    common_powers: list[float],
    shell_gate_powers: list[float],
    progresses: list[float],
    view_count: int,
    lambda_dssim: float,
    monotonic_weight: float = 1.0,
    smoothness_weight: float = 0.25,
    active_anchors: list[int] | None = None,
    source_path: str = "",
) -> dict[str, Any]:
    common_powers = sorted(set(float(value) for value in common_powers))
    shell_gate_powers = sorted(set(float(value) for value in shell_gate_powers))
    if not common_powers or min(common_powers) <= 0.0:
        raise ValueError("Every common power must be positive.")
    if not shell_gate_powers or min(shell_gate_powers) <= 0.0:
        raise ValueError("Every shell gate power must be positive.")
    if not 0.0 <= float(lambda_dssim) <= 1.0:
        raise ValueError("lambda_dssim must lie in [0, 1].")
    if float(monotonic_weight) < 0.0 or float(smoothness_weight) < 0.0:
        raise ValueError("Trajectory penalty weights must be non-negative.")

    dataset, pipeline, scene, gaussians = load_scene(model_path, iteration, source_path)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    bundle = load_adapter_bundle(adapter_path, gaussian_count)
    bundle_anchors = sorted(set(int(value) for value in bundle.get("budgets", [])))
    if len(bundle_anchors) < 2 or bundle_anchors[-1] != gaussian_count:
        raise ValueError("Adapter anchors must end at the full Gaussian count.")
    anchors = _resolve_active_anchors(bundle_anchors, active_anchors, gaussian_count)
    stored_order = load_nested_order(
        os.path.join(model_path, "nested_order.pt"),
        gaussian_count,
        gaussians.get_xyz.device,
    )
    endpoint_cache: dict[tuple[int, str], torch.Tensor | None] = {}
    permutation = _band_opacity_permutation(
        gaussians, stored_order, bundle, bundle_anchors, endpoint_cache
    )
    evaluation_order = stored_order[permutation]
    views = _calibration_views(scene.getTrainCameras(), view_count)
    background = build_background(dataset)
    intervals = {}
    started = time.perf_counter()

    def view_losses(proxy: Any) -> list[float]:
        losses = []
        with torch.no_grad():
            for view in views:
                image = render(
                    view, proxy, pipeline, background, track_gradients=False
                )["render"]
                alpha_mask = view.alpha_mask_cuda()
                if alpha_mask is not None:
                    image = image * alpha_mask
                target = view.original_image_cuda()
                loss = _compute_reconstruction_loss(image, target, lambda_dssim)
                losses.append(float(loss.item()))
        return losses

    def robust_quartile(values: list[float]) -> float:
        count = max(1, (len(values) + 3) // 4)
        return sum(sorted(values, reverse=True)[:count]) / count

    for lower, upper in zip(anchors[:-1], anchors[1:]):
        budgets = _interval_budgets(lower, upper, progresses)
        actual_progresses = [
            (budget - lower) / (upper - lower) for budget in budgets
        ]
        with torch.no_grad():
            lower_proxy = build_adapted_prefix(
                gaussians, stored_order, bundle["adapters"][lower]
            )
            upper_proxy = (
                gaussians if upper == gaussian_count else build_adapted_prefix(
                    gaussians, stored_order, bundle["adapters"][upper]
                )
            )
        lower_view_losses = view_losses(lower_proxy)
        upper_view_losses = view_losses(upper_proxy)
        lower_loss = sum(lower_view_losses) / len(lower_view_losses)
        upper_loss = sum(upper_view_losses) / len(upper_view_losses)
        del lower_proxy
        if upper_proxy is not gaussians:
            del upper_proxy
        candidates = []
        for common_power in common_powers:
            for shell_gate_power in shell_gate_powers:
                budget_view_losses = []
                for budget in budgets:
                    payload, _alpha, _progress = _interpolated_payload(
                        gaussians,
                        stored_order,
                        permutation,
                        bundle,
                        budget,
                        lower,
                        upper,
                        endpoint_cache,
                        common_power,
                        shell_gate_power,
                    )
                    with torch.no_grad():
                        proxy = build_adapted_prefix(gaussians, evaluation_order, payload)
                    budget_view_losses.append(view_losses(proxy))
                    del proxy, payload
                budget_losses = [
                    sum(losses) / len(losses) for losses in budget_view_losses
                ]
                loss_curve = [lower_loss, *budget_losses, upper_loss]
                progress_curve = [0.0, *actual_progresses, 1.0]
                slopes = [
                    (loss_curve[index] - loss_curve[index + 1])
                    / (progress_curve[index + 1] - progress_curve[index])
                    for index in range(len(loss_curve) - 1)
                ]
                endpoint_slope = lower_loss - upper_loss
                mean_curve_monotonic_penalty = sum(
                    max(-slope, 0.0) for slope in slopes
                ) / len(slopes)
                mean_curve_smoothness_penalty = sum(
                    abs(slope - endpoint_slope) for slope in slopes
                ) / len(slopes)
                view_monotonic_penalties = []
                view_smoothness_penalties = []
                for view_index in range(len(views)):
                    view_curve = [
                        lower_view_losses[view_index],
                        *(
                            losses[view_index]
                            for losses in budget_view_losses
                        ),
                        upper_view_losses[view_index],
                    ]
                    view_slopes = [
                        (view_curve[index] - view_curve[index + 1])
                        / (progress_curve[index + 1] - progress_curve[index])
                        for index in range(len(view_curve) - 1)
                    ]
                    view_endpoint_slope = view_curve[0] - view_curve[-1]
                    view_monotonic_penalties.append(sum(
                        max(-slope, 0.0) for slope in view_slopes
                    ) / len(view_slopes))
                    view_smoothness_penalties.append(sum(
                        abs(slope - view_endpoint_slope)
                        for slope in view_slopes
                    ) / len(view_slopes))
                monotonic_penalty = robust_quartile(view_monotonic_penalties)
                smoothness_penalty = robust_quartile(view_smoothness_penalties)
                mean_train_loss = sum(budget_losses) / len(budget_losses)
                score = mean_train_loss
                score += float(monotonic_weight) * monotonic_penalty
                score += float(smoothness_weight) * smoothness_penalty
                candidates.append({
                    "common_power": common_power,
                    "shell_gate_power": shell_gate_power,
                    "mean_train_loss": mean_train_loss,
                    "mean_loss_curve": loss_curve,
                    "mean_curve_monotonic_penalty": mean_curve_monotonic_penalty,
                    "mean_curve_smoothness_penalty": mean_curve_smoothness_penalty,
                    "monotonic_penalty": monotonic_penalty,
                    "smoothness_penalty": smoothness_penalty,
                    "score": score,
                })
        best = min(
            candidates,
            key=lambda item: (
                item["score"],
                -item["common_power"],
                -item["shell_gate_power"],
            ),
        )
        intervals[_interval_key(lower, upper)] = {
            "lower_anchor": lower,
            "upper_anchor": upper,
            "calibration_budgets": budgets,
            "endpoint_mean_losses": [lower_loss, upper_loss],
            "common_power": best["common_power"],
            "shell_gate_power": best["shell_gate_power"],
            "mean_train_loss": best["mean_train_loss"],
            "monotonic_penalty": best["monotonic_penalty"],
            "smoothness_penalty": best["smoothness_penalty"],
            "score": best["score"],
            "candidates": candidates,
        }
        print(
            "Calibrated {}:{} common_power={} shell_gate_power={} loss={:.6f} score={:.6f}".format(
                lower,
                upper,
                best["common_power"],
                best["shell_gate_power"],
                best["mean_train_loss"],
                best["score"],
            ),
            flush=True,
        )

    result = {
        "format": INTERPOLATION_CURVE_FORMAT,
        "version": 1,
        "model_path": os.path.abspath(model_path),
        "iteration": int(iteration),
        "adapter_path": os.path.abspath(adapter_path),
        "bundle_anchors": bundle_anchors,
        "anchors": anchors,
        "constraints": {
            "positive_power": True,
            "endpoint_exact": True,
        },
        "calibration": {
            "split": "train",
            "view_count": len(views),
            "lambda_dssim": float(lambda_dssim),
            "loss_mode": "reconstruction",
            "monotonic_weight": float(monotonic_weight),
            "smoothness_weight": float(smoothness_weight),
            "progresses": [float(value) for value in progresses],
            "common_power_candidates": common_powers,
            "shell_gate_power_candidates": shell_gate_powers,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "intervals": intervals,
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
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--common_powers",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )
    parser.add_argument(
        "--shell_gate_powers",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    )
    parser.add_argument("--progresses", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--view_count", type=int, default=10)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--monotonic_weight", type=float, default=1.0)
    parser.add_argument("--smoothness_weight", type=float, default=0.25)
    parser.add_argument("--active_anchors", nargs="+", type=int)
    args = parser.parse_args()
    result = calibrate_interpolation(
        args.model_path,
        args.iteration,
        args.adapter_path,
        args.output,
        args.common_powers,
        args.shell_gate_powers,
        args.progresses,
        args.view_count,
        args.lambda_dssim,
        args.monotonic_weight,
        args.smoothness_weight,
        args.active_anchors,
    )
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "elapsed_seconds": result["calibration"]["elapsed_seconds"],
        "selected": {
            key: {
                "common_power": value["common_power"],
                "shell_gate_power": value["shell_gate_power"],
            }
            for key, value in result["intervals"].items()
        },
    }, indent=2))
