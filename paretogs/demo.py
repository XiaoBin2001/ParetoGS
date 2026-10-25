"""Render one fixed camera while a ParetoGS budget grows over a chosen range."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import cv2
import numpy as np
import torch

from gaussian_renderer import render
from utils.evaluation.render_metric import build_background
from paretogs.decoder import BudgetDecoder
from paretogs.scene import load_scene
from utils.video_writer import StreamingVideoWriter


def _frame_rgb(
    view: Any,
    gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        image = render(
            view,
            gaussians,
            pipeline,
            background,
            track_gradients=False,
            inference_only=True,
        )["render"].clamp(0.0, 1.0)
        alpha_mask = view.alpha_mask_cuda()
        if alpha_mask is not None:
            image = image * alpha_mask
    return (
        image.mul(255.0)
        .add_(0.5)
        .clamp_(0.0, 255.0)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _annotate(
    frame: np.ndarray,
    scene_label: str,
    camera_index: int,
    image_name: str,
    budget: int,
    full_count: int,
    progress_frame: int | None = None,
    progress_total_frames: int | None = None,
    progress_fps: int | None = None,
    progress_segments: int = 0,
) -> np.ndarray:
    frame = frame.copy()
    height, width = frame.shape[:2]
    ratio = float(budget) / float(full_count)
    video_progress = progress_frame is not None
    if video_progress and (progress_total_frames is None or progress_fps is None
                           or not 0 <= progress_frame < progress_total_frames
                           or progress_fps <= 0 or progress_segments < 1):
        raise ValueError("Invalid video progress metadata")
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 92), (0, 0, 0), thickness=-1)
    cv2.rectangle(overlay, (0, height - (54 if video_progress else 42)),
                  (width, height), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        "{}  |  test camera {}  |  {}".format(
            scene_label.upper(), camera_index, image_name
        ),
        (24, 35),
        font,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Gaussians {:,} / {:,}    Budget {:6.2f}%".format(
            budget, full_count, ratio * 100.0
        ),
        (24, 72),
        font,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left, right = 24, width - 24
    if video_progress:
        elapsed = int(progress_frame / progress_fps)
        duration = int(progress_total_frames / progress_fps)
        cv2.putText(
            frame,
            "VIDEO {}:{:02d} / {}:{:02d}".format(
                elapsed // 60, elapsed % 60, duration // 60, duration % 60
            ),
            (left, height - 28), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        top, bottom = height - 19, height - 11
        fill_ratio = (progress_frame + 1) / progress_total_frames
    else:
        top, bottom = height - 27, height - 15
        fill_ratio = ratio
    cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100), thickness=1)
    fill = left + int(round((right - left) * fill_ratio))
    if fill > left:
        cv2.rectangle(frame, (left, top), (fill, bottom), (77, 210, 255), thickness=-1)
    if video_progress:
        for segment in range(1, progress_segments):
            tick = left + int(round((right - left) * segment / progress_segments))
            cv2.line(frame, (tick, top - 2), (tick, bottom + 2), (210, 210, 210), 1)
    return frame


def render_budget_video(
    model_path: str,
    iteration: int,
    adapter_path: str,
    curve_path: str,
    test_camera: int,
    scene_label: str,
    output_path: str,
    levels: int,
    fps: int,
    frames_per_level: int,
    endpoint_hold_seconds: float,
    start_percent: float,
    end_percent: float,
    active_anchors: list[int] | None = None,
    runtime_dir: str = "",
) -> dict[str, Any]:
    if levels < 2:
        raise ValueError("levels must be at least 2.")
    if fps <= 0 or frames_per_level <= 0 or endpoint_hold_seconds < 0.0:
        raise ValueError("fps/frames_per_level must be positive and hold must be non-negative.")
    if not 0.0 < start_percent < end_percent <= 100.0:
        raise ValueError(
            "start_percent and end_percent must satisfy "
            "0 < start_percent < end_percent <= 100."
        )
    if runtime_dir:
        from paretogs.runtime import ParetoGSRuntime

        decoder = ParetoGSRuntime(runtime_dir)
        dataset, pipeline, scene = decoder.dataset, decoder.pipeline, decoder.scene
        gaussians, anchors = decoder.gaussians, decoder.anchors
        model_path = dataset.model_path
        iteration = int(decoder.manifest["iteration"])
        adapter_path = decoder.manifest["adapter"]
        curve_path = decoder.manifest["interpolation_curve"]
    else:
        dataset, pipeline, scene, gaussians = load_scene(model_path, iteration)
        decoder = BudgetDecoder.from_assets(
            gaussians, adapter_path, os.path.join(dataset.model_path, "nested_order.pt"),
            curve_path, active_anchors,
        )
        anchors = decoder.anchors
    views = scene.getTestCameras()
    if test_camera < 0 or test_camera >= len(views):
        raise ValueError("test_camera is outside the available test views.")
    view = views[test_camera]
    full_count = decoder.gaussian_count
    background = build_background(dataset)
    start_budget = int(round(full_count * start_percent / 100.0))
    end_budget = int(round(full_count * end_percent / 100.0))
    if start_budget < anchors[0]:
        raise ValueError(
            "start_percent resolves to {:,} Gaussians, below the first trained "
            "anchor {:,}.".format(start_budget, anchors[0])
        )
    budgets = [
        int(round(start_budget + (end_budget - start_budget) * index / (levels - 1)))
        for index in range(levels)
    ]
    budgets[0], budgets[-1] = start_budget, end_budget
    output_path = os.path.abspath(output_path)
    writer = StreamingVideoWriter(
        output_path,
        fps=fps,
        width=int(view.image_width),
        height=int(view.image_height),
        crf=16,
        preset="medium",
    )
    hold_frames = int(round(endpoint_hold_seconds * fps))
    started = time.perf_counter()
    try:
        for level_index, budget in enumerate(budgets):
            proxy, _state = decoder.decode(budget)
            frame = _frame_rgb(view, proxy, pipeline, background)
            frame = _annotate(
                frame,
                scene_label,
                test_camera,
                str(view.image_name),
                budget,
                full_count,
            )
            repeat = hold_frames if level_index in (0, len(budgets) - 1) else frames_per_level
            repeat = max(repeat, 1)
            for _ in range(repeat):
                writer.write(frame)
            if proxy is not None and proxy is not gaussians:
                del proxy
            if level_index % 10 == 0 or level_index == len(budgets) - 1:
                print(
                    "{}: level {}/{} budget {:,} ({:.2f}%)".format(
                        scene_label,
                        level_index,
                        len(budgets) - 1,
                        budget,
                        100.0 * budget / full_count,
                    ),
                    flush=True,
                )
    finally:
        writer.close()
    result = {
        "format": "paretogs_nested_budget_demo",
        "version": 2,
        "scene": scene_label,
        "model_path": os.path.abspath(model_path),
        "iteration": int(iteration),
        "adapter_path": os.path.abspath(adapter_path),
        "curve_path": os.path.abspath(curve_path) if curve_path else None,
        "test_camera_index": int(test_camera),
        "test_camera_uid": int(view.uid),
        "image_name": str(view.image_name),
        "anchors": anchors,
        "levels": int(levels),
        "start_percent": float(start_percent),
        "start_budget": int(start_budget),
        "end_percent": float(end_percent),
        "end_budget": int(end_budget),
        "schedule": "linear_budget_ratio_{:g}_to_{:g}_percent".format(
            start_percent, end_percent
        ),
        "below_first_anchor": "unsupported; every frame is at or above a trained anchor",
        "fps": int(fps),
        "frames_per_level": int(frames_per_level),
        "endpoint_hold_seconds": float(endpoint_hold_seconds),
        "video_frames": int(2 * max(hold_frames, 1) + (levels - 2) * frames_per_level),
        "duration_seconds": float(
            (2 * max(hold_frames, 1) + (levels - 2) * frames_per_level) / fps
        ),
        "render_elapsed_seconds": time.perf_counter() - started,
        "output_path": output_path,
        "output_bytes": os.path.getsize(output_path),
    }
    with open(output_path + ".json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--curve_path", default="")
    parser.add_argument("--test_camera", type=int, required=True)
    parser.add_argument("--scene_label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels", type=int, default=101)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames_per_level", type=int, default=2)
    parser.add_argument("--endpoint_hold_seconds", type=float, default=1.0)
    parser.add_argument("--start_percent", type=float, default=1.0)
    parser.add_argument("--end_percent", type=float, default=100.0)
    parser.add_argument("--runtime_dir", default="")
    parser.add_argument("--active_anchors", nargs="+", type=int)
    args = parser.parse_args()
    if not args.runtime_dir and (not args.model_path or not args.adapter_path):
        parser.error("Provide --runtime_dir or --model_path with --adapter_path.")
    print(json.dumps(render_budget_video(
        args.model_path,
        args.iteration,
        args.adapter_path,
        args.curve_path,
        args.test_camera,
        args.scene_label,
        args.output,
        args.levels,
        args.fps,
        args.frames_per_level,
        args.endpoint_hold_seconds,
        args.start_percent,
        args.end_percent,
        args.active_anchors, args.runtime_dir,
    ), indent=2))
