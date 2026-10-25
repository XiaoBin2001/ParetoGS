"""
Batch result aggregation helpers.

Post-processing writes one JSON summary per scene and split. This module reads
those summaries, turns them into CSV rows, retains every repeat and budget, and can rebuild total CSV files from already existing outputs.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Optional

from utils.batch_training.config import (
    build_scene_paths,
)
from utils.experiment_utils import load_json


def load_result_summary(output_dir: str, split_name: str) -> Optional[dict[str, Any]]:
    """
        Load one `result_<split>.json` file if it exists and contains data.
    """
    summary_path = os.path.join(output_dir, "result_{}.json".format(split_name))
    if not os.path.exists(summary_path):
        return None
    payload = load_json(summary_path, default={})
    return payload if isinstance(payload, dict) and payload else None


def collect_scene_result_rows(
    config_path: str,
    scene: dict[str, Any],
    output_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    """
        Convert one scene output directory into train/test CSV rows.
    """
    status = load_json(os.path.join(output_dir, "run_status.json"), default={})
    if status and (status.get("train") != "completed" or status.get("postprocess") != "completed"):
        return {"train": [], "test": []}
    params = load_json(os.path.join(output_dir, "training_parameters.json"), default={})
    common_row = {
        "config_name": Path(config_path).stem,
        "scene_name": str(scene["name"]),
        "output_name": os.path.basename(output_dir.rstrip(os.sep)),
        "repeat": status.get("repeat", 1),
        "seed": params.get("seed", 0),
        "record_path": os.path.abspath(output_dir),
        "model_path": os.path.abspath(output_dir),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ("train", "test"):
        summary = load_result_summary(output_dir, split_name)
        if summary is None or (status and split_name not in status.get("written_splits", [])):
            continue
        summaries = summary.get("budget_results", [summary])
        for metrics in summaries:
            row = dict(common_row)
            row["split"] = split_name
            row.update(metrics)
            rows_by_split[split_name].append(row)
    return rows_by_split


def write_result_total_csv(output_root: str, config_path: str, split_name: str, rows: list[dict[str, Any]]) -> Optional[str]:
    """
        Write rows for one split into the config-level total CSV file.
    """
    csv_path = os.path.join(output_root, "{}_result_{}_total.csv".format(Path(config_path).stem, split_name))
    if not rows and not os.path.exists(csv_path):
        return None
    ordered_fieldnames = [
        "config_name",
        "split",
        "iteration",
        "scene_name",
        "output_name",
        "repeat",
        "seed",
        "record_path",
        "model_path",
        "budget",
        "budget_ratio",
        "PSNR",
        "SSIM",
        "LPIPS",
        "FPS",
        "NUM",
        "Training_time",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            display = dict(row)
            for metric, decimals in (("PSNR", 3), ("SSIM", 5), ("LPIPS", 5), ("FPS", 1), ("Training_time", 2)):
                if display.get(metric) is not None:
                    display[metric] = format(float(display[metric]), ".{}f".format(decimals))
            writer.writerow(display)
    return csv_path


def extend_result_rows(
    result_rows_by_split: dict[str, list[dict[str, Any]]],
    scene_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """
        Append newly collected rows into the split-indexed row accumulator.
    """
    for split_name, rows in scene_rows.items():
        result_rows_by_split[split_name].extend(rows)


def collect_existing_total_rows(config_path: str, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read every independent repeat; output directories are never selected or merged."""
    rows = {"train": [], "test": []}
    for scene in config.get("scenes", []):
        for index in range(config.get("repeat", 1)):
            _, directory = build_scene_paths(config, scene, index)
            found = collect_scene_result_rows(config_path, scene, directory)
            for split_rows in found.values():
                for row in split_rows:
                    row["repeat"] = index + 1
            extend_result_rows(rows, found)
    return rows


def rebuild_result_total_csv(config_path: str, config: dict[str, Any]) -> None:
    for split, rows in collect_existing_total_rows(config_path, config).items():
        path = write_result_total_csv(config["output_root"], config_path, split, rows)
        if path is not None:
            print("Rebuilt {} total CSV to {}".format(split, path))
