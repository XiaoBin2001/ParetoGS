"""Package the complete inference assets in one relocatable directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from utils.artifact_utils import file_sha256
from paretogs.decoder import BudgetDecoder, _evaluation_order_hash
from paretogs.scene import load_model


def package_runtime(
    model_path: str,
    iteration: int,
    target_adapter: str,
    curve_path: str,
    active_anchors: list[int],
    output_dir: str,
) -> dict[str, object]:
    source = Path(model_path).resolve()
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError("Refusing to overwrite runtime: {}".format(output))
    _dataset, _pipeline, gaussians = load_model(str(source), iteration)
    iteration = int(gaussians._loaded_iteration)
    decoder = BudgetDecoder.from_assets(
        gaussians, target_adapter, str(source / "nested_order.pt"),
        curve_path, active_anchors,
    )
    ply = Path("point_cloud") / ("iteration_{}".format(iteration)) / "point_cloud.ply"
    # Publish only after all required assets have been copied and verified.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        assets = {
            "adapter.pt": Path(target_adapter),
            "interpolation_curve.json": Path(curve_path),
            "model/nested_order.pt": source / "nested_order.pt",
            "model/training_parameters.json": source / "training_parameters.json",
            str(Path("model") / ply): source / ply,
        }
        for name, path in assets.items():
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if file_sha256(path) != file_sha256(destination):
                raise IOError("Runtime copy verification failed: {}".format(name))
        manifest = {
            "format": "paretogs_runtime_bundle", "version": 1,
            "model_path": "model", "iteration": iteration,
            "gaussian_count": decoder.gaussian_count,
            "active_anchors": decoder.anchors,
            "adapter": "adapter.pt", "adapter_bytes": (temporary / "adapter.pt").stat().st_size,
            "interpolation_curve": "interpolation_curve.json",
            "nested_order": "model/nested_order.pt",
            "evaluation_order_sha256": _evaluation_order_hash(decoder.evaluation_order),
            "model_sha256": file_sha256(temporary / "model" / ply),
        }
        (temporary / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--target_adapter", required=True)
    parser.add_argument("--curve_path", required=True)
    parser.add_argument("--active_anchors", nargs="+", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    print(json.dumps(package_runtime(**vars(args)), indent=2))
