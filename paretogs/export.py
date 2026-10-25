"""Export one ParetoGS operating point as a standard 3DGS PLY."""
from __future__ import annotations

import argparse
import json
import os

from paretogs.decoder import BudgetDecoder
from paretogs.scene import load_model


def materialize_prefix(
    model_path: str,
    iteration: int,
    adapter_path: str,
    budget: int,
    output_path: str,
    curve_path: str = "",
    active_anchors: list[int] | None = None,
    runtime_dir: str = "",
) -> dict:
    output_path = os.path.abspath(output_path)
    order_source = "saved"
    decoder = None
    if runtime_dir:
        from paretogs.runtime import ParetoGSRuntime

        decoder = ParetoGSRuntime(runtime_dir)
        gaussians = decoder.gaussians
        model_path = decoder.dataset.model_path
        iteration = int(decoder.manifest["iteration"])
        adapter_path = decoder.manifest["adapter"]
        curve_path = decoder.manifest["interpolation_curve"]
    else:
        model_path = os.path.abspath(os.fspath(model_path))
        _dataset, _pipeline, gaussians = load_model(model_path, iteration)
        if adapter_path:
            decoder = BudgetDecoder.from_assets(
                gaussians, adapter_path, os.path.join(model_path, "nested_order.pt"),
                curve_path, active_anchors,
            )
    iteration = int(gaussians._loaded_iteration)
    source_ply = gaussians._loaded_ply_path
    if os.path.abspath(source_ply) == output_path:
        raise ValueError("Output PLY must differ from the source PLY.")
    count = int(gaussians.get_xyz.shape[0])
    if decoder is not None:
        output_gaussians, metadata = decoder.decode(budget)
    elif int(budget) == count:
        output_gaussians, metadata = gaussians, {"mode": "full_anchor", "budget": count}
    else:
        raise ValueError("An adapter or runtime bundle is required for a truncated budget.")
    output_gaussians.save_ply(output_path)
    interpolated = metadata["mode"] in {"scalar_curve"}
    result = {
        "model_path": model_path, "iteration": int(iteration),
        "budget": int(budget), "full_gaussian_count": count,
        "adapter_path": os.path.abspath(os.fspath(adapter_path)) if adapter_path else None,
        "curve_path": os.path.abspath(os.fspath(curve_path)) if curve_path else None,
        "adapted": int(budget) != count, "interpolated": interpolated,
        "interpolation": metadata if interpolated else None,
        "order_source": order_source, "output_path": output_path,
        "output_bytes": os.path.getsize(output_path),
    }
    with open(output_path + ".json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_dir", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curve_path", default="")
    parser.add_argument("--active_anchors", nargs="+", type=int)
    args = parser.parse_args()
    if not args.runtime_dir and not args.model_path:
        parser.error("Provide --runtime_dir or --model_path.")
    print(json.dumps(materialize_prefix(
        args.model_path, args.iteration, args.adapter_path, args.budget,
        args.output, args.curve_path,
        args.active_anchors, args.runtime_dir,
    ), indent=2))
