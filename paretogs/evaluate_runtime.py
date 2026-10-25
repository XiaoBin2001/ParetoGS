"""Evaluate a packaged ParetoGS runtime bundle."""
from __future__ import annotations

import argparse
import json
import os

from paretogs.evaluate import evaluate_interpolation
from paretogs.runtime import ParetoGSRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_dir", required=True)
    parser.add_argument("--budgets", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--view_indices", nargs="+", type=int)
    parser.add_argument("--artifact_root", default="")
    parser.add_argument("--save_image_budgets", nargs="*", type=int, default=[])
    args = parser.parse_args()
    runtime_dir = os.path.abspath(args.runtime_dir)
    runtime = ParetoGSRuntime(runtime_dir)
    manifest = runtime.manifest
    result = evaluate_interpolation(
        manifest["model_path"], int(manifest["iteration"]), manifest["adapter"],
        args.budgets, args.output, view_indices=args.view_indices, runtime=runtime,
        artifact_root=args.artifact_root,
        save_image_budgets=args.save_image_budgets,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
