"""Native asset loading, integrity checks and relative runtime paths."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
from typing import Any

def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_asset(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_torch_asset(path: Any, *args: Any, **kwargs: Any) -> Any:
    """Load tensor assets; trusted training checkpoints must opt out explicitly."""
    import torch
    kwargs.setdefault("weights_only", True)
    return torch.load(path, *args, **kwargs)


def load_runtime_manifest(runtime_dir: str | os.PathLike[str]) -> dict[str, Any]:
    directory = Path(os.fspath(runtime_dir)).absolute()
    manifest = load_json_asset(directory / "runtime_manifest.json")
    if manifest.get("format") != "paretogs_runtime_bundle" or int(manifest.get("version", 0)) != 1:
        raise ValueError("Unsupported ParetoGS runtime manifest.")
    for key in ("model_path", "adapter", "interpolation_curve", "nested_order"):
        if key in manifest:
            path = Path(os.fspath(manifest[key]))
            manifest[key] = str(path if path.is_absolute() else directory / path)
    if manifest.get("schedules"):
        raise ValueError("Per-Gaussian trajectory schedules are not part of the paper runtime.")
    return manifest
