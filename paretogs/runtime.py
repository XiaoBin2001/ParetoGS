"""Persistent decoder for a packaged ParetoGS runtime bundle."""
from __future__ import annotations

import os
from typing import Any

from paretogs.adapters import load_adapter_bundle
from utils.artifact_utils import file_sha256, load_runtime_manifest
from paretogs.decoder import (
    BudgetDecoder, _evaluation_order_hash, load_interpolation_curve,
)
from paretogs.prefix import load_nested_order
from paretogs.scene import load_model


class ParetoGSRuntime(BudgetDecoder):
    """Load resident assets; dataset cameras are optional and loaded on first use."""

    def __init__(self, runtime_dir: str, cache_anchors: int | None = 2) -> None:
        self.runtime_dir = os.path.abspath(os.fspath(runtime_dir))
        self.manifest = load_runtime_manifest(self.runtime_dir)
        expected_model = self.manifest.get("model_sha256")
        if expected_model is not None:
            model_ply = os.path.join(
                self.manifest["model_path"], "point_cloud",
                "iteration_{}".format(self.manifest["iteration"]), "point_cloud.ply",
            )
            if file_sha256(model_ply) != expected_model:
                raise ValueError("Runtime full model does not match its manifest.")
        self.dataset, self.pipeline, gaussians = load_model(
            self.manifest["model_path"], int(self.manifest["iteration"])
        )
        self.manifest["iteration"] = int(gaussians._loaded_iteration)
        self._scene = None
        gaussian_count = int(gaussians.get_xyz.shape[0])
        if int(self.manifest.get("gaussian_count", gaussian_count)) != gaussian_count:
            raise ValueError("Runtime manifest and model Gaussian counts differ.")
        anchors = [int(value) for value in self.manifest["active_anchors"]]
        bundle = load_adapter_bundle(self.manifest["adapter"], gaussian_count)
        curve = load_interpolation_curve(self.manifest["interpolation_curve"], anchors)
        order = load_nested_order(
            self.manifest.get("nested_order") or os.path.join(self.dataset.model_path, "nested_order.pt"),
            gaussian_count, gaussians.get_xyz.device,
        )
        super().__init__(gaussians, bundle, order, anchors, curve, cache_anchors)
        if _evaluation_order_hash(self.evaluation_order) != self.manifest["evaluation_order_sha256"]:
            raise ValueError("Runtime Gaussian order does not match its manifest.")

    @property
    def scene(self) -> Any:
        if self._scene is None:
            from scene import Scene

            self._scene = Scene(
                self.dataset, self.gaussians,
                load_iteration=int(self.manifest["iteration"]),
                shuffle=False, load_gaussians=False,
            )
        return self._scene
