"""Scene loading and differentiable Gaussian proxies used by ParetoGS."""
from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.artifact_utils import load_json_asset


def load_model_parameters(model_path: str) -> tuple[Any, Any]:
    model_path = os.path.abspath(os.fspath(model_path))
    parameters_path = os.path.join(model_path, "training_parameters.json")
    saved_args = load_json_asset(parameters_path)
    saved_args["model_path"] = model_path
    namespace = Namespace(**saved_args)
    parser = ArgumentParser(add_help=False)
    dataset = ModelParams(parser).extract(namespace)
    pipeline = PipelineParams(parser).extract(namespace)
    return dataset, pipeline


def load_model(model_path: str, iteration: int) -> tuple[Any, Any, GaussianModel]:
    """Load model parameters without reading dataset images or cameras."""
    dataset, pipeline = load_model_parameters(model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    if iteration == -1:
        from utils.general_utils import searchForMaxIteration

        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    ply_path = os.path.join(
        dataset.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply"
    )
    gaussians.load_ply(ply_path)
    for value in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                  gaussians._scaling, gaussians._rotation, gaussians._opacity):
        value.requires_grad_(False)
    gaussians._loaded_ply_path = ply_path
    gaussians._loaded_iteration = iteration
    return dataset, pipeline, gaussians


def load_scene(model_path: str, iteration: int, source_path: str = "") -> tuple[Any, Any, Scene, GaussianModel]:
    dataset, pipeline = load_model_parameters(model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    if source_path:
        dataset.source_path = source_path
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return dataset, pipeline, scene, gaussians
