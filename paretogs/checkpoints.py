"""Checkpoint layouts and persistent Gaussian identity alignment."""
from __future__ import annotations

import numbers
import os
from typing import Any

import torch

from utils.artifact_utils import load_torch_asset


RAW_PARAMETER_INDICES = (2, 3, 4, 5, 6, 7)
PARAMETER_NAMES = {
    2: "xyz_delta",
    3: "dc_delta",
    4: "rest_delta",
    5: "scale_delta",
    6: "rotation_delta",
    7: "opacity_delta",
}


RAW_FIELD_INDICES = {name: index for index, name in PARAMETER_NAMES.items()}
ANCHOR_STATE_FORMAT = "paretogs_anchor_state"


def _normalize_model_params(model_params: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(model_params, tuple) or len(model_params) != 15:
        raise ValueError("Expected a 15-field checkpoint: {}".format(path))
    if isinstance(model_params[12], dict) and isinstance(model_params[13], numbers.Real):
        return model_params
    if isinstance(model_params[13], dict) and isinstance(model_params[14], numbers.Real):
        return (
            *model_params[:8],
            None,
            model_params[10],
            model_params[11],
            model_params[12],
            model_params[13],
            model_params[14],
            None,
        )
    raise ValueError("Unsupported 15-field checkpoint layout: {}".format(path))


def _load_checkpoint(path: str) -> tuple[dict[str, Any], tuple[Any, ...]]:
    payload = load_torch_asset(path, map_location="cpu", weights_only=False)
    model_params = payload.get("model_params") if isinstance(payload, dict) else None
    return payload, _normalize_model_params(model_params, path)


def _decode_source_ids(model_params: tuple[Any, ...]) -> torch.Tensor:
    gaussian_count = int(model_params[2].shape[0])
    metadata = model_params[14]
    source_ids = metadata.get("source_ids") if isinstance(metadata, dict) else None
    if not torch.is_tensor(source_ids):
        return torch.arange(gaussian_count, dtype=torch.long)
    source_ids = source_ids.to(dtype=torch.long, device="cpu").flatten().contiguous()
    if int(source_ids.shape[0]) != gaussian_count:
        raise ValueError("Checkpoint source_ids do not match its Gaussian count.")
    if int(torch.unique(source_ids).numel()) != gaussian_count:
        raise ValueError("Checkpoint source_ids must be unique.")
    return source_ids


def load_anchor_state(path: str) -> dict[str, Any]:
    """Read parameters/IDs from a trusted snapshot or full training checkpoint."""
    payload = load_torch_asset(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == ANCHOR_STATE_FORMAT:
        if payload.get("version") != 1:
            raise ValueError("Unsupported anchor state version: {}".format(path))
        result = payload
    else:
        model = _normalize_model_params(
            payload.get("model_params") if isinstance(payload, dict) else None, path
        )
        result = {
            "format": ANCHOR_STATE_FORMAT,
            "version": 1,
            "iteration": int(payload.get("iteration", 0)),
            "parameters": {name: model[index] for index, name in PARAMETER_NAMES.items()},
            "source_ids": _decode_source_ids(model),
        }
    ids = result.get("source_ids")
    parameters = result.get("parameters", {})
    if (
        not torch.is_tensor(ids) or ids.ndim != 1 or ids.dtype != torch.long
        or torch.unique(ids).numel() != ids.numel()
        or set(parameters) != set(PARAMETER_NAMES.values())
        or any(not torch.is_tensor(value) or value.ndim < 2
               or value.shape[0] != ids.numel() for value in parameters.values())
    ):
        raise ValueError("Malformed anchor parameters or source IDs: {}".format(path))
    return result


def write_anchor_state(checkpoint_path: str, output_path: str) -> int:
    """Save raw parameters and IDs atomically, without optimizer/trace state."""
    state = load_anchor_state(checkpoint_path)
    state["parameters"] = {
        key: value.detach().cpu().clone().contiguous()
        for key, value in state["parameters"].items()
    }
    state["source_ids"] = state["source_ids"].clone().contiguous()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, output_path)
    return int(state["source_ids"].numel())


def _membership_indices(base_ids: torch.Tensor, member_ids: torch.Tensor) -> torch.Tensor:
    sorted_ids, sorted_indices = torch.sort(base_ids)
    positions = torch.searchsorted(sorted_ids, member_ids)
    if bool((positions >= int(sorted_ids.shape[0])).any().item()):
        raise ValueError("A child Gaussian ID is absent from the full model.")
    indices = sorted_indices[positions]
    if not torch.equal(base_ids[indices], member_ids):
        raise ValueError("A child Gaussian ID is absent from the full model.")
    return indices
