"""Runtime helpers for ParetoGS density-conditioned parameter sidecars."""
from __future__ import annotations

from utils.artifact_utils import load_torch_asset

import math
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any

import torch

from scene.gaussian_model import GaussianModel


ADAPTER_TENSOR_KEYS = (
    "opacity_delta",
    "scale_delta",
    "dc_delta",
    "xyz_delta",
    "rest_delta",
    "rotation_delta",
)
NESTED_ADAPTER_BUNDLE_FORMAT = "paretogs_nested_budget_adapters"
SHELLWISE_ADAPTER_BUNDLE_FORMAT = "paretogs_shellwise_lowrank_adapters"


def _quantize_blockwise_int8(tensor: torch.Tensor, block_size: int) -> dict[str, Any]:
    """Quantize rows independently in small blocks with one scale per field channel."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    source = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    rows = int(source.shape[0])
    flat = source.reshape(rows, -1)
    block_count = math.ceil(rows / block_size)
    padded_rows = block_count * block_size
    if padded_rows != rows:
        flat = torch.nn.functional.pad(flat, (0, 0, 0, padded_rows - rows))
    blocks = flat.reshape(block_count, block_size, -1)
    scales = blocks.abs().amax(dim=1).div(127.0)
    scales = torch.where(scales > 0.0, scales, torch.ones_like(scales))
    quantized = torch.round(blocks / scales[:, None, :]).clamp_(-127, 127).to(torch.int8)
    quantized = quantized.reshape(padded_rows, -1)[:rows].contiguous()
    return {
        "encoding": "blockwise_symmetric_int8",
        "shape": tuple(source.shape),
        "block_size": int(block_size),
        "values": quantized,
        "scales": scales.to(torch.float16),
    }


def _decode_adapter_tensor(
    value: torch.Tensor | dict[str, Any] | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, dict) and value.get("encoding") == "lowrank_matrix_int8":
        shape = tuple(value.get("shape", ()))
        coefficients = _decode_adapter_tensor(value.get("coefficients"), device, dtype)
        basis = value.get("basis")
        if coefficients is None or not torch.is_tensor(basis) or not shape:
            raise ValueError("Malformed low-rank adapter tensor.")
        basis = basis.to(device=device, dtype=dtype)
        if coefficients.ndim != 2 or basis.ndim != 2 or coefficients.shape[1] != basis.shape[0]:
            raise ValueError("Low-rank adapter coefficient/basis shapes do not match.")
        if int(coefficients.shape[0]) != int(shape[0]) or int(basis.shape[1]) != math.prod(shape[1:]):
            raise ValueError("Low-rank adapter tensor shape does not match its metadata.")
        return torch.matmul(coefficients, basis).reshape(shape)
    if not isinstance(value, dict) or value.get("encoding") != "blockwise_symmetric_int8":
        raise ValueError("Unsupported adapter tensor encoding.")
    quantized = value.get("values")
    scales = value.get("scales")
    shape = tuple(value.get("shape", ()))
    block_size = int(value.get("block_size", 0))
    if not torch.is_tensor(quantized) or not torch.is_tensor(scales) or block_size <= 0 or not shape:
        raise ValueError("Malformed blockwise INT8 adapter tensor.")
    rows = int(shape[0])
    flat = quantized.to(device=device, dtype=dtype).reshape(rows, -1)
    expanded_scales = scales.to(device=device, dtype=dtype).repeat_interleave(block_size, dim=0)[:rows]
    return (flat * expanded_scales).reshape(shape)


def encode_lowrank_matrix_int8(
    tensor: torch.Tensor,
    rank: int,
    block_size: int = 64,
) -> dict[str, Any]:
    """Compress a row-major tensor using an optimal uncentered channel subspace."""
    source = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    rows = int(source.shape[0])
    flat = source.reshape(rows, -1)
    channels = int(flat.shape[1])
    rank = int(rank)
    if rank <= 0 or rank > channels:
        raise ValueError("rank must be in [1, {}].".format(channels))
    gram = torch.matmul(flat.transpose(0, 1), flat)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    basis = eigenvectors[:, -rank:].transpose(0, 1).contiguous()
    coefficients = torch.matmul(flat, basis.transpose(0, 1)).contiguous()
    total_energy = float(eigenvalues.clamp_min(0.0).sum().item())
    retained_energy = float(eigenvalues[-rank:].clamp_min(0.0).sum().item())
    return {
        "encoding": "lowrank_matrix_int8",
        "shape": tuple(source.shape),
        "rank": rank,
        "coefficients": _quantize_blockwise_int8(coefficients, block_size),
        "basis": basis.to(torch.float16),
        "explained_energy_ratio": retained_energy / total_energy if total_energy > 0.0 else 1.0,
    }


def quantize_adapter_payload(
    payload: dict[str, Any],
    precision: str,
) -> dict[str, Any]:
    """Store raw residuals in FP32 or FP16 for the paper encoding comparison."""
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Raw residual precision must be fp32 or fp16; use fuse for the atlas.")
    result = {key: value for key, value in payload.items() if key not in ADAPTER_TENSOR_KEYS}
    result["storage_precision"] = precision
    dtype = torch.float32 if precision == "fp32" else torch.float16
    for key in ADAPTER_TENSOR_KEYS:
        value = payload.get(key)
        if value is not None and not torch.is_tensor(value):
            raise ValueError("Adapter field {} must be a tensor or None.".format(key))
        result[key] = None if value is None else value.detach().to(device="cpu", dtype=dtype).contiguous()
    return result


def decode_adapter_payload(
    payload: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    """Decode adapter tensors to the base model's runtime device and dtype."""
    return {
        key: _decode_adapter_tensor(payload.get(key), device, dtype)
        for key in ADAPTER_TENSOR_KEYS
    }


def build_adapted_prefix(
    gaussians: GaussianModel,
    order: torch.Tensor,
    adapter_payload: dict[str, Any],
    identity_order: bool = False,
) -> GaussianModel:
    """Materialize one adapted prefix without changing the shared base model."""
    budget = int(adapter_payload.get("budget", 0))
    gaussian_count = int(gaussians.get_xyz.shape[0])
    if budget <= 0 or budget >= gaussian_count:
        raise ValueError("Adapter budget must be in [1, {}).".format(gaussian_count))
    if order.ndim != 1 or int(order.shape[0]) != gaussian_count:
        raise ValueError("Nested order does not match the base Gaussian count.")
    indices = order[:budget].to(device=gaussians.get_xyz.device, dtype=torch.long)
    decoded = decode_adapter_payload(
        adapter_payload,
        gaussians.get_xyz.device,
        gaussians.get_xyz.dtype,
    )
    expected_shapes = {
        "opacity_delta": (budget, 1),
        "scale_delta": (budget, 3),
        "dc_delta": (budget, 1, 3),
        "xyz_delta": (budget, 3),
        "rest_delta": (budget,) + tuple(gaussians._features_rest.shape[1:]),
        "rotation_delta": (budget, 4),
    }
    for key, value in decoded.items():
        if value is not None and tuple(value.shape) != expected_shapes[key]:
            raise ValueError(
                "Adapter field {} has shape {}, expected {}.".format(
                    key, tuple(value.shape), expected_shapes[key]
                )
            )
    if decoded["opacity_delta"] is None or decoded["scale_delta"] is None:
        raise ValueError("Every adapter requires opacity_delta and scale_delta.")

    proxy = GaussianModel(gaussians.max_sh_degree)
    proxy.active_sh_degree = gaussians.active_sh_degree
    def select(value: torch.Tensor, field: str) -> torch.Tensor:
        if identity_order:
            selected = value[:budget]
            return selected.clone() if decoded[field] is None else selected
        return torch.index_select(value, 0, indices)

    proxy._xyz = select(gaussians._xyz, "xyz_delta")
    proxy._features_dc = select(gaussians._features_dc, "dc_delta")
    proxy._features_rest = select(gaussians._features_rest, "rest_delta")
    proxy._scaling = select(gaussians._scaling, "scale_delta") + decoded["scale_delta"]
    proxy._rotation = select(gaussians._rotation, "rotation_delta")
    proxy._opacity = select(gaussians._opacity, "opacity_delta") + decoded["opacity_delta"]
    if decoded["xyz_delta"] is not None:
        proxy._xyz = proxy._xyz + decoded["xyz_delta"]
    if decoded["dc_delta"] is not None:
        proxy._features_dc = proxy._features_dc + decoded["dc_delta"]
    if decoded["rest_delta"] is not None:
        proxy._features_rest = proxy._features_rest + decoded["rest_delta"]
    if decoded["rotation_delta"] is not None:
        proxy._rotation = proxy._rotation + decoded["rotation_delta"]
    return proxy


def _decode_shellwise_anchor_field(
    compact: dict[str, Any],
    field: str,
    anchor: int,
) -> torch.Tensor | None:
    shells = compact.get("fields", {}).get(field)
    if not shells:
        return None
    parts = []
    for shell in shells:
        if int(shell["end"]) > int(anchor):
            break
        state_anchors = [int(value) for value in shell["state_anchors"]]
        if int(anchor) not in state_anchors:
            raise ValueError(
                "Compact adapter shell ending at {} has no state for budget {}.".format(
                    shell["end"], anchor
                )
            )
        state_index = state_anchors.index(int(anchor))
        coefficients = shell["coefficients"][state_index].float()
        basis = torch.stack([
            _decode_adapter_tensor(component, "cpu", torch.float32)
            for component in shell["basis"]
        ])
        if int(coefficients.numel()) != int(basis.shape[0]):
            raise ValueError("Compact adapter coefficient and basis ranks do not match.")
        flat = torch.matmul(coefficients, basis.reshape(basis.shape[0], -1))
        parts.append(flat.reshape(basis.shape[1:]))
    return torch.cat(parts, dim=0)


class _ShellwiseAdapterMapping(Mapping[int, dict[str, Any]]):
    """Decode only the compact budget state requested by the caller."""

    def __init__(self, compact: dict[str, Any]) -> None:
        self.compact = compact
        self.budgets = tuple(sorted(int(value) for value in compact["budgets"])[:-1])
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.max_cached_anchors = 2

    def __getitem__(self, budget: int) -> dict[str, Any]:
        budget = int(budget)
        if budget not in self.budgets:
            raise KeyError(budget)
        if budget not in self.cache:
            payload = {
                "mode": "shellwise_lowrank_adapter",
                "budget": budget,
                "storage_precision": "decoded_fp32",
            }
            for field in ADAPTER_TENSOR_KEYS:
                payload[field] = _decode_shellwise_anchor_field(
                    self.compact, field, budget
                )
            self.cache[budget] = payload
            while len(self.cache) > self.max_cached_anchors:
                self.cache.popitem(last=False)
        self.cache.move_to_end(budget)
        return self.cache[budget]

    def __iter__(self) -> Iterator[int]:
        return iter(self.budgets)

    def __len__(self) -> int:
        return len(self.budgets)


def load_adapter_bundle(path: str, gaussian_count: int) -> dict[str, Any]:
    """Load and validate a multi-budget adapter sidecar."""
    bundle = load_torch_asset(path, map_location="cpu")
    if isinstance(bundle, dict) and bundle.get("format") == SHELLWISE_ADAPTER_BUNDLE_FORMAT:
        if int(bundle.get("version", 0)) != 1:
            raise ValueError("Unsupported compact shell-wise adapter version.")
        budgets = sorted(set(int(value) for value in bundle.get("budgets", [])))
        if len(budgets) < 2 or budgets[-1] != int(gaussian_count):
            raise ValueError(
                "Compact adapter anchors must end at loaded model count {}.".format(
                    gaussian_count
                )
            )
        if not isinstance(bundle.get("fields"), dict) or not bundle["fields"]:
            raise ValueError("Compact adapter contains no parameter fields.")
        if bundle["fields"].get("rest_gain_delta") is not None:
            raise ValueError("Legacy rest_gain fields are outside the paper parameter representation.")
        result = dict(bundle)
        result["budgets"] = budgets
        result["base_gaussian_count"] = int(gaussian_count)
        result["storage_precision"] = "int8-basis/fp16-coefficients"
        result["adapters"] = _ShellwiseAdapterMapping(bundle)
        result["path"] = path
        return result
    if not isinstance(bundle, dict) or bundle.get("format") != NESTED_ADAPTER_BUNDLE_FORMAT:
        raise ValueError("Not a ParetoGS nested adapter bundle: {}".format(path))
    if int(bundle.get("version", 0)) != 1:
        raise ValueError("Unsupported nested adapter bundle version.")
    if int(bundle.get("base_gaussian_count", 0)) != int(gaussian_count):
        raise ValueError(
            "Adapter bundle base count {} does not match loaded model count {}.".format(
                bundle.get("base_gaussian_count"), int(gaussian_count)
            )
        )
    raw_adapters = bundle.get("adapters")
    if not isinstance(raw_adapters, dict) or not raw_adapters:
        raise ValueError("Adapter bundle contains no budget adapters.")
    adapters = {}
    for raw_budget, payload in raw_adapters.items():
        budget = int(raw_budget)
        if not isinstance(payload, dict) or int(payload.get("budget", 0)) != budget:
            raise ValueError("Malformed adapter payload for budget {}.".format(budget))
        if budget <= 0 or budget >= int(gaussian_count) or budget in adapters:
            raise ValueError("Invalid or duplicate adapter budget {}.".format(budget))
        if payload.get("rest_gain_delta") is not None:
            raise ValueError("Legacy rest_gain fields are outside the paper parameter representation.")
        adapters[budget] = payload
    result = dict(bundle)
    result["adapters"] = adapters
    result["path"] = path
    return result


def tensor_storage_bytes(value: Any) -> int:
    """Count tensor payload bytes recursively, excluding serialization metadata."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_storage_bytes(item) for item in value)
    return 0
