"""Ordering and validation helpers for ParetoGS Gaussian prefixes."""
from __future__ import annotations

from utils.artifact_utils import load_torch_asset

import os
from typing import Any

import torch


NESTED_ORDER_FILENAME = "nested_order.pt"


def validate_prefix_budgets(budgets: list[int], gaussian_count: int) -> list[int]:
    """Validate requested prefix sizes against the loaded Gaussian collection."""
    invalid = [budget for budget in budgets if budget <= 0 or budget > int(gaussian_count)]
    if invalid:
        raise ValueError(
            "Prefix budgets {} exceed the valid range [1, {}].".format(invalid, int(gaussian_count))
        )
    return sorted(set(int(budget) for budget in budgets))


def _decode_saved_order(payload: Any) -> torch.Tensor:
    if isinstance(payload, dict) and bool(payload.get("identity", False)):
        count = int(payload.get("count", 0))
        if count <= 0:
            raise ValueError("Identity nested order requires a positive count.")
        return torch.arange(count, dtype=torch.long)
    order = payload.get("order") if isinstance(payload, dict) else payload
    if not torch.is_tensor(order):
        raise ValueError("Nested order file does not contain a tensor.")
    return order.to(dtype=torch.long, device="cpu").contiguous()


def load_nested_order(path: str, gaussian_count: int, device: torch.device | str) -> torch.Tensor:
    """Load and validate a saved permutation."""
    order = _decode_saved_order(load_torch_asset(path, map_location="cpu"))
    if order.ndim != 1 or int(order.shape[0]) != int(gaussian_count):
        raise ValueError(
            "Nested order length {} does not match Gaussian count {}.".format(
                tuple(order.shape), int(gaussian_count)
            )
        )
    if int(torch.unique(order).numel()) != int(gaussian_count):
        raise ValueError("Nested order must be a permutation without duplicate indices.")
    if int(order.min().item()) != 0 or int(order.max().item()) != int(gaussian_count) - 1:
        raise ValueError("Nested order indices are outside the loaded Gaussian range.")
    return order.to(device=device)


def save_nested_order(
    path: str,
    order: torch.Tensor,
    strategy: str = "opacity_descending_fixed",
) -> None:
    """Persist one fixed permutation plus minimal provenance metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_order = order.detach().to(dtype=torch.long, device="cpu")
    identity = torch.equal(cpu_order, torch.arange(int(cpu_order.shape[0]), dtype=torch.long))
    payload = {
        "version": 1,
        "strategy": str(strategy),
        "count": int(cpu_order.shape[0]),
        "identity": bool(identity),
    }
    if not identity:
        payload["order"] = cpu_order
    torch.save(payload, path)


def opacity_descending_order(gaussians: Any) -> torch.Tensor:
    """Return a deterministic descending-opacity permutation."""
    opacity = gaussians.get_opacity.detach().flatten()
    return torch.argsort(opacity, descending=True, stable=True)


def resolve_nested_order(
    model_path: str,
    gaussians: Any,
    allow_create: bool,
    restored_order_path: str = "",
) -> tuple[torch.Tensor, str]:
    """Load the saved nested permutation, optionally creating an opacity fallback."""
    order_path = os.path.join(model_path, NESTED_ORDER_FILENAME)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    if os.path.exists(order_path):
        return load_nested_order(order_path, gaussian_count, gaussians.get_xyz.device), "saved"

    restored_order_path = str(restored_order_path or "").strip()
    if restored_order_path and os.path.exists(restored_order_path):
        order = load_nested_order(restored_order_path, gaussian_count, gaussians.get_xyz.device)
        if allow_create:
            save_nested_order(order_path, order, strategy="restored_fixed")
        return order, "restored"

    if not allow_create:
        raise FileNotFoundError(
            "Nested model is missing its saved permutation: {}.".format(order_path)
        )
    order = opacity_descending_order(gaussians)
    save_nested_order(order_path, order, strategy="opacity_descending_fixed")
    return order, "opacity_descending_created"
