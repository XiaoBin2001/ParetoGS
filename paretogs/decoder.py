"""Shared differentiable budget decoding, ordering and trajectory contracts."""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from paretogs.adapters import ADAPTER_TENSOR_KEYS, _decode_adapter_tensor, build_adapted_prefix, load_adapter_bundle
from utils.artifact_utils import load_json_asset
from paretogs.prefix import load_nested_order


INTERPOLATION_CURVE_FORMAT = "paretogs_monotone_interpolation_curve"
DECODER_VARIANTS = ("full", "fixed_shared_state", "no_opacity_gate", "linear_progress", "full_parameters")
DEFAULT_COMMON_POWER = 4.0
DEFAULT_SHELL_GATE_POWER = 2.0
class EndpointCache(dict):
    """Keep complete endpoint fields for a bounded number of recently used anchors."""

    def __init__(self, max_anchors: int | None = 2) -> None:
        super().__init__()
        if max_anchors is not None and max_anchors < 2:
            raise ValueError("Endpoint caching requires at least two anchors.")
        self.max_anchors = max_anchors
        self._anchors: OrderedDict[int, None] = OrderedDict()

    def __getitem__(self, key: tuple[int, str]) -> Any:
        value = super().__getitem__(key)
        self._anchors.move_to_end(key[0])
        return value

    def __setitem__(self, key: tuple[int, str], value: Any) -> None:
        self._anchors[key[0]] = None
        self._anchors.move_to_end(key[0])
        super().__setitem__(key, value)
        if self.max_anchors is not None:
            while len(self._anchors) > self.max_anchors:
                anchor, _ = self._anchors.popitem(last=False)
                for old_key in [item for item in self if item[0] == anchor]:
                    super().__delitem__(old_key)

    def clear(self) -> None:
        super().clear()
        self._anchors.clear()


def _interval_key(lower: int, upper: int) -> str:
    return "{}:{}".format(int(lower), int(upper))


def load_interpolation_curve(path: str, anchors: list[int]) -> dict[str, Any]:
    if not path:
        return {}
    path = os.fspath(path)
    curve = load_json_asset(path)
    if curve.get("format") != INTERPOLATION_CURVE_FORMAT or int(curve.get("version", 0)) != 1:
        raise ValueError("Unsupported interpolation curve: {}".format(path))
    if [int(value) for value in curve.get("anchors", [])] != [int(value) for value in anchors]:
        raise ValueError("Interpolation curve anchors do not match the adapter bundle.")
    intervals = curve.get("intervals")
    if not isinstance(intervals, dict):
        raise ValueError("Interpolation curve has no interval parameters.")
    for lower, upper in zip(anchors[:-1], anchors[1:]):
        parameters = intervals.get(_interval_key(lower, upper))
        if not isinstance(parameters, dict):
            raise ValueError("Interpolation curve is missing interval {}:{}".format(lower, upper))
        if float(parameters.get("common_power", 0.0)) <= 0.0:
            raise ValueError("common_power must be positive for endpoint continuity.")
        if float(parameters.get("shell_gate_power", 0.0)) <= 0.0:
            raise ValueError("shell_gate_power must be positive for endpoint continuity.")
    curve["path"] = os.path.abspath(path)
    return curve


def interpolation_powers(
    curve: dict[str, Any],
    lower: int,
    upper: int,
) -> tuple[float, float]:
    if not curve:
        return DEFAULT_COMMON_POWER, DEFAULT_SHELL_GATE_POWER
    parameters = curve["intervals"][_interval_key(lower, upper)]
    return float(parameters["common_power"]), float(parameters["shell_gate_power"])


def _bracket(anchors: list[int], budget: int) -> tuple[int, int]:
    for lower, upper in zip(anchors[:-1], anchors[1:]):
        if lower < budget < upper:
            return lower, upper
    raise ValueError(
        "Budget {} must lie strictly between trained anchors {}.".format(budget, anchors)
    )


def _resolve_active_anchors(
    bundle_anchors: list[int],
    requested_anchors: list[int] | None,
    gaussian_count: int,
) -> list[int]:
    if requested_anchors is None:
        return bundle_anchors
    anchors = sorted(set(int(value) for value in requested_anchors))
    missing = [anchor for anchor in anchors if anchor not in bundle_anchors]
    if len(anchors) < 2 or missing:
        raise ValueError(
            "Active anchors must select at least two bundle anchors; missing={}.".format(missing)
        )
    return anchors


def _decode_endpoint_field(
    gaussians: Any,
    bundle: dict[str, Any],
    anchor: int,
    key: str,
    cache: dict[tuple[int, str], torch.Tensor | None],
) -> torch.Tensor | None:
    cache_key = (int(anchor), key)
    if cache_key in cache:
        return cache[cache_key]
    full_count = int(gaussians.get_xyz.shape[0])
    if anchor == full_count:
        shapes = {
            "opacity_delta": (anchor, 1),
            "scale_delta": (anchor, 3),
            "dc_delta": (anchor, 1, 3),
            "xyz_delta": (anchor, 3),
            "rest_delta": tuple(gaussians._features_rest.shape),
            "rotation_delta": (anchor, 4),
        }
        # A broadcast zero preserves the tensor contract without a full-model allocation.
        result = torch.zeros(
            (), device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype
        ).expand(shapes[key])
    else:
        result = _decode_adapter_tensor(
            bundle["adapters"][anchor].get(key),
            gaussians.get_xyz.device,
            gaussians.get_xyz.dtype,
        )
    cache[cache_key] = result
    return result


def _strategy_position_order(
    strategy_payload: dict[str, Any],
    start: int,
    upper: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(start, upper, device=device)
    saved_order = strategy_payload.get("position_order")
    if saved_order is None:
        return positions
    if not torch.is_tensor(saved_order):
        raise ValueError("Explicit shell position_order must be a tensor.")
    saved_order = saved_order.to(device=device, dtype=torch.long).flatten()
    if (
        int(saved_order.shape[0]) != upper - start
        or int(torch.unique(saved_order).numel()) != upper - start
        or int(saved_order.min().item()) != start
        or int(saved_order.max().item()) != upper - 1
    ):
        raise ValueError(
            "Explicit shell position_order is not a permutation of [{}, {}).".format(
                start, upper
            )
        )
    return saved_order


def _composed_trace_position_order(
    shell_strategies: dict[str, Any],
    bundle_anchors: list[int],
    start: int,
    upper: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Preserve fine trace order when deployment skips stored adapter anchors."""
    boundaries = [anchor for anchor in bundle_anchors if start < anchor <= upper]
    if not boundaries or boundaries[-1] != upper:
        return None
    parts = []
    lower = start
    for boundary in boundaries:
        strategy_payload = shell_strategies.get(_interval_key(lower, boundary), {})
        if strategy_payload.get("strategy") != "reverse_gns_prune_trace":
            return None
        parts.append(_strategy_position_order(
            strategy_payload, lower, boundary, device
        ))
        lower = boundary
    return torch.cat(parts)


def _band_opacity_permutation(
    gaussians: Any,
    order: torch.Tensor,
    bundle: dict[str, Any],
    anchors: list[int],
    cache: dict[tuple[int, str], torch.Tensor | None],
) -> torch.Tensor:
    """Order each trained shell by its trace when available, otherwise by opacity."""
    parts = []
    start = 0
    full_count = int(gaussians.get_xyz.shape[0])
    shell_strategies = bundle.get("shell_order_strategies", {})
    if not isinstance(shell_strategies, dict):
        raise ValueError("Adapter shell_order_strategies must be a mapping.")
    bundle_anchors = sorted(set(int(value) for value in bundle.get("budgets", [])))
    for upper in anchors:
        positions = torch.arange(start, upper, device=order.device)
        strategy_payload = shell_strategies.get(_interval_key(start, upper), {})
        strategy = strategy_payload.get("strategy")
        if strategy == "reverse_gns_prune_trace":
            parts.append(_strategy_position_order(
                strategy_payload, start, upper, order.device
            ))
            start = upper
            continue
        if strategy is None:
            composed = _composed_trace_position_order(
                shell_strategies, bundle_anchors, start, upper, order.device
            )
            if composed is not None:
                parts.append(composed)
                start = upper
                continue
        if strategy is not None:
            raise ValueError(
                "Unsupported shell ordering strategy {} for {}:{}.".format(
                    strategy, start, upper
                )
            )
        physical_indices = order[start:upper]
        raw_opacity = torch.index_select(
            gaussians._opacity, 0, physical_indices
        ).flatten()
        if upper != full_count:
            opacity_delta = _decode_endpoint_field(
                gaussians, bundle, upper, "opacity_delta", cache
            )
            if opacity_delta is None:
                raise ValueError("Anchor {} has no opacity delta.".format(upper))
            raw_opacity = raw_opacity + opacity_delta[start:upper].flatten()
        parts.append(positions[torch.argsort(raw_opacity, descending=True, stable=True)])
        start = upper
    if start < full_count:
        parts.append(torch.arange(start, full_count, device=order.device))
    permutation = torch.cat(parts)
    if int(torch.unique(permutation).numel()) != full_count:
        raise ValueError("Shell opacity ordering is not a permutation.")
    return permutation


def _interpolated_payload(
    gaussians: Any,
    order: torch.Tensor,
    position_permutation: torch.Tensor,
    bundle: dict[str, Any],
    budget: int,
    lower: int,
    upper: int,
    cache: dict[tuple[int, str], torch.Tensor | None],
    common_power: float = DEFAULT_COMMON_POWER,
    shell_gate_power: float = DEFAULT_SHELL_GATE_POWER,
) -> tuple[dict[str, Any], float, float]:
    """Interpolate common rows and fade in a shell with endpoint-safe powers."""
    if float(common_power) <= 0.0 or float(shell_gate_power) <= 0.0:
        raise ValueError("Interpolation powers must be positive.")
    linear_progress = (budget - lower) / (upper - lower)
    common_alpha = linear_progress ** float(common_power)
    shell_gate = linear_progress ** float(shell_gate_power)
    payload = _interpolated_payload_from_weights(
        gaussians,
        order,
        position_permutation,
        bundle,
        budget,
        lower,
        upper,
        cache,
        common_alpha,
        shell_gate,
    )
    return payload, common_alpha, linear_progress


def _interpolated_payload_from_weights(
    gaussians: Any,
    order: torch.Tensor,
    position_permutation: torch.Tensor,
    bundle: dict[str, Any],
    budget: int,
    lower: int,
    upper: int,
    cache: dict[tuple[int, str], torch.Tensor | None],
    common_alpha: float | torch.Tensor | Mapping[str, float | torch.Tensor],
    shell_gate: float | torch.Tensor,
) -> dict[str, Any]:
    """Build an intermediate state from explicit differentiable trajectory weights."""
    lower_positions = position_permutation[:lower]
    upper_positions = position_permutation[:budget]
    payload: dict[str, Any] = {
        "mode": "smooth_interpolated_density_conditioned_deltas",
        "budget": int(budget),
    }
    for key in ADAPTER_TENSOR_KEYS:
        lower_delta = _decode_endpoint_field(gaussians, bundle, lower, key, cache)
        upper_delta = _decode_endpoint_field(gaussians, bundle, upper, key, cache)
        if lower_delta is None and upper_delta is None:
            payload[key] = None
            continue
        if lower_delta is None or upper_delta is None:
            raise ValueError("Adapter field {} is inconsistent across anchors.".format(key))
        lower_delta = torch.index_select(lower_delta, 0, lower_positions)
        if upper == int(gaussians.get_xyz.shape[0]):
            upper_delta = upper_delta[:budget]
        else:
            upper_delta = torch.index_select(upper_delta, 0, upper_positions)
        field_alpha = (
            common_alpha[key] if isinstance(common_alpha, Mapping) else common_alpha
        )
        common = torch.lerp(lower_delta, upper_delta[:lower], field_alpha)
        tail = upper_delta[lower:budget]
        if key == "opacity_delta":
            tail_positions = position_permutation[lower:budget]
            tail_indices = order[tail_positions]
            base_opacity = torch.index_select(gaussians._opacity, 0, tail_indices)
            activated = (torch.sigmoid(base_opacity + tail) * shell_gate).clamp(
                1e-6, 1.0 - 1e-6
            )
            tail = torch.logit(activated) - base_opacity
        payload[key] = torch.cat((common, tail), dim=0).contiguous()
    return payload


def _evaluation_order_hash(order: torch.Tensor) -> str:
    values = order.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


@torch.no_grad()
def materialize_budget(
    gaussians: Any,
    bundle: dict[str, Any],
    stored_order: torch.Tensor,
    evaluation_order: torch.Tensor,
    permutation: torch.Tensor,
    anchors: list[int],
    curve: dict[str, Any],
    budget: int,
    endpoint_cache: dict,
    identity_order: bool = False,
    variant: str = "full",
) -> tuple[Any, dict[str, Any]]:
    """Materialize a budget using validated ordering and trajectory context."""
    gaussian_count = int(gaussians.get_xyz.shape[0])
    budget = int(budget)
    if not anchors[0] <= budget <= anchors[-1]:
        raise ValueError("Budget must lie in [{}, {}].".format(anchors[0], anchors[-1]))
    if budget == gaussian_count:
        return gaussians, {"mode": "full_anchor", "budget": budget}
    if budget in anchors:
        payload = {"mode": "decoded_anchor", "budget": budget}
        payload.update({
            key: _decode_endpoint_field(
                gaussians, bundle, budget, key, endpoint_cache
            )
            for key in ADAPTER_TENSOR_KEYS
        })
        proxy = build_adapted_prefix(
            gaussians, stored_order, payload,
            identity_order=identity_order,
        )
        return proxy, {"mode": "trained_anchor", "budget": budget}

    lower, upper = _bracket(anchors, budget)
    common_power, shell_power = interpolation_powers(curve, lower, upper)
    if variant == "linear_progress":
        common_power, shell_power = 1.0, 1.0
    progress = (budget - lower) / (upper - lower)
    alpha = 0.0 if variant == "fixed_shared_state" else progress ** common_power
    gate = 1.0 if variant == "no_opacity_gate" else progress ** shell_power
    payload = _interpolated_payload_from_weights(
        gaussians, stored_order, permutation, bundle,
        budget, lower, upper, endpoint_cache, alpha, gate,
    )
    if variant == "full_parameters":
        # Same member IDs, using the original Full parameters without residuals or gating.
        payload = {"budget": budget, **{
            key: None if value is None else torch.zeros_like(value)
            for key, value in payload.items() if key in ADAPTER_TENSOR_KEYS
        }}
    metadata = {
        "mode": "scalar_curve", "variant": variant, "linear_progress": progress,
        "common_power": common_power, "common_alpha": alpha,
        "shell_gate_power": shell_power, "shell_opacity_gate": gate,
    }
    proxy = build_adapted_prefix(gaussians, evaluation_order, payload)
    return proxy, {"budget": budget, "interval": [lower, upper], **metadata}


class BudgetDecoder:
    """Shared inference path for paper anchors, scalar clocks and component ablations."""

    @staticmethod
    def from_assets(
        gaussians: Any,
        adapter_path: str,
        order_path: str,
        curve_path: str = "",
        active_anchors: list[int] | None = None,
        cache_anchors: int | None = 2,
    ) -> BudgetDecoder:
        """Bind serialized assets to an already loaded model without loading cameras."""
        count = int(gaussians.get_xyz.shape[0])
        bundle = load_adapter_bundle(adapter_path, count)
        anchors = _resolve_active_anchors(sorted(bundle["budgets"]), active_anchors, count)
        order = load_nested_order(order_path, count, gaussians.get_xyz.device)
        return BudgetDecoder(
            gaussians, bundle, order, anchors,
            load_interpolation_curve(curve_path, anchors),
            cache_anchors=cache_anchors,
        )

    def __init__(
        self,
        gaussians: Any,
        bundle: dict[str, Any],
        stored_order: torch.Tensor,
        anchors: list[int] | None = None,
        curve: dict[str, Any] | None = None,
        cache_anchors: int | None = 2,
        endpoint_cache: dict | None = None,
    ) -> None:
        self.gaussians = gaussians
        self.gaussian_count = int(gaussians.get_xyz.shape[0])
        self.bundle = bundle
        bundle_anchors = sorted(set(int(value) for value in bundle["budgets"]))
        if len(bundle_anchors) < 2 or bundle_anchors[0] <= 0 or bundle_anchors[-1] != self.gaussian_count:
            raise ValueError("Adapter anchors must be positive and end at the full model.")
        self.anchors = _resolve_active_anchors(bundle_anchors, anchors, self.gaussian_count)
        self.curve = curve or {}
        self.stored_order = stored_order
        if stored_order.ndim != 1 or int(stored_order.numel()) != self.gaussian_count:
            raise ValueError("Stored order does not match the model.")
        self.endpoint_cache = endpoint_cache if endpoint_cache is not None else EndpointCache(cache_anchors)
        adapter_mapping = bundle["adapters"]
        if hasattr(adapter_mapping, "max_cached_anchors"):
            adapter_mapping.max_cached_anchors = cache_anchors or len(bundle_anchors)
        with torch.no_grad():
            self.permutation = _band_opacity_permutation(
                gaussians, stored_order, bundle, bundle_anchors, self.endpoint_cache
            )
            self.evaluation_order = stored_order[self.permutation]
            self.identity_order = torch.equal(
                stored_order,
                torch.arange(self.gaussian_count, device=stored_order.device),
            )
    def decode(self, budget: int, variant: str = "full") -> tuple[Any, dict[str, Any]]:
        if variant not in DECODER_VARIANTS:
            raise ValueError("Unknown decoder variant: {}".format(variant))
        return materialize_budget(
            self.gaussians, self.bundle, self.stored_order, self.evaluation_order,
            self.permutation, self.anchors, self.curve,
            budget, self.endpoint_cache, self.identity_order, variant,
        )

    def clear_cache(self) -> None:
        self.endpoint_cache.clear()
        adapters = self.bundle["adapters"]
        if hasattr(adapters, "cache"):
            adapters.cache.clear()
