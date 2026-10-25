"""Encode a shellwise residual atlas from fixed ParetoGS anchor states."""
from __future__ import annotations

from utils.artifact_utils import load_torch_asset

import argparse
import json
import os
import time
from typing import Any

import torch

from paretogs.adapters import (
    ADAPTER_TENSOR_KEYS,
    NESTED_ADAPTER_BUNDLE_FORMAT,
    _decode_adapter_tensor,
    _quantize_blockwise_int8,
    encode_lowrank_matrix_int8,
    tensor_storage_bytes,
)


class ShellwiseAdapter:
    """Factor fixed anchor states on CPU before quantizing their bases."""

    def __init__(
        self,
        bundle: dict[str, Any],
        sh_rank: int = 36,
    ) -> None:
        self.anchors = sorted(int(value) for value in bundle["budgets"])
        self.nonfull_anchors = self.anchors[:-1]
        self.shell_order_strategies = dict(bundle.get("shell_order_strategies", {}))
        self.rank = len(self.nonfull_anchors)
        self.sh_rank = int(sh_rank)
        source_adapters = {int(key): value for key, value in bundle["adapters"].items()}
        self.bases = {}
        self.sh_coefficients = {}
        self.sh_channel_bases = {}
        self.coefficients = {}
        self.shells: dict[str, list[dict[str, Any]]] = {}

        for field in ADAPTER_TENSOR_KEYS:
            first = source_adapters[self.nonfull_anchors[0]].get(field)
            if first is None:
                continue
            decoded = {
                anchor: _decode_adapter_tensor(
                    source_adapters[anchor].get(field), "cpu", torch.float32
                )
                for anchor in self.nonfull_anchors
            }
            field_shells = []
            start = 0
            for shell_index, end in enumerate(self.nonfull_anchors):
                state_anchors = [anchor for anchor in self.nonfull_anchors if anchor >= end]
                matrix = torch.stack([
                    decoded[anchor][start:end].reshape(-1)
                    for anchor in state_anchors
                ])
                gram = torch.matmul(matrix, matrix.transpose(0, 1))
                _eigenvalues, eigenvectors = torch.linalg.eigh(gram)
                shell_rank = len(state_anchors)
                coefficients = eigenvectors[:, -shell_rank:].contiguous()
                basis = torch.matmul(coefficients.transpose(0, 1), matrix).reshape(
                    (shell_rank, end - start) + tuple(decoded[state_anchors[0]].shape[1:])
                )
                name = "{}_{}".format(field, shell_index)
                shell_sh_rank = self.sh_rank
                if field == "rest_delta" and shell_sh_rank > 0:
                    spatial_coefficients = []
                    channel_bases = []
                    for component in basis:
                        flat_component = component.reshape(component.shape[0], -1)
                        channel_rank = min(shell_sh_rank, int(flat_component.shape[1]))
                        channel_gram = torch.matmul(
                            flat_component.transpose(0, 1), flat_component
                        )
                        _channel_values, channel_vectors = torch.linalg.eigh(channel_gram)
                        channel_basis = channel_vectors[:, -channel_rank:].transpose(0, 1)
                        spatial_coefficients.append(
                            torch.matmul(flat_component, channel_basis.transpose(0, 1))
                        )
                        channel_bases.append(channel_basis)
                    self.sh_coefficients[name] = torch.stack(spatial_coefficients)
                    self.sh_channel_bases[name] = torch.stack(channel_bases)
                else:
                    self.bases[name] = basis
                self.coefficients[name] = coefficients
                field_shells.append({
                    "name": name,
                    "start": start,
                    "end": end,
                    "state_anchors": state_anchors,
                })
                start = end
            self.shells[field] = field_shells


    def _basis_tensor(self, name: str) -> torch.Tensor:
        if name in self.sh_coefficients:
            coefficients = self.sh_coefficients[name]
            channel_basis = self.sh_channel_bases[name]
            flat = torch.matmul(coefficients, channel_basis)
            return flat.reshape(
                coefficients.shape[0], coefficients.shape[1], 15, 3
            )
        return self.bases[name]


    def compact_payload(
        self,
        block_size: int,
    ) -> dict[str, Any]:
        fields = {}
        for field, shells in self.shells.items():
            encoded_shells = []
            for shell_index, shell in enumerate(shells):
                basis = self._basis_tensor(shell["name"]).detach().cpu()
                shell_sh_rank = self.sh_rank
                if field == "rest_delta" and shell_sh_rank > 0:
                    encoded_basis = [
                        encode_lowrank_matrix_int8(component, shell_sh_rank, block_size)
                        for component in basis
                    ]
                else:
                    encoded_basis = [
                        _quantize_blockwise_int8(component, block_size)
                        for component in basis
                    ]
                encoded_shells.append({
                    "start": int(shell["start"]),
                    "end": int(shell["end"]),
                    "state_anchors": list(shell["state_anchors"]),
                    "basis": encoded_basis,
                    "coefficients": self.coefficients[shell["name"]].detach().cpu().half(),
                    "sh_basis_rank": shell_sh_rank if field == "rest_delta" else None,
                    "basis_storage_precision": "int8",
                })
            fields[field] = encoded_shells
        payload = {
            "format": "paretogs_shellwise_lowrank_adapters",
            "version": 1,
            "rank": self.rank,
            "sh_basis_rank": self.sh_rank or None,
            "budgets": self.anchors,
            "shell_order_strategies": self.shell_order_strategies,
            "fields": fields,
        }
        payload["tensor_storage_bytes"] = tensor_storage_bytes(payload)
        return payload


@torch.no_grad()
def encode_atlas(
    model_path: str,
    iteration: int,
    input_bundle_path: str,
    block_size: int,
    output_path: str,
    sh_rank: int = 36,
    evaluate: bool = False,
) -> dict[str, Any]:
    """Encode fixed states; render only when evaluation is explicitly requested."""
    model_path = os.path.abspath(os.fspath(model_path))
    input_bundle_path = os.path.abspath(os.fspath(input_bundle_path))
    output_path = os.path.abspath(output_path)
    source_bundle = load_torch_asset(input_bundle_path, map_location="cpu")
    if source_bundle.get("format") != NESTED_ADAPTER_BUNDLE_FORMAT:
        raise ValueError("Unsupported input adapter bundle: {}".format(input_bundle_path))
    source_bytes = int(source_bundle.get("tensor_storage_bytes") or tensor_storage_bytes(source_bundle))
    anchors = sorted(int(value) for value in source_bundle["budgets"])
    factorization_started = time.perf_counter()
    adapter = ShellwiseAdapter(source_bundle, sh_rank)
    factorization_seconds = time.perf_counter() - factorization_started
    serialization_started = time.perf_counter()
    compact = adapter.compact_payload(block_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(compact, output_path)
    serialization_seconds = time.perf_counter() - serialization_started
    evaluation_seconds = 0.0
    test_results = None
    if evaluate:
        from paretogs.evaluate import evaluate_interpolation

        evaluation_started = time.perf_counter()
        test_results = evaluate_interpolation(
            model_path, iteration, output_path, anchors, output_path + ".evaluation.json"
        )["results"]
        evaluation_seconds = time.perf_counter() - evaluation_started
    result = {
        "model_path": os.path.abspath(model_path),
        "iteration": int(iteration),
        "input_bundle": os.path.abspath(input_bundle_path),
        "rank": adapter.rank,
        "sh_basis_rank": (
            adapter.sh_rank or None
        ),
        "training_elapsed_seconds": 0.0,
        "factorization_elapsed_seconds": factorization_seconds,
        "serialization_elapsed_seconds": serialization_seconds,
        "evaluation_elapsed_seconds": evaluation_seconds,
        "compact_tensor_bytes": int(compact["tensor_storage_bytes"]),
        "source_tensor_bytes": source_bytes,
        "storage_ratio": compact["tensor_storage_bytes"] / max(
            source_bytes, 1
        ),
        "evaluation_performed": bool(evaluate),
        "test_serialized_anchors": test_results,
        "compact_path": os.path.abspath(output_path),
    }
    with open(output_path + ".json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--input_bundle", required=True)
    parser.add_argument("--block_size", default=64, type=int)
    parser.add_argument("--sh_rank", default=36, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate", action="store_true", help="Explicitly evaluate the serialized adapter after compression.")
    args = parser.parse_args()
    print(json.dumps(encode_atlas(
        args.model_path,
        args.iteration,
        args.input_bundle,
        args.block_size,
        args.output,
        args.sh_rank,
        args.evaluate,
    ), indent=2))
