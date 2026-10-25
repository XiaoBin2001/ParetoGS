"""
Optimizer and checkpoint helpers for GaussianModel.

The Gaussian model changes tensor sizes during densification and pruning. These
helpers build optimizer parameter groups, restore matching optimizer state from
checkpoints, and replace optimizer-owned tensors whenever the model structure
changes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.general_utils import get_expon_lr_func

try:
    from utils.rasterizer_backend import GaussianAdam
except Exception:
    GaussianAdam = None


class GaussianModelOptimizerMixin:
    """
        Mixin that owns optimizer setup, restore, and tensor replacement logic.
    """

    def _ensure_optimizer_parameter_layouts(self) -> None:
        """
            Make optimizer-updated parameters contiguous before optimizer creation.
        """
        parameter_names = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ]
        for parameter_name in parameter_names:
            parameter = getattr(self, parameter_name)
            if not isinstance(parameter, nn.Parameter):
                continue
            if parameter.is_contiguous():
                continue
            setattr(
                self,
                parameter_name,
                nn.Parameter(parameter.detach().contiguous().requires_grad_(parameter.requires_grad)),
            )

    def _get_optimizer_step_value(self, stored_state: dict[str, object] | None) -> int:
        """
            Read an optimizer step counter whether it is stored as a tensor or int.
        """
        if stored_state is None:
            return 0
        step_value = stored_state.get("step")
        if step_value is None:
            return 0
        if torch.is_tensor(step_value):
            return int(step_value.item())
        return int(step_value)

    def _build_optimizer_param_groups(self, training_args: Any) -> list[dict[str, object]]:
        """
            Build one optimizer parameter group per Gaussian tensor family.

            Separate groups keep learning rates independent for xyz, color,
            opacity, scaling, and rotation.
        """
        lr_rate = float(getattr(training_args, "lr_rate", 1.0))
        param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * lr_rate, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * lr_rate, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr * lr_rate / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * lr_rate, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * lr_rate, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * lr_rate, "name": "rotation"}
        ]
        return param_groups

    def _configure_optimizers(self, training_args: Any) -> None:
        """
            Recreate optimizers and learning-rate schedules from current tensors.

            This is called during initial setup and checkpoint restoration.
        """
        self._ensure_optimizer_parameter_layouts()
        param_groups = self._build_optimizer_param_groups(training_args)
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "ours_adam":
            if GaussianAdam is None:
                raise RuntimeError("GaussianAdam is not available, please install the correct rasterizer build.")
            self.optimizer = GaussianAdam(param_groups, lr=0.0, eps=1e-15, max_steps=int(getattr(training_args, "iterations", 30000)))
        else:
            raise ValueError("Unsupported optimizer_type: {}".format(self.optimizer_type))

        lr_rate = float(getattr(training_args, "lr_rate", 1.0))
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * lr_rate,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * lr_rate,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
    def _restore_optimizer_state(self, opt_dict: dict[str, object]) -> None:
        """
            Restore checkpoint optimizer state by parameter group name.

            Tensor states are copied only when their shape still matches the
            current parameter, which avoids invalid momentum after SH expansion
            or other structure changes.
        """
        current_state = self.optimizer.state_dict()
        checkpoint_groups = opt_dict.get("param_groups", [])
        checkpoint_state = opt_dict.get("state", {})
        checkpoint_groups_by_name = {group.get("name"): group for group in checkpoint_groups}
        current_tensor_by_name = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }
        restored_state: dict[int, dict[str, torch.Tensor]] = {}
        for current_group in current_state["param_groups"]:
            group_name = current_group.get("name")
            checkpoint_group = checkpoint_groups_by_name.get(group_name)
            if checkpoint_group is None:
                continue
            for field_name, field_value in checkpoint_group.items():
                if field_name == "params":
                    continue
                current_group[field_name] = field_value
            checkpoint_param_id = checkpoint_group["params"][0]
            current_param_id = current_group["params"][0]
            state_entry = checkpoint_state.get(checkpoint_param_id)
            if state_entry is None:
                continue
            exp_avg = state_entry.get("exp_avg")
            if exp_avg is not None and tuple(exp_avg.shape) != tuple(current_tensor_by_name[group_name].shape):
                continue
            normalized_state_entry: dict[str, torch.Tensor | object] = {}
            for state_name, state_value in state_entry.items():
                if not torch.is_tensor(state_value):
                    normalized_state_entry[state_name] = state_value
                    continue
                normalized_state_entry[state_name] = state_value.contiguous()
            restored_state[current_param_id] = normalized_state_entry

        current_state["state"] = restored_state
        self.optimizer.load_state_dict(current_state)

    def restore(self, model_args: tuple[Any, ...], training_args: Any) -> None:
        """
            Restore model tensors, gradient accumulators, and optimizer state.
        """
        if len(model_args) != 15:
            raise ValueError("Only the 15-field 3DGS checkpoint format is supported; got {} fields.".format(len(model_args)))
        (
            self.active_sh_degree,
            self.max_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            _discarded_aux,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            source_metadata,
        ) = model_args
        model_device = torch.device("cuda")
        for parameter_name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ):
            checkpoint_tensor = getattr(self, parameter_name)
            setattr(
                self,
                parameter_name,
                nn.Parameter(
                    checkpoint_tensor.detach().to(device=model_device).contiguous().requires_grad_(True)
                ),
            )
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum.to(device=model_device)
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs.to(device=model_device)
        self.denom = denom.to(device=model_device)
        self._restore_optimizer_state(opt_dict)
        self.restore_source_metadata(source_metadata)

    def training_setup(self, training_args: Any) -> None:
        """
            Initialize accumulators and optimizers for the current Gaussian tensors.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._configure_optimizers(training_args)

    def update_learning_rate(self, iteration: int) -> float | None:
        """
            Update the xyz learning rate for the current iteration.
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> dict[str, nn.Parameter]:
        """
            Replace one optimizer parameter tensor and reset its momentum buffers.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                tensor = tensor.contiguous()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @torch.no_grad()
    def reset_optimizer_state_rows(self, indices: torch.Tensor) -> None:
        """Clear per-Gaussian optimizer momentum for rows whose parameters changed in place."""
        if indices.numel() == 0:
            return
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is None:
                continue
            for state_name, state_value in stored_state.items():
                if self._is_per_gaussian_optimizer_state(state_name, state_value):
                    state_value[indices] = 0

    @staticmethod
    def _structure_tensor_nbytes(tensor: torch.Tensor, first_dim: int) -> int:
        """Return the bytes required by `tensor` when its first dimension becomes `first_dim`."""
        row_elements = tensor.numel() // max(int(tensor.shape[0]), 1)
        return int(first_dim) * int(row_elements) * int(tensor.element_size())

    @staticmethod
    def _is_per_gaussian_optimizer_state(state_name: str, state_value: object) -> bool:
        """Identify optimizer buffers that follow the Gaussian parameter shape."""
        return state_name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and torch.is_tensor(state_value)

    @staticmethod
    def _copy_selected_cuda(
        destination: torch.Tensor,
        source: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> None:
        """Gather retained rows directly into an already allocated CUDA destination."""
        if keep_indices.numel() == 0:
            return
        torch.index_select(source, 0, keep_indices, out=destination[: keep_indices.shape[0]])

    @staticmethod
    def _copy_selected_cpu_chunked(
        destination: torch.Tensor,
        source: torch.Tensor,
        keep_indices: torch.Tensor,
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Gather retained CUDA rows into CPU memory without a full-size CUDA temporary."""
        if keep_indices.numel() == 0:
            return
        row_bytes = max((source.numel() // max(int(source.shape[0]), 1)) * source.element_size(), 1)
        rows_per_chunk = max(int(chunk_bytes) // int(row_bytes), 1)
        for start in range(0, int(keep_indices.shape[0]), rows_per_chunk):
            end = min(start + rows_per_chunk, int(keep_indices.shape[0]))
            selected_chunk = torch.index_select(source, 0, keep_indices[start:end])
            destination[start:end].copy_(selected_chunk, non_blocking=False)
            del selected_chunk

    @staticmethod
    def _copy_cuda_tensor_to_cpu_chunked(
        destination: torch.Tensor,
        source: torch.Tensor,
        chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Copy a CUDA tensor to CPU in bounded first-dimension chunks."""
        if source.shape[0] == 0:
            return
        row_bytes = max((source.numel() // int(source.shape[0])) * source.element_size(), 1)
        rows_per_chunk = max(int(chunk_bytes) // int(row_bytes), 1)
        for start in range(0, int(source.shape[0]), rows_per_chunk):
            end = min(start + rows_per_chunk, int(source.shape[0]))
            destination[start:end].copy_(source[start:end], non_blocking=False)

    def _estimate_structure_destination_bytes(
        self,
        final_count: int,
        source_parameters: dict[str, torch.Tensor],
        source_states: dict[str, dict[str, object] | None],
        reset_stats: bool,
    ) -> int:
        """Estimate bytes for a complete GPU-side structure destination."""
        required_bytes = 0
        for group_name, parameter in source_parameters.items():
            required_bytes += self._structure_tensor_nbytes(parameter, final_count)
            stored_state = source_states[group_name]
            if stored_state is None:
                continue
            for state_name, state_value in stored_state.items():
                if self._is_per_gaussian_optimizer_state(state_name, state_value):
                    required_bytes += self._structure_tensor_nbytes(state_value, final_count)
                elif state_name == "last_seen_step" and torch.is_tensor(state_value):
                    required_bytes += int(final_count) * int(state_value.element_size())
        if reset_stats:
            required_bytes += int(final_count) * 3 * torch.empty((), dtype=torch.float32).element_size()
        else:
            required_bytes += sum(
                self._structure_tensor_nbytes(stat_tensor, final_count)
                for stat_tensor in (self.xyz_gradient_accum, self.xyz_gradient_accum_abs, self.denom)
            )
        return required_bytes

    def _build_structure_cuda(
        self,
        keep_indices: torch.Tensor,
        extension_dicts: list[dict[str, torch.Tensor]],
        source_parameters: dict[str, torch.Tensor],
        source_states: dict[str, dict[str, object] | None],
        final_count: int,
        reset_stats: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object] | None], tuple[torch.Tensor, ...]]:
        """Build a complete replacement structure on CUDA while the old structure remains valid."""
        extension_counts = [int(extension["xyz"].shape[0]) for extension in extension_dicts]
        ordered_names = sorted(
            source_parameters,
            key=lambda name: source_parameters[name].numel(),
            reverse=True,
        )
        new_parameters: dict[str, torch.Tensor] = {}
        new_states: dict[str, dict[str, object] | None] = {}

        for group_name in ordered_names:
            source_parameter = source_parameters[group_name]
            destination_shape = (int(final_count),) + tuple(source_parameter.shape[1:])
            destination = torch.empty(destination_shape, dtype=source_parameter.dtype, device=source_parameter.device)
            self._copy_selected_cuda(destination, source_parameter, keep_indices)
            offset = int(keep_indices.shape[0])
            for extension, extension_count in zip(extension_dicts, extension_counts):
                destination[offset : offset + extension_count].copy_(extension[group_name])
                offset += extension_count
            new_parameters[group_name] = destination

            source_state = source_states[group_name]
            if source_state is None:
                new_states[group_name] = None
                continue
            destination_state: dict[str, object] = {}
            current_step = self._get_optimizer_step_value(source_state)
            for state_name, state_value in source_state.items():
                if self._is_per_gaussian_optimizer_state(state_name, state_value):
                    state_shape = (int(final_count),) + tuple(state_value.shape[1:])
                    state_destination = torch.empty(
                        state_shape,
                        dtype=state_value.dtype,
                        device=state_value.device,
                    )
                    self._copy_selected_cuda(state_destination, state_value, keep_indices)
                    state_destination[int(keep_indices.shape[0]) :].zero_()
                    destination_state[state_name] = state_destination
                elif state_name == "last_seen_step" and torch.is_tensor(state_value):
                    state_destination = torch.empty(
                        (int(final_count),),
                        dtype=state_value.dtype,
                        device=state_value.device,
                    )
                    self._copy_selected_cuda(state_destination, state_value, keep_indices)
                    state_destination[int(keep_indices.shape[0]) :].fill_(current_step)
                    destination_state[state_name] = state_destination
                else:
                    destination_state[state_name] = state_value
            new_states[group_name] = destination_state

        if reset_stats:
            new_stats = tuple(
                torch.zeros((int(final_count), 1), dtype=torch.float32, device=self.get_xyz.device)
                for _ in range(3)
            )
        else:
            new_stats_list = []
            for source_stat in (self.xyz_gradient_accum, self.xyz_gradient_accum_abs, self.denom):
                destination_shape = (int(final_count),) + tuple(source_stat.shape[1:])
                destination = torch.zeros(destination_shape, dtype=source_stat.dtype, device=source_stat.device)
                self._copy_selected_cuda(destination, source_stat, keep_indices)
                new_stats_list.append(destination)
            new_stats = tuple(new_stats_list)
        return new_parameters, new_states, new_stats

    def _build_structure_cpu(
        self,
        keep_indices: torch.Tensor,
        extension_dicts: list[dict[str, torch.Tensor]],
        source_parameters: dict[str, torch.Tensor],
        source_states: dict[str, dict[str, object] | None],
        final_count: int,
        reset_stats: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object] | None], tuple[torch.Tensor, ...] | None]:
        """Build a complete replacement structure in CPU memory using bounded CUDA temporaries."""
        extension_counts = [int(extension["xyz"].shape[0]) for extension in extension_dicts]
        new_parameters: dict[str, torch.Tensor] = {}
        new_states: dict[str, dict[str, object] | None] = {}

        for group_name, source_parameter in source_parameters.items():
            destination_shape = (int(final_count),) + tuple(source_parameter.shape[1:])
            destination = torch.empty(destination_shape, dtype=source_parameter.dtype, device="cpu")
            self._copy_selected_cpu_chunked(destination, source_parameter, keep_indices)
            offset = int(keep_indices.shape[0])
            for extension, extension_count in zip(extension_dicts, extension_counts):
                self._copy_cuda_tensor_to_cpu_chunked(
                    destination[offset : offset + extension_count],
                    extension[group_name],
                )
                offset += extension_count
            new_parameters[group_name] = destination

            source_state = source_states[group_name]
            if source_state is None:
                new_states[group_name] = None
                continue
            destination_state: dict[str, object] = {}
            current_step = self._get_optimizer_step_value(source_state)
            for state_name, state_value in source_state.items():
                if self._is_per_gaussian_optimizer_state(state_name, state_value):
                    state_shape = (int(final_count),) + tuple(state_value.shape[1:])
                    state_destination = torch.empty(state_shape, dtype=state_value.dtype, device="cpu")
                    self._copy_selected_cpu_chunked(state_destination, state_value, keep_indices)
                    state_destination[int(keep_indices.shape[0]) :].zero_()
                    destination_state[state_name] = state_destination
                elif state_name == "last_seen_step" and torch.is_tensor(state_value):
                    state_destination = torch.empty((int(final_count),), dtype=state_value.dtype, device="cpu")
                    self._copy_selected_cpu_chunked(state_destination, state_value, keep_indices)
                    state_destination[int(keep_indices.shape[0]) :].fill_(current_step)
                    destination_state[state_name] = state_destination
                else:
                    destination_state[state_name] = state_value
            new_states[group_name] = destination_state

        if reset_stats:
            new_stats = None
        else:
            cpu_stats = []
            for source_stat in (self.xyz_gradient_accum, self.xyz_gradient_accum_abs, self.denom):
                destination_shape = (int(final_count),) + tuple(source_stat.shape[1:])
                destination = torch.zeros(destination_shape, dtype=source_stat.dtype, device="cpu")
                self._copy_selected_cpu_chunked(destination, source_stat, keep_indices)
                cpu_stats.append(destination)
            new_stats = tuple(cpu_stats)
        return new_parameters, new_states, new_stats

    def _bind_structure(
        self,
        raw_parameters: dict[str, torch.Tensor],
        states: dict[str, dict[str, object] | None],
        stats: tuple[torch.Tensor, ...],
    ) -> None:
        """Atomically re-key optimizer state and bind a fully built Gaussian structure."""
        optimizable_tensors: dict[str, nn.Parameter] = {}
        for group in self.optimizer.param_groups:
            old_parameter = group["params"][0]
            self.optimizer.state.pop(old_parameter, None)
            new_parameter = nn.Parameter(raw_parameters[group["name"]].requires_grad_(True))
            group["params"][0] = new_parameter
            stored_state = states[group["name"]]
            if stored_state is not None:
                self.optimizer.state[new_parameter] = stored_state
            optimizable_tensors[group["name"]] = new_parameter

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum, self.xyz_gradient_accum_abs, self.denom = stats

    @torch.no_grad()
    def rebuild_optimizer_structure(
        self,
        keep_mask: torch.Tensor,
        extension_dicts: list[dict[str, torch.Tensor]] | None = None,
        reset_stats: bool = True,
        backend: str = "auto",
    ) -> str:
        """
        Rebuild parameters and optimizer state once from retained old rows plus new Gaussian batches.

        The GPU backend builds a complete destination before rebinding any model field. The CPU backend stages
        the same destination in host memory, releases the old CUDA structure, and uploads a compact replacement.
        """
        if extension_dicts is None:
            extension_dicts = []
        if backend not in {"auto", "gpu", "cpu"}:
            raise ValueError("Unsupported structure rebuild backend: {}".format(backend))
        old_count = int(self.get_xyz.shape[0])
        model_device = self.get_xyz.device
        if keep_mask.dtype != torch.bool or keep_mask.ndim != 1 or int(keep_mask.shape[0]) != old_count:
            raise ValueError("keep_mask must be a 1D bool tensor matching the current Gaussian count.")
        if keep_mask.device != self.get_xyz.device:
            raise ValueError("keep_mask must be on the same CUDA device as the Gaussian parameters.")

        expected_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        for extension in extension_dicts:
            if set(extension) != expected_names:
                raise ValueError("Each Gaussian extension must contain exactly {}.".format(sorted(expected_names)))
            extension_count = int(extension["xyz"].shape[0])
            for group_name in expected_names:
                if int(extension[group_name].shape[0]) != extension_count:
                    raise ValueError("Extension tensors must share the same first dimension.")

        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).contiguous()
        extension_count = sum(int(extension["xyz"].shape[0]) for extension in extension_dicts)
        final_count = int(keep_indices.shape[0]) + int(extension_count)
        if final_count <= 0:
            raise ValueError("A structure rebuild cannot remove every Gaussian.")

        source_parameters = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }
        source_states = {
            group["name"]: self.optimizer.state.get(group["params"][0], None)
            for group in self.optimizer.param_groups
        }

        torch.cuda.empty_cache()
        required_bytes = self._estimate_structure_destination_bytes(
            final_count,
            source_parameters,
            source_states,
            reset_stats,
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.get_xyz.device)
        safety_bytes = max(512 * 1024 * 1024, required_bytes // 20)
        selected_backend = backend
        if selected_backend == "auto":
            selected_backend = "gpu" if required_bytes + safety_bytes <= free_bytes else "cpu"

        gpu_build_failed = False
        if selected_backend == "gpu":
            try:
                new_parameters, new_states, new_stats = self._build_structure_cuda(
                    keep_indices,
                    extension_dicts,
                    source_parameters,
                    source_states,
                    final_count,
                    reset_stats,
                )
            except torch.cuda.OutOfMemoryError:
                if backend == "gpu":
                    raise
                selected_backend = "cpu"
                gpu_build_failed = True

        if gpu_build_failed:
            torch.cuda.empty_cache()

        if selected_backend == "cpu":
            cpu_parameters, cpu_states, cpu_stats = self._build_structure_cpu(
                keep_indices,
                extension_dicts,
                source_parameters,
                source_states,
                final_count,
                reset_stats,
            )

            placeholder_parameters = {
                group_name: nn.Parameter(
                    torch.empty(
                        (0,) + tuple(parameter.shape[1:]),
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ).requires_grad_(True)
                )
                for group_name, parameter in source_parameters.items()
            }
            for group in self.optimizer.param_groups:
                old_parameter = group["params"][0]
                self.optimizer.state.pop(old_parameter, None)
                group["params"][0] = placeholder_parameters[group["name"]]
            del old_parameter
            self._xyz = placeholder_parameters["xyz"]
            self._features_dc = placeholder_parameters["f_dc"]
            self._features_rest = placeholder_parameters["f_rest"]
            self._opacity = placeholder_parameters["opacity"]
            self._scaling = placeholder_parameters["scaling"]
            self._rotation = placeholder_parameters["rotation"]
            self.xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32, device=model_device)
            self.xyz_gradient_accum_abs = torch.empty((0, 1), dtype=torch.float32, device=model_device)
            self.denom = torch.empty((0, 1), dtype=torch.float32, device=model_device)

            source_parameters.clear()
            source_states.clear()
            for extension in extension_dicts:
                extension.clear()
            extension_dicts.clear()
            keep_indices = None
            torch.cuda.empty_cache()

            new_parameters = {
                group_name: tensor.to(device=model_device, non_blocking=False)
                for group_name, tensor in cpu_parameters.items()
            }
            new_states = {}
            for group_name, stored_state in cpu_states.items():
                if stored_state is None:
                    new_states[group_name] = None
                    continue
                uploaded_state = {}
                for state_name, state_value in stored_state.items():
                    if self._is_per_gaussian_optimizer_state(state_name, state_value) or (
                        state_name == "last_seen_step" and torch.is_tensor(state_value)
                    ):
                        uploaded_state[state_name] = state_value.to(device=model_device, non_blocking=False)
                    else:
                        uploaded_state[state_name] = state_value
                new_states[group_name] = uploaded_state
            if reset_stats:
                new_stats = tuple(
                    torch.zeros((int(final_count), 1), dtype=torch.float32, device=model_device)
                    for _ in range(3)
                )
            else:
                assert cpu_stats is not None
                new_stats = tuple(stat.to(device=model_device, non_blocking=False) for stat in cpu_stats)

        self._bind_structure(new_parameters, new_states, new_stats)
        source_parameters.clear()
        source_states.clear()
        for extension in extension_dicts:
            extension.clear()
        extension_dicts.clear()
        del keep_indices
        torch.cuda.empty_cache()
        return selected_backend


    def update_opacity_lr(self, rate: float) -> None:
        """
            Let GNS raise the opacity learning rate during regularized pruning and restore it afterward.
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group["lr"] = param_group["lr"] * float(rate)
