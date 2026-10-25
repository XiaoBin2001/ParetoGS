"""
GaussianModel structure-changing operations.

These methods modify the number of Gaussians or reset their trainable tensors.
They also keep optimizer state aligned with the new tensors, so training can
continue after cloning, splitting, or pruning.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from utils.general_utils import build_rotation, inverse_sigmoid


class GaussianModelDensificationMixin:
    """
        Mixin that adds structure updates to GaussianModel.
    """

    def reset_opacity(self, min_opacity: float = 0.01) -> None:
        """
            Clamp all activated opacities to a maximum value and update optimizer state.
        """
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def _build_clone_additions(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
    ) -> dict[str, torch.Tensor] | None:
        """Build the logical clone batch without changing model or optimizer storage."""
        selected_pts_mask = torch.norm(grads, dim=-1) >= float(grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * float(scene_extent),
        )
        if not bool(selected_pts_mask.any().item()):
            return None
        return {
            "xyz": self._xyz[selected_pts_mask],
            "f_dc": self._features_dc[selected_pts_mask],
            "f_rest": self._features_rest[selected_pts_mask],
            "opacity": self._opacity[selected_pts_mask],
            "scaling": self._scaling[selected_pts_mask],
            "rotation": self._rotation[selected_pts_mask],
        }

    def _build_split_additions(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        num_children: int = 2,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Build standard split children and return the old-parent removal mask."""
        old_count = int(self.get_xyz.shape[0])
        padded_grad = torch.zeros((old_count,), dtype=torch.float32, device=self.get_xyz.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= float(grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * float(scene_extent),
        )
        if not bool(selected_pts_mask.any().item()):
            return selected_pts_mask, None

        return selected_pts_mask, self._build_standard_split_additions(selected_pts_mask, num_children)

    def _build_standard_split_additions(
        self,
        selected_pts_mask: torch.Tensor,
        num_children: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Build original stochastic Split children for an already selected parent mask."""

        stds = self.get_scaling[selected_pts_mask].repeat(int(num_children), 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[selected_pts_mask]).repeat(int(num_children), 1, 1)
        new_xyz = (
            torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(int(num_children), 1)
        )
        additions = {
            "xyz": new_xyz,
            "f_dc": self._features_dc[selected_pts_mask].repeat(int(num_children), 1, 1),
            "f_rest": self._features_rest[selected_pts_mask].repeat(int(num_children), 1, 1),
            "opacity": self._opacity[selected_pts_mask].repeat(int(num_children), 1),
            "scaling": self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask].repeat(int(num_children), 1) / (0.8 * int(num_children))
            ),
            "rotation": self._rotation[selected_pts_mask].repeat(int(num_children), 1),
        }
        return additions

    def _commit_densification_plan(
        self,
        old_remove_mask: torch.Tensor,
        additions: list[dict[str, torch.Tensor]],
        reset_stats: bool,
    ) -> str | None:
        """Apply one logical clone/split/prune plan with a single storage rebuild."""
        has_removals = bool(old_remove_mask.any().item())
        has_additions = any(int(extension["xyz"].shape[0]) > 0 for extension in additions)
        if not has_removals and not has_additions:
            if reset_stats:
                self.xyz_gradient_accum.zero_()
                self.xyz_gradient_accum_abs.zero_()
                self.denom.zero_()
            return None
        return self.rebuild_optimizer_structure(
            ~old_remove_mask,
            extension_dicts=additions,
            reset_stats=reset_stats,
        )

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
    ) -> None:
        """
            Run original 3DGS densification, then prune Gaussians that meet the pruning rules.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        clone_additions = self._build_clone_additions(grads, max_grad, extent)
        split_parent_mask, split_additions = self._build_split_additions(grads, max_grad, extent)
        old_opacity_prune_mask = (self.get_opacity < float(min_opacity)).squeeze()
        additions = [extension for extension in (clone_additions, split_additions) if extension is not None]
        self._commit_densification_plan(
            torch.logical_or(split_parent_mask, old_opacity_prune_mask),
            additions,
            reset_stats=True,
        )


    def add_densification_stats(self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor) -> None:
        """
            Accumulate normal screen-space gradient magnitudes for visible Gaussians.
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_abs(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
        sync_default_accumulator: bool = False,
        abs_grad_reduce: str = "mean",
    ) -> None:
        """
            Collect `abs(dx)` and `abs(dy)` written by CUDA backward for ImprovedGS densification scoring. When `sync_default_accumulator=True`, also update `xyz_gradient_accum` so ABSGS can reuse the baseline `densify_and_prune()` path.
        """
        grad_values = torch.norm(viewspace_point_tensor.grad[update_filter, 2:4], dim=-1, keepdim=True)
        if abs_grad_reduce == "max":
            self.xyz_gradient_accum_abs[update_filter] = torch.maximum(
                self.xyz_gradient_accum_abs[update_filter],
                grad_values,
            )
            if sync_default_accumulator:
                self.xyz_gradient_accum[update_filter] = torch.maximum(
                    self.xyz_gradient_accum[update_filter],
                    grad_values,
                )
            self.denom[update_filter] = 1
            return

        self.xyz_gradient_accum_abs[update_filter] += grad_values
        if sync_default_accumulator:
            self.xyz_gradient_accum[update_filter] += grad_values
        self.denom[update_filter] += 1


    def only_prune(self, min_opacity: float, percent: bool = False) -> None:
        """
            Provide a lightweight pruning entry point that RAP and GNS can call without running densification.
        """
        if self.get_opacity.numel() == 0:
            return
        if percent:
            opacity_array = self.get_opacity.detach().flatten()
            threshold = torch.quantile(opacity_array, float(min_opacity))
            prune_mask = (self.get_opacity < threshold).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        if prune_mask.any():
            self.prune_points(prune_mask)

    def densify_and_prune_improved(
        self,
        scores: torch.Tensor | None,
        min_opacity: float,
        budget: int,
        opt: Any,
        iteration: int,
        scene_extent: float,
    ) -> None:
        """
            Select candidate Gaussians with absolute screen-space gradients, sample them under the active budget, then build either long-axis or standard Split children. In late training, fall back to gradient scores and relax the threshold while the budget is not full.
        """
        grad_values = self.xyz_gradient_accum_abs / self.denom
        grad_values[grad_values.isnan()] = 0.0

        min_grad = float(opt.densify_grad_threshold)
        late_densify_iter = max(int(opt.densify_until_iter) - 100, int(opt.densify_from_iter))
        if scores is None or iteration > late_densify_iter:
            scores = grad_values.squeeze(-1)
            if self.get_opacity.shape[0] < budget and iteration > late_densify_iter:
                min_grad = min_grad / 1.5

        grad_qualifiers = torch.where(torch.norm(grad_values, dim=-1) >= min_grad, True, False)
        total_candidates = int(grad_qualifiers.sum().item())
        current_points = int(self.get_xyz.shape[0])
        current_budget = min(int(budget), total_candidates + current_points)
        split_budget = current_budget - current_points
        split_parent_mask = torch.zeros((current_points,), dtype=torch.bool, device=self.get_xyz.device)
        split_additions = None
        if split_budget > 0:
            if bool(getattr(opt, "use_las", True)):
                split_parent_mask, split_additions, _ = self._build_long_axis_split(
                    scores,
                    split_budget,
                    grad_qualifiers,
                    opt.split_distance,
                    opt.opacity_reduction,
                )
            else:
                padded_importance = torch.zeros(
                    (current_points,),
                    dtype=torch.float32,
                    device=self.get_xyz.device,
                )
                padded_importance[: scores.shape[0]] = scores.detach().float().clamp_min(0)
                padded_importance[~grad_qualifiers] = 0
                positive_count = int((padded_importance > 0).sum().item())
                if positive_count > 0:
                    sample_count = min(int(split_budget), positive_count)
                    selected_indices = torch.multinomial(padded_importance, sample_count, replacement=False)
                    split_parent_mask[selected_indices] = True
                    split_additions = self._build_standard_split_additions(split_parent_mask)

        if iteration < late_densify_iter:
            old_opacity_prune_mask = (self.get_opacity < float(min_opacity)).squeeze()
        else:
            old_opacity_prune_mask = torch.zeros_like(split_parent_mask)
        additions = [] if split_additions is None else [split_additions]
        self._commit_densification_plan(
            torch.logical_or(split_parent_mask, old_opacity_prune_mask),
            additions,
            reset_stats=True,
        )

    def _build_long_axis_split(
        self,
        scores: torch.Tensor,
        budget: int,
        filter_mask: torch.Tensor,
        split_distance: float,
        opacity_reduction: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, int]:
        """
            Build long-axis split children and return their old-parent removal mask without mutating storage.
        """
        empty_parent_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
        if budget <= 0 or scores.numel() == 0 or not torch.any(filter_mask):
            return empty_parent_mask, None, 0

        padded_importance = torch.zeros(
            (self.get_xyz.shape[0]),
            dtype=torch.float32,
            device=self.get_xyz.device,
        )
        padded_importance[:scores.shape[0]] = scores.detach().float().clamp_min(0)
        padded_importance[~filter_mask] = 0
        positive_count = int((padded_importance > 0).sum().item())
        if positive_count == 0:
            return empty_parent_mask, None, 0

        budget = min(int(budget), positive_count)
        selected_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=torch.bool)
        selected_pts_mask[selected_indices] = True

        stds = self.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)
        axis_offsets = stds * axis_mask * 3.0 * float(split_distance)
        axis_offsets = torch.cat([axis_offsets, -axis_offsets], dim=0)

        rotation_mats = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        parent_xyz = self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_xyz = torch.bmm(rotation_mats, axis_offsets.unsqueeze(-1)).squeeze(-1) + parent_xyz

        split_distance_sq = float(split_distance) * float(split_distance)
        rate_w = max(1.0 - float(split_distance), 1e-6)
        rate_h = math.sqrt(max(1.0 - split_distance_sq, 1e-6))
        new_scales = stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h
        new_scaling = self.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * float(opacity_reduction)).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

        additions = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        return selected_pts_mask, additions, budget
