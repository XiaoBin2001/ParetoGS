"""
Command-line argument system.

Each parameter group declares defaults as class attributes. ParamGroup then
turns those attributes into argparse options and later extracts only the fields
that belong to that group. This keeps the training entry point readable while
still producing normal Python objects for the rest of the code.
"""
from __future__ import annotations
import json
from argparse import ArgumentParser, Namespace
import os
import sys


def parse_bool_arg(raw_value: str | bool) -> bool:
    """
        Parse flexible CLI boolean text into real `True` or `False` values.

        Argparse passes strings for values such as `--flag false`; this helper
        accepts common spellings and raises a clear error for anything else.
    """
    if isinstance(raw_value, bool):
        return raw_value
    normalized = str(raw_value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("Unsupported boolean value: {}".format(raw_value))


class GroupParams:
    """
        Simple object used to hold extracted parameters as attributes.

        The project historically reads options with dot access, so this class
        intentionally stays empty and lets ParamGroup attach fields dynamically.
    """
    pass


class ParamGroup:
    """
        Base class that registers one group of argparse options from defaults.
    """
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False) -> None:
        self._group_name = name
        self._fill_none = fill_none
        if not hasattr(self, "_fields"):
            self._fields: dict[str, object] = {}
        self._register_group(parser)

    def _register_group(self, parser: ArgumentParser) -> None:
        """
            Convert fields in `self._fields` into argparse options.

            A leading underscore means the option also gets a one-letter short
            form, for example `_source_path` becomes `--source_path` and `-s`.
        """
        group = parser.add_argument_group(self._group_name)
        for raw_key, raw_value in self._fields.items():
            shorthand = False
            key = raw_key
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            value = raw_value if not self._fill_none else None
            value_type = type(raw_value)
            if shorthand:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        "-" + key[0:1],
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=value_type)
            else:
                if value_type is bool or raw_value is None:
                    group.add_argument(
                        "--" + key,
                        default=value,
                        nargs="?",
                        const=True,
                        type=parse_bool_arg,
                    )
                else:
                    group.add_argument("--" + key, default=value, type=value_type)

    def extract(self, args: Namespace) -> GroupParams:
        """
            Copy only this group's parsed fields from the shared argparse namespace.
        """
        group = GroupParams()
        for raw_key, raw_default in self._fields.items():
            key = raw_key[1:] if raw_key.startswith("_") else raw_key
            value = getattr(args, key, raw_default)
            if self._fill_none and value is None:
                value = raw_default
            setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    """
        Dataset and output-path options used before the scene is created.
    """
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.sh_degree = 3  # Maximum SH degree
        self._source_path = ""  # Dataset path
        self._model_path = ""  # Output model path
        self._images = "images"  # Image subdirectory
        self._resolution = -1  # Training resolution scale
        self._white_background = False  # Use a white background
        self.init_type = "sfm"  # Initialization type
        self.max_initial_points = -1  # Maximum initial point cloud size; -1 keeps all points
        self.data_device = "cpu"  # Image data loading device
        self.eval = True  # Use evaluation split
        fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        self._fields = fields
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args: Namespace) -> GroupParams:
        """
            Extract model options and normalize the source path to an absolute path.
        """
        group = super().extract(args)
        group.source_path = os.path.abspath(os.fspath(group.source_path))
        if group.model_path:
            group.model_path = os.fspath(group.model_path)
        return group


class PipelineParams(ParamGroup):
    """
        Renderer behavior options passed to the Gaussian rasterizer.
    """
    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        self.debug = False  # Renderer debug switch
        self.antialiasing = False  # Enable antialiasing
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Pipeline Parameters", sentinel)


class OptimizationBaseParams(ParamGroup):
    """
        Common optimization settings shared by all training methods.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.iterations = 30_000  # Total training iterations
        self.position_lr_init = 0.00004  # Initial position learning rate
        self.position_lr_final = 0.000002  # Final position learning rate
        self.position_lr_delay_mult = 0.01  # Position LR delay multiplier
        self.position_lr_max_steps = 30_000  # Position LR decay steps
        self.lr_rate = 1.0  # Global LR multiplier
        self.feature_lr = 0.0025  # SH feature learning rate
        self.opacity_lr = 0.025  # Opacity learning rate
        self.initial_opacity = 0.1  # Initial opacity
        self.scaling_lr = 0.005  # Scale learning rate
        self.rotation_lr = 0.001  # Rotation learning rate
        self.percent_dense = 0.01  # Clone/split scale boundary
        self.lambda_dssim = 0.2  # DSSIM loss weight
        self.densification_interval = 100  # Densification interval
        self.opacity_reset_interval = 3000  # Opacity reset interval
        self.densify_from_iter = 500  # Densification start iteration
        self.densify_until_iter = 15_000  # Densification end iteration
        self.densify_grad_threshold = 0.00025  # Densification gradient threshold
        self.min_opacity = 0.005  # Minimum opacity pruning threshold
        self.random_background = False  # Use random background
        self.optimizer_type = "ours_adam"  # Dense optimizer type
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Optimization Base Parameters")


class TrainingMethodParams(ParamGroup):
    """
        Method selector and ImprovedGS component switches.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.training_method = "improvedgs"  # Overall training strategy
        self.use_las = True  # Enable ImprovedGS long-axis split
        self.use_rap = True  # Enable ImprovedGS RAP
        self.use_eas = True  # Enable ImprovedGS edge importance
        self.use_mu = True  # Enable ImprovedGS MU update cadence
        self.abs_grad_reduce = "mean"  # ABS gradient reduction: mean or max
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Training Method Parameters")


class ImprovedGSParams(ParamGroup):
    """
        ImprovedGS structure-growth, budget, RAP, and MU schedule options.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.edge_sample_cams = 10  # Edge-scoring sampled camera count
        self.improvedgs_reset_min_opacity = 0.05  # RAP reset opacity upper bound
        self.split_distance = 0.45  # Long-axis split offset ratio
        self.opacity_reduction = 0.6  # Opacity decay after split
        self.budget = 300_0000  # Gaussian budget
        self.budget_multiplier = 3.0  # Budget warmup multiplier
        self.budget_warmup_until_offset = 500  # Budget warmup end offset
        self.rap_prune_ratio = 0.2  # RAP pruning ratio
        self.rap_prune_offset = 300  # Pruning delay after RAP reset
        self.rap_rounds = 2  # Maximum RAP rounds
        self.mu_start_iter = 20_000  # MU first-stage start iteration
        self.mu_interval = 5  # MU first-stage step interval
        self.mu_second_start_iter = 24_000  # MU second-stage start iteration
        self.mu_second_interval = 20  # MU second-stage step interval
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "ImprovedGS Parameters")


class GNSParams(ParamGroup):
    """
        GNS opacity regularization and learning-rate scaling settings.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.gns_opacity_reg = 0.002  # GNS opacity regularization weight
        self.gns_opacity_lr_scale = 5.0  # GNS opacity learning-rate multiplier
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "GNS Parameters")






class RegPruneParams(ParamGroup):
    """
        Final pruning budget settings used by regularized pruning methods.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.final_budget = 60_0000  # Target Gaussian count for pruning
        self.final_budget_mode = "rate"  # Target budget mode
        self.final_rate = 0.2  # Retention ratio in rate mode
        self.reg_prune_from_iter = 15_000  # Regularized pruning start iteration
        self.reg_prune_until_iter = 20_000  # Regularized pruning end iteration
        self._fields = {key: value for key, value in vars(self).items() if not key.startswith("__")}
        super().__init__(parser, "Regularized Pruning Parameters")


class OptimizationParams:
    """
        Wrapper that registers and merges all optimization-related groups.
    """
    def __init__(self, parser: ArgumentParser) -> None:
        self.groups = [
            OptimizationBaseParams(parser),
            TrainingMethodParams(parser),
            ImprovedGSParams(parser),
            GNSParams(parser),
            RegPruneParams(parser),
        ]

    def extract(self, args: Namespace) -> GroupParams:
        """
            Merge all optimization parameter groups into one object for training.
        """
        merged = GroupParams()
        for group in self.groups:
            partial = group.extract(args)
            for key, value in vars(partial).items():
                setattr(merged, key, value)
        return merged




def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Recover omitted model/renderer options from training_parameters.json."""
    args = parser.parse_args(sys.argv[1:])
    model_path = getattr(args, "model_path", None)
    saved = {}
    if model_path:
        args.model_path = os.fspath(model_path)
        parameter_path = os.path.join(args.model_path, "training_parameters.json")
        if os.path.isfile(parameter_path):
            with open(parameter_path, encoding="utf-8") as stream:
                saved = json.load(stream)
    merged = {key: saved[key] for key in vars(args) if key in saved}
    merged.update({key: value for key, value in vars(args).items() if value is not None})
    return Namespace(**merged)
