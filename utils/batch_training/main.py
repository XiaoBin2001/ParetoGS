"""Config-driven training, postprocessing, summary with independent repeats."""
from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import sys
import time

from utils.batch_training.config import (
    build_postprocess_payload, build_python_command, build_scene_paths, build_train_payload,
    filter_config_scenes, load_batch_config, parse_args, parse_scene_filter, resolve_config_paths,
    validate_batch_config,
)
from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import install_signal_handlers, run_command
from utils.batch_training.results import rebuild_result_total_csv
from utils.experiment_utils import atomic_json, backup_batch_training_config, load_json, validate_gns_start
from utils.batch_training.paretogs import validate_build_config, validate_runtime_output, run_paretogs_config
from utils.resource_limits import check_resource_support, resolve_resource_limits


def _model_tasks(config: dict, scenes: list[dict]) -> list[tuple]:
    tasks = []
    for index in range(config.get("repeat", 1)):
        for scene in scenes:
            data, output = build_scene_paths(config, scene, index)
            tasks.append((index + 1, scene, str(Path(data).absolute()), str(Path(output).absolute())))
    return tasks


def _postprocess(config: dict, scene: dict, data: str, output: Path, state: dict,
                 gpu, dry_run: bool) -> int:
    python = config.get("python_executable", sys.executable)
    if config.get("pipeline") == "paretogs_full":
        from utils.batch_training.config import merge_dict
        payload = merge_dict(config.get("postprocess_args", {}), scene.get("postprocess_args", {}))
        payload["model_path"] = str(output)
        command = [python, "-m", "paretogs.postprocess"]
        from utils.batch_training.config import dict_to_cli_args
        command.extend(dict_to_cli_args(payload))
    else:
        payload = build_postprocess_payload(config, scene, data, str(output),
                                            use_saved=not dry_run or (output / "training_parameters.json").is_file())
        command = build_python_command(python, "postprocess.py", payload)
    state["required_splits"] = (["train"] if payload.get("include_train", config.get("pipeline") == "paretogs_full") else []) + (
        [] if payload.get("skip_test", False) else ["test"])
    return _run_stage(config, command, gpu, output, state, "postprocess", dry_run)


def _preflight(config: dict, scenes: list[dict], postprocess_only: bool, dry_run: bool) -> list[tuple]:
    if config.get("pipeline") == "paretogs_experiment":
        if postprocess_only:
            raise ValueError("Paper experiments support full execution or --rebuild-summary; omit --postprocess-only.")
        from paretogs.experiments import validate_experiment
        return [task[:4] for task in validate_experiment(dict(config, scenes=scenes), dry_run)]
    tasks = _model_tasks(config, scenes)
    if config.get("pipeline") == "paretogs_full":
        if not postprocess_only:
            validate_build_config(dict(config, scenes=scenes), dry_run)
        else:
            for _, _, _, output in tasks:
                validate_runtime_output(Path(output))
        from paretogs.postprocess import parse_args as parse_postprocess_args
        from utils.batch_training.config import dict_to_cli_args, merge_dict
        for _, scene, _, output in tasks:
            payload = merge_dict(config.get("postprocess_args", {}), scene.get("postprocess_args", {}))
            parse_postprocess_args(["--model_path", output, *dict_to_cli_args(payload)])
    elif postprocess_only:
        for _, scene, data, output in tasks:
            root = Path(output)
            if not (root / "training_parameters.json").is_file():
                raise ValueError("Missing saved training parameters: {}".format(root))
            payload = build_postprocess_payload(config, scene, data, output)
            iteration = payload.get("iteration", -1)
            models = list((root / "point_cloud").glob("iteration_*/point_cloud.ply")) if iteration == -1 else [
                root / "point_cloud" / ("iteration_{}".format(iteration)) / "point_cloud.ply"]
            if not models or not all(path.is_file() for path in models):
                raise ValueError("Missing saved model for postprocessing: {}".format(root))
    else:
        for _, scene, data, output in tasks:
            validate_gns_start(output, Namespace(**build_train_payload(config, scene, data, output)))
    limits = config.get("resource_limits")
    if limits is not None:
        (resolve_resource_limits if dry_run else check_resource_support)(limits)
    return tasks


def _run_stage(config: dict, command: list[str], gpu, output: Path, state: dict, stage: str, dry_run: bool) -> int:
    started_ns = time.time_ns()
    if not dry_run:
        state[stage] = "running"
        if stage == "train":
            state["postprocess"] = "pending"
        if stage == "postprocess":
            state["written_splits"] = []
        atomic_json(output / "run_status.json", state)
    code = 1
    try:
        code = run_command(command, gpu_id=gpu, dry_run=dry_run,
                           resource_limits=config.get("resource_limits"),
                           resource_path=str(output / (stage + "_resources.json")))
        return code
    finally:
        if not dry_run:
            state[stage] = "completed" if code == 0 else "failed"
            state[stage + "_exit_code"] = code
            if stage == "postprocess" and code == 0:
                state["written_splits"] = [split for split in ("train", "test")
                                            if (output / ("result_" + split + ".json")).is_file()
                                            and (output / ("result_" + split + ".json")).stat().st_mtime_ns >= started_ns]
            atomic_json(output / "run_status.json", state)


def run_single_config(config_path: str, dry_run: bool, scene_filter: set[str], rebuild_total: bool,
                      overrides: dict | None = None, postprocess_only: bool = False) -> int:
    config = load_batch_config(config_path)
    config.update(overrides or {})
    validate_batch_config(config, scene_filter)
    scenes = filter_config_scenes(config, scene_filter)
    if config.get("pipeline") == "paretogs_experiment":
        if postprocess_only:
            raise ValueError("Paper experiment configs already evaluate saved builds.")
        from paretogs.experiments import run_config
        return run_config(config_path, dict(config, scenes=scenes), dry_run, rebuild_total)
    if rebuild_total:
        # Rebuild from all independent repeat records without loading models.
        from utils.batch_training.results import collect_existing_total_rows
        collect_existing_total_rows(config_path, dict(config, scenes=scenes))
        if dry_run:
            print("Dry run: would rebuild total CSV for {}".format(config_path))
        else:
            rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
        return 0
    tasks = _preflight(config, scenes, postprocess_only, dry_run)
    if config.get("pipeline") == "paretogs_full" and not postprocess_only:
        code = run_paretogs_config(config_path, config, dry_run, scene_filter, False,
                                  postprocess=_postprocess)
        if not dry_run:
            rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
        return code

    resource_limits = config.get("resource_limits")
    if resource_limits is not None:
        (resolve_resource_limits if dry_run else check_resource_support)(resource_limits)
    if not dry_run and not postprocess_only:
        Path(config["output_root"]).mkdir(parents=True, exist_ok=True)
        backup = backup_batch_training_config(config, config_path)
        if backup is not None:
            print("Backed up batch training config to {}".format(backup))
    first_failure = 0
    for repeat, scene, data_dir, output_dir in tasks:
        lock = None
        output = Path(output_dir)
        state = (load_json(str(output / "run_status.json"), default={}) if postprocess_only else {})
        state.update(repeat=repeat, seed=0, scene=scene["name"], model_path=output_dir)
        if postprocess_only:
            state.setdefault("train", "completed")
        try:
            gpu, lock = select_scene_gpu(config, scene["name"], config.get("gpu_id"), dry_run)
            python = config.get("python_executable", sys.executable)
            if not postprocess_only:
                payload = build_train_payload(config, scene, data_dir, output_dir)
                code = _run_stage(config, build_python_command(python, "train.py", payload), gpu,
                                  output, state, "train", dry_run)
                if code:
                    first_failure = first_failure or code
                    if config.get("stop_on_error", True):
                        if not dry_run:
                            rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
                        return code
                    continue
            if not postprocess_only and not config.get("run_postprocess", True):
                if not dry_run:
                    state["postprocess"] = "skipped"
                    atomic_json(output / "run_status.json", state)
                continue
            code = _postprocess(config, scene, data_dir, output, state, gpu, dry_run)
            if code:
                first_failure = first_failure or code
                if config.get("stop_on_error", True):
                    if not dry_run:
                        rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
                    return code
        finally:
            release_gpu_lock(lock)
    if not dry_run:
        rebuild_result_total_csv(config_path, dict(config, scenes=scenes))
    return first_failure


def main() -> int:
    install_signal_handlers()
    args = parse_args()
    try:
        scene_filter = parse_scene_filter(args.only_scene)
        paths = resolve_config_paths(args.config)
        overrides = {key: getattr(args, key) for key in ("data_root", "output_root", "gpu_id")
                     if getattr(args, key, None) is not None}
        # Every config is checked before backups, GPU locks or child processes.
        seen_outputs = set()
        for path in paths:
            config = load_batch_config(path)
            config.update(overrides)
            try:
                validate_batch_config(config, scene_filter)
                if config.get("pipeline") == "paretogs_experiment" and args.rebuild_total:
                    from paretogs.experiments import validate_experiment
                    validate_experiment(dict(config, scenes=filter_config_scenes(config, scene_filter)), args.dry_run, True)
                if not args.rebuild_total:
                    tasks = _preflight(config, filter_config_scenes(config, scene_filter), args.postprocess_only, args.dry_run)
                    for _, _, _, output in tasks:
                        if str(Path(output).resolve()) in seen_outputs:
                            raise ValueError("Multiple configs target the same output: {}".format(output))
                        seen_outputs.add(str(Path(output).resolve()))
                    limits = config.get("resource_limits")
                    if limits is not None:
                        (resolve_resource_limits if args.dry_run else check_resource_support)(limits)
            except (ValueError, OSError) as exc:
                raise ValueError("{}: {}".format(path, exc)) from exc
        first_failure = 0
        for path in paths:
            print("Processing config {}".format(path))
            code = run_single_config(path, args.dry_run, scene_filter, args.rebuild_total, overrides,
                                     args.postprocess_only)
            if code:
                first_failure = first_failure or code
                if load_batch_config(path).get("stop_on_error", True):
                    return code
        return first_failure
    except (ValueError, OSError) as exc:
        print("Batch workflow error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
