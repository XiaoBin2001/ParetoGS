"""Config-driven paper evaluations sharing the deployment decoder and metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from utils.batch_training.config import build_output_name, build_scene_paths, filter_config_scenes
from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import run_command
from utils.experiment_utils import atomic_json

KINDS = ("components", "encoding", "anchor_density", "switch_timing", "budget_curve", "independent_gns")
VARIANTS = ("full", "fixed_shared_state", "no_opacity_gate", "linear_progress", "full_parameters")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def experiment_tasks(config):
    for index in range(config.get("repeat", 1)):
        for scene in config["scenes"]:
            data, output = build_scene_paths(config, scene, index)
            name = scene.get("build_name", build_output_name(config, scene, index))
            yield index + 1, scene, str(Path(data).absolute()), str(Path(output).absolute()), Path(config["build_root"]) / name


def midpoint_budgets(anchors):
    result = [round((lo + hi) / 2) for lo, hi in zip(anchors[:-1], anchors[1:])]
    if any(not lo < mid < hi for lo, mid, hi in zip(anchors[:-1], result, anchors[1:])):
        raise ValueError("An anchor interval has no integer midpoint.")
    return result


def curve_budgets(anchors, subdivisions):
    """Split each anchor interval evenly, keeping each shared endpoint once."""
    if len(anchors) < 2 or anchors != sorted(set(anchors)):
        raise ValueError('Anchors must be strictly increasing.')
    if type(subdivisions) is not int or subdivisions < 1:
        raise ValueError('subdivisions must be a positive integer.')
    budgets = []
    for index, (lower, upper) in enumerate(zip(anchors[:-1], anchors[1:])):
        budgets.extend(round(lower + (upper - lower) * step / subdivisions)
                       for step in range(0 if index == 0 else 1, subdivisions + 1))
    if budgets != sorted(set(budgets)):
        raise ValueError('Curve subdivisions collapse to duplicate integer budgets.')
    return budgets


def independent_training_args(settings, full, budget, source, output):
    """Every target starts independently from the same rebased Full checkpoint."""
    training = settings['training']
    start = output / ('start_chkpnt{}.pth'.format(training['start_iteration']))
    args = dict(training.get('train_args', {}))
    args.update(source_path=source, model_path=str(output / 'model'),
                iterations=training['end_iteration'], position_lr_max_steps=training['position_lr_max_steps'],
                checkpoint_iterations=[training['end_iteration']], training_method='gns',
                budget=full, final_budget_mode='num', final_budget=budget,
                reg_prune_from_iter=training['prune_from_iteration'],
                reg_prune_until_iter=training['prune_until_iteration'],
                start_checkpoint_dir=str(output), start_checkpoint_file=start.name)
    return args


def density_budgets(anchor_sets, count):
    lo, hi = anchor_sets[0][0], anchor_sets[0][-1]
    if any(anchors[0] != lo or anchors[-1] != hi for anchors in anchor_sets):
        raise ValueError("Anchor-density variants must share their endpoints.")
    forbidden = set(x for anchors in anchor_sets for x in anchors)
    values = [round(math.exp(math.log(lo) + math.log(hi / lo) * i / (count + 1)))
              for i in range(1, count + 1)]
    # Integer rounding can hit an anchor on small fixtures. Move to its nearest unused neighbour.
    selected = set()
    for value in values:
        distance = 0
        while True:
            candidates = [value - distance, value + distance]
            valid = next((x for x in candidates if lo < x < hi and x not in forbidden | selected), None)
            if valid is not None:
                selected.add(valid)
                break
            distance += 1
            if distance > hi - lo:
                raise ValueError("Too few distinct non-anchor budgets in the requested window.")
    return sorted(selected)


def validate_experiment(config, dry_run=False, rebuild=False):
    settings = config.get("experiment", {})
    if not isinstance(settings, dict):
        raise ValueError("experiment must be a JSON object.")
    kind = settings.get("kind")
    if kind not in KINDS:
        raise ValueError("experiment.kind must be one of {}".format(KINDS))
    if "save_images" in settings and type(settings["save_images"]) is not bool:
        raise ValueError("save_images must be a JSON boolean.")
    if "view_indices" in settings and (not isinstance(settings["view_indices"], list) or any(type(i) is not int or i < 0 for i in settings["view_indices"])):
        raise ValueError("view_indices must contain nonnegative integers.")
    variants = settings.get("variants", [])
    if not isinstance(variants, list) or any(not isinstance(v, str) for v in variants) or len(variants) != len(set(variants)):
        raise ValueError("variants must be distinct strings.")
    allowed = {"kind", "variants", "view_indices", "save_images", "anchor_sets", "samples", "calibration", "sequence", "repetitions", "subdivisions", "training"}
    if set(settings) - allowed:
        raise ValueError("Unknown experiment options: {}".format(sorted(set(settings) - allowed)))
    if not isinstance(config.get("build_root"), str) or not config["build_root"]:
        raise ValueError("Paper experiments require build_root.")
    if kind == "components" and (not settings.get("variants") or set(settings["variants"]) - set(VARIANTS)):
        raise ValueError("Invalid component variants.")
    if kind == "encoding" and settings.get("variants") != ["fp32", "fp16", "atlas"]:
        raise ValueError("Encoding comparison uses fp32, fp16, atlas with one shared curve.")
    if kind == "anchor_density":
        sets = settings.get("anchor_sets", {})
        if not sets or type(settings.get("samples")) is not int or settings["samples"] < 1:
            raise ValueError("Anchor density requires anchor_sets and positive samples.")
        for name, ratios in sets.items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid anchor-set name.")
            if len(ratios) < 2 or ratios != sorted(set(ratios)) or not 0 < ratios[0] < ratios[-1] <= 1:
                raise ValueError("Invalid anchor ratios.")
        calibration = settings.get("calibration")
        fields = {"common_powers", "shell_gate_powers", "progresses", "view_count", "lambda_dssim", "monotonic_weight", "smoothness_weight"}
        if not isinstance(calibration, dict) or set(calibration) != fields:
            raise ValueError("Anchor density requires complete calibration settings.")
        for key in ("common_powers", "shell_gate_powers", "progresses"):
            if not isinstance(calibration[key], list) or not calibration[key] or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in calibration[key]):
                raise ValueError("Invalid calibration candidates.")
        if any(x >= 1 for x in calibration["progresses"]) or type(calibration["view_count"]) is not int or calibration["view_count"] < 1:
            raise ValueError("Invalid calibration sampling.")
        if not 0 <= calibration["lambda_dssim"] <= 1 or calibration["monotonic_weight"] < 0 or calibration["smoothness_weight"] < 0:
            raise ValueError("Invalid calibration weights.")
    if kind == "switch_timing":
        if not settings.get("sequence") or any(not 0 < x <= 1 for x in settings["sequence"]):
            raise ValueError("Switch timing requires a ratio sequence in (0, 1].")
        if type(settings.get("repetitions")) is not int or settings["repetitions"] < 1:
            raise ValueError("Timing repetitions must be positive.")
    if kind == 'budget_curve' and (type(settings.get('subdivisions')) is not int or settings['subdivisions'] < 1):
        raise ValueError('Budget curves require positive integer subdivisions.')
    if kind == 'independent_gns':
        training = settings.get('training')
        fields = {'start_iteration', 'end_iteration', 'prune_from_iteration', 'prune_until_iteration', 'position_lr_max_steps', 'train_args'}
        if not isinstance(training, dict) or set(training) != fields:
            raise ValueError('Independent GNS requires a complete training protocol.')
        if any(type(training[key]) is not int or training[key] < 1 for key in fields - {'train_args'}):
            raise ValueError('Independent GNS iterations must be positive integers.')
        if not training['start_iteration'] < training['prune_from_iteration'] <= training['prune_until_iteration'] < training['end_iteration']:
            raise ValueError('Independent GNS requires start < prune_from <= prune_until < end.')
        if not isinstance(training['train_args'], dict):
            raise ValueError('Independent GNS train_args must be an object.')
        reserved = {'source_path', 'model_path', 'iterations', 'position_lr_max_steps', 'checkpoint_iterations',
                    'training_method', 'budget', 'final_budget_mode', 'final_budget', 'reg_prune_from_iter',
                    'reg_prune_until_iter', 'start_checkpoint_dir', 'start_checkpoint_file', 'resume_training'}
        if set(training['train_args']) & reserved:
            raise ValueError('Put GNS schedule and generated paths in the training protocol, not train_args.')
    from utils.batch_training.paretogs import validate_runtime_output
    tasks = list(experiment_tasks(config))
    for _, _, data, output, build in tasks:
        if (Path(output).resolve() == build.resolve() or build.resolve() in Path(output).resolve().parents
                or Path(output).resolve() in build.resolve().parents):
            raise ValueError("Experiment output must be separate from its input build.")
        if rebuild:
            continue
        if not dry_run and Path(output).exists():
            raise FileExistsError("Refusing to overwrite experiment output: {}".format(output))
        if dry_run and not build.exists():
            continue
        if not dry_run and not Path(data).is_dir():
            raise FileNotFoundError(data)
        manifest = validate_runtime_output(build)
        anchors = manifest["active_anchors"]
        if kind in {"components", "encoding", "independent_gns"}:
            midpoint_budgets(anchors)
        if kind == 'budget_curve':
            curve_budgets(anchors, settings['subdivisions'])
        if kind == 'independent_gns':
            checkpoint = build / 'backbone' / ('chkpnt{}.pth'.format(manifest['iteration']))
            if not checkpoint.is_file():
                raise FileNotFoundError('Independent GNS requires the retained Full checkpoint: ' + str(checkpoint))
        if kind in {"encoding", "anchor_density"}:
            if not (build / "hierarchy/model/nested_adapters_fp16_full.pt").is_file():
                raise FileNotFoundError("Paper comparison requires retained FP16 anchor residuals.")
        if kind == "encoding":
            summary = load_json(build / "hierarchy/pipeline_summary.json")
            paths = [summary["full_anchor_state"], summary["checkpoint_nested_order"], *summary["anchor_states"].values()]
            paths.extend(str(Path(stage["model_path"]) / "gns_prune_trace.pt") for stage in summary["stages"])
            for path in paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
        if kind == "anchor_density":
            sets = [[round(manifest["gaussian_count"] * ratio) for ratio in ratios]
                    for ratios in settings["anchor_sets"].values()]
            if any(set(values) - set(anchors) or len(set(values)) != len(values) for values in sets):
                raise ValueError("Anchor sets must select stored, distinct anchors.")
            density_budgets(sets, settings["samples"])
        if kind == "switch_timing" and min(settings["sequence"]) * manifest["gaussian_count"] < anchors[0] - 0.5:
            raise ValueError("Switch sequence requests a budget below the smallest stored anchor.")
    return tasks


def summarize(config):
    rows, tasks, pooled_switches = [], [], []
    for repeat, scene, _, output, _ in experiment_tasks(config):
        path = Path(output) / "result.json"
        task = {"scene": scene["name"], "repeat": repeat, "status": "missing"}
        if path.is_file():
            try:
                result = load_json(path)
                status = result["status"]
                task["status"] = status if status in {"complete", "failed"} else "incomplete"
                if status == "complete":
                    summary = result["summary"]
                    if not isinstance(summary, dict) or not summary or any(not isinstance(v, dict) or not v for v in summary.values()):
                        raise ValueError("Completed result has no valid summary.")
                    samples = [sample["prepare_ms"] for sample in result.get("samples", [])]
                    rows.extend({"scene": scene["name"], "repeat": repeat, "variant": variant,
                                 **values, "status": "complete"} for variant, values in summary.items())
                    pooled_switches.extend(samples)
                elif result.get("error"):
                    task["reason"] = result["error"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                task.update(status="incomplete", reason=str(error))
        else:
            task["reason"] = "result.json is missing"
        tasks.append(task)
        if task["status"] != "complete":
            rows.append({**task, "variant": ""})
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["scene", "repeat", "variant"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    completed_rows = [row for row in rows if row["status"] == "complete"]
    for variant in dict.fromkeys(row["variant"] for row in completed_rows):
        group = [row for row in completed_rows if row["variant"] == variant]
        means[variant] = {key: statistics.mean(row[key] for row in group)
                          for key in columns if key not in {"scene", "repeat", "variant"} and all(isinstance(row.get(key), (float, int)) for row in group)}
    counts = {status: sum(task['status'] == status for task in tasks)
              for status in ('complete', 'failed', 'missing', 'incomplete')}
    report = {"scene_repeat_rows": len(completed_rows), "means": means,
              "expected_tasks": len(tasks), "completed": counts.pop('complete'), **counts,
              "tasks": tasks, "complete": bool(tasks) and all(task['status'] == 'complete' for task in tasks),
              "aggregation_scope": "completed_tasks_only",
              "aggregation": "mean budgets within each scene, then equal scene/repeat weight"}
    if pooled_switches:
        import torch
        values = torch.tensor(pooled_switches, dtype=torch.float64)
        report["pooled_switches"] = {"count": len(pooled_switches), "median_ms": values.quantile(.5).item(),
                                     "p95_ms": values.quantile(.95).item(), "maximum_ms": values.max().item()}
    atomic_json(root / "summary.json", report)
    print('Paper summary: {} ({}/{} completed; {} failed, {} missing, {} incomplete).'.format(
        'complete' if report['complete'] else 'partial', report['completed'], report['expected_tasks'],
        report['failed'], report['missing'], report['incomplete']))
    return report


def run_config(config_path, config, dry_run=False, rebuild=False):
    tasks = validate_experiment(config, dry_run, rebuild)
    if rebuild:
        if dry_run:
            print("Would rebuild paper experiment summaries in", config["output_root"])
        else:
            return 0 if summarize(config)['complete'] else 1
        return 0
    first_failure = 0
    for repeat, scene, data, output, build in tasks:
        lock = None
        try:
            gpu, lock = select_scene_gpu(config, scene["name"], config.get("gpu_id"), dry_run)
            command = [config.get("python_executable") or sys.executable, "-m", "paretogs.experiments",
                       "--config", config_path, "--build", str(build.absolute()), "--output", output,
                       "--source_path", data]
            if dry_run:
                print(json.dumps({'experiment': config['experiment'], 'build': str(build), 'output': output,
                                  'source_path': data, 'repeat': repeat}))
            code = run_command(command, gpu_id=gpu, dry_run=dry_run, resource_limits=config.get("resource_limits"),
                               resource_path=str(Path(output) / "resources.json"))
            if code:
                if not dry_run:
                    summarize(config)
                first_failure = first_failure or code
                if config.get("stop_on_error", True):
                    return code
        finally:
            release_gpu_lock(lock)
    if not dry_run:
        report = summarize(config)
        if not report['complete']:
            return first_failure or 1
    return first_failure


def _raw_encoding(build, output):
    import torch
    from paretogs.train_hierarchy import _build_order_and_adapters
    from paretogs.prefix import load_nested_order
    from paretogs.adapters import ADAPTER_TENSOR_KEYS
    summary = load_json(build / "hierarchy/pipeline_summary.json")
    traces = {int(stage["budget"]): str(Path(stage["model_path"]) / "gns_prune_trace.pt") for stage in summary["stages"]}
    order, bundle = _build_order_and_adapters(summary["full_anchor_state"],
        {int(k): v for k, v in summary["anchor_states"].items()}, summary["budgets"], "fp32", traces)
    saved_order = load_nested_order(summary["checkpoint_nested_order"], len(order), "cpu")
    if not torch.equal(order, saved_order):
        raise ValueError("Reconstructed residual order differs from the deployed model.")
    fp16 = torch.load(build / "hierarchy/model/nested_adapters_fp16_full.pt", map_location="cpu", weights_only=False)
    for anchor in bundle["budgets"][:-1]:
        for field in ADAPTER_TENSOR_KEYS:
            if not torch.equal(bundle["adapters"][anchor][field].half(), fp16["adapters"][anchor][field]):
                raise ValueError("FP32 reconstruction does not reproduce the retained FP16 residuals.")
    path = output / "residuals_fp32.pt"
    torch.save(bundle, path)
    return path


def switch_timing(runtime, settings):
    import torch
    from gaussian_renderer import render
    from paretogs.adapters import ADAPTER_TENSOR_KEYS
    from paretogs.decoder import _decode_endpoint_field
    from utils.evaluation.render_metric import build_background
    view = runtime.scene.getTestCameras()[0]
    background = build_background(runtime.dataset)
    for _ in range(2):
        render(view, runtime.gaussians, runtime.pipeline, background, track_gradients=False)
    torch.cuda.synchronize()
    predecode_started = time.perf_counter()
    for anchor in runtime.anchors:
        for field in ADAPTER_TENSOR_KEYS:
            _decode_endpoint_field(runtime.gaussians, runtime.bundle, anchor, field, runtime.endpoint_cache)
    torch.cuda.synchronize()
    predecode_seconds = time.perf_counter() - predecode_started
    decoded_bytes = sum(value.numel() * value.element_size() for (anchor, _), value in runtime.endpoint_cache.items()
                        if anchor != runtime.gaussian_count and value is not None)
    rows, previous = [], 1.0
    for repeat in range(settings["repetitions"]):
        for index, ratio in enumerate(settings["sequence"]):
            if ratio == previous:
                continue
            budget = round(runtime.gaussian_count * ratio)
            torch.cuda.synchronize()
            started = time.perf_counter()
            proxy, metadata = runtime.decode(budget)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            render(view, proxy, runtime.pipeline, background, track_gradients=False)
            rows.append({"repeat": repeat, "index": index, "from_ratio": previous, "ratio": ratio,
                         "budget": budget, "prepare_ms": elapsed, "decoder": metadata})
            previous = ratio
            del proxy
    if not rows:
        raise ValueError("Switch sequence contains no changed requests.")
    values = torch.tensor([row["prepare_ms"] for row in rows], dtype=torch.float64)
    return {"samples": rows, "view": view.image_name, "device": torch.cuda.get_device_name(),
            "scope": "CUDA synchronized parameter preparation only; two Full warmup renders and all-anchor predecode excluded; one render after each timed request", 
            "predecode_seconds": predecode_seconds, "decoded_endpoint_bytes": decoded_bytes,
            "summary": {"resident": {"mean_ms": values.mean().item(), "median_ms": values.quantile(.5).item(),
                                      "p95_ms": values.quantile(.95).item(), "maximum_ms": values.max().item(), "switches": len(rows)}}}


def run_experiment(config, build, output, source_path):
    import torch
    from paretogs.runtime import ParetoGSRuntime
    from paretogs.decoder import BudgetDecoder, load_interpolation_curve
    from paretogs.evaluate import evaluate_interpolation
    settings = config["experiment"]
    kind = settings["kind"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", {"experiment": settings, "build": str(build), "source_path": source_path})
    if kind == 'independent_gns':
        try:
            result = run_independent_gns(settings, build, output, source_path)
        except Exception as error:
            atomic_json(output / 'result.json', {'status': 'failed', 'kind': kind, 'error': str(error)})
            raise
        result.update(status='complete', kind=kind)
        atomic_json(output / 'result.json', result)
        return
    runtime = ParetoGSRuntime(str(build / "runtime"), cache_anchors=None if kind == "switch_timing" else 2)
    runtime.dataset.source_path = source_path
    manifest = runtime.manifest
    if kind == "switch_timing":
        result = switch_timing(runtime, settings)
    elif kind == 'budget_curve':
        budgets = curve_budgets(runtime.anchors, settings['subdivisions'])
        evaluation = evaluate_interpolation(manifest['model_path'], manifest['iteration'], manifest['adapter'],
            budgets, str(output / 'metrics.json'), runtime=runtime, include_per_view=True,
            view_indices=settings.get('view_indices'), artifact_root=str(output / 'test'),
            save_image_budgets=budgets if settings.get('save_images', False) else [])
        result = {'evaluation_budgets': budgets, 'summary': {'full': {
            key: statistics.mean(evaluation['results'][str(k)]['metrics'][key] for k in budgets)
            for key in ('PSNR', 'SSIM', 'LPIPS')}}}
    else:
        budgets = midpoint_budgets(runtime.anchors)
        variants = settings.get("variants", list(settings.get("anchor_sets", {})))
        adapters = {name: manifest["adapter"] for name in variants}
        active_sets = {}
        if kind == "encoding":
            adapters.update(fp32=str(_raw_encoding(build, output)), fp16=str(build / "hierarchy/model/nested_adapters_fp16_full.pt"))
        if kind == "anchor_density":
            active_sets = {name: [round(runtime.gaussian_count * ratio) for ratio in ratios]
                           for name, ratios in settings["anchor_sets"].items()}
            budgets = density_budgets(list(active_sets.values()), settings["samples"])
        result = {"summary": {}, "evaluation_budgets": budgets}
        for name in variants:
            curve = runtime.curve
            anchors = active_sets.get(name, manifest["active_anchors"])
            if kind == "anchor_density":
                from paretogs.calibrate import calibrate_interpolation
                curve_path = output / name / "curve.json"
                calibrate_interpolation(manifest["model_path"], manifest["iteration"], str(build / "hierarchy/model/nested_adapters_fp16_full.pt"), str(curve_path),
                                        active_anchors=anchors, source_path=source_path, **settings["calibration"])
                curve = load_interpolation_curve(str(curve_path), anchors)
            from paretogs.adapters import load_adapter_bundle
            decoder = BudgetDecoder(runtime.gaussians, load_adapter_bundle(adapters[name], runtime.gaussian_count),
                                    runtime.stored_order, anchors=anchors, curve=curve)
            if not torch.equal(decoder.evaluation_order, runtime.evaluation_order) and kind in {"components", "anchor_density"}:
                raise ValueError("Ablation changed the member ordering.")
            decoder.dataset, decoder.pipeline, decoder.scene = runtime.dataset, runtime.pipeline, runtime.scene
            requested = sorted(set(budgets + (anchors if kind == "anchor_density" else [])))
            evaluation = evaluate_interpolation(manifest["model_path"], manifest["iteration"], adapters[name], requested,
                str(output / name / "metrics.json"), runtime=decoder, include_per_view=True,
                view_indices=settings.get("view_indices"), artifact_root=str(output / name / "test"),
                save_image_budgets=requested if settings.get("save_images", False) else [],
                variant=name if kind == "components" else "full")
            result["summary"][name] = {key: statistics.mean(evaluation["results"][str(k)]["metrics"][key] for k in budgets)
                                       for key in ("PSNR", "SSIM", "LPIPS")}
            if kind == "encoding":
                result["summary"][name]["additional_state_MB"] = Path(adapters[name]).stat().st_size / 1e6
            del decoder
    result.update(status="complete", kind=kind)
    atomic_json(output / "result.json", result)


def run_independent_gns(settings, build, output, source_path):
    from paretogs.train_hierarchy import _rebase_checkpoint
    from paretogs.checkpoints import write_anchor_state
    from utils.batch_training.config import dict_to_cli_args
    from utils.batch_training.paretogs import validate_runtime_output

    manifest = validate_runtime_output(build)
    iteration = manifest['iteration']
    source = build / 'backbone' / ('chkpnt{}.pth'.format(iteration))
    full = manifest['gaussian_count']
    budgets = midpoint_budgets(manifest['active_anchors'])
    training = settings['training']
    result = {'evaluation_budgets': budgets, 'independent_from_full': True,
              'source_checkpoint': str(source), 'summary': {}}
    repo = Path(__file__).resolve().parents[1]
    for budget in budgets:
        root = output / ('budget_{}'.format(budget))
        root.mkdir()
        start = root / ('start_chkpnt{}.pth'.format(training['start_iteration']))
        checkpoint = root / 'model' / ('chkpnt{}.pth'.format(training['end_iteration']))
        anchor = root / 'anchor_state.pt'
        _rebase_checkpoint(str(source), str(start), training['start_iteration'])
        args = independent_training_args(settings, full, budget, source_path, root)
        command = [sys.executable, str(repo / 'train.py'), *dict_to_cli_args(args)]
        print(' '.join(command), flush=True)
        # Inherit the worker process group, GPU, affinity and memory limits.
        subprocess.run(command, cwd=repo, check=True)
        actual = write_anchor_state(str(checkpoint), str(anchor))
        if actual != budget:
            raise ValueError('Independent GNS produced {} Gaussians; expected {}.'.format(actual, budget))
        command = [sys.executable, '-m', 'paretogs.evaluate_model',
                   '--model_path', str(build / 'backbone'), '--iteration', str(iteration),
                   '--output', str(root / 'metrics.json'), '--artifact_root', str(root / 'test'),
                   '--anchor_state_path', str(anchor), '--source_path', source_path, '--include_per_view']
        if settings.get('save_images', False):
            command.append('--save_images')
        if settings.get('view_indices'):
            command.extend(['--view_indices', *map(str, settings['view_indices'])])
        print(' '.join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)
        evaluation = load_json(root / 'metrics.json')
        parameters = load_json(root / 'model/training_parameters.json')
        result['summary'][str(budget)] = {key: evaluation['metrics'][key] for key in ('PSNR', 'SSIM', 'LPIPS')}
        result['summary'][str(budget)].update(budget=budget, ratio=budget / full,
                                              training_seconds=parameters['elapsed_seconds'])
        # The complete final checkpoint and compact state have both been verified.
        start.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "build", "output", "source_path"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    run_experiment(load_json(args.config), Path(args.build), Path(args.output), args.source_path)
