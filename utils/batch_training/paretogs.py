"""Build complete ParetoGS assets through the existing stage entry points."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from utils.artifact_utils import file_sha256
from utils.batch_training.config import build_scene_paths, build_train_payload, dict_to_cli_args, filter_config_scenes
from utils.batch_training.gpu import release_gpu_lock, select_scene_gpu
from utils.batch_training.processes import run_command
from utils.experiment_utils import backup_batch_training_config, atomic_json
from utils.resource_limits import check_resource_support, resolve_resource_limits


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _full_count(path: Path) -> int:
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Full model is not a PLY: {}'.format(path))
        count = None
        for line in stream:
            if line.startswith(b'element vertex '):
                count = int(line.split()[-1])
            if line.strip() == b'end_header':
                break
        if count is None or count <= 0:
            raise ValueError('Full model has no positive Gaussian count: {}'.format(path))
        return count


def _budgets(count: int, ratios: list[float]) -> list[int]:
    values = [int(round(count * ratio)) for ratio in ratios]
    if values[0] <= 0 or values != sorted(set(values)) or values[-1] != count:
        raise ValueError('Anchor ratios do not give distinct positive budgets ending at full.')
    return values


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get('scenes') or not isinstance(config['scenes'], list):
        raise ValueError('ParetoGS config requires a nonempty scenes list.')
    settings = config['paretogs']
    allowed = {
        'hierarchy': {'iteration', 'rebase_iteration', 'post_prune_iterations', 'precision'},
        'compression': {'sh_rank', 'block_size'},
        'calibration': {'common_powers', 'shell_gate_powers', 'progresses', 'view_count', 'lambda_dssim', 'monotonic_weight', 'smoothness_weight'},
    }
    if set(settings) != {'anchor_ratios', *allowed}:
        raise ValueError('ParetoGS settings require anchor_ratios, hierarchy, compression and calibration.')
    for key, fields in allowed.items():
        if set(settings[key]) != fields:
            raise ValueError('Unexpected or missing ParetoGS {} settings.'.format(key))
    ratios = settings['anchor_ratios']
    if (not isinstance(ratios, list) or len(ratios) < 2
            or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in ratios)
            or ratios != sorted(set(ratios)) or ratios[0] <= 0 or ratios[-1] != 1):
        raise ValueError('Anchor ratios must increase strictly from a positive value to 1.')
    h = settings['hierarchy']
    if h['precision'] != 'fp16' or h['post_prune_iterations'] <= 0 or h['iteration'] - h['post_prune_iterations'] <= h['rebase_iteration']:
        raise ValueError('ParetoGS full build requires FP16 teacher and a nonempty pruning/fixed-member schedule.')
    if settings['compression']['block_size'] <= 0:
        raise ValueError('Compression block sizes must be positive.')
    if not 1 <= settings['compression']['sh_rank'] <= 45:
        raise ValueError('SH channel rank must be in [1,45].')
    calibration = settings['calibration']
    if any(not calibration[key] or any(x <= 0 for x in calibration[key])
           for key in ['common_powers', 'shell_gate_powers']):
        raise ValueError('Calibration requires positive p/q candidates.')
    if (not calibration['progresses'] or any(not 0 < x < 1 for x in calibration['progresses'])
            or calibration['view_count'] <= 0 or not 0 <= calibration['lambda_dssim'] <= 1
            or calibration['monotonic_weight'] < 0 or calibration['smoothness_weight'] < 0):
        raise ValueError('Invalid calibration views, samples or loss weights.')
    return settings


def validate_runtime_output(root: Path) -> dict:
    """Check every required runtime asset before scheduling GPU evaluation."""
    from utils.artifact_utils import load_runtime_manifest
    manifest = load_runtime_manifest(root / "runtime")
    paths = [Path(manifest[key]) for key in ("adapter", "interpolation_curve", "nested_order")]
    model = Path(manifest["model_path"])
    paths.extend([model / "training_parameters.json", model / "point_cloud" /
                  ("iteration_{}".format(manifest["iteration"])) / "point_cloud.ply"])
    for path in paths:
        if not path.is_file():
            raise ValueError("Incomplete ParetoGS runtime; missing {}".format(path))
    return manifest


def validate_build_config(config: dict[str, Any], dry_run: bool) -> list[tuple]:
    settings = _validate(config)
    if config.get("resume_training", False):
        raise ValueError("resume_training applies to single-stage training; use --postprocess-only for a completed ParetoGS build. Interrupted multi-stage builds require a new output path.")
    scenes = config['scenes']
    repeat = config.get('repeat', 1)
    jobs = []
    for run_index in range(repeat):
        for scene in scenes:
            data_dir, output_dir = build_scene_paths(config, scene, run_index)
            full = Path(output_dir) / 'backbone'
            train = build_train_payload(config, scene, data_dir, str(full))
            iterations = int(train['iterations'])
            if train.get('training_method') != 'improvedgs':
                raise ValueError('Full backbone must use ImprovedGS.')
            if train.get('checkpoint_iterations') != [iterations] or train.get('save_iterations') != [iterations]:
                raise ValueError('Full backbone must save its final PLY and checkpoint.')
            if iterations < settings['hierarchy']['iteration'] or int(train['budget']) <= 0:
                raise ValueError('Full training length/budget does not support this hierarchy.')
            if not dry_run and not Path(data_dir).is_dir():
                raise FileNotFoundError('Scene data is missing: {}'.format(data_dir))
            if Path(output_dir).exists() and not dry_run:
                raise FileExistsError('Refusing to overwrite a ParetoGS build: {}'.format(output_dir))
            jobs.append((run_index + 1, scene, str(Path(output_dir).absolute()), train))
    return jobs


def run_paretogs_config(config_path: str, config: dict[str, Any], dry_run: bool,
                        scene_filter: set[str], rebuild_total: bool, postprocess=None) -> int:
    """Build each repeat independently through the original method stages."""
    config = dict(config, scenes=filter_config_scenes(config, scene_filter))
    settings = _validate(config)
    jobs = validate_build_config(config, dry_run)
    limits = config.get('resource_limits')
    python = config.get('python_executable', sys.executable)
    if not dry_run:
        Path(config['output_root']).mkdir(parents=True, exist_ok=True)
        backup_batch_training_config(config, config_path)
    first_failure = 0
    summaries = []
    for repeat, scene, output_dir, train in jobs:
        root = Path(output_dir)
        full, hierarchy = root/'backbone', root/'hierarchy'
        model, temporary, runtime = hierarchy/'model', root/'encoding', root/'runtime'
        full_iterations = int(train['iterations'])
        nominal_count = int(train['budget'])
        budgets = _budgets(nominal_count, settings['anchor_ratios'])
        clock_report, curve, adapter = temporary/'calibration_search.json', temporary/'curve.json', temporary/'adapter.pt'
        lock = None
        phase_times = {}
        started = time.perf_counter()
        state = {"repeat": repeat, "seed": 0, "scene": scene['name'], "model_path": str(root),
                 "train": "running", "postprocess": "pending"}
        if not dry_run:
            atomic_json(root / "run_status.json", state)
        try:
            gpu, lock = select_scene_gpu(config, scene['name'], config.get('gpu_id'), dry_run)

            def execute(phase: str, module: str | None, payload: dict[str, Any]) -> None:
                command = [python, '-m', module] if module else [python, 'train.py']
                command.extend(dict_to_cli_args(payload))
                begin = time.perf_counter()
                code = run_command(command, gpu_id=gpu, dry_run=dry_run, resource_limits=limits,
                                   resource_path=str(root/(phase+'_resources.json')))
                phase_times[phase] = time.perf_counter() - begin
                if code:
                    raise RuntimeError('ParetoGS {} failed with exit code {}'.format(phase, code), code)

            execute('full', None, train)
            if not dry_run:
                count = _full_count(full/'point_cloud'/('iteration_{}'.format(full_iterations))/'point_cloud.ply')
                if count > nominal_count:
                    raise ValueError('Full Gaussian count exceeds the configured budget.')
                budgets = _budgets(count, settings['anchor_ratios'])
                full_time = float(json.loads((full/'training_parameters.json').read_text())['elapsed_seconds'])
                _write_json(root/'training_summary.json', {
                    'scene': scene['name'], 'status': 'building_anchors',
                    'full_training_seconds': full_time, 'actual_full_gaussians': count,
                    'configured_full_budget': nominal_count, 'anchors': budgets,
                })
            else:
                print('ParetoGS {}: nominal anchor preview {}; actual counts will use the trained full PLY.'.format(scene['name'], budgets))
            execute('hierarchy', 'paretogs.train_hierarchy', {
                'source_path': train['source_path'], 'output_root': str(hierarchy),
                'budgets': budgets, 'source_budget': nominal_count,
                'full_checkpoint': str(full/('chkpnt{}.pth'.format(full_iterations))),
                'full_model_path': str(full), 'full_iteration': full_iterations,
                'data_device': train.get('data_device', 'cpu'),
                **settings['hierarchy'],
            })
            if not dry_run:
                built = json.loads((hierarchy/'pipeline_summary.json').read_text())
                if built['budgets'] != budgets:
                    raise ValueError('Constructed hierarchy budgets differ from the requested full-model ratios.')
            execute('compression', 'paretogs.fuse', {
                'model_path': str(model), 'iteration': full_iterations,
                'input_bundle': str(model/'nested_adapters_fp16_full.pt'),
                **settings['compression'], 'output': str(adapter),
            })
            execute('calibration', 'paretogs.calibrate', {
                'model_path': str(model), 'iteration': full_iterations,
                'adapter_path': str(model/'nested_adapters_fp16_full.pt'),
                'output': str(clock_report), **settings['calibration'],
            })
            if not dry_run:
                fitted = json.loads(clock_report.read_text())
                compact = {key: fitted[key] for key in ['format', 'version', 'anchors']}
                compact['intervals'] = {key: {field: value[field] for field in ['common_power', 'shell_gate_power']}
                                        for key, value in fitted['intervals'].items()}
                _write_json(curve, compact)
            execute('package', 'paretogs.package_runtime', {
                'model_path': str(model), 'iteration': full_iterations, 'target_adapter': str(adapter),
                'curve_path': str(curve), 'active_anchors': budgets, 'output_dir': str(runtime),
            })
            if not dry_run:
                if file_sha256(adapter) != file_sha256(runtime/'adapter.pt'):
                    raise ValueError('Packaged atlas differs from the encoded source.')
                adapter.unlink()  # The verified runtime copy is the retained deployment asset.
                anchor_time = sum(float(json.loads((Path(s['model_path'])/'training_parameters.json').read_text())['elapsed_seconds'])
                                  for s in built['stages'])
                summary = {
                    'scene': scene['name'], 'repeat': repeat, 'status': 'complete', 'gpu': gpu,
                    'configured_full_budget': nominal_count, 'actual_full_gaussians': budgets[-1], 'anchors': budgets,
                    'full_training_seconds': full_time, 'anchor_chain_training_seconds': anchor_time,
                    'total_training_seconds': full_time+anchor_time,
                    'construction_wall_seconds': time.perf_counter()-started,
                    'phase_wall_seconds': phase_times, 'evaluation_seconds': 0.0,
                    'runtime_dir': str(runtime.resolve()),
                    'training_time_definition': 'loop_excluding_reporting_including_ply_and_checkpoint',
                    'training_time_note': 'Training-loop elapsed_seconds excludes reporting and includes PLY exports and checkpoint writes. Anchor times are aggregated. evaluation_seconds refers only to the final evaluation sweep.',
                }
                _write_json(root/'training_summary.json', summary)
                summaries.append(summary)
                state.update(train="completed", train_exit_code=0)
                atomic_json(root / "run_status.json", state)
            if config.get('run_postprocess', False) and postprocess is not None:
                code = postprocess(config, scene, train['source_path'], root, state, gpu, dry_run)
                if code:
                    raise RuntimeError('ParetoGS postprocessing failed', code)
                if not dry_run:
                    summaries[-1] = json.loads((root / 'training_summary.json').read_text())
            elif not dry_run:
                state['postprocess'] = 'skipped'
                atomic_json(root / "run_status.json", state)
        except KeyboardInterrupt:
            return 130
        except (RuntimeError, ValueError, OSError) as error:
            print('ParetoGS {} failed: {}'.format(scene['name'], error), file=sys.stderr)
            code = int(error.args[1]) if len(error.args) == 2 and isinstance(error.args[1], int) else 1
            first_failure = first_failure or code
            if not dry_run:
                if state['train'] != 'completed':
                    state.update(train='failed', train_exit_code=code)
                atomic_json(root / 'run_status.json', state)
                _write_json(root/'build_failure.json', {'scene': scene['name'], 'error': str(error), 'phase_wall_seconds': phase_times})
            if config.get('stop_on_error', True):
                return code
        finally:
            release_gpu_lock(lock)
    if not dry_run:
        _write_json(Path(config['output_root'])/'training_summary.json', {'scenes': summaries, 'exit_code': first_failure})
    return first_failure
