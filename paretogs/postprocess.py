"""Evaluate all requested budgets of one complete ParetoGS build."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


def _bool(value: str) -> bool:
    if value.lower() not in {'true', 'false'}:
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True, help='Complete build directory containing runtime/')
    parser.add_argument('--budget_ratios', nargs='+', type=float, default=[1, .3, .1, .05, .01, .001])
    for name, default in [('include_train', True), ('skip_test', False),
                          ('no_save_test_images', False), ('no_save_train_images', True)]:
        parser.add_argument('--' + name, type=_bool, nargs='?', const=True, default=default)
    for name in ('fps_warmup_rounds', 'fps_measure_rounds'):
        parser.add_argument('--' + name, type=int, default=5)
    # Unspecified settings come from the model's saved effective parameters.
    for name in ('source_path', 'images', 'data_device'):
        parser.add_argument('--' + name)
    parser.add_argument('--resolution', type=int)
    for name in ('white_background', 'eval', 'debug', 'antialiasing'):
        parser.add_argument('--' + name, type=_bool, nargs='?', const=True)
    args = parser.parse_args(argv)
    if any(not math.isfinite(r) or not 0 < r <= 1 for r in args.budget_ratios):
        parser.error('budget_ratios must be finite values in (0, 1]')
    if args.fps_warmup_rounds < 0 or args.fps_measure_rounds <= 0:
        parser.error('FPS requires nonnegative warmup and positive measurement rounds')
    if not args.include_train and args.skip_test:
        parser.error('At least one evaluation split is required')
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    from paretogs.runtime import ParetoGSRuntime
    from paretogs.evaluate import evaluate_interpolation
    from utils.experiment_utils import atomic_json, load_json
    from utils.general_utils import safe_state

    safe_state(False)
    root = Path(args.model_path).absolute()
    runtime = ParetoGSRuntime(str(root / 'runtime'))
    for target, fields in [(runtime.dataset, ('source_path', 'images', 'resolution', 'data_device',
                                              'white_background', 'eval')),
                            (runtime.pipeline, ('debug', 'antialiasing'))]:
        for name in fields:
            value = getattr(args, name)
            if value is not None:
                setattr(target, name, value)
    count = runtime.gaussian_count
    budgets = sorted(set(round(count * ratio) for ratio in args.budget_ratios))
    if budgets[0] < runtime.anchors[0]:
        raise ValueError('Requested budget is below the smallest trained anchor: {}'.format(runtime.anchors[0]))
    summary = load_json(str(root / 'training_summary.json'), default={})
    seconds = summary.get('total_training_seconds')
    minutes = float(seconds) / 60 if seconds is not None else None
    manifest = runtime.manifest
    splits = (['train'] if args.include_train else []) + ([] if args.skip_test else ['test'])
    for index, split in enumerate(splits):
        save_images = not getattr(args, 'no_save_' + split + '_images')
        path = root / ('result_' + split + '.json')
        result = evaluate_interpolation(
            manifest['model_path'], int(manifest['iteration']), manifest['adapter'], budgets, str(path),
            evaluation_split=split, runtime=runtime, artifact_root=str(root / 'evaluation' / split),
            save_image_budgets=budgets if save_images else [],
            fps_warmup_rounds=args.fps_warmup_rounds, fps_measure_rounds=args.fps_measure_rounds,
            training_time_minutes=minutes, opacity_root=str(root / 'evaluation') if index == 0 else '',
        )
        result['budget_results'] = [
            dict(value['metrics'], budget=int(budget), budget_ratio=int(budget) / count,
                 full_gaussian_count=count)
            for budget, value in result.pop('results').items()
        ]
        result['evaluation_settings'] = {
            name: getattr(runtime.dataset, name) for name in
            ('source_path', 'images', 'resolution', 'data_device', 'white_background', 'eval')
        }
        result['evaluation_settings'].update(
            {name: getattr(runtime.pipeline, name) for name in ('debug', 'antialiasing')})
        result['evaluation_settings'].update(fps_warmup_rounds=args.fps_warmup_rounds,
                                             fps_measure_rounds=args.fps_measure_rounds)
        atomic_json(path, result)
    if summary:
        summary['evaluation_seconds'] = time.perf_counter() - started
        atomic_json(root / 'training_summary.json', summary)
        # Keep the existing batch-level view consistent after postprocess-only.
        batch_path = root.parent / 'training_summary.json'
        batch = load_json(str(batch_path), default={})
        changed = False
        for entry in batch.get('scenes', []):
            if entry.get('runtime_dir') == str((root / 'runtime').resolve()):
                entry['evaluation_seconds'] = summary['evaluation_seconds']
                changed = True
        if changed:
            atomic_json(batch_path, batch)


if __name__ == '__main__':
    main()
