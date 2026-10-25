# Configurations and paper protocols

English | [简体中文](README.zh-CN.md)

The [main README](../README.md) takes you through a first complete run. Use this guide when you want to change a build or choose a paper experiment.

Run commands from the repository root with `python run.py -c <config>`. Dataset and output paths are resolved from that directory. You can override them with `--data_root` and `--output_root`, choose a GPU with `--gpu_id`, and select a scene with `--only-scene`. Add `--dry-run` to check the queue before starting.

## Complete model

The root [training_config.json](../training_config.json) runs the complete pipeline. [paretogs_main_table.json](paretogs_main_table.json) uses the same method/scenes with a separate output root and GPU setting. Both build a 30K ImprovedGS Full model, train nested GNS anchors, encode residuals, calibrate p/q on training views, and package `runtime/`.

Anchor ratios are relative to the actual Full count:

```text
0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
0.064, 0.128, 0.256, 0.512, 1.0
```

`paretogs.hierarchy` defines the GNS schedule and FP16 residual storage. `paretogs.compression` sets SH rank 36 and INT8 block size 64; state rank remains full. `paretogs.calibration` defines training-view sampling, p/q candidates, and the calibration objective. The Full backbone uses LAS, RAP, EAS, and MU as specified in `train_args`.

Final evaluation is controlled by `run_postprocess` and `postprocess_args`, or can be run separately with `--postprocess-only`. The main-table ratios are `[1, 0.3, 0.1, 0.05, 0.01, 0.001]`. Keep the same config/output root when postprocessing. For a relocated dataset, set `scenes[].postprocess_args.source_path` to that scene's new path, or use the [single-build evaluation command](../paretogs/README.md).

## Paper experiments

These configs set `pipeline: paretogs_experiment`. Each scene reads an existing complete build from `build_root` and writes measurements to its independent `output_root`.

| Configuration | Protocol | Required build assets |
| :---: | :---: | :---: |
| [paper_components.json](paper_components.json) | Five variants at ten adjacent-anchor midpoints | `runtime/` |
| [paper_encoding.json](paper_encoding.json) | Raw FP32, FP16, and encoded atlas at the same budgets and p/q | `runtime/` and raw `hierarchy/` states |
| [paper_anchor_density.json](paper_anchor_density.json) | Garden: five, three, or two anchors; 50 common non-anchor budgets | `runtime/` and raw `hierarchy/` states |
| [paper_switch_timing.json](paper_switch_timing.json) | All anchor endpoints resident; 108 changed-budget requests per scene | `runtime/` |
| [paper_budget_curve.json](paper_budget_curve.json) | Bicycle/Garden/Stump: ten subdivisions per anchor interval, 101 budgets | `runtime/` |
| [paper_independent_gns.json](paper_independent_gns.json) | The same three scenes: independent GNS at adjacent-anchor midpoints | `runtime/` and the `backbone/` Full checkpoint |

```bash
python run.py -c configs/paper_encoding.json --data_root /path/to/data --gpu_id 0 --dry-run
python run.py -c configs/paper_encoding.json --data_root /path/to/data --gpu_id 0
python run.py -c configs/paper_encoding.json --rebuild-summary
```

The default build location is `output/paretogs`. If using `configs/paretogs_main_table.json`, change the experiment's `build_root` to `output/paretogs_main_table`. With `repeat > 1`, inputs and outputs use the corresponding numbered scene names. To evaluate one existing build repeatedly, set a scene's `build_name`, for example `{"name": "bicycle", "build_name": "bicycle"}`. Every repeat is retained.

FP32 encoding is reconstructed from raw anchor parameters and persistent Gaussian IDs, rather than upcast from FP16 residuals. Anchor-density variants recalibrate the retained anchor subsets. Switch timing excludes loading and all-anchor predecode, measures CUDA-synchronized parameter preparation, and renders once after each timed request. Quantiles pool all measured switches; this protocol differs from the runtime's default two-endpoint cache.

Independent GNS starts every target budget from the same Full checkpoint using `experiment.training`. Each target retains its final checkpoint, anchor state, and metrics. The experiment does training; the dense budget curve only evaluates an existing runtime.

Each scene/repeat saves its config and result. `summary.csv` includes task status; `summary.json` records expected, completed, failed, missing, and incomplete tasks. Aggregates use completed tasks only and explicitly mark partial results. `--rebuild-summary` reads existing records without training or rendering and returns nonzero when results are incomplete.

## Single-stage training and parameter reference

[3dgs.json](3dgs.json), [absgs.json](absgs.json), and [improvedgs.json](improvedgs.json) expose the retained training backbones. Their shipped repeat count is 3; all numbered results remain. GNS requires `train_args.start_checkpoint_file` and optionally `start_checkpoint_dir`, or an existing run with top-level `resume_training: true`.

[total_config.json](../total_config.json) lists grouped defaults and runner examples. [total_config.schema.json](../total_config.schema.json) lists string choices. Copy relevant fields into a runnable config, using `scenes[].train_args` or `scenes[].postprocess_args` for per-scene overrides.

## Evaluation, timing, and resources

Training uses the code's default evaluation nodes every 3000 iterations; configs do not override this schedule. Training-report LPIPS and final evaluation are separate settings. The complete build disables training-report LPIPS by default; final evaluation includes it.

`elapsed_seconds` excludes training-time reporting and includes PLY/checkpoint writes. PLY time is also recorded in `ply_save_seconds`; `training_time_definition` identifies this convention. Total build wall time additionally includes loading, hierarchy construction, encoding, calibration, and packaging. Full exports its final PLY and checkpoint; GNS stages retain checkpoints and anchor states without a PLY per stage.

`resource_limits` uses Linux interfaces and applies to runner subprocesses. CPU percentages control affinity and threads. RAM percentages use the host/cgroup capacity; `guard` monitors process-tree RSS and `strict` requires a writable cgroup v2 memory controller. Omit the object to run without these limits. Windows mode automatically sets this object to `null` in the loaded config and reports that CPU/RAM limits are disabled; it does not emulate cgroups or silently substitute another memory limit. Direct calls to `train.py` or individual modules do not apply runner limits.
