# Use and export a ParetoGS model

English | [简体中文](README.zh-CN.md)

Once a build has finished, you can choose another budget, export a PLY, or move the model to another machine. This guide covers those steps. For installation and building, see the [main README](../README.md); for paper experiments, see the [configuration guide](../configs/README.md). Run commands from the repository root.

## Portable runtime

A complete build produces this directory:

```text
runtime/
  runtime_manifest.json
  adapter.pt
  interpolation_curve.json
  model/
    training_parameters.json
    nested_order.pt
    point_cloud/iteration_<N>/point_cloud.ply
```

Copy the entire directory to move the continuous-budget model. It contains all model assets needed for decoding and PLY export. Camera-based rendering and evaluation also need the source dataset.

```python
from paretogs.runtime import ParetoGSRuntime

runtime = ParetoGSRuntime("output/paretogs/bicycle/runtime")
budget = runtime.anchors[0]  # Choose an integer between the smallest anchor and Full.
gaussians, metadata = runtime.decode(budget)
```

`decode` uses the shared `BudgetDecoder` implementation and returns a Gaussian proxy accepted by the project renderer. The model and encoded assets stay resident; cameras load only when `runtime.scene` is accessed. Default endpoint caching retains at most two anchors. `cache_anchors=None` keeps all endpoints, as used by the paper's switch-timing experiment.

For explicit model/assets integration, `BudgetDecoder.from_assets(gaussians, adapter_path, order_path, curve_path)` provides the same decoder. The calibrated curve is included in packaged runtimes.

## PLY export and evaluation

Choose budgets within the runtime's trained range; the values below are examples and must fit the selected scene.

```bash
python -m paretogs.export \
  --runtime_dir output/paretogs/bicycle/runtime \
  --budget 180000 --output output/bicycle_180000.ply

# Evaluate selected budgets on all held-out views.
python -m paretogs.evaluate_runtime \
  --runtime_dir output/paretogs/bicycle/runtime \
  --budgets 30000 180000 --output output/bicycle_selected_budgets.json
```

`evaluate_runtime` uses the saved dataset location. Its optional `--view_indices` selects a subset; omit it for complete held-out evaluation. The main-table evaluation, including image export and both splits, is `python run.py --postprocess-only` with the build's config.

For a moved dataset, the complete-build evaluator accepts an explicit path:

```bash
python -m paretogs.postprocess \
  --model_path output/paretogs/bicycle \
  --source_path /path/to/data/bicycle
```

`python -m paretogs.demo --help` exposes the optional fixed-camera budget-sweep video. It uses the same decoder.

## Which files should I keep?

Keep `hierarchy/` for raw FP32/FP16 reconstruction and anchor-density experiments, and the Full checkpoint in `backbone/` for independent GNS comparisons. `anchor_state.pt` stores parameters and Gaussian IDs; it does not contain optimizer state and is not a training checkpoint.

For inference and export, the complete `runtime/` directory is enough. Keep the dataset as well if you need to render or evaluate its camera views. Complete multi-stage builds do not automatically resume an interrupted schedule; use a new output directory when rebuilding.

## Where to look in the code

If you want to adapt the implementation, these are the main starting points:

| Modules | Responsibility |
| :---: | :---: |
| `train_hierarchy.py`, `checkpoints.py` | GNS hierarchy, checkpoint transfer, and anchor states |
| `prefix.py`, `scene.py` | Gaussian membership/order and model loading |
| `adapters.py`, `fuse.py` | Residual organization, state/SH encoding, and quantization |
| `calibrate.py` | Training-view p/q calibration |
| `decoder.py`, `runtime.py` | Shared budget decoding and portable runtime loading |
| `package_runtime.py` | Package and verify the deployment assets |
| `evaluate.py`, `evaluate_model.py`, `evaluate_runtime.py`, `postprocess.py` | Explicit-asset, independent-model, runtime, and main-table evaluation |
| `experiments.py` | Config-driven paper experiments and summaries |
| `export.py`, `demo.py` | PLY export and budget-sweep video |

Encoding through `fuse` does not implicitly render; evaluation is explicit. All paper experiment variants reuse these implementations.
