<h1 align="center">
  ParetoGS<br>
  <sub>High-Quality Continuous-Budget 3DGS</sub>
</h1>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

## 1. About ParetoGS

**ParetoGS: High-Quality Continuous-Budget 3D Gaussian Splatting with Sparse Adapted Anchors**

Xiaobin Deng · Changyu Diao<sup>*</sup> · Min Li · Duanqing Xu<sup>*</sup><br>
Zhejiang University · <sup>*</sup>Corresponding authors

[Project page](https://xiaobin2001.github.io/ParetoGS-Web/) · [Paper (PDF)](https://xiaobin2001.github.io/ParetoGS-Web/assets/paper.pdf) · [arXiv](https://arxiv.org/abs/2610.34235)

ParetoGS builds one 3D Gaussian Splatting (3DGS) scene model that can be decoded at different Gaussian budgets. Once it is ready, you can choose a budget to suit the device or application and obtain the corresponding Gaussian parameters.

The method optimizes a small set of nested models, called anchors, and connects the budgets between them. A compact residual atlas stores the adapted states; calibration on training views controls how neighboring anchors are interpolated. Together, these support continuous budget selection with a small set of explicitly optimized models.

The official implementation, full build pipeline, and paper experiments are here. To make your first model with a budget you can change, keep reading. The paper covers the method and experimental details; visit the project page for visual examples and budget-switching demos.

**Keywords:** 3D Gaussian Splatting (3DGS) · novel view synthesis · scalable Gaussian splatting · continuous budget · level of detail (LoD) · Gaussian pruning

## 2. Install and run

### Set up the environment

We tested this release on **Ubuntu 22.04 LTS, PyTorch 2.5.1 (CUDA 12.1), and an NVIDIA GeForce RTX 4090 with 24 GB VRAM**, using Python 3.10.19 and CUDA Toolkit 12.1.

You do not need to copy these versions exactly. You can use your own CUDA-enabled PyTorch environment with a matching torchvision, CUDA toolkit (`nvcc`), and C++ compiler; make sure the three bundled CUDA extensions below can compile and import in that environment. If you need to install PyTorch, see the [PyTorch installation page](https://pytorch.org/get-started/locally/). For a new environment, Python 3.10–3.12 fits the repository's Python syntax and NumPy 1.x dependency.

> **Windows mode:** the scripts automatically detect native Windows and disable Linux-only `resource_limits` (CPU affinity and both `guard`/`strict` RAM limits), with a startup notice. Tasks then run without these CPU/RAM limits. Process startup/cleanup and GPU lock files use Windows-compatible interfaces. CUDA compilation and training in this mode have not yet been tested on a Windows machine. Building extensions on Windows also requires MSVC C++ Build Tools and `nvcc` in your build environment.

The remaining Python dependencies are `numpy<2` (NumPy 1.x), `Pillow`, `opencv-python`, `plyfile`, `tqdm`, and `psutil`; extension builds also use `setuptools` and `wheel`. We keep the NumPy 1.x compatibility constraint without pinning exact package versions. [Ninja is optional](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.BuildExtension): it speeds up compilation, and PyTorch's extension builder falls back to the standard backend when it is absent.

With your environment active and PyTorch installed, you can choose and install these packages yourself, or use the list in [requirements.txt](requirements.txt) from the repository root:

```bash
python -m pip install -r requirements.txt
```

Then install the three bundled extensions in the same environment:

```bash
python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
python -m pip install ./submodules/simple-knn --no-build-isolation --no-deps
python -m pip install ./submodules/fused-ssim --no-build-isolation --no-deps
```

If compatible `simple_knn` and `fused_ssim` packages are already installed, you can skip the last two commands. LPIPS downloads pretrained weights on first use; prepare the Torch cache beforehand for offline evaluation.

### Prepare your data

Use undistorted COLMAP scenes with this layout:

```text
data/
  bicycle/
    images/
    sparse/0/
  garden/
    images/
    sparse/0/
  ...
```

The paper preset contains 13 scenes: `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `bonsai`, `counter`, `kitchen`, `room`, `playroom`, `drjohnson`, `train`, and `truck`. Names are case-sensitive. The default `resolution: -1` downsizes images wider than 1600 pixels to that width; every eighth sorted image is held out for testing.

### Build and evaluate the complete model

[training_config.json](training_config.json) runs the main-table method: train the ImprovedGS Full model, build the GNS anchor hierarchy, encode it, calibrate interpolation, and package the runtime. Replace the data path and choose your GPU:

```bash
# Preview the queue without starting training.
python run.py --data_root /path/to/data --gpu_id 0 --dry-run

# Build the complete continuous-budget models.
python run.py --data_root /path/to/data --gpu_id 0

# Render and evaluate the six main-table budgets.
python run.py --postprocess-only --gpu_id 0
```

For a first run on just Bicycle, add `--only-scene bicycle` to each command. Final evaluation is separate by default; set `run_postprocess: true` if you prefer to run it immediately after each build. Keep the same config and output root when building and evaluating.

Results appear under `output/paretogs/<scene>/`. The `runtime/` directory is the portable continuous-budget model. After evaluation, `result_train.json` and `result_test.json` contain multi-budget metrics, and `evaluation/` contains rendered test images and per-view results. Requested budgets are 100%, 30%, 10%, 5%, 1%, and 0.1% of the actual Full Gaussian count.

> Tip: once the model is built, try another evaluation budget—there is no need to rebuild it for each choice.

## 3. Your options

### Choose the build settings

The default is the paper's complete setup: a 30K ImprovedGS Full model, 11 anchors spanning 0.1%–100% of its actual Gaussian count, encoded residuals, and interpolation calibrated on training views. Training-time evaluation retains its 3K-iteration schedule.

Edit `training_config.json` for your own runs, or use `python run.py -c <config>`. [configs/paretogs_main_table.json](configs/paretogs_main_table.json) preserves the same paper method with a separate output directory and GPU setting.

| Config field | What you can explore |
| :---: | :---: |
| `paretogs.anchor_ratios` | Anchor budgets as fractions of the actual Full count |
| `paretogs.hierarchy` | GNS adaptation schedules and residual storage precision |
| `paretogs.compression` | SH rank and INT8 block size; defaults are `36` and `64` |
| `paretogs.calibration` | Training-view sampling, candidate p/q powers, and calibration weights |
| `postprocess_args.budget_ratios` | Budgets to render and evaluate after building |

Changing evaluation budgets lets you inspect more points on an existing model. Changing anchors, encoding, or calibration defines a different build or paper experiment. The [configuration guide](configs/README.md) explains the inputs and protocols.

[total_config.json](total_config.json) is the grouped parameter reference; [total_config.schema.json](total_config.schema.json) lists string-valued choices. Copy the fields you need into a runnable config. On Linux, the shipped `resource_limits` sets CPU and host RAM budgets to 45%; adjust or remove it for your machine. The default `guard` mode stops a task when monitored RSS exceeds its budget; it is not a hard memory cap.

### Use the model at another budget

Copy the complete `runtime/` directory when moving a model. It includes the Full parameters, Gaussian order, encoded states, and calibration curve needed for decoding and PLY export. Rendering also needs cameras from the dataset.

```python
from paretogs.runtime import ParetoGSRuntime

runtime = ParetoGSRuntime("output/paretogs/bicycle/runtime")
print(runtime.anchors)  # Anchor counts, from smallest to Full.
budget = runtime.anchors[0]
gaussians, metadata = runtime.decode(budget)
```

Choose any integer Gaussian count within the trained range to decode another budget. A PLY exported at one budget can be used as a standalone model. The [runtime and export guide](paretogs/README.md) covers export commands, evaluation, and cache options.

### Try the paper experiments

All six experiments use `run.py -c <config>`:

| Experiment | Config |
| :---: | :---: |
| Method components | [paper_components.json](configs/paper_components.json) |
| FP32 / FP16 / encoded atlas | [paper_encoding.json](configs/paper_encoding.json) |
| Number of anchors | [paper_anchor_density.json](configs/paper_anchor_density.json) |
| Budget-switch latency | [paper_switch_timing.json](configs/paper_switch_timing.json) |
| Dense quality–budget curves | [paper_budget_curve.json](configs/paper_budget_curve.json) |
| Independent GNS models | [paper_independent_gns.json](configs/paper_independent_gns.json) |

After building the source models, for example:

```bash
python run.py -c configs/paper_components.json --data_root /path/to/data --gpu_id 0
python run.py -c configs/paper_components.json --rebuild-summary
```

The default `build_root` is `output/paretogs`; update it if your builds are elsewhere. Some experiments also need raw anchors in `hierarchy/` or the Full checkpoint in `backbone/`; the [configuration guide](configs/README.md) lists these dependencies. Independent GNS includes training; the budget-curve experiment evaluates an existing runtime.

`repeat` defaults to `1`. Larger values keep independent `<scene>-1`, `<scene>-2`, … outputs and all their results. Use a new output directory to rerun a complete build or experiment. Single-stage training supports `checkpoint_iterations` and `resume_training`; complete multi-stage builds do not automatically resume an interrupted schedule.

## 4. Code origins and acknowledgements

ParetoGS is developed from the [ImprovedGS open-source repository](https://github.com/XiaoBin2001/Improved-GS). It uses ImprovedGS for the Full model and GNS for progressive pruning and adaptation, then adds the anchor encoding, interpolation, calibration, and runtime implementation of ParetoGS.

For this release, we streamlined the shared configuration workflow for building, evaluating, and running paper experiments. The CUDA extension has its own import name (`diff_gaussian_rasterization_paretogs`), the bundled GLM dependency is replaced by the small set of math operations used by the renderer, and RGB-only rendering can skip unused depth computation. Runtime packaging, saved settings, and per-repeat summaries make models easier to use and inspect.

We thank the ImprovedGS authors and the contributors to 3D Gaussian Splatting, AbsGS, Speedy-Splat, simple-knn, fused-ssim, and LPIPS for the implementations and ideas this work builds on. Original copyright and dependency notices are retained in the source.

## 5. Citation and license

If this work helps your research, please cite:

```bibtex
@misc{deng2026paretogshighqualitycontinuousbudget3d,
      title={ParetoGS: High-Quality Continuous-Budget 3D Gaussian Splatting with Sparse Adapted Anchors},
      author={Xiaobin Deng and Changyu Diao and Min Li and Duanqing Xu},
      year={2026},
      eprint={2610.34235},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.34235},
}
```

See [ImprovedGS](https://github.com/XiaoBin2001/Improved-GS) for the upstream paper and citation.

- Original code and algorithm implementations introduced by ParetoGS are licensed under [MIT](LICENSE.md).
- Inherited code follows its original licenses, primarily those of [3DGS](LICENSE.md#3dgs-license) and [ImprovedGS](LICENSE.md#improvedgs-license).
