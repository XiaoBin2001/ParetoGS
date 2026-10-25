<h1 align="center">
  ParetoGS<br>
  <sub>高质量、连续预算的 3DGS</sub>
</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

## 1. 认识 ParetoGS

**ParetoGS: High-Quality Continuous-Budget 3D Gaussian Splatting with Sparse Adapted Anchors**

Xiaobin Deng · Changyu Diao<sup>*</sup> · Min Li · Duanqing Xu<sup>*</sup><br>
浙江大学 · <sup>*</sup>通讯作者

[项目页](https://xiaobin2001.github.io/ParetoGS-Web/zh.html) · [论文 PDF](https://xiaobin2001.github.io/ParetoGS-Web/assets/paper.pdf) · arXiv：待更新

ParetoGS 希望让一个 3D Gaussian Splatting（3DGS）场景模型适应不同的高斯数量预算。模型构建完成后，就可以根据设备或应用的需要选择预算，解码出对应的高斯参数。

方法先优化少量相互嵌套的模型，称为锚点，再连接锚点之间的预算。紧凑的残差图谱保存适配后的状态，训练视图上的标定控制相邻锚点的插值方式。这样，只需显式优化少量模型，就可以支持连续预算选择。

这里放的是官方完整实现、构建流程和论文实验。想先做出一个能切换预算的模型，接着往下看；方法定义和实验细节交给论文；更多案例和预算切换演示可以去项目页看看。

**关键词：** 3D 高斯泼溅（3D Gaussian Splatting / 3DGS）· 新视角合成 · 可伸缩高斯泼溅 · 连续预算 · 细节层次（LoD）· 高斯剪枝

## 2. 安装与运行

### 准备环境

本开源版本的实际测试环境为 **Ubuntu 22.04 LTS、PyTorch 2.5.1（CUDA 12.1）、NVIDIA GeForce RTX 4090 24 GB**，使用 Python 3.10.19 和 CUDA Toolkit 12.1。

不必照搬这些版本。你也可以沿用自己的 CUDA 版 PyTorch 环境，搭配兼容的 torchvision、CUDA Toolkit（含 `nvcc`）和 C++ 编译器，确认下方三个随库 CUDA 扩展可以在该环境中编译、导入即可。需要安装 PyTorch 时，可以参考 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。如果新建环境，建议选择 Python 3.10–3.12，以兼顾本库的 Python 语法和 NumPy 1.x 依赖。

> **Windows 模式：**脚本会自动识别原生 Windows，关闭依赖 Linux 的 `resource_limits`（CPU 亲和性及 `guard`/`strict` 内存限制），并在启动时提示；此时任务不受这些 CPU/内存限制。进程启动/清理和 GPU 锁文件使用适用于 Windows 的接口。本模式尚未在 Windows 实机上验证 CUDA 编译与训练。Windows 下编译扩展还需准备 MSVC C++ Build Tools，并确保构建环境可以找到 `nvcc`。

其余 Python 依赖为 `numpy<2`（NumPy 1.x）、`Pillow`、`opencv-python`、`plyfile`、`tqdm`、`psutil`；编译扩展还需要 `setuptools` 和 `wheel`。保留 NumPy 1.x 的兼容约束，其余不固定具体版本。[Ninja 是可选的](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.BuildExtension)，用于加快编译；未安装时，PyTorch 的扩展构建工具会使用标准构建后端。

激活环境、装好 PyTorch 后，如果希望自己选择上述 Python 包的版本，可以自行安装；也可以在仓库根目录使用 [requirements.txt](requirements.txt) 直接安装：

```bash
python -m pip install -r requirements.txt
```

然后在同一环境中安装三个随库扩展：

```bash
python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
python -m pip install ./submodules/simple-knn --no-build-isolation --no-deps
python -m pip install ./submodules/fused-ssim --no-build-isolation --no-deps
```

如果已有环境中安装了兼容的 `simple_knn` 和 `fused_ssim`，可以跳过最后两条命令。LPIPS 首次使用时会下载预训练权重；离线评测前，请先准备好对应的 Torch 缓存。

### 准备数据

使用完成去畸变的 COLMAP 场景，按下面的方式摆放：

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

论文预设包含 13 个场景：`bicycle`、`flowers`、`garden`、`stump`、`treehill`、`bonsai`、`counter`、`kitchen`、`room`、`playroom`、`drjohnson`、`train` 和 `truck`。场景名区分大小写。默认 `resolution: -1` 会将宽度超过 1600 像素的图像缩放至该宽度；按排序后的图像每八张留出一张作为测试视图。

### 构建并评测完整模型

[training_config.json](training_config.json) 已配置论文主表流程：训练 ImprovedGS Full 模型、建立 GNS 锚点层级、编码、标定插值，最后打包 runtime。替换数据路径并选择 GPU 即可：

```bash
# 先预览运行队列，此命令不会开始训练。
python run.py --data_root /path/to/data --gpu_id 0 --dry-run

# 构建完整的连续预算模型。
python run.py --data_root /path/to/data --gpu_id 0

# 渲染并评测论文主表的六个预算。
python run.py --postprocess-only --gpu_id 0
```

第一次尝试时，可以在每条命令后加上 `--only-scene bicycle`，先跑一个场景。默认将最终评测作为独立步骤；如果希望每个场景构建后立即评测，将 `run_postprocess` 设为 `true` 即可。构建与评测请使用同一份配置和输出目录。

结果保存在 `output/paretogs/<scene>/`，其中 `runtime/` 就是可搬运的连续预算模型。评测完成后，`result_train.json` 和 `result_test.json` 保存多预算指标，`evaluation/` 保存测试渲染图像及逐视图结果。默认评测预算为实际 Full 高斯数量的 100%、30%、10%、5%、1% 和 0.1%。

> 小提示：模型建好后，换个评测预算就能继续探索，不用每次都重新构建。

## 3. 试试这些选项

### 调整模型构建设置

默认配置对应论文完整方法：30K 次迭代的 ImprovedGS Full 模型、覆盖其实际高斯数量 0.1%–100% 的 11 个锚点、残差编码，以及基于训练视图的插值标定。训练中评测仍每 3K 次迭代执行一次。

日常使用可以编辑 `training_config.json`，也可以通过 `python run.py -c <config>` 选择配置。[configs/paretogs_main_table.json](configs/paretogs_main_table.json) 保留相同的论文方法，使用单独的输出目录和 GPU 设置。

| 配置字段 | 可以调整什么 |
| :---: | :---: |
| `paretogs.anchor_ratios` | 锚点预算，相对于实际 Full 数量的比例 |
| `paretogs.hierarchy` | GNS 适配日程和残差存储精度 |
| `paretogs.compression` | SH 秩与 INT8 分块大小，默认分别为 `36` 和 `64` |
| `paretogs.calibration` | 训练视图采样、候选 p/q 指数和标定权重 |
| `postprocess_args.budget_ratios` | 构建后希望渲染与评测的预算 |

修改评测预算，可以查看同一个已有模型上的更多位置；修改锚点、编码或标定设置，则对应另一套构建或论文实验。[配置说明](configs/README.zh-CN.md) 中介绍了各自的输入与协议。

[total_config.json](total_config.json) 是分组参数参考，[total_config.schema.json](total_config.schema.json) 列出字符串选项的可选值。把需要的字段复制到运行配置中即可。在 Linux 上，随附配置中的 `resource_limits` 将 CPU 和主机内存预算设为 45%，可以按自己的机器调整或移除。默认 `guard` 模式在监测到 RSS 超限时停止任务，并非硬性内存上限。

### 换一个预算使用模型

搬运模型时，请复制完整的 `runtime/` 目录。它包含解码与 PLY 导出需要的 Full 参数、高斯顺序、编码状态和标定曲线。渲染还需要数据集中的相机信息。

```python
from paretogs.runtime import ParetoGSRuntime

runtime = ParetoGSRuntime("output/paretogs/bicycle/runtime")
print(runtime.anchors)  # 从最小锚点到 Full 的高斯数量。
budget = runtime.anchors[0]
gaussians, metadata = runtime.decode(budget)
```

在训练覆盖的范围内，选择任意整数高斯数量即可解码另一个预算。按某个预算导出的 PLY 可以作为独立模型使用。[模型使用与导出说明](paretogs/README.zh-CN.md) 提供了导出、评测和缓存设置的具体示例。

### 运行论文实验

六类实验都使用 `run.py -c <config>`：

| 实验 | 配置 |
| :---: | :---: |
| 方法组件 | [paper_components.json](configs/paper_components.json) |
| FP32 / FP16 / 编码图谱 | [paper_encoding.json](configs/paper_encoding.json) |
| 锚点数量 | [paper_anchor_density.json](configs/paper_anchor_density.json) |
| 预算切换延迟 | [paper_switch_timing.json](configs/paper_switch_timing.json) |
| 密集质量—预算曲线 | [paper_budget_curve.json](configs/paper_budget_curve.json) |
| 独立 GNS 模型 | [paper_independent_gns.json](configs/paper_independent_gns.json) |

准备好源模型后，例如：

```bash
python run.py -c configs/paper_components.json --data_root /path/to/data --gpu_id 0
python run.py -c configs/paper_components.json --rebuild-summary
```

默认 `build_root` 为 `output/paretogs`；如果模型在其他位置，请相应修改。部分实验还需要 `hierarchy/` 中的原始锚点，或 `backbone/` 中的 Full 检查点，具体见[配置说明](configs/README.zh-CN.md)。独立 GNS 实验会进行训练；预算曲线实验只评测已有 runtime。

`repeat` 默认为 `1`。增大后会保留 `<scene>-1`、`<scene>-2` 等独立目录及全部结果。重新构建或测量时请使用新的输出目录。单阶段训练支持 `checkpoint_iterations` 和 `resume_training`；完整多阶段构建不提供中断后的自动接续。

## 4. 代码来源与致谢

ParetoGS 从 [ImprovedGS 开源代码库](https://github.com/XiaoBin2001/Improved-GS) 发展而来，使用 ImprovedGS 训练 Full 模型，通过 GNS 逐步剪枝与适配，再加入 ParetoGS 的锚点编码、插值、标定和 runtime 实现。

这次开源整理也简化了构建、评测和论文实验共用的配置流程。CUDA 扩展使用独立导入名 `diff_gaussian_rasterization_paretogs`；原来随库携带的 GLM 被替换为渲染器实际需要的一小组数学运算；纯 RGB 渲染可以跳过不需要的深度计算。独立运行包、保存的配置和逐次重复汇总，也方便大家使用模型、查看结果。

感谢 ImprovedGS 作者，以及 3D Gaussian Splatting、AbsGS、Speedy-Splat、simple-knn、fused-ssim 和 LPIPS 的贡献者提供的实现与研究思路。源码中保留了原有版权声明和依赖说明。

## 5. 引用与许可

BibTeX：待更新。

上游论文及引用信息见 [ImprovedGS](https://github.com/XiaoBin2001/Improved-GS)。

- ParetoGS 本工作新增的代码与算法实现采用 [MIT 许可](LICENSE.md)。
- 继承部分遵循原有工作的许可，主要为 [3DGS](LICENSE.md#3dgs-license) 和 [ImprovedGS](LICENSE.md#improvedgs-license)。
