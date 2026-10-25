# 使用与导出 ParetoGS 模型

[English](README.md) | 简体中文

构建完成后，你可以选择新的预算、导出 PLY，或把模型搬到另一台机器上。这份说明介绍这些操作。安装与构建见[主 README](../README.zh-CN.md)，论文实验见[配置说明](../configs/README.zh-CN.md)。下面的命令都在仓库根目录执行。

## 使用可搬运的 runtime

完整构建会产生下面的目录：

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

搬运连续预算模型时，复制整个目录即可。其中包含解码与 PLY 导出需要的全部模型资产；基于相机的渲染和评测还需要源数据集。

```python
from paretogs.runtime import ParetoGSRuntime

runtime = ParetoGSRuntime("output/paretogs/bicycle/runtime")
budget = runtime.anchors[0]  # 可选择最小锚点到 Full 之间的任意整数数量。
gaussians, metadata = runtime.decode(budget)
```

`decode` 复用共用的 `BudgetDecoder` 实现，返回可直接交给项目渲染器的高斯代理对象。模型与编码资产保持常驻；只有访问 `runtime.scene` 时才加载相机。默认端点缓存最多保留两个锚点。设置 `cache_anchors=None` 可以保留全部端点，论文的切换计时实验采用这种方式。

如果已有显式模型与资产，可以用 `BudgetDecoder.from_assets(gaussians, adapter_path, order_path, curve_path)` 接入同一解码器。打包的 runtime 已包含标定曲线。

## 导出 PLY 与评测

预算应在当前模型的训练范围内。下面的数量只是示例，请按所选场景调整。

```bash
python -m paretogs.export \
  --runtime_dir output/paretogs/bicycle/runtime \
  --budget 180000 --output output/bicycle_180000.ply

# 在全部测试视图上评测指定预算。
python -m paretogs.evaluate_runtime \
  --runtime_dir output/paretogs/bicycle/runtime \
  --budgets 30000 180000 --output output/bicycle_selected_budgets.json
```

`evaluate_runtime` 使用模型保存的数据集位置。可选参数 `--view_indices` 用于挑选视图；完整测试集评测时省略它。包含图像导出以及训练、测试两个划分的主表评测，使用构建时的配置执行 `python run.py --postprocess-only`。

数据集搬了位置时，完整构建的评测器也支持显式指定路径：

```bash
python -m paretogs.postprocess \
  --model_path output/paretogs/bicycle \
  --source_path /path/to/data/bicycle
```

想生成固定相机下的预算变化视频，可以从 `python -m paretogs.demo --help` 查看参数；视频使用相同的解码器。

## 哪些文件需要保留？

原始 FP32/FP16 重建和锚点密度实验需要 `hierarchy/`；独立 GNS 对照需要 `backbone/` 中的 Full 检查点。`anchor_state.pt` 保存参数和高斯 ID，不含优化器状态，因此不是训练检查点。

仅做推理和导出时，完整的 `runtime/` 目录就够了。如果需要按数据集相机渲染或评测，也请保留数据集。完整多阶段构建不会自动恢复中断的日程，重新构建时使用新的输出目录。

## 想改代码时，从哪里看起？

可以按你想调整的功能找到对应模块：

| 模块 | 主要职责 |
| :---: | :---: |
| `train_hierarchy.py`、`checkpoints.py` | GNS 层级、检查点传递和锚点状态 |
| `prefix.py`、`scene.py` | 高斯成员关系、顺序和模型加载 |
| `adapters.py`、`fuse.py` | 残差组织、状态/SH 编码与量化 |
| `calibrate.py` | 基于训练视图的 p/q 标定 |
| `decoder.py`、`runtime.py` | 共用预算解码与独立运行包加载 |
| `package_runtime.py` | 打包并核验部署资产 |
| `evaluate.py`、`evaluate_model.py`、`evaluate_runtime.py`、`postprocess.py` | 显式资产、独立模型、runtime 和主表评测 |
| `experiments.py` | 配置驱动的论文实验及汇总 |
| `export.py`、`demo.py` | PLY 导出和预算变化视频 |

`fuse` 编码不会隐式启动渲染；评测需要单独调用。论文各实验变体复用这些实现。
