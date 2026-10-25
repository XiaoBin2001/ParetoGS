# 配置与论文实验

[English](README.md) | 简体中文

[主 README](../README.zh-CN.md) 介绍了第一次完整运行的方法。如果想调整构建设置，或选择某项论文实验，可以接着看这里。

所有命令都在仓库根目录执行，使用 `python run.py -c <config>`。数据和输出路径相对于这个目录解析；可以用 `--data_root`、`--output_root` 覆盖路径，用 `--gpu_id` 选择 GPU，用 `--only-scene` 选择场景。加上 `--dry-run`，就能在启动前预览队列。

## 构建完整模型

根目录的 [training_config.json](../training_config.json) 运行完整流程。[paretogs_main_table.json](paretogs_main_table.json) 使用相同的方法和场景，另设输出目录与 GPU。两者都会训练 30K ImprovedGS Full 模型、建立嵌套 GNS 锚点、编码残差、用训练视图标定 p/q，并打包 `runtime/`。

锚点比例相对于实际 Full 高斯数量定义：

```text
0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
0.064, 0.128, 0.256, 0.512, 1.0
```

`paretogs.hierarchy` 定义 GNS 日程和 FP16 残差存储；`paretogs.compression` 默认使用 SH 秩 36、INT8 分块大小 64，状态秩保持完整；`paretogs.calibration` 定义训练视图采样、p/q 候选和标定目标。Full 训练骨架按照 `train_args` 启用 LAS、RAP、EAS 和 MU。

最终评测由 `run_postprocess` 和 `postprocess_args` 控制，也可以通过 `--postprocess-only` 单独启动。主表预算比例为 `[1, 0.3, 0.1, 0.05, 0.01, 0.001]`。评测时沿用原来的配置与输出目录。如果数据集搬了位置，可以设置 `scenes[].postprocess_args.source_path` 为该场景的新路径，或使用[单模型评测命令](../paretogs/README.zh-CN.md)。

## 选择论文实验

下面的配置设置了 `pipeline: paretogs_experiment`。每个场景从 `build_root` 读取已有完整构建，将测量结果写到单独的 `output_root`。

| 配置 | 实验协议 | 需要保留的构建产物 |
| :---: | :---: | :---: |
| [paper_components.json](paper_components.json) | 在十个相邻锚点中间预算上比较五种变体 | `runtime/` |
| [paper_encoding.json](paper_encoding.json) | 同预算、同 p/q 下比较原始 FP32、FP16 和编码图谱 | `runtime/` 与原始 `hierarchy/` 状态 |
| [paper_anchor_density.json](paper_anchor_density.json) | Garden 的五、三、两个锚点变体；50 个共用非锚点预算 | `runtime/` 与原始 `hierarchy/` 状态 |
| [paper_switch_timing.json](paper_switch_timing.json) | 全部锚点端点常驻；每场景 108 次预算变化请求 | `runtime/` |
| [paper_budget_curve.json](paper_budget_curve.json) | Bicycle/Garden/Stump，每个锚点区间十等分，共 101 个预算 | `runtime/` |
| [paper_independent_gns.json](paper_independent_gns.json) | 同三个场景，在相邻锚点中间预算独立训练 GNS | `runtime/` 与 `backbone/` 的 Full 检查点 |

```bash
python run.py -c configs/paper_encoding.json --data_root /path/to/data --gpu_id 0 --dry-run
python run.py -c configs/paper_encoding.json --data_root /path/to/data --gpu_id 0
python run.py -c configs/paper_encoding.json --rebuild-summary
```

默认构建目录为 `output/paretogs`。如果使用 `configs/paretogs_main_table.json` 构建，请把实验的 `build_root` 改为 `output/paretogs_main_table`。当 `repeat > 1` 时，输入和输出按对应的场景编号配对。想对同一份已有构建重复测量，可以设置场景的 `build_name`，例如 `{"name": "bicycle", "build_name": "bicycle"}`。每次重复都会保留。

FP32 编码从原始锚点参数和持久高斯 ID 重建，不是将 FP16 残差升精度。锚点密度实验会对保留的锚点子集重新标定。切换计时排除加载和全锚点预解码，使用 CUDA 同步测量参数准备时间，并在每次计时请求后渲染一次；分位数由全部测量样本计算。这与 runtime 默认只缓存两个端点的使用方式不同。

独立 GNS 按 `experiment.training` 的设置，让每个目标预算从同一 Full 检查点出发，并保存最终检查点、锚点状态和指标。这项实验会训练；密集预算曲线则只评测已有 runtime。

每个场景、每次重复都会保存配置和结果。`summary.csv` 包含任务状态；`summary.json` 记录预期、完成、失败、缺失和不完整任务数。汇总只聚合已完成任务，并明确标注部分结果。`--rebuild-summary` 只读取已有记录，不训练或渲染；结果不完整时返回非零退出码。

## 单阶段训练与参数参考

[3dgs.json](3dgs.json)、[absgs.json](absgs.json) 和 [improvedgs.json](improvedgs.json) 提供保留的训练骨架配置。它们随附的重复次数为 3，所有编号结果都会保留。GNS 需要 `train_args.start_checkpoint_file`，可以配合 `start_checkpoint_dir`；也可以对已有运行设置顶层 `resume_training: true` 续训。

[total_config.json](../total_config.json) 列出分组默认值和运行示例，[total_config.schema.json](../total_config.schema.json) 列出字符串可选值。把需要的字段复制到运行配置即可；逐场景的训练和后处理覆盖分别放在 `scenes[].train_args`、`scenes[].postprocess_args`。

## 评测、计时与资源

训练沿用代码中每 3000 次迭代的默认评测节点，配置不覆盖这个日程。训练日志中的 LPIPS 和最终评测单独控制：完整构建默认关闭训练日志 LPIPS，最终评测包含该指标。

`elapsed_seconds` 排除训练中评测，包含 PLY 和检查点写入；PLY 时间另记在 `ply_save_seconds` 中，`training_time_definition` 说明计时口径。完整构建墙钟时间还包括加载、层级构建、编码、标定和打包。Full 导出最终 PLY 与检查点；GNS 阶段保留检查点和锚点状态，不逐阶段导出 PLY。

`resource_limits` 使用 Linux 接口，作用于运行器的子进程。CPU 百分比控制亲和性和线程数；内存百分比按主机/cgroup 容量计算，`guard` 监控进程树 RSS，`strict` 需要可写的 cgroup v2 内存控制器。可以移除该对象来关闭这些限制。Windows 模式会在读取配置时自动将该对象设为 `null`，并提示 CPU/内存限制已关闭；不模拟 cgroup，也不替换成另一种内存限制。直接调用 `train.py` 或独立模块时，不应用运行器的资源限制。
