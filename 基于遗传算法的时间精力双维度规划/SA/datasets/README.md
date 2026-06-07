# 任务数据集说明

## 有没有现成公开数据集？

**没有**与本项目完全匹配的公开标准数据集。本项目需要同时包含：

- 个人学习任务（时长、类型、优先级、截止时间）
- 日内精力曲线匹配
- 固定大学课程表 / 午休等硬约束

相近但不可直接使用的资源：

| 资源 | 差异 |
|------|------|
| [ITC Timetabling](https://www.itc2007.org/) | 课程排课，无个人精力维度 |
| Job Shop / Flow Shop 标准算例 | 工业机器调度，无截止日与精力 |
| 日历/待办 API 数据 | 极少标注 `task_type` 与精力需求 |

因此采用 **可复现合成数据** + **真实风格模板（realistic 场景）**。

## 目录结构

```
datasets/
  README.md
  benchmark/
    medium_seed42.json      # 默认单实例（valid.py 默认加载）
    manifest.json           # 多实例清单（--pack 生成）
    light_inst00_seed1000.json
    ...
```

## 生成数据

```bash
# 默认中等负载 12 任务
python generate_dataset.py

# 指定场景
python generate_dataset.py --scenario heavy --n-tasks 18 --seed 7

# 批量 benchmark 包（论文多算例）
python generate_dataset.py --pack --instances 5
```

## 在 valid.py 中使用

```bash
# 默认：抽样 5 个 benchmark JSON + 快速模式（不含 SA，SA 由队友单独实验）
python valid.py

# 跑全部实例（较慢）
python valid.py --full --runs 5

# 只跑 medium 场景
python valid.py --scenario medium

# 指定单个文件
python valid.py --dataset datasets/benchmark/medium_seed42.json

# 快速试跑
python valid.py --limit 2 --fast --runs 2
```

对比算法：**GA vs RS / EDF / HERF / WSPT**（不含 SA）。

图表与 CSV 输出到 `benchmark_output/`（英文标签，避免乱码）。

JSON 字段与 `Task` 类一致：`name`, `duration`, `task_type`, `deadline`, `priority`。
