#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键生成并保存任务数据集。

示例：
  python generate_dataset.py
  python generate_dataset.py --scenario heavy --n-tasks 18 --seed 7
  python generate_dataset.py --pack --instances 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataset import (
    DEFAULT_DATASET,
    DATASET_DIR,
    build_dataset_payload,
    ensure_default_dataset,
    generate_benchmark_pack,
    save_dataset,
)


def _progress(cur, total, msg):
    pct = 100 * cur / total
    print(f"\r[{cur}/{total}] ({pct:5.1f}%) {msg}", end="", flush=True)
    if cur >= total:
        print()


def main():
    parser = argparse.ArgumentParser(description="生成时间-精力调度任务数据集")
    parser.add_argument("--scenario", default="medium", choices=["light", "medium", "heavy", "realistic"])
    parser.add_argument("--n-tasks", type=int, default=None, help="任务数（不指定则按场景随机）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--pack", action="store_true", help="生成多实例 benchmark 包")
    parser.add_argument("--instances", type=int, default=5, help="每场景实例数（--pack 时有效）")
    args = parser.parse_args()

    if args.pack:
        print(f"正在生成 benchmark 包 -> {DATASET_DIR / 'benchmark'}")
        manifest = generate_benchmark_pack(
            instances_per_scenario=args.instances,
            seed_base=args.seed,
            on_progress=_progress,
        )
        print(f"完成。清单: {manifest}")
        return 0

    if args.output:
        out = args.output
    elif args.scenario == "medium" and args.seed == 42 and args.n_tasks in (None, 12):
        out = str(DEFAULT_DATASET)
    else:
        n = args.n_tasks or "auto"
        out = str(DATASET_DIR / "benchmark" / f"{args.scenario}_n{n}_seed{args.seed}.json")

    payload = build_dataset_payload(n_tasks=args.n_tasks, seed=args.seed, scenario=args.scenario)
    path = save_dataset(payload, out)
    print(f"已保存: {path}")
    print(f"  场景={args.scenario}, 任务数={payload['meta']['n_tasks']}, seed={args.seed}")
    print(f"  说明: {payload['meta']['description']}")
    print("\n在 valid.py 中使用:")
    print(f"  python valid.py --dataset \"{path}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
