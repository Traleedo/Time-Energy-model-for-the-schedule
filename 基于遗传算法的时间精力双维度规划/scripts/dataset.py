#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务数据集：生成、保存、加载
================================================================================
关于「有没有现成数据集」：
  本课题是「个人任务 + 日内精力曲线 + 大学课程表」的联合调度，公开标准库中
  **没有**直接可用的同类数据集。相近领域仅有：
    - 课程表调度 (ITC University Timetabling) — 无精力维度、非个人任务
    - 作业车间调度 (Job Shop) — 无截止日与精力曲线
  因此采用 **可复现的合成数据生成器** + **可选真实任务模板**，并持久化为 JSON。

用法：
  python generate_dataset.py                    # 生成默认基准包
  python generate_dataset.py --scenario heavy   # 单场景
  from dataset import load_tasks_from_file
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from 新建文件夹.config import CONFIG
from 新建文件夹.task import Task

# 项目根目录下的数据目录
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
BENCHMARK_DIR = DATASET_DIR / "benchmark"
DEFAULT_DATASET = BENCHMARK_DIR / "medium_seed42.json"
MANIFEST_FILE = BENCHMARK_DIR / "manifest.json"

# 与 Task.estimate_energy 一致的任务类型
TASK_TYPES = ["coding", "review", "writing", "learning", "admin", "meeting"]
DURATIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# 场景配置：控制任务数、截止紧迫度、优先级分布
SCENARIOS: dict[str, dict[str, Any]] = {
    "light": {
        "description": "轻负载：任务少、截止宽松，适合验证可行性",
        "n_tasks_range": (6, 10),
        "deadline_day_range": (3, 7),
        "deadline_hour_range": (18.0, 23.0),
        "priority_range": (0.5, 1.5),
        "tight_ratio": 0.1,
    },
    "medium": {
        "description": "中等负载：论文/课程设计常用规模",
        "n_tasks_range": (10, 15),
        "deadline_day_range": (1, 7),
        "deadline_hour_range": (12.0, 23.5),
        "priority_range": (0.5, 2.0),
        "tight_ratio": 0.35,
    },
    "heavy": {
        "description": "重负载：任务多、截止紧，考验逾期与过载惩罚",
        "n_tasks_range": (15, 22),
        "deadline_day_range": (0, 5),
        "deadline_hour_range": (10.0, 22.0),
        "priority_range": (0.8, 2.5),
        "tight_ratio": 0.55,
    },
    "realistic": {
        "description": "贴近 main.py 中真实课设/作业混合场景",
        "fixed_tasks": True,
    },
}


def _base_date() -> datetime:
    return datetime.strptime(CONFIG.START_DATE, "%Y-%m-%d")


def _realistic_tasks() -> list[dict]:
    """与 main.py 多日示例一致的真实风格任务。"""
    base = _base_date()
    return [
        {"name": "多媒体", "duration": 1.0, "task_type": "coding",
         "deadline": (base + timedelta(days=6)).replace(hour=23, minute=59).strftime("%Y-%m-%d %H:%M"),
         "priority": 0.3},
        {"name": "计算机中的数学", "duration": 0.5, "task_type": "learning",
         "deadline": (base + timedelta(days=7)).replace(hour=23, minute=59).strftime("%Y-%m-%d %H:%M"),
         "priority": 0.3},
        {"name": "批改网作文", "duration": 0.7, "task_type": "writing",
         "deadline": (base + timedelta(days=6)).replace(hour=23, minute=59).strftime("%Y-%m-%d %H:%M"),
         "priority": 0.7},
        {"name": "操作系统", "duration": 2.0, "task_type": "coding",
         "deadline": (base + timedelta(days=2)).replace(hour=23, minute=59).strftime("%Y-%m-%d %H:%M"),
         "priority": 1.0},
    ]


def generate_task_records(
    n_tasks: int | None = None,
    seed: int = 42,
    scenario: str = "medium",
) -> tuple[list[dict], dict]:
    """
    生成符合项目约束的任务记录（字典列表 + 元信息）。

    约束对齐：
      - duration, task_type -> energy_req 由 Task 类自动估算
      - deadline 在 [START_DATE, START_DATE + PLANNING_DAYS] 内
      - priority, name 字段完整
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景 {scenario}，可选: {list(SCENARIOS.keys())}")

    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    base = _base_date()
    planning_days = CONFIG.PLANNING_DAYS

    if cfg.get("fixed_tasks"):
        records = _realistic_tasks()
        n_tasks = len(records)
    else:
        lo, hi = cfg["n_tasks_range"]
        n_tasks = n_tasks or int(rng.integers(lo, hi + 1))
        d_lo, d_hi = cfg["deadline_day_range"]
        h_lo, h_hi = cfg["deadline_hour_range"]
        p_lo, p_hi = cfg["priority_range"]
        tight_ratio = cfg["tight_ratio"]

        records = []
        for i in range(n_tasks):
            t_type = str(rng.choice(TASK_TYPES))
            duration = float(rng.choice(DURATIONS))
            if rng.random() < tight_ratio:
                dl_day = int(rng.integers(0, max(2, d_hi // 2 + 1)))
                dl_hour = float(rng.uniform(h_lo, (h_lo + h_hi) / 2))
            else:
                dl_day = int(rng.integers(d_lo, min(planning_days, d_hi) + 1))
                dl_hour = float(rng.uniform(h_lo, h_hi))
            dl_day = min(dl_day, planning_days)
            h, m = int(dl_hour), int(round((dl_hour % 1) * 60))
            deadline = base + timedelta(days=dl_day, hours=h, minutes=m)
            records.append({
                "name": f"{t_type}_{i}",
                "duration": duration,
                "task_type": t_type,
                "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
                "priority": round(float(rng.uniform(p_lo, p_hi)), 3),
            })

    meta = {
        "version": 1,
        "scenario": scenario,
        "description": cfg["description"],
        "n_tasks": n_tasks,
        "seed": seed,
        "start_date": CONFIG.START_DATE,
        "planning_days": planning_days,
        "multi_day_mode": CONFIG.MULTI_DAY_MODE,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return records, meta


def build_dataset_payload(
    n_tasks: int | None = None,
    seed: int = 42,
    scenario: str = "medium",
) -> dict:
    """构造可序列化的完整数据集 JSON 对象。"""
    records, meta = generate_task_records(n_tasks=n_tasks, seed=seed, scenario=scenario)
    return {"meta": meta, "tasks": records}


def save_dataset(payload: dict, path: str | Path) -> Path:
    """保存数据集到 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_dataset(path: str | Path) -> dict:
    """从 JSON 加载原始数据集。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {path}\n"
            f"请先运行: python generate_dataset.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def records_to_tasks(records: list[dict]) -> list[Task]:
    """将 JSON 记录转为 Task 对象列表。"""
    tasks = []
    for rec in records:
        tasks.append(Task(
            name=rec["name"],
            duration=float(rec["duration"]),
            task_type=rec["task_type"],
            deadline=rec["deadline"],
            priority=float(rec.get("priority", 1.0)),
        ))
    return tasks


def load_tasks_from_file(path: str | Path | None = None) -> tuple[list[Task], dict]:
    """
    加载数据集并返回 (tasks, meta)。

    Parameters
    ----------
    path : 数据集路径，默认 datasets/benchmark/medium_seed42.json
    """
    path = Path(path) if path else DEFAULT_DATASET
    payload = load_dataset(path)
    tasks = records_to_tasks(payload["tasks"])
    meta = payload.get("meta", {})
    meta["source_file"] = str(path)
    return tasks, meta


def generate_benchmark_pack(
    scenarios: list[str] | None = None,
    instances_per_scenario: int = 100,
    seed_base: int = 1000,
    output_dir: str | Path | None = None,
    on_progress=None,
) -> Path:
    """
    生成多实例基准包 + manifest.json，供 valid.py 批量实验选用。

    Returns
    -------
    Path  manifest 文件路径
    """
    output_dir = Path(output_dir or DATASET_DIR / "benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = scenarios or ["light", "medium", "heavy"]
    manifest = {"version": 1, "instances": []}
    total = len(scenarios) * instances_per_scenario
    step = 0

    for scenario in scenarios:
        for k in range(instances_per_scenario):
            seed = seed_base + step
            n_tasks = None
            payload = build_dataset_payload(n_tasks=n_tasks, seed=seed, scenario=scenario)
            fname = f"{scenario}_inst{k:02d}_seed{seed}.json"
            fpath = output_dir / fname
            save_dataset(payload, fpath)
            manifest["instances"].append({
                "file": fname,
                "scenario": scenario,
                "seed": seed,
                "n_tasks": payload["meta"]["n_tasks"],
            })
            step += 1
            if on_progress:
                on_progress(step, total, f"已生成 {fname}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def _is_task_dataset(path: Path) -> bool:
    """判断 JSON 是否为任务数据集（含 tasks 列表）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data.get("tasks"), list) and len(data["tasks"]) > 0
    except (json.JSONDecodeError, OSError):
        return False


def list_benchmark_datasets(
    directory: str | Path | None = None,
    scenario: str | None = None,
) -> list[Path]:
    """
    列出 benchmark 目录下可用于 valid.py 的 JSON（排除 manifest.json）。

    Parameters
    ----------
    scenario : 可选过滤，如 'light' / 'medium' / 'heavy'
    """
    directory = Path(directory or BENCHMARK_DIR)
    if not directory.exists():
        return []
    files = sorted(
        p for p in directory.glob("*.json")
        if p.name != "manifest.json" and _is_task_dataset(p)
    )
    if scenario:
        prefix = scenario.lower() + "_"
        files = [p for p in files if p.name.startswith(prefix) or f"_{scenario}" in p.name]
    return files


def list_available_datasets(directory: str | Path | None = None) -> list[Path]:
    """列出 datasets 目录下所有任务 JSON（不含 manifest）。"""
    directory = Path(directory or DATASET_DIR)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*.json")
        if p.name != "manifest.json" and _is_task_dataset(p)
    )


def ensure_default_dataset() -> Path:
    """若默认数据集不存在则自动创建。"""
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    payload = build_dataset_payload(n_tasks=12, seed=42, scenario="medium")
    return save_dataset(payload, DEFAULT_DATASET)
