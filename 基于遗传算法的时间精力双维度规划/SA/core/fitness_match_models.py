#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精力匹配函数库 — 多种策略（供调度与 PPT 对比）

当前项目默认使用 linear_l1：match = 1 - |req - slot|

为何总适应度常为负？
  fitness = Σ(match)·w_energy + spread_bonus − deadline_pen − overload − overtime
  惩罚项（尤其逾期）量级大，会把正奖励抵消，故 raw fitness 常为负。
  PPT 展示建议用：① 纯匹配度 0~1；② 奖励/惩罚分解；③ 归一化 0~100 分。
"""

from __future__ import annotations

import numpy as np

MATCH_MODEL_CATALOG = [
    {
        "id": "linear_l1",
        "name_zh": "线性 L1（默认）",
        "name_en": "Linear L1",
        "shape_zh": "对称 V 形，偏差越大分越低",
        "formula_zh": "1 − |req − slot|",
        "color": "#2ca02c",
        "ppt_note": "实现简单，与代码中 decode 选槽一致",
    },
    {
        "id": "squared_l2",
        "name_zh": "平方 L2",
        "name_en": "Squared L2",
        "shape_zh": "抛物线型，小偏差宽容、大偏差惩罚更重",
        "formula_zh": "1 − (req − slot)²",
        "color": "#1f77b4",
        "ppt_note": "光滑可导，利于梯度类方法（展示用）",
    },
    {
        "id": "gaussian",
        "name_zh": "高斯核",
        "name_en": "Gaussian kernel",
        "shape_zh": "钟形，峰在 req=slot，远处快速衰减",
        "formula_zh": "exp(−(req−slot)² / 2σ²)",
        "color": "#ff7f0e",
        "ppt_note": "σ 控制容忍带宽，σ≈0.2 时常用",
    },
    {
        "id": "exponential",
        "name_zh": "指数衰减",
        "name_en": "Exponential decay",
        "shape_zh": "尖峰 + 长尾，比高斯衰减更陡",
        "formula_zh": "exp(−β·|req − slot|)",
        "color": "#9467bd",
        "ppt_note": "β 越大对错位越敏感",
    },
    {
        "id": "hinge",
        "name_zh": "铰链/阈值",
        "name_en": "Hinge (margin)",
        "shape_zh": "容差带内满分，带外线性下降",
        "formula_zh": "|Δ|≤ε → 1，否则线性罚",
        "color": "#d62728",
        "ppt_note": "模拟「差不多就行」的容忍区",
    },
    {
        "id": "asymmetric",
        "name_zh": "非对称（槽不足重罚）",
        "name_en": "Asymmetric (under-slot)",
        "shape_zh": "slot≥req 时接近满分；slot<req 按比例降分",
        "formula_zh": "slot<req: slot/req；否则 1−0.3·(slot−req)",
        "color": "#8c564b",
        "ppt_note": "强调高难度任务不能放在低精力时段",
    },
]

_DEFAULT_SIGMA = 0.15
_DEFAULT_BETA = 4.0
_DEFAULT_MARGIN = 0.15


def list_match_model_ids() -> list[str]:
    return [m["id"] for m in MATCH_MODEL_CATALOG]


def get_match_meta(model_id: str) -> dict:
    for m in MATCH_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    raise KeyError(f"未知匹配策略: {model_id}，可选: {list_match_model_ids()}")


def compute_energy_match(
    req: float | np.ndarray,
    slot: float | np.ndarray,
    model_id: str = "linear_l1",
    *,
    sigma: float = _DEFAULT_SIGMA,
    beta: float = _DEFAULT_BETA,
    margin: float = _DEFAULT_MARGIN,
) -> float | np.ndarray:
    """单任务精力匹配分，输出裁剪到 [0, 1]。"""
    req = np.asarray(req, dtype=float)
    slot = np.asarray(slot, dtype=float)
    diff = req - slot
    ad = np.abs(diff)

    if model_id == "linear_l1":
        out = 1.0 - ad
    elif model_id == "squared_l2":
        out = 1.0 - diff ** 2
    elif model_id == "gaussian":
        out = np.exp(-0.5 * (diff / max(sigma, 1e-6)) ** 2)
    elif model_id == "exponential":
        out = np.exp(-beta * ad)
    elif model_id == "hinge":
        out = np.where(
            ad <= margin,
            1.0,
            np.clip(1.0 - (ad - margin) / max(1.0 - margin, 1e-6), 0.0, 1.0),
        )
    elif model_id == "asymmetric":
        out = np.where(
            slot < req,
            np.clip(slot / np.maximum(req, 1e-6), 0.0, 1.0),
            np.clip(1.0 - 0.3 * (slot - req), 0.0, 1.0),
        )
    else:
        raise KeyError(f"未知匹配策略: {model_id}")

    return np.clip(out, 0.0, 1.0)


def aggregate_match(matches: list[float] | np.ndarray) -> dict:
    arr = np.asarray(matches, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "sum": 0.0, "pct_100": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "sum": float(np.sum(arr)),
        "pct_100": float(np.mean(arr) * 100.0),
    }


def decompose_fitness(
    schedule: list[dict],
    weights: dict,
    match_model_id: str = "linear_l1",
    planning_days: int = 7,
    start_date=None,
    daily_max_tasks: int = 2,
) -> dict:
    """
    将适应度拆成可展示分量（PPT 用）。
    schedule: [{'task', 'start', 'end'}, ...]
    """
    from datetime import datetime, timedelta

    if start_date is None:
        start_date = datetime.strptime("2026-04-06", "%Y-%m-%d")
    w = weights
    matches = []
    daily_task_count = {}
    used_days = set()

    for item in schedule:
        day_key = item["start"].date()
        daily_task_count[day_key] = daily_task_count.get(day_key, 0) + 1
        used_days.add(day_key)

    deadline_pen = 0.0
    overtime_pen = 0.0
    max_end = start_date + timedelta(days=planning_days)

    for item in schedule:
        task = item["task"]
        start, end = item["start"], item["end"]
        slot_e = item.get("slot_energy")
        if slot_e is None:
            from .energy_curve import EnergyCurve
            curve = EnergyCurve(model_id="default")
            slot_e = curve.get_average_energy(start, end)
        req = task.energy_req
        m = float(compute_energy_match(req, slot_e, match_model_id))
        matches.append(m)

        if end > task.deadline:
            delay_h = (end - task.deadline).total_seconds() / 3600
            deadline_pen += delay_h * w.get("deadline_penalty", 8.0) * task.priority
        if end > max_end:
            overtime_pen += (end - max_end).total_seconds() / 3600 * w.get("overtime_penalty", 5.0)

    overload_pen = 0.0
    for _day, cnt in daily_task_count.items():
        if cnt > daily_max_tasks:
            overload_pen += (cnt - daily_max_tasks) ** 2 * w.get("daily_overload_penalty", 3.0)

    spread_bonus = len(used_days) * 2.0
    energy_sum = float(np.sum(matches))
    energy_weighted = energy_sum * w.get("energy_match", 2.5)
    penalty_total = deadline_pen + overload_pen + overtime_pen
    reward_total = energy_weighted + spread_bonus
    raw_fitness = reward_total - penalty_total

    agg = aggregate_match(matches)
    reward_penalty_ratio = (
        100.0 * reward_total / (reward_total + penalty_total + 1e-9)
        if (reward_total + penalty_total) > 0
        else 0.0
    )

    return {
        "match_model_id": match_model_id,
        "energy_match_mean": agg["mean"],
        "energy_match_pct": agg["pct_100"],
        "energy_sum": energy_sum,
        "energy_weighted": energy_weighted,
        "spread_bonus": spread_bonus,
        "deadline_penalty": deadline_pen,
        "overload_penalty": overload_pen,
        "overtime_penalty": overtime_pen,
        "penalty_total": penalty_total,
        "reward_total": reward_total,
        "raw_fitness": raw_fitness,
        "display_score_100": reward_penalty_ratio,
        "n_tasks": len(schedule),
    }
