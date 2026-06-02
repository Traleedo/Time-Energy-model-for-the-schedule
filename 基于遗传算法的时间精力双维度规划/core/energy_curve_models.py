#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内精力曲线 — 多方案模型库（供调度实验与 PPT 展示）

方案说明：
  default   项目默认：晨峰 + 午后回落 + 傍晚次峰（双正弦）
  lark      早晨型：6–10 点高，下午持续走低
  owl       夜晚型：上午低迷，17–22 点高
  flat      平稳型：全天波动小（对照实验）
  bimodal   双峰型：上午峰 + 晚间峰，午后谷底
  gaussian  单峰高斯：可调峰时（默认 10:00）
"""

from __future__ import annotations

import numpy as np

# 每条曲线: (id, 中文名, 英文名, 形状简述, 公式简述, 颜色)
ENERGY_MODEL_CATALOG = [
    {
        "id": "default",
        "name_zh": "默认双正弦",
        "name_en": "Default (dual-sine)",
        "shape_zh": "上午升高 → 午后小低谷 → 傍晚回升",
        "formula_zh": "0.5 + 0.3·sin((h-6)π/12) + 0.1·sin((h-14)π/8)",
        "color": "#2ca02c",
    },
    {
        "id": "lark",
        "name_zh": "早晨型",
        "name_en": "Morning Lark",
        "shape_zh": "清晨快速爬升，午后持续下降",
        "formula_zh": "峰位 ~8:00，高斯衰减至晚间",
        "color": "#1f77b4",
    },
    {
        "id": "owl",
        "name_zh": "夜晚型",
        "name_en": "Night Owl",
        "shape_zh": "上午低迷，傍晚至夜间精力充沛",
        "formula_zh": "峰位 ~20:00，上午维持低位",
        "color": "#9467bd",
    },
    {
        "id": "bimodal",
        "name_zh": "经典双峰",
        "name_en": "Bimodal",
        "shape_zh": "上午峰 + 午后谷 + 晚间次峰",
        "formula_zh": "两个高斯峰(9:00, 19:00)叠加",
        "color": "#ff7f0e",
    },
    {
        "id": "gaussian",
        "name_zh": "单峰高斯",
        "name_en": "Single Gaussian",
        "shape_zh": "单一对称峰，钟形曲线",
        "formula_zh": "峰位 10:00，σ≈2.5h",
        "color": "#d62728",
    },
    {
        "id": "flat",
        "name_zh": "平稳对照",
        "name_en": "Flat baseline",
        "shape_zh": "近似水平，轻微随机起伏",
        "formula_zh": "0.55 + 0.05·sin(h)",
        "color": "#7f7f7f",
    },
]


def _clip_energy(values: np.ndarray, low: float = 0.1, high: float = 1.0) -> np.ndarray:
    return np.clip(values, low, high)


def compute_energy_values(hours: np.ndarray, model_id: str = "default") -> np.ndarray:
    """按方案 ID 计算 0~24h 精力值序列。"""
    h = np.asarray(hours, dtype=float)
    mid = model_id.lower().strip()

    if mid == "default":
        val = 0.5 + 0.3 * np.sin((h - 6) * np.pi / 12) + 0.1 * np.sin((h - 14) * np.pi / 8)

    elif mid == "lark":
        # 早晨型：峰在 8 点，σ=3h
        val = 0.25 + 0.75 * np.exp(-0.5 * ((h - 8.0) / 3.0) ** 2)
        val = val - 0.15 * np.exp(-0.5 * ((h - 20.0) / 4.0) ** 2)

    elif mid == "owl":
        # 夜晚型：峰在 20 点
        val = 0.35 + 0.15 * np.exp(-0.5 * ((h - 9.0) / 4.0) ** 2)
        val = val + 0.55 * np.exp(-0.5 * ((h - 20.0) / 2.8) ** 2)

    elif mid == "bimodal":
        morning = 0.55 * np.exp(-0.5 * ((h - 9.0) / 2.2) ** 2)
        evening = 0.50 * np.exp(-0.5 * ((h - 19.0) / 2.5) ** 2)
        dip = 0.35 * np.exp(-0.5 * ((h - 14.0) / 1.8) ** 2)
        val = 0.20 + morning + evening - dip

    elif mid == "gaussian":
        val = 0.15 + 0.85 * np.exp(-0.5 * ((h - 10.0) / 2.5) ** 2)

    elif mid == "flat":
        val = 0.55 + 0.05 * np.sin(h * np.pi / 12)

    else:
        raise ValueError(f"未知精力曲线方案: {model_id}，可选: {[m['id'] for m in ENERGY_MODEL_CATALOG]}")

    return _clip_energy(val)


def get_model_meta(model_id: str) -> dict:
    for m in ENERGY_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    raise KeyError(model_id)


def list_model_ids() -> list[str]:
    return [m["id"] for m in ENERGY_MODEL_CATALOG]


def peak_valley_summary(hours: np.ndarray, values: np.ndarray) -> dict:
    """峰/谷时刻与数值，用于 PPT 表格。"""
    i_peak = int(np.argmax(values))
    i_valley = int(np.argmin(values))
    return {
        "peak_hour": float(hours[i_peak]),
        "peak_value": float(values[i_peak]),
        "valley_hour": float(hours[i_valley]),
        "valley_value": float(values[i_valley]),
        "mean": float(np.mean(values)),
    }
