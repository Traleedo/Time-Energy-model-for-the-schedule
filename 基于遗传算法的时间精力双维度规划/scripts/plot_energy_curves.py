#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成多方案日内精力曲线对比图（PPT 用）

用法:
  python plot_energy_curves.py
  python plot_energy_curves.py --models default,lark,owl,bimodal
  python plot_energy_curves.py --output ppt_assets/energy_curves
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from 新建文件夹.energy_curve_models import (
    ENERGY_MODEL_CATALOG,
    compute_energy_values,
    get_model_meta,
    list_model_ids,
    peak_valley_summary,
)

OUTPUT_DEFAULT = Path(__file__).resolve().parent.parent / "ppt_assets" / "energy_curves"
WORK_START, WORK_END = 6.0, 23.0  # 展示用工作时段 shading


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })


def _shade_work_hours(ax, y_min=0, y_max=1.05):
    ax.axvspan(WORK_START, WORK_END, alpha=0.08, color="#4C72B0", label="_work")
    ax.axvline(WORK_START, color="#4C72B0", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(WORK_END, color="#4C72B0", ls=":", lw=0.8, alpha=0.5)


def plot_overlay(hours, model_ids, save_path: Path):
    """一张图叠加所有曲线 — 适合 PPT「方案对比」页。"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _shade_work_hours(ax)

    for mid in model_ids:
        meta = get_model_meta(mid)
        vals = compute_energy_values(hours, mid)
        ax.plot(
            hours, vals, lw=2.2, color=meta["color"],
            label=f"{meta['name_zh']} ({meta['name_en']})",
        )

    ax.set_xlim(0, 24)
    ax.set_ylim(0.05, 1.05)
    ax.set_xlabel("时刻 (小时)", fontsize=12)
    ax.set_ylabel("精力水平 (0~1)", fontsize=12)
    ax.set_title("日内精力曲线 — 多方案对比", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.text(0.5, 0.02, "阴影区: 典型学习/工作时段 (6:00–23:00)", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  叠加对比图: {save_path}")


def plot_grid(hours, model_ids, save_path: Path):
    """分面子图 — 适合 PPT「各曲线形状」页。"""
    n = len(model_ids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, mid in zip(axes, model_ids):
        meta = get_model_meta(mid)
        vals = compute_energy_values(hours, mid)
        summary = peak_valley_summary(hours, vals)

        _shade_work_hours(ax)
        ax.fill_between(hours, vals, alpha=0.25, color=meta["color"])
        ax.plot(hours, vals, lw=2.5, color=meta["color"])
        ax.scatter(
            [summary["peak_hour"], summary["valley_hour"]],
            [summary["peak_value"], summary["valley_value"]],
            c=["#C00000", "#0070C0"], s=45, zorder=5,
        )
        ax.annotate(
            f"峰 {summary['peak_hour']:.1f}h",
            (summary["peak_hour"], summary["peak_value"]),
            textcoords="offset points", xytext=(4, 6), fontsize=8, color="#C00000",
        )
        ax.annotate(
            f"谷 {summary['valley_hour']:.1f}h",
            (summary["valley_hour"], summary["valley_value"]),
            textcoords="offset points", xytext=(4, -12), fontsize=8, color="#0070C0",
        )

        ax.set_xlim(0, 24)
        ax.set_ylim(0.05, 1.05)
        ax.set_title(f"{meta['name_zh']}\n{meta['shape_zh']}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, 25, 4))

    for ax in axes[len(model_ids):]:
        ax.set_visible(False)

    fig.supxlabel("时刻 (小时)", fontsize=12)
    fig.supylabel("精力水平 (0~1)", fontsize=12)
    fig.suptitle("各方案精力曲线形状（单图）", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  分面形状图: {save_path}")


def plot_shape_legend_card(save_path: Path):
    """文字说明卡片 — 适合 PPT 备注页。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    lines = ["日内精力曲线方案一览", ""]
    for m in ENERGY_MODEL_CATALOG:
        lines.append(f"■ {m['name_zh']} ({m['name_en']})")
        lines.append(f"    形状: {m['shape_zh']}")
        lines.append(f"    公式: {m['formula_zh']}")
        lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11,
            transform=ax.transAxes)
    fig.savefig(save_path, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"  方案说明卡: {save_path}")


def export_summary_table(hours, model_ids, save_path: Path):
    rows = []
    for mid in model_ids:
        meta = get_model_meta(mid)
        vals = compute_energy_values(hours, mid)
        s = peak_valley_summary(hours, vals)
        rows.append({
            "方案ID": mid,
            "中文名": meta["name_zh"],
            "英文名": meta["name_en"],
            "形状描述": meta["shape_zh"],
            "公式": meta["formula_zh"],
            "峰值时刻": f"{s['peak_hour']:.1f}h",
            "峰值": f"{s['peak_value']:.3f}",
            "谷值时刻": f"{s['valley_hour']:.1f}h",
            "谷值": f"{s['valley_value']:.3f}",
            "日均": f"{s['mean']:.3f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  参数表 CSV: {save_path}")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="生成多方案精力曲线 PPT 配图")
    p.add_argument("--models", type=str, default="all",
                   help="逗号分隔方案ID，或 all")
    p.add_argument("--output", type=str, default=str(OUTPUT_DEFAULT))
    p.add_argument("--step", type=float, default=0.1, help="采样步长(小时)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_style()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.models.strip().lower() == "all":
        model_ids = list_model_ids()
    else:
        model_ids = [x.strip() for x in args.models.split(",") if x.strip()]

    hours = np.arange(0, 24 + args.step / 2, args.step)
    print(f"生成 {len(model_ids)} 种精力曲线 -> {out.resolve()}")

    plot_overlay(hours, model_ids, out / "energy_curves_overlay.png")
    plot_grid(hours, model_ids, out / "energy_curves_grid.png")
    plot_shape_legend_card(out / "energy_curves_legend.png")
    df = export_summary_table(hours, model_ids, out / "energy_curves_summary.csv")

    print("\n方案摘要:")
    print(df.to_string(index=False))
    print("\n完成。可将 overlay / grid 图插入 PPT。")
