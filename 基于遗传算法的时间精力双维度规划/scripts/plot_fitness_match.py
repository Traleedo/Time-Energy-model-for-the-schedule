#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成多种「精力匹配 / 适应度」策略对比图（PPT 用）

用法:
  python plot_fitness_match.py
  python plot_fitness_match.py --dataset datasets/benchmark/medium_seed42.json
  python plot_fitness_match.py --output ppt_assets/fitness_match
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

from 新建文件夹.fitness_match_models import (
    MATCH_MODEL_CATALOG,
    compute_energy_match,
    decompose_fitness,
    get_match_meta,
    list_match_model_ids,
)
from scripts.dataset import load_tasks_from_file

OUTPUT_DEFAULT = Path(__file__).resolve().parent.parent / "ppt_assets" / "fitness_match"
REQ_SAMPLES = [0.3, 0.5, 0.7, 0.9]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })


def plot_match_curves(save_path: Path, model_ids: list[str]):
    """固定任务需求，匹配分随槽位精力变化 — 展示各策略曲线形状。"""
    slot = np.linspace(0, 1, 200)
    n = len(model_ids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, mid in zip(axes, model_ids):
        meta = get_match_meta(mid)
        for req in REQ_SAMPLES:
            y = compute_energy_match(req, slot, mid)
            ax.plot(slot, y, lw=2, label=f"任务需求 req={req:.1f}")
        ax.set_title(f"{meta['name_zh']}\n{meta['formula_zh']}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        ax.axvline(0.5, color="gray", ls=":", alpha=0.4)
    for ax in axes[len(model_ids):]:
        ax.set_visible(False)
    fig.supxlabel("时段精力 slot", fontsize=12)
    fig.supylabel("匹配分 match (0~1)", fontsize=12)
    fig.suptitle("不同匹配策略：匹配分随槽位精力变化", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  匹配曲线分面图: {save_path}")


def plot_match_heatmaps(save_path: Path, model_ids: list[str]):
    """二维热力图：横轴 slot，纵轴 req。"""
    req = np.linspace(0.05, 0.95, 80)
    slot = np.linspace(0.05, 0.95, 80)
    R, S = np.meshgrid(req, slot)
    n = len(model_ids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, mid in zip(axes, model_ids):
        meta = get_match_meta(mid)
        Z = compute_energy_match(R, S, mid)
        im = ax.imshow(
            Z, origin="lower", aspect="auto",
            extent=[slot.min(), slot.max(), req.min(), req.max()],
            cmap="YlGnBu", vmin=0, vmax=1,
        )
        ax.set_title(meta["name_zh"], fontsize=11)
        ax.set_xlabel("slot")
        ax.set_ylabel("req")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(model_ids):]:
        ax.set_visible(False)
    fig.suptitle("匹配分热力图（颜色越亮匹配越好）", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  匹配热力图: {save_path}")


def _schedule_with_slot_energy(engine, schedule):
    out = []
    for item in schedule:
        se = engine.energy_curve.get_energy_datetime(item["start"])
        d = dict(item)
        d["slot_energy"] = se
        out.append(d)
    return out


def plot_strategy_bars_on_schedule(engine, schedule, save_path: Path, model_ids: list[str]):
    """同一排程，换不同匹配函数看平均匹配分（均为 0~1 正分）。"""
    sched = _schedule_with_slot_energy(engine, schedule)
    means = []
    labels = []
    for mid in model_ids:
        meta = get_match_meta(mid)
        ms = [
            float(compute_energy_match(it["task"].energy_req, it["slot_energy"], mid))
            for it in sched
        ]
        means.append(np.mean(ms) * 100)
        labels.append(meta["name_zh"])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [get_match_meta(m)["color"] for m in model_ids]
    bars = ax.bar(labels, means, color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_ylabel("平均精力匹配度 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("同一调度方案 · 不同匹配策略下的平均匹配分\n（均为 0~100 正值，便于 PPT 对比）", fontweight="bold")
    ax.axhline(np.mean(means), color="#666", ls="--", lw=1, label=f"均值 {np.mean(means):.1f}%")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.legend()
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  同方案策略柱状图: {save_path}")


def plot_fitness_decomposition(decomp: dict, save_path: Path):
    """奖励 vs 惩罚分解 + 原始适应度 vs 归一化展示分。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：堆叠分解
    ax = axes[0]
    pos_labels = ["精力匹配\n(加权)", "跨日分散\n奖励"]
    pos_vals = [decomp["energy_weighted"], decomp["spread_bonus"]]
    neg_labels = ["截止逾期", "日过载", "规划超期"]
    neg_vals = [
        decomp["deadline_penalty"],
        decomp["overload_penalty"],
        decomp["overtime_penalty"],
    ]
    x = 0
    bottom = 0
    for lab, v in zip(pos_labels, pos_vals):
        if v > 0:
            ax.bar(x, v, bottom=bottom, label=lab, color="#5b9bd5", width=0.5)
            bottom += v
    top_pos = bottom
    bottom = 0
    for lab, v in zip(neg_labels, neg_vals):
        if v > 0:
            ax.bar(x, -v, bottom=-bottom, label=lab, color="#c55a5a", width=0.5)
            bottom += v
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["适应度分量"])
    ax.set_ylabel("分值（加权后）")
    ax.set_title(
        f"适应度分解\n原始 fitness = {decomp['raw_fitness']:.1f}（可为负）",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)

    # 右：两种可读指标
    ax2 = axes[1]
    cats = ["纯精力匹配\n(0~100%)", "奖励/(奖励+惩罚)\n展示分", "原始 fitness\n(可负)"]
    vals = [
        decomp["energy_match_pct"],
        decomp["display_score_100"],
        max(min(decomp["raw_fitness"] / 50 + 50, 100), 0),  # 示意映射到 0~100 柱高
    ]
    colors = ["#70ad47", "#ffc000", "#a5a5a5"]
    bars = ax2.bar(cats, vals, color=colors, width=0.55)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("展示用分值")
    ax2.set_title("PPT 推荐展示方式", fontweight="bold")
    raw_labels = [
        f"{decomp['energy_match_pct']:.1f}%",
        f"{decomp['display_score_100']:.1f}",
        f"{decomp['raw_fitness']:.1f}",
    ]
    for b, v, label in zip(bars, vals, raw_labels):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2, label, ha="center", fontsize=10)
    ax2.text(
        0.5, -0.22,
        "说明：原始 fitness 因逾期惩罚大而常为负；汇报时用左侧分解或纯匹配百分比更清晰。",
        transform=ax2.transAxes, ha="center", fontsize=9, color="#444",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  适应度分解图: {save_path}")


def plot_raw_vs_normalized_table(rows: list[dict], save_path: Path):
    """多策略：同一排程的 raw fitness 与展示分对比。"""
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(df) + 2)))
    ax.axis("off")
    cols = ["策略", "平均匹配%", "奖励合计", "惩罚合计", "raw fitness", "展示分(0~100)"]
    table_data = [
        [
            r["name_zh"],
            f"{r['energy_match_pct']:.1f}",
            f"{r['reward_total']:.1f}",
            f"{r['penalty_total']:.1f}",
            f"{r['raw_fitness']:.1f}",
            f"{r['display_score_100']:.1f}",
        ]
        for r in rows
    ]
    tbl = ax.table(
        cellText=table_data, colLabels=cols, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    ax.set_title("同一排程 · 不同匹配策略下的适应度对比", fontsize=13, fontweight="bold", pad=20)
    fig.savefig(save_path, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"  对比表图: {save_path}")
    return df


def plot_overlay_linear_vs_gaussian(save_path: Path):
    """突出默认线性 vs 高斯 — 单页 PPT。"""
    slot = np.linspace(0, 1, 200)
    req = 0.7
    fig, ax = plt.subplots(figsize=(8, 5))
    for mid, lw in [("linear_l1", 2.5), ("gaussian", 2.5), ("asymmetric", 2.0)]:
        meta = get_match_meta(mid)
        y = compute_energy_match(req, slot, mid)
        ax.plot(slot, y, color=meta["color"], lw=lw, label=meta["name_zh"])
    ax.axvline(req, color="#333", ls=":", alpha=0.6, label=f"任务需求 req={req}")
    ax.set_xlabel("时段精力 slot")
    ax.set_ylabel("匹配分")
    ax.set_title(f"三种典型策略对比（任务需求 req={req}）", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  三策略叠加图: {save_path}")


def plot_legend_card(save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    lines = [
        "精力匹配与适应度 — PPT 说明",
        "",
        "1. 为何 fitness 常为负？",
        "   适应度 = 精力奖励 + 分散奖励 − 逾期惩罚 − 过载惩罚 − 超期惩罚",
        "   逾期惩罚系数大(×8×优先级×延迟小时)，容易超过正奖励。",
        "",
        "2. 是否仍用线性匹配？",
        "   代码默认：linear_l1，match = 1 − |req − slot|",
        "   还可选：平方L2、高斯核、指数衰减、铰链阈值、非对称(槽不足重罚)",
        "",
        "3. PPT 建议展示：",
        "   · 匹配曲线/热力图 → 讲「策略形状」",
        "   · 平均匹配% (0~100) → 讲「排程质量」",
        "   · 奖励/惩罚分解 → 讲「为何 raw fitness 为负」",
        "   · 展示分 = 奖励/(奖励+惩罚)×100 → 综合可读分",
        "",
    ]
    for m in MATCH_MODEL_CATALOG:
        lines.append(f"■ {m['name_zh']}: {m['formula_zh']} — {m['ppt_note']}")
    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", fontsize=10.5, transform=ax.transAxes)
    fig.savefig(save_path, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"  说明卡: {save_path}")


def run_with_dataset(dataset_path: Path, output: Path):
    from core.schedule_engine import MultiDayScheduleEngine
    from core.ga_solver import ScheduleGA
    from 新建文件夹.energy_curve import EnergyCurve

    tasks, meta = load_tasks_from_file(str(dataset_path))
    start_date = meta.get("start_date", "2026-04-06")
    planning_days = meta.get("planning_days", 7)
    multi_day = meta.get("multi_day_mode", True)
    cfg = {
        "START_DATE": start_date,
        "PLANNING_DAYS": planning_days,
        "MULTI_DAY_MODE": multi_day,
        "DAILY_WORK_START": 8.0,
        "DAILY_WORK_END": 20.0,
        "DECODE_STRATEGY": "best_fit",
        "SLOT_SCAN_STEP": 0.5,
        "STOCHASTIC_TOP_K": 8,
        "DAILY_MAX_TASKS": 2,
        "WEIGHTS": {
            "daily_overload_penalty": 3.0,
            "energy_match": 2.5,
            "deadline_penalty": 8.0,
            "overtime_penalty": 5.0,
            "priority_reward": 2.0,
        },
        "ENERGY_MATCH_MODE": "linear_l1",
        "POP_SIZE": 30,
        "GENERATIONS": 40,
        "CROSS_RATE": 0.8,
        "MUT_RATE": 0.2,
        "TOURNAMENT_K": 3,
        "MUT_SCRAMBLE_RATE": 0.15,
        "GA_SEED_HEURISTICS": True,
        "GA_LOCAL_SEARCH_STEPS": 40,
        "ENABLE_COURSE_SCHEDULE": True,
        "COURSE_SCHEDULE": {},
        "TIME_SEGMENT_MODE": False,
    }

    curve = EnergyCurve()
    engine = MultiDayScheduleEngine(tasks, curve, config=cfg)
    ga = ScheduleGA(tasks, curve, config=cfg)
    best, _ = ga.run(verbose=False)
    schedule, _ = engine.evaluate_individual(best)
    return engine, schedule


def parse_args():
    p = argparse.ArgumentParser(description="生成精力匹配/适应度策略 PPT 配图")
    p.add_argument("--dataset", type=str, default="datasets/benchmark/medium_seed42.json")
    p.add_argument("--output", type=str, default=str(OUTPUT_DEFAULT))
    p.add_argument("--strategies", type=str, default="all")
    p.add_argument("--skip-ga", action="store_true", help="跳过 GA，用 HERF 顺序快速解码")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_style()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.strategies.strip().lower() == "all":
        model_ids = list_match_model_ids()
    else:
        model_ids = [x.strip() for x in args.strategies.split(",") if x.strip()]

    print(f"生成 {len(model_ids)} 种匹配策略配图 -> {out.resolve()}")

    plot_match_curves(out / "match_curves_grid.png", model_ids)
    plot_match_heatmaps(out / "match_heatmaps.png", model_ids)
    plot_overlay_linear_vs_gaussian(out / "match_three_compare.png")
    plot_legend_card(out / "fitness_match_legend.png")

    ds = Path(args.dataset)
    if ds.exists():
        if args.skip_ga:
            from core.schedule_engine import MultiDayScheduleEngine
            from 新建文件夹.energy_curve import EnergyCurve
            tasks, meta = load_tasks_from_file(str(ds))
            curve = EnergyCurve()
            skip_cfg = {
                "START_DATE": meta.get("start_date", "2026-04-06"),
                "PLANNING_DAYS": meta.get("planning_days", 7),
                "MULTI_DAY_MODE": meta.get("multi_day_mode", True),
                "DAILY_WORK_START": 8.0,
                "DAILY_WORK_END": 20.0,
                "DECODE_STRATEGY": "best_fit",
                "SLOT_SCAN_STEP": 0.5,
                "STOCHASTIC_TOP_K": 8,
                "DAILY_MAX_TASKS": 2,
                "WEIGHTS": {"daily_overload_penalty": 3.0, "energy_match": 2.5, "deadline_penalty": 8.0, "overtime_penalty": 5.0, "priority_reward": 2.0},
                "ENERGY_MATCH_MODE": "linear_l1",
                "ENABLE_COURSE_SCHEDULE": False,
                "TIME_SEGMENT_MODE": False,
            }
            engine = MultiDayScheduleEngine(tasks, curve, config=skip_cfg)
            perm = sorted(range(len(tasks)), key=lambda i: -tasks[i].energy_req)
            schedule = engine.decode_schedule(perm)
        else:
            print("  运行 GA 获取示例排程（约 30s）...")
            engine, schedule = run_with_dataset(ds, out)

        plot_strategy_bars_on_schedule(
            engine, schedule, out / "match_strategy_bars.png", model_ids
        )

        sched_enriched = _schedule_with_slot_energy(engine, schedule)
        weights = engine.config["WEIGHTS"] if isinstance(engine.config, dict) else engine.config.WEIGHTS
        daily_max = engine.config.get("DAILY_MAX_TASKS", 2) if isinstance(engine.config, dict) else getattr(engine.config, 'DAILY_MAX_TASKS', 2)
        decomp_default = decompose_fitness(
            sched_enriched,
            weights,
            "linear_l1",
            engine.planning_days,
            engine.start_date,
            daily_max,
        )
        plot_fitness_decomposition(decomp_default, out / "fitness_decomposition.png")

        rows = []
        for mid in model_ids:
            meta = get_match_meta(mid)
            d = decompose_fitness(
                sched_enriched,
                weights,
                mid,
                engine.planning_days,
                engine.start_date,
                daily_max,
            )
            d["name_zh"] = meta["name_zh"]
            rows.append(d)

        df = plot_raw_vs_normalized_table(rows, out / "fitness_strategy_table.png")
        df_out = pd.DataFrame(rows)[
            ["name_zh", "match_model_id", "energy_match_pct", "reward_total",
             "penalty_total", "raw_fitness", "display_score_100"]
        ]
        csv_path = out / "fitness_match_summary.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  CSV: {csv_path}")
        print("\n" + df_out.to_string(index=False))
    else:
        print(f"  跳过排程对比（未找到数据集 {ds}）")

    print("\n完成。建议 PPT 页：heatmaps / match_curves / decomposition / strategy_bars")
