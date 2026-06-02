#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA/SA 全息对比仪表盘
=====================
六种昼夜节律 × 六种匹配算法 × GA/SA 双求解器
通过入口函数 run_dashboard(solver='ga'|'sa') 切换求解器

实验设计:
  Phase A — 六种精力曲线对比 (default, lark, owl, bimodal, gaussian, flat)
  Phase B — 六种匹配算法对比 (linear_l1, squared_l2, gaussian, exponential, hinge, asymmetric)
  Phase C — GA vs SA 对比
  Phase D — 节律×匹配 交叉热力图 (36组合)

生成图表:
  01_energy_curves.png          六种昼夜节律曲线
  02_chronotype_metrics.png     六种作息人群的指标对比
  03_chronotype_radar.png       每种作息人群的能力雷达图
  04_chronotype_heatmap.png     各种作息人群的任务时段分布
  05_match_models.png           六种匹配算法对比
  06_cross_heatmap.png          节律×匹配 交叉热力图
  07_ga_vs_sa.png               GA vs SA 全维度对比
  08_lark_vs_owl.png            早起鸟 vs 夜猫子 深度对比
  09_findings.png               自动生成的有趣发现
  10_convergence.png            收敛曲线合集
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json

from core.config import CONFIG, Config
from core.energy_curve import EnergyCurve
from core.energy_curve_models import ENERGY_MODEL_CATALOG, compute_energy_values
from core.slot_scheduler import SlotScheduler
from core.slot_ga_solver import SlotGA
from core.fitness_match_models import MATCH_MODEL_CATALOG, compute_energy_match

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "datasets" / "benchmark"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_output"

ENERGY_MODELS = [m["id"] for m in ENERGY_MODEL_CATALOG]  # 6
MATCH_MODELS = [m["id"] for m in MATCH_MODEL_CATALOG]    # 6

ECOLOR = {m["id"]: m["color"] for m in ENERGY_MODEL_CATALOG}
ENAME = {m["id"]: m["name_zh"] for m in ENERGY_MODEL_CATALOG}
MCOLOR = {m["id"]: m["color"] for m in MATCH_MODEL_CATALOG}
MNAME = {m["id"]: m["name_zh"] for m in MATCH_MODEL_CATALOG}

SCENARIO_COLORS = {"light": "#2ca02c", "medium": "#ff7f0e", "heavy": "#d62728"}


# ═══════════════════════════════════════════════════════════════════
# 数据收集
# ═══════════════════════════════════════════════════════════════════

def _collect_instances(fast: bool = True):
    """收集 benchmark 实例。fast=True 时每场景取 1 个 (共 3 个)。"""
    manifest = BENCHMARK_DIR / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        records = data.get("instances", [])
        paths = [(str(BENCHMARK_DIR / r["file"]), r.get("scenario", "unknown"))
                 for r in records if (BENCHMARK_DIR / r["file"]).exists()]
    else:
        paths = []
        for fp in sorted(BENCHMARK_DIR.glob("*_inst*_seed*.json")):
            sc = "light" if "light" in fp.name else "medium" if "medium" in fp.name else "heavy"
            paths.append((str(fp), sc))

    if fast:
        seen = {}
        for p, sc in paths:
            if sc not in seen:
                seen[sc] = p
        paths = [(p, sc) for sc, p in sorted(seen.items())]
    return paths


def load_tasks_safe(path: str):
    """加载任务，处理不同来源的 dataset 模块"""
    try:
        from SA.scripts.dataset import load_tasks_from_file
        return load_tasks_from_file(path)
    except Exception:
        pass
    # fallback: 用 json 直接加载
    with open(path) as f:
        data = json.load(f)
    from core.task import Task
    tasks = [Task(**t) if isinstance(t, dict) else t for t in data.get("tasks", data)]
    return tasks, data.get("meta", {})


def run_one(path: str, *, energy_model: str = "default",
            match_model: str = "gaussian", solver: str = "ga",
            config=None):
    """Run one optimization on one instance.

    Returns (scheduler, solver_obj, schedule, history, metrics)
    """
    if config is None:
        config = CONFIG

    tasks, meta = load_tasks_safe(path)

    # Build config with requested match model
    if isinstance(config, Config):
        cfg = config
    else:
        cfg = Config()

    # Override energy match mode
    old_match = cfg.ENERGY_MATCH_MODE
    cfg.ENERGY_MATCH_MODE = match_model

    energy_curve = EnergyCurve(model_id=energy_model)
    scheduler = SlotScheduler(tasks, energy_curve, cfg)

    if solver == "ga":
        slv = SlotGA(scheduler, config=cfg)
    elif solver == "sa":
        from solvers.slot_sa import SlotSA
        slv = SlotSA(scheduler, config=cfg)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    best_solution, history = slv.run(verbose=False)
    schedule = scheduler.assign(best_solution)
    metrics = scheduler.evaluate_slots(best_solution)

    # Restore
    cfg.ENERGY_MATCH_MODE = old_match

    return scheduler, slv, schedule, history, metrics


# ═══════════════════════════════════════════════════════════════════
# 图表 1: 六种昼夜节律曲线
# ═══════════════════════════════════════════════════════════════════

def plot_energy_curves(save_to: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.linspace(0, 24, 300)
    for m in ENERGY_MODEL_CATALOG:
        vals = compute_energy_values(hours, m["id"])
        ax.plot(hours, vals, color=m["color"], linewidth=2.2, label=f'{m["name_zh"]} ({m["name_en"]})')
        # Mark peak
        i_peak = np.argmax(vals)
        ax.annotate(f'{hours[i_peak]:.0f}h',
                    (hours[i_peak], vals[i_peak]),
                    fontsize=7, color=m["color"], fontweight="bold",
                    textcoords="offset points", xytext=(0, 6), ha="center")

    # 标注工作时段
    ax.axvspan(8, 22, facecolor="green", alpha=0.04)
    ax.axvspan(13, 14, facecolor="orange", alpha=0.08)
    ax.text(10.5, 0.96, "工作时间 8:00-22:00", fontsize=8, color="#888", ha="center")
    ax.text(13.5, 0.12, "午休", fontsize=7, color="#b08800", ha="center")

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("小时 (0-24h)", fontsize=11)
    ax.set_ylabel("精力水平", fontsize=11)
    ax.set_title("六种昼夜节律（精力曲线）—— 六种「人设」", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.2)

    # 加注解读文字
    notes = [
        "[Lark] 早晨型: 6-10点高峰，适合早起处理高难度任务",
        "[Owl] 夜晚型: 17-22点高峰，上午不振",
        "[Bimodal] 双峰型: 上午+晚间双峰，午后有低谷",
        "[Gaussian] 单峰型: 10点单峰，下午平缓下降",
        "[Flat] 平稳型: 全天波动小，适合均匀分配",
    ]
    for i, note in enumerate(notes):
        ax.text(0.02, 0.95 - i * 0.06, note, transform=ax.transAxes,
                fontsize=7.5, color="#555", va="top")

    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 01_energy_curves -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 2: 六种作息人群的指标对比
# ═══════════════════════════════════════════════════════════════════

def plot_chronotype_metrics(all_results: dict, save_to: Path):
    """all_results: {energy_model_id: [metrics_dict, ...]}"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    mkeys = ["energy_match", "on_time_rate", "composite", "fitness", "load_std", "makespan"]
    titles = ["精力匹配度", "准时率", "综合分", "适应度", "负载标准差", "工期 (h)"]
    em_ids = ENERGY_MODELS

    for ax, mkey, title in zip(axes.flat, mkeys, titles):
        box_data = []
        for em in em_ids:
            vals = [m[mkey] for m in all_results.get(em, [])]
            if vals:
                box_data.append(vals)
            else:
                box_data.append([0])
        bp = ax.boxplot(box_data, tick_labels=[ENAME[e] for e in em_ids],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))
        for patch, em in zip(bp["boxes"], em_ids):
            patch.set_facecolor(ECOLOR[em])
            patch.set_alpha(0.6)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("六种作息人群的全指标对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 02_chronotype_metrics -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 3: 每种作息人群的能力雷达图
# ═══════════════════════════════════════════════════════════════════

def plot_chronotype_radar(all_results: dict, save_to: Path):
    n = len(ENERGY_MODELS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5),
                             subplot_kw={"projection": "polar"})
    axes = axes.flatten() if n > 1 else [axes]

    categories = ["精力匹配", "准时率", "负载均衡", "工期效率", "综合分"]
    N = len(categories)

    for i, em in enumerate(ENERGY_MODELS):
        ax = axes[i]
        metrics_list = all_results.get(em, [])
        if not metrics_list:
            ax.set_title(f"{ENAME[em]}\n(无数据)", fontweight="bold")
            continue

        avg_em_val = np.mean([m["energy_match"] for m in metrics_list])
        avg_otr = np.mean([m["on_time_rate"] for m in metrics_list])
        avg_lstd = np.mean([m["load_std"] for m in metrics_list])
        avg_mk = np.mean([m["makespan"] for m in metrics_list])
        avg_comp = np.mean([m["composite"] for m in metrics_list])

        load_balance = max(0, 1.0 - min(avg_lstd / 4.0, 1.0))
        makespan_eff = max(0, 1.0 - min(avg_mk / 200.0, 1.0))
        values = [avg_em_val, avg_otr, load_balance, makespan_eff, avg_comp]

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
        vals_plot = values + values[:1]

        ax.fill(angles, vals_plot, color=ECOLOR[em], alpha=0.2)
        ax.plot(angles, vals_plot, color=ECOLOR[em], linewidth=2)
        ax.set_xticks(angles[:N])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
        ax.set_title(f"{ENAME[em]}\n综合分={avg_comp:.3f}", fontweight="bold",
                     color=ECOLOR[em], fontsize=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("六种作息人群的能力雷达图", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 03_chronotype_radar -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 4: 各种作息人群的任务时段分布热力图
# ═══════════════════════════════════════════════════════════════════

def _collect_schedule_details(scheduler, schedule):
    """Extract per-task timing details."""
    records = []
    for item in schedule:
        t = item["task"]
        s, e = item["start"], item["end"]
        hour = s.hour + s.minute / 60
        slot_e = scheduler.energy_curve.get_average_energy(s, e)
        m = float(compute_energy_match(t.energy_req, slot_e, "gaussian"))
        records.append({
            "type": t.task_type,
            "start_h": hour,
            "duration": t.duration,
            "energy_req": t.energy_req,
            "slot_e": slot_e,
            "match": m,
            "on_time": e <= t.deadline,
            "priority": t.priority,
        })
    return records


def plot_chronotype_heatmap(all_schedules: dict, save_to: Path):
    """all_schedules: {energy_model_id: [(scheduler, schedule), ...]}"""
    hour_bins = [8, 10, 12, 14, 16, 18, 20, 22]
    bin_labels = ["8-10", "10-12", "12-14", "14-16", "16-18", "18-20", "20-22"]
    task_types = ["coding", "writing", "meeting", "admin", "review", "learning"]
    type_labels = ["编程", "写作", "会议", "行政", "复习", "学习"]

    n_em = len(ENERGY_MODELS)
    cols = 3
    rows = (n_em + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes = axes.flatten() if n_em > 1 else [axes]

    for i, em in enumerate(ENERGY_MODELS):
        ax = axes[i]
        sched_list = all_schedules.get(em, [])
        if not sched_list:
            ax.set_title(f"{ENAME[em]}\n(无数据)")
            continue

        # Aggregate all tasks across instances
        heatmap = np.zeros((len(task_types), len(bin_labels)))
        total_tasks = 0
        for scheduler, schedule in sched_list:
            records = _collect_schedule_details(scheduler, schedule)
            for rec in records:
                total_tasks += 1
                h = rec["start_h"]
                ti = task_types.index(rec["type"]) if rec["type"] in task_types else -1
                if ti < 0:
                    continue
                for j in range(len(hour_bins) - 1):
                    if hour_bins[j] <= h < hour_bins[j + 1]:
                        heatmap[ti, j] += 1
                        break

        # Normalize by column (show preference)
        heatmap_norm = heatmap / (heatmap.sum(axis=0, keepdims=True) + 0.01)

        im = ax.imshow(heatmap_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=7)
        ax.set_yticks(range(len(task_types)))
        ax.set_yticklabels(type_labels, fontsize=8)
        ax.set_title(f"{ENAME[em]} (n={total_tasks})", fontweight="bold", color=ECOLOR[em])

        for ti in range(len(task_types)):
            for bj in range(len(bin_labels)):
                if heatmap[ti, bj] > 0:
                    ax.text(bj, ti, f"{int(heatmap[ti,bj])}", ha="center", va="center",
                            fontsize=6, color="white" if heatmap_norm[ti, bj] > 0.25 else "black")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("各种作息人群的任务时段分布热力图\n(数字=任务数, 颜色=列归一化占比)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 04_chronotype_heatmap -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 5: 六种匹配算法对比
# ═══════════════════════════════════════════════════════════════════

def plot_match_models(all_results: dict, save_to: Path):
    """all_results: {match_model_id: [metrics_dict, ...]}"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    mkeys = ["energy_match", "on_time_rate", "composite", "fitness", "load_std", "makespan"]
    titles = ["精力匹配度", "准时率", "综合分", "适应度", "负载标准差", "工期 (h)"]
    mm_ids = MATCH_MODELS

    for ax, mkey, title in zip(axes.flat, mkeys, titles):
        box_data = []
        for mm in mm_ids:
            vals = [m[mkey] for m in all_results.get(mm, [])]
            box_data.append(vals if vals else [0])
        bp = ax.boxplot(box_data, tick_labels=[MNAME[m] for m in mm_ids],
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))
        for patch, mm in zip(bp["boxes"], mm_ids):
            patch.set_facecolor(MCOLOR[mm])
            patch.set_alpha(0.6)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)

    fig.suptitle("六种匹配算法的全指标对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 05_match_models -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 6: 节律 × 匹配 交叉热力图 (36组合)
# ═══════════════════════════════════════════════════════════════════

def plot_cross_heatmap(cross_data: dict, save_to: Path):
    """cross_data: {(energy_model, match_model): metrics}"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mkeys = ["composite", "energy_match", "on_time_rate", "fitness", "load_std", "makespan"]
    titles = ["综合分", "精力匹配度", "准时率", "适应度", "负载标准差", "工期"]

    for ax, mkey, title in zip(axes.flat, mkeys, titles):
        mat = np.zeros((len(ENERGY_MODELS), len(MATCH_MODELS)))
        for i, em in enumerate(ENERGY_MODELS):
            for j, mm in enumerate(MATCH_MODELS):
                m = cross_data.get((em, mm))
                if m:
                    mat[i, j] = m[mkey]

        im = ax.imshow(mat, cmap="RdYlGn" if mkey != "makespan" and mkey != "load_std" else "RdYlGn_r",
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(MATCH_MODELS)))
        ax.set_xticklabels([MNAME[m] for m in MATCH_MODELS], rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(ENERGY_MODELS)))
        ax.set_yticklabels([ENAME[e] for e in ENERGY_MODELS], fontsize=8)
        ax.set_title(title, fontweight="bold")

        # Annotate best
        best_idx = np.unravel_index(np.argmax(mat) if mkey != "makespan" and mkey != "load_std"
                                    else np.argmin(mat), mat.shape)
        for i in range(len(ENERGY_MODELS)):
            for j in range(len(MATCH_MODELS)):
                ax.text(j, i, f"{mat[i,j]:.3f}" if mkey in ("composite", "energy_match", "on_time_rate")
                        else f"{mat[i,j]:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] > np.median(mat) else "black")
        # Highlight best cell
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=3, linestyle="--"))

        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("六种节律 × 六种匹配算法 = 36 种组合交叉热力图\n(虚线框=最优组合)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 06_cross_heatmap -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 7: GA vs SA 对比
# ═══════════════════════════════════════════════════════════════════

def plot_ga_vs_sa(ga_results: list[dict], sa_results: list[dict], save_to: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    mkeys = ["energy_match", "on_time_rate", "composite", "fitness", "load_std", "makespan"]
    titles = ["精力匹配度", "准时率", "综合分", "适应度", "负载标准差", "工期 (h)"]

    for ax, mkey, title in zip(axes.flat, mkeys, titles):
        ga_vals = [m[mkey] for m in ga_results]
        sa_vals = [m[mkey] for m in sa_results]
        data = [ga_vals, sa_vals]
        bp = ax.boxplot(data, tick_labels=["GA", "SA"], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][1].set_facecolor("#ff7f0e")
        for b in bp["boxes"]:
            b.set_alpha(0.6)

        # Individual points
        for i, vals in enumerate(data):
            x = np.random.normal(i + 1, 0.04, len(vals))
            ax.scatter(x, vals, alpha=0.5, s=30, color=["#1f77b4", "#ff7f0e"][i], edgecolors="white")
            ax.text(i + 1, np.mean(vals), f"μ={np.mean(vals):.3f}", ha="center", fontsize=8,
                    fontweight="bold", va="bottom")

        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    # Convergence subplot
    fig.delaxes(axes[1, 2])  # can't easily, so we'll just use different layout
    fig.suptitle("GA vs SA 全维度对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 07_ga_vs_sa -> {save_to.name}")


def plot_ga_vs_sa_v2(ga_results: list[dict], sa_results: list[dict],
                     ga_histories: list[list], sa_histories: list[list],
                     save_to: Path):
    """Full GA vs SA with convergence curves."""
    fig = plt.figure(figsize=(18, 12))

    # 左上到右下各子图
    metrics_config = [
        (0, 3, 0, 2, "energy_match", "精力匹配度"),
        (1, 3, 0, 2, "on_time_rate", "准时率"),
        (2, 3, 0, 2, "composite", "综合分"),
        (0, 3, 2, 4, "fitness", "适应度"),
        (1, 3, 2, 4, "load_std", "负载标准差"),
        (2, 3, 2, 4, "makespan", "工期 (h)"),
    ]

    gs = fig.add_gridspec(4, 3)
    for col, ncols, row, nrows, mkey, title in metrics_config:
        ax = fig.add_subplot(gs[row:row + nrows, col:col + ncols])
        ga_vals = [m[mkey] for m in ga_results]
        sa_vals = [m[mkey] for m in sa_results]
        data = [ga_vals, sa_vals]
        bp = ax.boxplot(data, tick_labels=["GA", "SA"], patch_artist=True, widths=0.4)
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][1].set_facecolor("#ff7f0e")
        for b in bp["boxes"]:
            b.set_alpha(0.6)
        for i, vals in enumerate(data):
            if vals:
                x_jitter = np.random.normal(i + 1, 0.03, len(vals))
                ax.scatter(x_jitter, vals, alpha=0.4, s=25,
                          color=["#1f77b4", "#ff7f0e"][i], edgecolors="white")
        ax.set_title(f"{title}: GA={np.mean(ga_vals):.3f} vs SA={np.mean(sa_vals):.3f}",
                     fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    # Convergence curves (bottom row, spanning)
    ax_conv = fig.add_subplot(gs[3, :])
    for hist in ga_histories:
        ax_conv.plot(hist, color="#1f77b4", alpha=0.25, linewidth=0.5)
    for hist in sa_histories:
        ax_conv.plot(hist, color="#ff7f0e", alpha=0.25, linewidth=0.5)
    # Medians
    if ga_histories:
        max_len = max(len(h) for h in ga_histories)
        padded = np.full((len(ga_histories), max_len), np.nan)
        for i, h in enumerate(ga_histories):
            padded[i, :len(h)] = h
        ax_conv.plot(np.nanmedian(padded, axis=0), color="#1f77b4", linewidth=2, label="GA 中位")
    if sa_histories:
        max_len = max(len(h) for h in sa_histories)
        padded = np.full((len(sa_histories), max_len), np.nan)
        for i, h in enumerate(sa_histories):
            padded[i, :len(h)] = h
        ax_conv.plot(np.nanmedian(padded, axis=0), color="#ff7f0e", linewidth=2, label="SA 中位")
    ax_conv.set_xlabel("迭代步数 (GA=代数, SA=温度循环)", fontsize=10)
    ax_conv.set_ylabel("Fitness", fontsize=10)
    ax_conv.set_title("收敛曲线对比 (细线=单次运行, 粗线=中位)", fontweight="bold")
    ax_conv.legend(fontsize=9)
    ax_conv.grid(True, alpha=0.2)

    fig.suptitle("GA vs SA 全维度对比 + 收敛曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 07_ga_vs_sa -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 8: 早起鸟 vs 夜猫子深度对比
# ═══════════════════════════════════════════════════════════════════

def plot_lark_vs_owl(all_schedules: dict, all_metrics: dict, save_to: Path):
    """Deep comparison of lark vs owl chronotypes."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    lark_sched = all_schedules.get("lark", [])
    owl_sched = all_schedules.get("owl", [])

    # 1: 任务时段分布叠加直方图
    ax = axes[0, 0]
    for label, sched_list, color in [("早晨型", lark_sched, "#1f77b4"), ("夜晚型", owl_sched, "#9467bd")]:
        all_hours = []
        for scheduler, schedule in sched_list:
            for item in schedule:
                all_hours.append(item["start"].hour + item["start"].minute / 60)
        ax.hist(all_hours, bins=np.linspace(8, 22, 15), alpha=0.5, color=color, label=label, edgecolor="white")
    ax.axvspan(13, 14, facecolor="orange", alpha=0.1)
    ax.set_xlabel("时段 (h)"); ax.set_ylabel("任务数")
    ax.set_title("任务开始时段分布")
    ax.legend(); ax.grid(True, alpha=0.2)

    # 2: 分任务类型的平均匹配度
    ax = axes[0, 1]
    task_types = ["coding", "writing", "meeting", "admin", "review", "learning"]
    type_labels = ["编程", "写作", "会议", "行政", "复习", "学习"]
    x = np.arange(len(task_types))
    w = 0.35

    for label, sched_list, color, offset in [("早晨型", lark_sched, "#1f77b4", -w/2),
                                              ("夜晚型", owl_sched, "#9467bd", w/2)]:
        type_matches = {t: [] for t in task_types}
        for scheduler, schedule in sched_list:
            records = _collect_schedule_details(scheduler, schedule)
            for rec in records:
                if rec["type"] in type_matches:
                    type_matches[rec["type"]].append(rec["match"])
        means = [np.mean(type_matches[t]) if type_matches[t] else 0 for t in task_types]
        errs = [np.std(type_matches[t]) if type_matches[t] else 0 for t in task_types]
        ax.bar(x + offset, means, w, yerr=errs, color=color, alpha=0.7, label=label, capsize=3)

    ax.set_xticks(x); ax.set_xticklabels(type_labels, fontsize=8)
    ax.set_ylabel("平均精力匹配度"); ax.set_title("各任务类型的精力匹配")
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")

    # 3: 各时段的精力匹配度曲线
    ax = axes[0, 2]
    hours = np.arange(8, 22.5, 0.5)
    for label, sched_list, color in [("早晨型", lark_sched, "#1f77b4"), ("夜晚型", owl_sched, "#9467bd")]:
        hour_matches = {h: [] for h in hours}
        for scheduler, schedule in sched_list:
            records = _collect_schedule_details(scheduler, schedule)
            for rec in records:
                h_bin = round(rec["start_h"] * 2) / 2
                if h_bin in hour_matches:
                    hour_matches[h_bin].append(rec["match"])
        means = [np.mean(hour_matches[h]) if hour_matches[h] else np.nan for h in hours]
        ax.plot(hours, means, color=color, linewidth=2, label=label)
        ax.fill_between(hours, means, alpha=0.1, color=color)
    ax.axvspan(13, 14, facecolor="orange", alpha=0.08)
    ax.set_xlabel("时段 (h)"); ax.set_ylabel("平均匹配度")
    ax.set_title("全天各时段的精力匹配质量")
    ax.legend(); ax.grid(True, alpha=0.2)

    # 4: 高/低精力需求任务分布
    ax = axes[1, 0]
    for label, sched_list, color in [("早晨型", lark_sched, "#1f77b4"), ("夜晚型", owl_sched, "#9467bd")]:
        high_hours, low_hours = [], []
        for scheduler, schedule in sched_list:
            for item in schedule:
                t = item["task"]
                h = item["start"].hour + item["start"].minute / 60
                if t.energy_req >= 0.7:
                    high_hours.append(h)
                else:
                    low_hours.append(h)
        ax.hist(high_hours, bins=np.linspace(8, 22, 12), alpha=0.4, color=color,
                label=f"{label} 高需求(≥0.7)", edgecolor="white", histtype="stepfilled")
        ax.hist(low_hours, bins=np.linspace(8, 22, 12), alpha=0.2, color=color,
                label=f"{label} 低需求(<0.7)", edgecolor="white", hatch="//")
    ax.axvspan(13, 14, facecolor="orange", alpha=0.1)
    ax.set_xlabel("时段 (h)"); ax.set_ylabel("任务数")
    ax.set_title("高/低精力需求任务的时段分布")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.2)

    # 5: 准时率对比
    ax = axes[1, 1]
    for i, (label, sched_list, color) in enumerate([
        ("早晨型", lark_sched, "#1f77b4"), ("夜晚型", owl_sched, "#9467bd")]):
        on_time_rates = []
        for scheduler, schedule in sched_list:
            records = _collect_schedule_details(scheduler, schedule)
            if records:
                on_time_rates.append(np.mean([r["on_time"] for r in records]))
        if on_time_rates:
            ax.bar(i, np.mean(on_time_rates), color=color, alpha=0.7)
            ax.errorbar(i, np.mean(on_time_rates), yerr=np.std(on_time_rates), color="black", capsize=5)
            ax.text(i, np.mean(on_time_rates) / 2, f"{np.mean(on_time_rates):.1%}", ha="center",
                    fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["早晨型(Lark)", "夜晚型(Owl)"])
    ax.set_ylabel("准时率"); ax.set_title("准时率对比")
    ax.grid(True, alpha=0.2, axis="y")

    # 6: 核心指标横向对比
    ax = axes[1, 2]
    ax.axis("off")
    lines = []
    for mkey, label in [("energy_match", "精力匹配"), ("on_time_rate", "准时率"),
                         ("composite", "综合分"), ("fitness", "适应度")]:
        lark_vals = [m[mkey] for m in all_metrics.get("lark", [])]
        owl_vals = [m[mkey] for m in all_metrics.get("owl", [])]
        if lark_vals and owl_vals:
            lark_m, owl_m = np.mean(lark_vals), np.mean(owl_vals)
            better = "早晨型胜" if lark_m > owl_m else "夜晚型胜"
            lines.append(f"{label}: 早晨={lark_m:.3f}  夜晚={owl_m:.3f}  → {better}")
        else:
            lines.append(f"{label}: 无数据")
    for i, line in enumerate(lines):
        ax.text(0.05, 0.95 - i * 0.12, line, fontsize=10, fontweight="bold",
                transform=ax.transAxes, va="top")
    ax.set_title("核心指标对比总结", fontweight="bold")

    fig.suptitle("早起鸟 vs 夜猫子 —— 两种极端作息的全维度对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 08_lark_vs_owl -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 9: 自动生成的有趣发现
# ═══════════════════════════════════════════════════════════════════

def _generate_findings(chrono_metrics: dict, all_schedules: dict, ga_vs_sa: dict,
                       cross_data: dict, match_metrics: dict = None) -> list[str]:
    """Auto-generate interesting observations from experimental data."""
    findings = []

    # 1. Best chronotype
    best_comp = -1
    best_em = None
    for em in ENERGY_MODELS:
        vals = [m["composite"] for m in chrono_metrics.get(em, [])]
        if vals and np.mean(vals) > best_comp:
            best_comp = np.mean(vals)
            best_em = em

    findings.append(
        f"[最佳] 综合分最高的作息: {ENAME.get(best_em, best_em)} (composite={best_comp:.3f}) — "
        f"在所有指标上表现最均衡"
    )

    # 2. Lark vs Owl specific
    lark_em = [m["energy_match"] for m in chrono_metrics.get("lark", [])]
    owl_em = [m["energy_match"] for m in chrono_metrics.get("owl", [])]
    lark_ot = [m["on_time_rate"] for m in chrono_metrics.get("lark", [])]
    owl_ot = [m["on_time_rate"] for m in chrono_metrics.get("owl", [])]

    if lark_em and owl_em:
        if np.mean(lark_em) > np.mean(owl_em):
            findings.append(
                f"[Lark>Owl] 早晨型在精力匹配上优于夜晚型 "
                f"({np.mean(lark_em):.3f} vs {np.mean(owl_em):.3f}) — "
                f"工作时段(8-22)的精力曲线更匹配标准任务的精力需求"
            )
        else:
            findings.append(
                f"[Owl>Lark] 夜晚型在精力匹配上优于早晨型 "
                f"({np.mean(owl_em):.3f} vs {np.mean(lark_em):.3f}) — "
                f"夜猫子在晚间集中精力处理高难度任务有优势"
            )

    if lark_ot and owl_ot:
        findings.append(
            f"[守时] 准时率对比: 早晨型 {np.mean(lark_ot):.1%} vs 夜晚型 {np.mean(owl_ot):.1%} — "
            f"{'早晨型' if np.mean(lark_ot) > np.mean(owl_ot) else '夜晚型'}更守时"
        )

    # 3. Best match model
    best_mm_comp = -1
    best_mm = None
    if match_metrics:
        for mm in MATCH_MODELS:
            vals = [m["composite"] for m in match_metrics.get(mm, [])]
            if vals and np.mean(vals) > best_mm_comp:
                best_mm_comp = np.mean(vals)
                best_mm = mm
    if best_mm:
        findings.append(
            f"[匹配] 最优匹配算法: {MNAME.get(best_mm, best_mm)} (composite={best_mm_comp:.3f})"
        )

    # 4. GA vs SA
    ga_comp = [m["composite"] for m in ga_vs_sa.get("ga", [])]
    sa_comp = [m["composite"] for m in ga_vs_sa.get("sa", [])]
    if ga_comp and sa_comp:
        winner = "GA" if np.mean(ga_comp) > np.mean(sa_comp) else "SA"
        findings.append(
            f"[GAvsSA] {winner}胜出, GA={np.mean(ga_comp):.3f} vs SA={np.mean(sa_comp):.3f}"
        )

    # 5. Best cross combination
    if cross_data:
        best_cross = max(cross_data.items(), key=lambda kv: kv[1].get("composite", 0))
        (best_ec, best_mc), best_cross_met = best_cross
        findings.append(
            f"[最佳组合] {ENAME.get(best_ec, best_ec)} x {MNAME.get(best_mc, best_mc)} "
            f"(composite={best_cross_met['composite']:.3f}) — "
            f"正确的精力曲线+匹配算法组合可以事半功倍"
        )

    # 6. Flat baseline insight
    flat_comp = [m["composite"] for m in chrono_metrics.get("flat", [])]
    if flat_comp and best_comp > 0:
        findings.append(
            f"[对照] 平稳型(Flat)作为对照: composite={np.mean(flat_comp):.3f} — "
            f"没有精力波动的假设下，系统仍能通过匹配算法优化，"
            f"但比最优节律({ENAME.get(best_em,'')})低 {(best_comp - np.mean(flat_comp)) / best_comp * 100:.1f}%"
        )

    # 7. Night owl task placement
    owl_sched = all_schedules.get("owl", [])
    if owl_sched:
        night_count = 0
        total_count = 0
        for scheduler, schedule in owl_sched:
            for item in schedule:
                h = item["start"].hour + item["start"].minute / 60
                total_count += 1
                if h >= 18:
                    night_count += 1
        if total_count > 0:
            pct = night_count / total_count * 100
            findings.append(
                f"[夜猫子] 夜猫子有 {pct:.0f}% 的任务安排在 18:00 之后 — "
                f"他们的'黄金时段'是傍晚到夜间"
            )

    # 8. Bimodal balance
    bimodal_vals = [m["load_std"] for m in chrono_metrics.get("bimodal", [])]
    if bimodal_vals:
        findings.append(
            f"[双峰] 双峰型(Bimodal)的负载标准差均值={np.mean(bimodal_vals):.2f} — "
            f"早晚两段峰值天然引导任务分散，有助于负载均衡"
        )

    return findings


def plot_findings(findings: list[str], save_to: Path):
    fig, ax = plt.subplots(figsize=(14, len(findings) * 0.8 + 2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.97, "自动发现的有趣结论", ha="center", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.91, "(基于本次实验数据的统计分析)", ha="center", fontsize=9, color="#888",
            transform=ax.transAxes)

    for i, finding in enumerate(findings):
        y = 0.84 - i * 0.09
        # Color box
        colors = ["#e3f2fd", "#f3e5f5", "#e8f5e9", "#fff3e0", "#fce4ec", "#e0f7fa",
                  "#fff8e1", "#f1f8e9", "#ede7f6", "#fbe9e7"]
        box_color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.9, 0.07, transform=ax.transAxes,
                                   facecolor=box_color, edgecolor="#ddd", linewidth=0.5,
                                   zorder=0))
        ax.text(0.08, y, finding, fontsize=10, transform=ax.transAxes, va="center",
                wrap=True)

    fig.suptitle("", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 09_findings -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 图表 10: 收敛曲线合集
# ═══════════════════════════════════════════════════════════════════

def plot_convergence(all_histories: dict, all_metrics: dict, save_to: Path):
    """all_histories: {energy_model_id: [history_list, ...]}"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, em in enumerate(ENERGY_MODELS):
        ax = axes[i // 3, i % 3]
        histories = all_histories.get(em, [])
        if not histories:
            ax.set_title(f"{ENAME[em]} (无数据)")
            continue

        for hist in histories:
            ax.plot(hist, color=ECOLOR[em], alpha=0.3, linewidth=0.5)

        # Median
        max_len = max(len(h) for h in histories)
        padded = np.full((len(histories), max_len), np.nan)
        for j, h in enumerate(histories):
            padded[j, :len(h)] = h
        median = np.nanmedian(padded, axis=0)
        ax.plot(median, color="black", linewidth=2, label="中位")

        # Final composite
        comp_vals = [m["composite"] for m in all_metrics.get(em, [])]
        ax.set_title(f"{ENAME[em]} (μ comp={np.mean(comp_vals):.3f})"
                     if comp_vals else ENAME[em],
                     color=ECOLOR[em], fontweight="bold")
        ax.set_xlabel("代数"); ax.set_ylabel("Fitness")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    fig.suptitle("六种作息人群的 GA 收敛曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 10_convergence -> {save_to.name}")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def run_dashboard(solver: str = "ga", fast: bool = True, output_dir: str = None):
    """运行全息对比仪表盘

    Args:
        solver: 'ga' 或 'sa'
        fast: True=每场景1实例(共3), False=全量15实例
        output_dir: 输出目录, 默认 benchmark_output

    实验流程:
        Phase A — 六种精力曲线, 使用默认匹配算法(gaussian)
        Phase B — 六种匹配算法, 使用默认精力曲线(default)
        Phase C — GA vs SA 对比
        Phase D — 36组合交叉热力图 (仅1个medium实例)
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    instances = _collect_instances(fast=fast)

    solver_label = solver.upper()
    prefix = f"dashboard_{solver.lower()}"

    print("=" * 70)
    print(f"GA/SA 全息仪表盘 — solver={solver_label}, instances={len(instances)}")
    print("=" * 70)

    # ── Phase A: 六种精力曲线 ──
    print("\n" + "=" * 50)
    print("Phase A: 六种昼夜节律对比 (match_model=gaussian)")
    print("=" * 50)

    phase_a_metrics = defaultdict(list)
    phase_a_schedules = defaultdict(list)
    phase_a_histories = defaultdict(list)

    for em in ENERGY_MODELS:
        for path, scenario in instances:
            name = Path(path).stem
            print(f"  [{ENAME[em]}] {name} ({scenario})...", end=" ")
            try:
                sched, slv, schedule, history, metrics = run_one(
                    path, energy_model=em, match_model="gaussian", solver=solver
                )
                phase_a_metrics[em].append(metrics)
                phase_a_schedules[em].append((sched, schedule))
                phase_a_histories[em].append(history)
                print(f"comp={metrics['composite']:.3f} em={metrics['energy_match']:.3f}")
            except Exception as e:
                print(f"FAIL: {e}")

    # ── Phase B: 六种匹配算法 ──
    print("\n" + "=" * 50)
    print("Phase B: 六种匹配算法对比 (energy_model=default)")
    print("=" * 50)

    phase_b_metrics = defaultdict(list)

    for mm in MATCH_MODELS:
        for path, scenario in instances:
            name = Path(path).stem
            print(f"  [{MNAME[mm]}] {name} ({scenario})...", end=" ")
            try:
                _, _, _, _, metrics = run_one(
                    path, energy_model="default", match_model=mm, solver=solver
                )
                phase_b_metrics[mm].append(metrics)
                print(f"comp={metrics['composite']:.3f} em={metrics['energy_match']:.3f}")
            except Exception as e:
                print(f"FAIL: {e}")

    # ── Phase C: GA vs SA (run both) ──
    print("\n" + "=" * 50)
    print("Phase C: GA vs SA 对比")
    print("=" * 50)

    ga_metrics, sa_metrics = [], []
    ga_histories, sa_histories = [], []

    for s in ["ga", "sa"]:
        for path, scenario in instances:
            name = Path(path).stem
            print(f"  [{s.upper()}] {name} ({scenario})...", end=" ")
            try:
                _, _, _, history, metrics = run_one(
                    path, energy_model="default", match_model="gaussian", solver=s
                )
                if s == "ga":
                    ga_metrics.append(metrics)
                    ga_histories.append(history)
                else:
                    sa_metrics.append(metrics)
                    sa_histories.append(history)
                print(f"comp={metrics['composite']:.3f}")
            except Exception as e:
                print(f"FAIL: {e}")

    ga_vs_sa_data = {"ga": ga_metrics, "sa": sa_metrics}

    # ── Phase D: 36组合交叉热力图 (仅1个medium实例) ──
    print("\n" + "=" * 50)
    print("Phase D: 节律×匹配 交叉热力图 (1个medium实例)")
    print("=" * 50)

    cross_data = {}
    medium_inst = [p for p, s in instances if s == "medium"]
    cross_path = medium_inst[0] if medium_inst else instances[0][0]
    cross_name = Path(cross_path).stem
    print(f"  使用实例: {cross_name}")

    for em in ENERGY_MODELS:
        for mm in MATCH_MODELS:
            print(f"  {ENAME[em]} × {MNAME[mm]} ...", end=" ")
            try:
                _, _, _, _, metrics = run_one(
                    cross_path, energy_model=em, match_model=mm, solver=solver
                )
                cross_data[(em, mm)] = metrics
                print(f"comp={metrics['composite']:.3f}")
            except Exception as e:
                print(f"FAIL: {e}")

    # ── 生成图表 ──
    print("\n" + "=" * 50)
    print("生成图表...")
    print("=" * 50)

    # 1. 精力曲线对比
    plot_energy_curves(out / f"{prefix}_01_energy_curves.png")

    # 2. 六种作息人群指标对比
    if phase_a_metrics:
        plot_chronotype_metrics(phase_a_metrics, out / f"{prefix}_02_chronotype_metrics.png")

    # 3. 能力雷达图
    if phase_a_metrics:
        plot_chronotype_radar(phase_a_metrics, out / f"{prefix}_03_chronotype_radar.png")

    # 4. 任务时段热力图
    if phase_a_schedules:
        plot_chronotype_heatmap(phase_a_schedules, out / f"{prefix}_04_chronotype_heatmap.png")

    # 5. 匹配算法对比
    if phase_b_metrics:
        plot_match_models(phase_b_metrics, out / f"{prefix}_05_match_models.png")

    # 6. 交叉热力图
    if cross_data:
        plot_cross_heatmap(cross_data, out / f"{prefix}_06_cross_heatmap.png")

    # 7. GA vs SA
    if ga_metrics and sa_metrics:
        plot_ga_vs_sa_v2(ga_metrics, sa_metrics, ga_histories, sa_histories,
                         out / f"{prefix}_07_ga_vs_sa.png")
    elif ga_metrics:
        # Only GA data available (when solver='ga'), show GA-only metrics
        plot_ga_vs_sa(ga_metrics, sa_metrics or ga_metrics,
                      out / f"{prefix}_07_ga_vs_sa.png")

    # 8. Lark vs Owl
    if phase_a_schedules.get("lark") and phase_a_schedules.get("owl"):
        plot_lark_vs_owl(phase_a_schedules, phase_a_metrics,
                         out / f"{prefix}_08_lark_vs_owl.png")

    # 9. 有趣发现
    findings = _generate_findings(phase_a_metrics, phase_a_schedules,
                                  ga_vs_sa_data, cross_data,
                                  match_metrics=phase_b_metrics)
    plot_findings(findings, out / f"{prefix}_09_findings.png")

    # 10. 收敛曲线
    if phase_a_histories:
        plot_convergence(phase_a_histories, phase_a_metrics,
                         out / f"{prefix}_10_convergence.png")

    # Print findings to console too
    print("\n" + "=" * 50)
    print("自动发现的有趣结论:")
    print("=" * 50)
    for f_text in findings:
        print(f"  {f_text}")

    print(f"\n全部图表已保存到: {out}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GA/SA 全息对比仪表盘")
    parser.add_argument("--solver", type=str, default="ga", choices=["ga", "sa"],
                        help="求解器: ga 或 sa")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="快速模式 (3实例, 默认)")
    parser.add_argument("--full", action="store_true",
                        help="全量模式 (15实例)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    fast = not args.full
    run_dashboard(solver=args.solver, fast=fast, output_dir=args.output)
