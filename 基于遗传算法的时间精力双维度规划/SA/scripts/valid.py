#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调度算法对比实验（valid.py）
================================================================================
稳定接口见 benchmark_api.py；本脚本负责 CLI、汇总表与 PPT 图表。

对比原则：仅改变任务排列求解器，共用 decode + fitness。

用法:
  python valid.py                              # GA + 基线，默认 5 实例，并行
  python valid.py --fast --workers 6
  python valid.py --algorithms GA,HERF,WSPT
  python valid.py --primary GA --algorithms GA,SA,HERF --solver SA=solvers/sa_template.py
  python valid.py --dataset datasets/benchmark/medium_seed42.json --runs 3

队友 SA:
  复制 solvers/sa_template.py，实现 solve() 后:
  python valid.py --algorithms GA,SA,HERF --solver SA=solvers/sa_team.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.benchmark_api import (
    ALGORITHM_DOCS,
    ALGO_COLORS,
    DEFAULT_ALGORITHMS,
    SLOT_ALGORITHMS,
    BenchmarkRunner,
    aggregate_cross_dataset,
    aggregate_results,
    algo_display_order,
    get_benchmark_config,
    get_slot_benchmark_config,
)
from scripts.dataset import (
    BENCHMARK_DIR,
    ensure_default_dataset,
    list_benchmark_datasets,
    load_tasks_from_file,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_output"
N_RUNS = 5
DEFAULT_INSTANCE_LIMIT = 5


def log(msg):
    print(msg, flush=True)


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 150,
    })


def _save_fig(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _algo_means(raw_df, order):
    means = raw_df.groupby("algorithm").mean(numeric_only=True)
    return means.reindex([a for a in order if a in means.index])


def parse_solver_specs(specs: list[str]) -> dict[str, str]:
    """解析 --solver SA=path/to.py"""
    out = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--solver 格式应为 NAME=path，收到: {s}")
        name, path = s.split("=", 1)
        out[name.strip().upper()] = path.strip()
    return out


def parse_algorithms(primary: str, algorithms_arg: str | None, solvers: dict, mode: str = "permutation") -> tuple[str, tuple[str, ...]]:
    if algorithms_arg:
        algos = [a.strip().upper() for a in algorithms_arg.split(",") if a.strip()]
    else:
        algos = list(SLOT_ALGORITHMS if mode == "slot" else DEFAULT_ALGORITHMS)
    primary = primary.upper()
    if primary not in algos:
        algos = [primary] + algos
    for name in solvers:
        if name not in algos:
            algos.append(name)
    return primary, tuple(dict.fromkeys(algos))


def print_algorithm_docs(algorithms: tuple[str, ...]):
    log("\n" + "=" * 70)
    log("算法说明")
    log("=" * 70)
    for key in algorithms:
        doc = ALGORITHM_DOCS.get(key, {"name": key, "pros": "-", "cons": "-"})
        log(f"\n【{key}】{doc.get('name', key)}")
        log(f"  优势：{doc.get('pros', '-')}")
        log(f"  缺点：{doc.get('cons', '-')}")


def print_ranking_summary(raw_df, primary: str = "GA"):
    log("\n" + "=" * 70)
    log(f"指标排名（主算法: {primary}）")
    log("=" * 70)
    means = raw_df.groupby("algorithm").mean(numeric_only=True)
    for col, asc in [
        ("fitness", False), ("composite", False), ("energy_match", False),
        ("on_time_rate", False), ("load_std", True), ("makespan", True),
    ]:
        if col not in means.columns:
            continue
        ranked = means[col].sort_values(ascending=asc)
        rank_str = " > ".join(f"{a}({v:.3f})" for a, v in ranked.items())
        if primary in ranked.index:
            rank = list(ranked.index).index(primary) + 1
            mark = " ← 主算法第1" if rank == 1 else f" (主算法第{rank}名)"
        else:
            mark = ""
        log(f"  {col}: {rank_str}{mark}")


def statistical_tests(raw_df, primary: str = "GA"):
    log("\n" + "=" * 70)
    log(f"统计检验 ({primary} vs 其他, 配对 t 检验——按(dataset, run)对齐)")
    log("=" * 70)
    pri_df = raw_df[raw_df["algorithm"] == primary].sort_values(["dataset", "run"])
    if pri_df.empty:
        log(f"  无 {primary} 结果，跳过")
        return
    others = [a for a in raw_df["algorithm"].unique() if a != primary]
    for algo in others:
        other_df = raw_df[raw_df["algorithm"] == algo].sort_values(["dataset", "run"])
        merged = pri_df[["dataset", "run"]].merge(
            other_df[["dataset", "run"]], on=["dataset", "run"], how="inner"
        )
        n_pairs = len(merged)
        if n_pairs < 2:
            log(f"  {primary} vs {algo}: 有效配对数不足 (n={n_pairs})，跳过")
            continue
        # 按 (dataset, run) 提取对齐后的值
        pri_vals = pri_df.set_index(["dataset", "run"]).loc[
            pd.MultiIndex.from_frame(merged[["dataset", "run"]])
        ]
        other_vals = other_df.set_index(["dataset", "run"]).loc[
            pd.MultiIndex.from_frame(merged[["dataset", "run"]])
        ]
        for metric in ["fitness", "energy_match", "on_time_rate"]:
            p_vals = pri_vals[metric].values.astype(float)
            o_vals = other_vals[metric].values.astype(float)
            _, p = stats.ttest_rel(p_vals, o_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            log(f"  {primary} vs {algo} | {metric:15s} | p={p:.4f} {sig} | n={n_pairs}")


def plot_comparison_dashboard(raw_df, save_path, primary: str = "GA", title_suffix: str = ""):
    order = algo_display_order(raw_df["algorithm"].unique().tolist(), primary)
    means = _algo_means(raw_df, order)
    algos = list(means.index)
    x = np.arange(len(algos))
    colors = [ALGO_COLORS.get(a, "#888888") for a in algos]
    edge_colors = ["#111111" if a == primary else "#666666" for a in algos]
    lw = [2.2 if a == primary else 0.6 for a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ((ax1, ax2), (ax3, ax4)) = axes

    # 1. Composite score
    comp = means["composite"].values
    bars1 = ax1.bar(x, comp, color=colors, edgecolor=edge_colors, linewidth=lw)
    ax1.set_xticks(x); ax1.set_xticklabels(algos, fontsize=11)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Composite", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # 2. Fitness
    fit = means["fitness"].values
    bars2 = ax2.bar(x, fit, color=colors, edgecolor=edge_colors, linewidth=lw)
    ax2.set_xticks(x); ax2.set_xticklabels(algos, fontsize=11)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_title("Fitness", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    # 3. Energy + On-time
    w = 0.35
    ax3.bar(x - w/2, means["energy_match"].values, w, label="Energy match", color="#5b9bd5", edgecolor="#33333355", linewidth=0.5)
    ax3.bar(x + w/2, means["on_time_rate"].values, w, label="On-time", color="#ed7d31", edgecolor="#33333355", linewidth=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(algos)
    ax3.set_ylim(0, 1.08)
    ax3.set_title("Energy & On-time", fontsize=13, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax3.grid(axis="y", alpha=0.2)

    # 4. Load std + Makespan (lower = better, inverted)
    load_vals = means["load_std"].values
    span_vals = means["makespan"].values / 200.0  # normalize to 0-1
    ax4.bar(x - w/2, load_vals, w, label="Load std", color="#9467bd", edgecolor="#33333355", linewidth=0.5)
    ax4.bar(x + w/2, span_vals, w, label="Span/200", color="#bcbd22", edgecolor="#33333355", linewidth=0.5)
    ax4.set_xticks(x); ax4.set_xticklabels(algos)
    ax4.set_title("Load & Span (lower better)", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax4.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, save_path)
    log(f"Dashboard: {save_path}")


def plot_fitness_boxplot(raw_df, save_path, primary: str = "GA"):
    stochastic = [a for a in raw_df["algorithm"].unique() if raw_df[raw_df["algorithm"] == a].shape[0] > 1]
    if primary not in stochastic and primary in raw_df["algorithm"].values:
        stochastic = [primary] + [a for a in stochastic if a != primary]

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_data, labels, colors_used = [], [], []
    for a in algo_display_order(stochastic, primary):
        sub = raw_df[raw_df["algorithm"] == a]
        if len(sub) == 0:
            continue
        plot_data.append(sub["fitness"].values)
        labels.append(a)
        colors_used.append(ALGO_COLORS.get(a, "#888"))

    if not plot_data:
        plt.close(fig); return

    bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                     widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], colors_used):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color("#111111"); median.set_linewidth(1.5)

    ax.set_ylabel("Fitness")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, save_path)
    log(f"Fitness: {save_path}")


def plot_metrics_bars(raw_df, save_path, primary: str = "GA"):
    order = algo_display_order(raw_df["algorithm"].unique().tolist(), primary)
    means = _algo_means(raw_df, order)
    metrics = [
        ("composite", "Composite", False),
        ("energy_match", "Energy match", False),
        ("on_time_rate", "On-time rate", False),
        ("load_std", "Load std", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    algos = list(means.index)
    y = np.arange(len(algos))

    for ax, (col, title, invert) in zip(axes.flat, metrics):
        if col not in means.columns:
            continue
        vals = means[col].values
        idx = np.argsort(vals) if invert else np.argsort(-vals)
        bar_algos = [algos[i] for i in idx]
        bar_vals = vals[idx]
        colors = [ALGO_COLORS.get(a, "#888") for a in bar_algos]
        ax.barh(y, bar_vals, color=colors, edgecolor="#33333377", height=0.55, linewidth=0.5)
        ax.set_yticks(y); ax.set_yticklabels(bar_algos, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    _save_fig(fig, save_path)
    log(f"Metrics: {save_path}")


def generate_all_plots(raw_df, out_dir: Path, tag: str = "", primary: str = "GA"):
    suffix = f"_{tag}" if tag else ""
    plot_comparison_dashboard(raw_df, out_dir / f"dashboard{suffix}.png", primary=primary)
    plot_fitness_boxplot(raw_df, out_dir / f"fitness{suffix}.png", primary=primary)
    plot_metrics_bars(raw_df, out_dir / f"metrics{suffix}.png", primary=primary)


def resolve_dataset_paths(args) -> list[Path]:
    if args.dataset:
        p = Path(args.dataset)
        if not p.is_file():
            raise FileNotFoundError(f"数据集不存在: {p}")
        return [p]

    if args.regen or not list_benchmark_datasets():
        log("benchmark 目录无数据，正在生成默认集...")
        ensure_default_dataset()

    paths = list_benchmark_datasets(scenario=args.scenario)
    if not paths:
        raise FileNotFoundError(
            f"在 {BENCHMARK_DIR} 下未找到任务 JSON。\n请运行: python generate_dataset.py --pack"
        )
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="调度算法对比（benchmark_api 稳定接口 + 多进程加速）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python valid.py
  python valid.py --workers 6 --fast
  python valid.py --primary GA --algorithms GA,HERF,WSPT,RS
  python valid.py --algorithms GA,SA,HERF --solver SA=solvers/sa_template.py
  python valid.py --dataset datasets/benchmark/medium_seed42.json --runs 3
  python valid.py --mode slot --algorithms SLOT-GA,SLOT-RS,EDF,HERF,WSPT --fast
        """,
    )
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--scenario", type=str, default=None,
                   choices=["light", "medium", "heavy", "realistic"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--runs", type=int, default=N_RUNS)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--detail-plots", action="store_true")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--workers", type=int, default=0,
                   help="并行进程数，0=自动(≤8)")
    p.add_argument("--primary", type=str, default="GA",
                   help="主算法（PPT 高亮与检验对照），默认 GA")
    p.add_argument("--algorithms", type=str, default=None,
                   help="逗号分隔，如 GA,SA,HERF,WSPT,EDF,RS")
    p.add_argument("--solver", action="append", default=[], metavar="NAME=PATH",
                   help="自定义求解器，可多次指定，如 --solver SA=solvers/sa_team.py")
    p.add_argument("--sequential", action="store_true",
                   help="单进程运行（调试）")
    p.add_argument("--mode", type=str, default="permutation",
                   choices=["permutation", "slot"],
                   help="permutation=传统排列式解码, slot=时段直接分配解码")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    solvers = parse_solver_specs(args.solver)

    # Slot mode defaults
    if args.mode == "slot":
        if args.primary == "GA":  # default not changed by user
            args.primary = "SLOT-GA"

    primary, algorithms = parse_algorithms(args.primary, args.algorithms, solvers, mode=args.mode)

    if not args.dataset and not args.full and args.limit == 0:
        args.limit = DEFAULT_INSTANCE_LIMIT
        if not args.fast:
            args.fast = True

    paths = resolve_dataset_paths(args)
    workers = 1 if args.sequential else args.workers

    log("=" * 70)
    log("调度算法对比实验")
    log("=" * 70)
    log(f"模式: {'SLOT (直接空位分配)' if args.mode == 'slot' else 'PERMUTATION (排列+解码)'}")
    log(f"主算法: {primary} | 参与: {', '.join(algorithms)}")
    if solvers:
        log(f"自定义求解器: {solvers}")
    log(f"数据: {len(paths)} 个实例 | workers={'1(顺序)' if workers == 1 else (workers or 'auto')}")

    if args.mode == "slot":
        bench_config = get_slot_benchmark_config(fast=args.fast)
    else:
        bench_config = get_benchmark_config(fast=args.fast)
    log(f"GA 参数: POP={bench_config['POP_SIZE']}, GEN={bench_config['GENERATIONS']}, "
        f"DECODE={bench_config.get('DECODE_STRATEGY')}")

    print_algorithm_docs(algorithms)

    runner = BenchmarkRunner(
        primary=primary,
        algorithms=algorithms,
        solvers=solvers,
        workers=workers,
        fast=args.fast,
        config=bench_config,
    )

    t_start = time.time()
    raw_df = runner.run(paths, n_runs=args.runs, show_progress=True)
    log(f"\n全部实验耗时: {time.time() - t_start:.1f}s")

    if "dataset" in raw_df.columns and raw_df["dataset"].nunique() > 1:
        log("\n" + "=" * 70)
        log("跨实例汇总")
        log("=" * 70)
        cross = aggregate_cross_dataset(raw_df)
        log(cross.to_string())
        cross.to_csv(out_dir / "benchmark_cross_dataset.csv", encoding="utf-8-sig")

    order = algo_display_order(algorithms, primary)
    summary_df, _ = aggregate_results(raw_df, algo_order=order)
    log("\n" + "=" * 70)
    log("算法对比汇总")
    log("=" * 70)
    log(summary_df.to_string())

    summary_df.to_csv(out_dir / "benchmark_results.csv", encoding="utf-8-sig")
    raw_df.to_csv(out_dir / "benchmark_raw_runs.csv", index=False, encoding="utf-8-sig")
    try:
        summary_df.to_latex(out_dir / "table_results.tex")
    except Exception:
        pass
    log(f"\n已导出: {out_dir.resolve()}")

    algo_means = raw_df.groupby("algorithm", as_index=False).mean(numeric_only=True)
    print_ranking_summary(algo_means, primary=primary)
    statistical_tests(raw_df, primary=primary)

    if not args.no_plot:
        log("\n生成 PPT 图表...")
        tag = "all" if len(paths) > 1 else paths[0].stem
        generate_all_plots(algo_means, out_dir, tag=tag, primary=primary)
        if args.detail_plots and raw_df["dataset"].nunique() > 1:
            for ds_name, sub in raw_df.groupby("dataset"):
                sub_mean = sub.groupby("algorithm", as_index=False).mean(numeric_only=True)
                generate_all_plots(
                    sub_mean, out_dir / "by_dataset", tag=ds_name, primary=primary
                )

    log("\n实验完成。")
