#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定对比实验接口（供 valid.py / 队友 SA / 外部脚本调用）
================================================================================
原则：仅改变「任务排列」求解器，共用 MultiDayScheduleEngine.decode + fitness。

快速开始:
    from benchmark_api import BenchmarkRunner, DEFAULT_BASELINES

    runner = BenchmarkRunner(
        primary="GA",
        algorithms=["GA", "HERF", "WSPT", "SA"],
        solvers={"SA": "solvers/sa_template.py"},
        workers=4,
        fast=True,
    )
    df = runner.run(paths=["datasets/benchmark/medium_seed42.json"], n_runs=3)

队友 SA:
    1. 复制 solvers/sa_template.py -> solvers/sa_team.py
    2. 实现 solve(tasks, config, seed) -> list[int]  # 任务下标排列
    3. python valid.py --algorithms GA,SA,HERF --solver SA=solvers/sa_team.py
"""

from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG
from core.energy_curve import EnergyCurve
from core.schedule_engine import MultiDayScheduleEngine
from core.ga_solver import ScheduleGA
from core.fitness_match_models import compute_energy_match
from scripts.dataset import load_tasks_from_file

# ---------------------------------------------------------------------------
# 内置算法
# ---------------------------------------------------------------------------
BASELINE_METHODS = frozenset({"RS", "EDF", "HERF", "WSPT"})
BUILTIN_STOCHASTIC = frozenset({"RS", "GA", "SLOT-GA", "SLOT-RS"})
DEFAULT_BASELINES = ("EDF", "HERF", "WSPT", "RS")
DEFAULT_ALGORITHMS = ("GA", "WSPT", "EDF", "HERF", "RS")
SLOT_ALGORITHMS = ("SLOT-GA", "SLOT-RS", "EDF", "HERF", "WSPT")

ALGO_COLORS = {
    "GA": "#2ca02c",
    "SLOT-GA": "#17becf",
    "SLOT-RS": "#bcbd22",
    "SA": "#e377c2",
    "WSPT": "#1f77b4",
    "EDF": "#ff7f0e",
    "HERF": "#d62728",
    "RS": "#7f7f7f",
}

ALGORITHM_DOCS: dict[str, dict[str, str]] = {
    "RS": {
        "name": "Random Search",
        "pros": "无偏下限基线。",
        "cons": "方差大、性能差。",
    },
    "EDF": {
        "name": "Earliest Deadline First",
        "pros": "利于准时率与降低逾期惩罚。",
        "cons": "忽视精力曲线。",
    },
    "HERF": {
        "name": "Highest Energy Requirement First",
        "pros": "高精力任务优先，energy_match 常较高。",
        "cons": "易忽视紧迫截止。",
    },
    "WSPT": {
        "name": "Weighted Shortest Processing Time",
        "pros": "经典加权短作业优先。",
        "cons": "长任务可能饥饿。",
    },
    "GA": {
        "name": "Genetic Algorithm",
        "pros": "种群搜索，兼顾时间-精力适应度。",
        "cons": "计算开销较大。",
    },
    "SA": {
        "name": "Simulated Annealing (custom)",
        "pros": "单解邻域搜索，易接入自定义邻域。",
        "cons": "需调温参；由队友模块实现。",
    },
    "SLOT-GA": {
        "name": "Slot-based Genetic Algorithm",
        "pros": "直接进化开始时间，解码器不做全局优化，算法贡献清晰可见。",
        "cons": "搜索空间更大（连续值），收敛需更多代数。",
    },
    "SLOT-RS": {
        "name": "Slot Random (one-shot)",
        "pros": "纯随机一次分配，不作任何优化，真实下限基线。",
        "cons": "大量冲突逾期，energy_match 也靠运气。",
    },
}

PermutationSolver = Callable[..., list[int]]
_REGISTRY: dict[str, dict[str, Any]] = {}


def register_solver(
    name: str,
    solve_fn: PermutationSolver,
    *,
    stochastic: bool = True,
    color: Optional[str] = None,
    doc: Optional[dict[str, str]] = None,
) -> None:
    """注册自定义求解器（如 SA）。solve_fn(tasks, config, seed) -> permutation"""
    _REGISTRY[name.upper()] = {
        "solve": solve_fn,
        "stochastic": stochastic,
        "builtin": False,
    }
    if color:
        ALGO_COLORS[name.upper()] = color
    if doc:
        ALGORITHM_DOCS[name.upper()] = doc


def load_solver_from_file(path: str | Path) -> PermutationSolver:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"求解器脚本不存在: {path}")
    mod_name = f"_solver_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise AttributeError(f"{path} 须定义 solve(tasks, config, seed) -> list[int]")
    return mod.solve  # type: ignore[return-value]


def _register_builtin(name: str, stochastic: bool) -> None:
    _REGISTRY[name] = {"builtin": True, "stochastic": stochastic, "method": name}


for _n, _stoch in [("EDF", False), ("HERF", False), ("WSPT", False), ("RS", True), ("GA", True), ("SLOT-GA", True), ("SLOT-RS", True)]:
    _register_builtin(_n, _stoch)


# ---------------------------------------------------------------------------
# 排列生成 / 求解
# ---------------------------------------------------------------------------

def generate_baseline_permutations(tasks, method: str, rng=None) -> list[int]:
    n = len(tasks)
    indices = list(range(n))
    rng = rng or np.random.default_rng()
    if method == "RS":
        return rng.permutation(n).tolist()
    if method == "EDF":
        return sorted(indices, key=lambda i: (tasks[i].deadline, -tasks[i].priority))
    if method == "HERF":
        return sorted(indices, key=lambda i: (-tasks[i].energy_req, -tasks[i].priority))
    if method == "WSPT":
        return sorted(indices, key=lambda i: -(tasks[i].priority / tasks[i].duration))
    raise ValueError(f"未知基线: {method}")


def run_ga_solver(tasks, config: dict, seed: int) -> list[int]:
    random.seed(seed)
    np.random.seed(seed)
    curve = EnergyCurve()
    ga = ScheduleGA(tasks, curve, config=config)
    best, _ = ga.run(verbose=False)
    return best


# ---------------------------------------------------------------------------
# Slot-based solver helpers (direct slot assignment, no decoder)
# ---------------------------------------------------------------------------

def _make_slot_scheduler(tasks, config):
    from core.slot_scheduler import SlotScheduler
    curve = EnergyCurve()
    return SlotScheduler(tasks, curve, config)


def _run_slot_ga(tasks, config: dict, seed: int) -> list[int]:
    """Run standalone SlotGA, return slot-index chromosome."""
    random.seed(seed)
    np.random.seed(seed)
    from core.slot_ga_solver import SlotGA
    sched = _make_slot_scheduler(tasks, config)
    ga = SlotGA(sched, config=config)
    best, _ = ga.run(verbose=False)
    return best


def _run_slot_rs(tasks, config: dict, seed: int) -> dict:
    """Pure random slot assignment — one shot, no search."""
    rng = np.random.default_rng(seed)
    sched = _make_slot_scheduler(tasks, config)
    t0 = time.time()
    ind = [rng.integers(0, len(sched.valid_slots[i])) for i in range(len(tasks))]
    metrics = sched.evaluate_slots(ind)
    metrics['runtime'] = time.time() - t0
    return metrics


def _run_slot_baseline(tasks, config: dict, method: str) -> dict:
    """Deterministic baseline (EDF/HERF/WSPT) — each task independently picks a slot."""
    sched = _make_slot_scheduler(tasks, config)
    if method == "EDF":
        slot_indices = sched.edf_slots()
    elif method == "HERF":
        slot_indices = sched.herf_slots()
    elif method == "WSPT":
        slot_indices = sched.wspt_slots()
    else:
        raise ValueError(f"Unknown slot baseline: {method}")
    return sched.evaluate_slots(slot_indices)


def resolve_permutation(
    algorithm: str,
    tasks,
    config: dict,
    seed: int,
    custom_solvers: Optional[dict[str, PermutationSolver]] = None,
) -> list[int]:
    algo = algorithm.upper()
    custom_solvers = custom_solvers or {}

    if algo in BASELINE_METHODS:
        rng = np.random.default_rng(seed) if algo == "RS" else None
        return generate_baseline_permutations(tasks, algo, rng=rng)
    if algo == "GA":
        return run_ga_solver(tasks, config, seed)
    if algo in custom_solvers:
        return custom_solvers[algo](tasks, config, seed)
    if algo in _REGISTRY and not _REGISTRY[algo].get("builtin"):
        return _REGISTRY[algo]["solve"](tasks, config, seed)
    raise ValueError(f"未注册算法: {algo}，可用: {list_algorithms()}")


def list_algorithms() -> list[str]:
    return sorted(set(_REGISTRY.keys()) | BASELINE_METHODS | {"GA"})


def is_stochastic(algorithm: str) -> bool:
    algo = algorithm.upper()
    if algo in _REGISTRY:
        return bool(_REGISTRY[algo].get("stochastic", True))
    return algo in BUILTIN_STOCHASTIC


def algo_display_order(algorithms: list[str], primary: str = "GA") -> list[str]:
    primary = primary.upper()
    rest = sorted(
        [a.upper() for a in algorithms if a.upper() != primary],
        key=lambda a: (a not in DEFAULT_ALGORITHMS, a),
    )
    if primary in [a.upper() for a in algorithms]:
        return [primary] + rest
    return rest


# ---------------------------------------------------------------------------
# 评估（统一指标，供 PPT 图表）
# ---------------------------------------------------------------------------

def evaluate_permutation(permutation: list[int], engine: MultiDayScheduleEngine) -> dict[str, float]:
    t0 = time.time()
    schedule, fitness = engine.evaluate_individual(permutation)
    n = len(schedule)
    energy_matches, on_time_count, daily_load = [], 0, {}

    for item in schedule:
        task = item["task"]
        start, end = item["start"], item["end"]
        daily_load[start.date()] = daily_load.get(start.date(), 0) + 1
        slot_e = engine.energy_curve.get_energy_datetime(start)
        mode = engine._cfg('ENERGY_MATCH_MODE', 'gaussian')
        energy_matches.append(float(compute_energy_match(task.energy_req, slot_e, mode)))
        if end <= task.deadline:
            on_time_count += 1

    if n > 0:
        makespan = (
            max(x["end"] for x in schedule) - min(x["start"] for x in schedule)
        ).total_seconds() / 3600
        energy_match = float(np.mean(energy_matches))
        on_time_rate = on_time_count / n
        load_std = float(np.std(list(daily_load.values())))
        composite = (
            0.45* energy_match
            + 0.45* on_time_rate
            + 0.05 * (1.0 - min(load_std / 2.0, 1.0))
            + 0.05 * (1.0 - min(makespan / 200.0, 1.0))
        )
        return {
            "fitness": float(fitness),
            "energy_match": energy_match,
            "on_time_rate": on_time_rate,
            "load_std": load_std,
            "makespan": makespan,
            "composite": composite,
            "runtime": time.time() - t0,
        }
    return {
        "fitness": float(fitness),
        "energy_match": 0.0,
        "on_time_rate": 0.0,
        "load_std": 0.0,
        "makespan": 0.0,
        "composite": 0.0,
        "runtime": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def get_benchmark_config(fast: bool = False) -> dict:
    """构建用于 benchmark 的 dict 配置（从全局 Config 对象取值）"""
    cfg = {
        "POP_SIZE": CONFIG.POP_SIZE,
        "GENERATIONS": CONFIG.GENERATIONS,
        "MUT_RATE": CONFIG.MUT_RATE,
        "CROSS_RATE": CONFIG.CROSS_RATE,
        "TOURNAMENT_K": CONFIG.TOURNAMENT_K,
        "MULTI_DAY_MODE": CONFIG.MULTI_DAY_MODE,
        "PLANNING_DAYS": CONFIG.PLANNING_DAYS,
        "START_DATE": CONFIG.START_DATE,
        "DAYS_OF_WEEK": list(CONFIG.DAYS_OF_WEEK),
        "DAILY_WORK_START": CONFIG.DAILY_WORK_START,
        "DAILY_WORK_END": CONFIG.DAILY_WORK_END,
        "SLOT_DURATION": CONFIG.SLOT_DURATION,
        "TIME_SEGMENT_MODE": CONFIG.TIME_SEGMENT_MODE,
        "SEGMENTS": dict(CONFIG.SEGMENTS),
        "ENABLE_COURSE_SCHEDULE": CONFIG.ENABLE_COURSE_SCHEDULE,
        "COURSE_SCHEDULE": dict(CONFIG.COURSE_SCHEDULE),
        "DAILY_MAX_TASKS": CONFIG.DAILY_MAX_TASKS,
        "DECODE_STRATEGY": "best_fit",
        "SLOT_SCAN_STEP": 15 / 60,  # 15分钟
        "STOCHASTIC_TOP_K": CONFIG.STOCHASTIC_TOP_K,
        "MUT_SCRAMBLE_RATE": CONFIG.MUT_SCRAMBLE_RATE,
        "WEIGHTS": dict(CONFIG.WEIGHTS),
        "ENERGY_MATCH_MODE": CONFIG.ENERGY_MATCH_MODE,
        "GA_SEED_HEURISTICS": True,
        "GA_LOCAL_SEARCH_STEPS": CONFIG.GA_LOCAL_SEARCH_STEPS,
    }
    if fast:
        cfg["POP_SIZE"] = CONFIG.POP_SIZE
        cfg["GENERATIONS"] = CONFIG.GENERATIONS
    else:
        cfg["POP_SIZE"] = CONFIG.POP_SIZE
        cfg["GENERATIONS"] = CONFIG.GENERATIONS
    return cfg


def get_slot_benchmark_config(fast: bool = False) -> dict:
    """Build benchmark config for slot-based mode (direct slot assignment)."""
    cfg = get_benchmark_config(fast=fast)
    cfg["DECODE_STRATEGY"] = "slot"  # triggers slot dispatch in execute_job
    return cfg


def apply_dataset_meta(config: dict, meta: dict) -> dict:
    cfg = copy.deepcopy(config)
    if meta.get("start_date"):
        cfg["START_DATE"] = meta["start_date"]
    if meta.get("planning_days"):
        cfg["PLANNING_DAYS"] = meta["planning_days"]
    if meta.get("multi_day_mode") is not None:
        cfg["MULTI_DAY_MODE"] = meta["multi_day_mode"]
    return cfg


# ---------------------------------------------------------------------------
# 并行任务
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkJob:
    dataset_path: str
    dataset_stem: str
    scenario: str
    algorithm: str
    run: int
    seed: int
    config: dict
    solver_paths: dict = field(default_factory=dict)


def build_jobs(
    dataset_paths: list[Path],
    algorithms: list[str],
    config: dict,
    n_runs: int,
    task_seed: int = 42,
    solver_paths: Optional[dict[str, str]] = None,
) -> list[BenchmarkJob]:
    solver_paths = solver_paths or {}
    jobs: list[BenchmarkJob] = []
    for path in dataset_paths:
        _, meta = load_tasks_from_file(str(path))
        seed_base = int(meta.get("seed", task_seed))
        cfg = apply_dataset_meta(config, meta)
        stem = path.stem
        scenario = meta.get("scenario", "")

        for algo in algorithms:
            algo_u = algo.upper()
            runs = n_runs  # 所有算法统一 runs 数以支持配对检验
            for r in range(runs):
                if algo_u == "RS":
                    run_seed = seed_base + r
                elif algo_u in BUILTIN_STOCHASTIC or is_stochastic(algo_u):
                    run_seed = seed_base + 1000 + r
                else:
                    run_seed = seed_base
                jobs.append(
                    BenchmarkJob(
                        dataset_path=str(path.resolve()),
                        dataset_stem=stem,
                        scenario=scenario,
                        algorithm=algo_u,
                        run=r if runs > 1 else 0,
                        seed=run_seed,
                        config=cfg,
                        solver_paths=solver_paths,
                    )
                )
    return jobs


def _load_custom_solvers(solver_paths: dict[str, str]) -> dict[str, PermutationSolver]:
    out = {}
    for name, path in solver_paths.items():
        out[name.upper()] = load_solver_from_file(path)
    return out


def _run_random_search(tasks, engine, n_evals: int, rng) -> dict[str, float]:
    """等预算随机搜索：生成 n_evals 个随机排列，返回最优。"""
    best_metrics = None
    best_fitness = -float("inf")
    t0 = time.time()
    n = len(tasks)

    for _ in range(n_evals):
        perm = rng.permutation(n).tolist()
        metrics = evaluate_permutation(perm, engine)
        if metrics["fitness"] > best_fitness:
            best_fitness = metrics["fitness"]
            best_metrics = metrics

    best_metrics["runtime"] = time.time() - t0
    return best_metrics


def execute_job(job: BenchmarkJob) -> dict[str, Any]:
    """单 job 执行（供进程池调用，须在模块顶层可 pickle）。"""
    tasks, meta = load_tasks_from_file(job.dataset_path)
    cfg = apply_dataset_meta(job.config, meta)
    algo = job.algorithm.upper()

    # --- Slot-based algorithms (direct slot assignment) ---
    if algo == "SLOT-GA":
        chrom = _run_slot_ga(tasks, cfg, job.seed)
        sched = _make_slot_scheduler(tasks, cfg)
        metrics = sched.evaluate_slots(chrom)
    elif algo == "SLOT-RS":
        metrics = _run_slot_rs(tasks, cfg, job.seed)
    elif algo in ("EDF", "HERF", "WSPT") and cfg.get("DECODE_STRATEGY") == "slot":
        metrics = _run_slot_baseline(tasks, cfg, algo)
    # --- Permutation-based algorithms ---
    else:
        curve = EnergyCurve()
        engine = MultiDayScheduleEngine(tasks, curve, config=cfg)
        custom = _load_custom_solvers(job.solver_paths)
        if job.algorithm == "RS":
            n_evals = cfg.get("POP_SIZE", 24) * cfg.get("GENERATIONS", 40)
            rng = np.random.default_rng(job.seed)
            metrics = _run_random_search(tasks, engine, n_evals, rng)
        else:
            perm = resolve_permutation(
                job.algorithm, tasks, cfg, job.seed, custom_solvers=custom
            )
            metrics = evaluate_permutation(perm, engine)

    return {
        "algorithm": job.algorithm,
        "run": job.run,
        "dataset": job.dataset_stem,
        "scenario": job.scenario,
        **metrics,
    }


def run_jobs(
    jobs: list[BenchmarkJob],
    workers: int = 1,
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
) -> list[dict]:
    if not jobs:
        return []
    workers = max(1, workers)
    results: list[dict] = []
    total = len(jobs)

    if workers == 1:
        for i, job in enumerate(jobs):
            rec = execute_job(job)
            results.append(rec)
            if on_progress:
                on_progress(i + 1, total, rec)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute_job, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            done += 1
            if on_progress:
                on_progress(done, total, rec)
    return results


# ---------------------------------------------------------------------------
# 高层 Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRunner:
  """
  稳定对比入口。

  primary: 主算法（PPT 高亮、统计检验对照），默认 GA
  algorithms: 参与对比的算法列表
  solvers: 自定义算法 -> .py 路径，如 {"SA": "solvers/sa_team.py"}
  workers: 并行进程数，0 表示自动 min(cpu, jobs)
  """

  primary: str = "GA"
  algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
  solvers: dict[str, str] = field(default_factory=dict)
  workers: int = 0
  fast: bool = True
  config: Optional[dict] = None

  def __post_init__(self):
    self.primary = self.primary.upper()
    self.algorithms = tuple(a.upper() for a in self.algorithms)
    for name, path in self.solvers.items():
      register_solver(name, load_solver_from_file(path), stochastic=True)

  def build_config(self) -> dict:
    return self.config or get_benchmark_config(fast=self.fast)

  def run(
      self,
      paths: list[str | Path],
      n_runs: int = 5,
      show_progress: bool = True,
  ) -> pd.DataFrame:
    paths = [Path(p) for p in paths]
    cfg = self.build_config()
    jobs = build_jobs(paths, list(self.algorithms), cfg, n_runs, solver_paths=self.solvers)
    n_workers = self.workers or min(os.cpu_count() or 4, len(jobs), 8)

    def _progress(done, total, rec):
      if not show_progress:
        return
      msg = f"{rec['dataset'][:12]} | {rec['algorithm']}#{rec['run']} fit={rec['fitness']:.0f}"
      try:
        from tqdm import tqdm
        if not hasattr(_progress, "bar"):
          _progress.bar = tqdm(total=total, desc="Benchmark", file=sys.stdout)
        _progress.bar.update(1)
        _progress.bar.set_postfix_str(msg[:45])
        if done >= total:
          _progress.bar.close()
      except ImportError:
        print(f"\r  [{done}/{total}] {msg}    ", end="", flush=True)
        if done >= total:
          print(flush=True)

    t0 = time.time()
    records = run_jobs(jobs, workers=n_workers, on_progress=_progress if show_progress else None)
    if show_progress:
      print(f"  并行 workers={n_workers}, 耗时 {time.time() - t0:.1f}s, jobs={len(jobs)}")
    return pd.DataFrame(records)


def aggregate_results(raw_df: pd.DataFrame, algo_order: Optional[list[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["fitness", "energy_match", "on_time_rate", "load_std", "makespan", "composite", "runtime"]
    order = algo_order or algo_display_order(raw_df["algorithm"].unique().tolist())
    rows = []
    for algo in order:
        sub = raw_df[raw_df["algorithm"] == algo]
        if sub.empty:
            continue
        row = {"algorithm": algo}
        for m in metrics:
            if m not in sub.columns:
                continue
            std = sub[m].std() if len(sub) > 1 else 0.0
            row[m] = f"{sub[m].mean():.3f}±{std:.3f}"
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("algorithm") if rows else pd.DataFrame()
    return summary, raw_df


def aggregate_cross_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["fitness", "energy_match", "on_time_rate", "load_std", "makespan", "composite"]
    return raw_df.groupby("algorithm")[cols].agg(["mean", "std"]).round(3)
