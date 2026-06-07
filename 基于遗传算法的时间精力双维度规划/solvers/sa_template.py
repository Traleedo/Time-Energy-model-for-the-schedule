#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟退火 (SA) 求解器模板 — 供队友替换实现

接口（必须）:
    solve(tasks, config, seed) -> list[int]
        返回任务下标的一个排列，例如 [2,0,1,3,...]

与 GA/基线公平对比:
    - 只输出排列，不要自行 decode
    - 适应度由 MultiDayScheduleEngine.evaluate_individual 统一计算

接入:
    python valid.py --algorithms GA,SA,HERF --solver SA=solvers/sa_template.py
"""

from __future__ import annotations

import math
import random

from 新建文件夹.energy_curve import EnergyCurve
from core.schedule_engine import MultiDayScheduleEngine


def solve(tasks, config, seed: int) -> list[int]:
    """
    默认 SA：交换邻域 + 指数降温（示例实现，可整段替换）
    """
    random.seed(seed)

    n = len(tasks)
    if n <= 1:
        return list(range(n))

    curve = EnergyCurve()
    engine = MultiDayScheduleEngine(tasks, curve, config=config)

    def fitness(perm):
        return engine.calculate_fitness(perm)

    current = list(range(n))
    random.shuffle(current)
    best = current[:]
    f_cur = fitness(current)
    f_best = f_cur

    # 温参与迭代（可在队友版本中调参）
    t_max = 1.0
    t_min = 1e-3
    alpha = 0.92
    steps_per_temp = max(20, n * 4)
    max_outer = int(config.get("SA_OUTER_LOOPS", 80))

    t = t_max
    for _ in range(max_outer):
        for _ in range(steps_per_temp):
            i, j = random.sample(range(n), 2)
            if i == j:
                continue
            trial = current[:]
            trial[i], trial[j] = trial[j], trial[i]
            f_new = fitness(trial)
            delta = f_new - f_cur
            if delta > 0 or random.random() < math.exp(delta / max(t, 1e-9)):
                current = trial
                f_cur = f_new
                if f_cur > f_best:
                    best = current[:]
                    f_best = f_cur
        t *= alpha
        if t < t_min:
            break

    return best
