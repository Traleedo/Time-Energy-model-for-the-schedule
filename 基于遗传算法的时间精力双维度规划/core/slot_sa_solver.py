"""Slot-based Simulated Annealing Solver."""
import copy
import math
import random
import numpy as np
from slot_scheduler import SlotScheduler


class SlotSA:
    """Directly optimizes slot indices using Simulated Annealing.

    核心逻辑：
    1. 初始化解：使用高精力优先启发式 (HERF) 或随机解。
    2. 邻域操作：随机扰动某个任务的槽索引（在合法槽列表中前后滑动）。
    3. 接受准则：Metropolis 准则（允许一定概率接受劣解以跳出局部最优）。
    """

    def __init__(self, scheduler, config=None):
        self.scheduler = scheduler
        self.tasks = scheduler.tasks
        self._n = len(self.tasks)

        # SA 核心参数
        self.initial_temp = config.get('SA_INIT_TEMP', 100.0) if isinstance(config, dict) else getattr(config,
                                                                                                       'SA_INIT_TEMP',
                                                                                                       100.0)
        self.cooling_rate = config.get('SA_COOLING_RATE', 0.995) if isinstance(config, dict) else getattr(config,
                                                                                                          'SA_COOLING_RATE',
                                                                                                          0.995)
        self.min_temp = 0.1
        self.max_iter = config.get('SA_MAX_ITER', 2000) if isinstance(config, dict) else getattr(config, 'SA_MAX_ITER',
                                                                                                 2000)

        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []

    def _calculate_fitness(self, individual):
        """计算适应度"""
        return self.scheduler.fitness(self.scheduler.assign(individual))

    def _get_neighbor(self, ind):
        """生成邻域解：随机扰动一个任务的槽位索引"""
        neighbor = ind[:]  # 浅拷贝列表

        # 随机选择一个任务
        task_idx = random.randint(0, self._n - 1)

        n_slots = len(self.scheduler.valid_slots[task_idx])
        if n_slots <= 1:
            return neighbor

        current_slot_idx = neighbor[task_idx]

        # 策略：70% 概率小幅度扰动（前后滑动），30% 概率大范围随机跳转
        if random.random() < 0.7:
            delta = random.randint(-5, 5)
            new_idx = max(0, min(n_slots - 1, current_slot_idx + delta))
        else:
            new_idx = random.randint(0, n_slots - 1)

        neighbor[task_idx] = new_idx
        return neighbor

    def run(self, verbose=True):
        # 1. 初始化当前解 (使用启发式解 HERF 加速收敛)
        try:
            current_solution = self.scheduler.herf_slots()
        except Exception:
            current_solution = [random.randint(0, len(self.scheduler.valid_slots[i]) - 1) for i in range(self._n)]

        current_fitness = self._calculate_fitness(current_solution)

        best_solution = current_solution[:]
        best_fitness = current_fitness

        T = self.initial_temp
        t0 = 0  # 迭代计数

        if verbose:
            print(f"[SlotSA] tasks={self._n} | Init_Fit={current_fitness:.2f} | T_start={T}")

        while T > self.min_temp and t0 < self.max_iter:
            # 2. 产生新解
            new_solution = self._get_neighbor(current_solution)
            new_fitness = self._calculate_fitness(new_solution)

            delta_e = new_fitness - current_fitness

            # 3. Metropolis 接受准则
            accept = False
            if delta_e > 0:
                accept = True
            else:
                prob = math.exp(delta_e / T)
                if random.random() < prob:
                    accept = True

            if accept:
                current_solution = new_solution
                current_fitness = new_fitness

            # 更新全局最优
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution[:]
                self.history.append(best_fitness)

            # 4. 降温
            T *= self.cooling_rate
            t0 += 1

            if verbose and t0 % 500 == 0:
                print(f"  SA Iter {t0} | T={T:.2f} | Best={best_fitness:.2f}")

        if verbose:
            print(f"[SlotSA DONE] best_fitness={best_fitness:.2f}")

        return best_solution, self.history