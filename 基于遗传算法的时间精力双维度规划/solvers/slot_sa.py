"""Slot-based Simulated Annealing — direct slot-index assignment."""

import copy
import math
import random
import numpy as np


class SlotSA:
    """Simulated annealing that directly assigns tasks to time slots.

    Same interface as SlotGA: chromosome[i] = index into valid_slots[i].
    """

    def __init__(self, scheduler, config=None):
        self.scheduler = scheduler
        self.tasks = scheduler.tasks
        self._n = len(self.tasks)

        self.t_max = scheduler._c('SA_T_MAX', 2.0)
        self.t_min = scheduler._c('SA_T_MIN', 0.001)
        self.alpha = scheduler._c('SA_ALPHA', 0.92)
        self.steps_per_temp = scheduler._c('SA_STEPS_PER_TEMP', max(20, self._n * 4))
        self.max_outer = scheduler._c('SA_OUTER_LOOPS', 80)
        self.local_steps = scheduler._c('SA_LOCAL_SEARCH_STEPS', 50)

        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []

    def _random_slot(self, task_idx):
        n_slots = len(self.scheduler.valid_slots[task_idx])
        return random.randint(0, n_slots - 1) if n_slots > 0 else 0

    def _fitness(self, individual):
        return self.scheduler.fitness(self.scheduler.assign(individual))

    def evaluate(self, individual):
        return self.scheduler.evaluate_slots(individual)

    def _heuristic_individual(self):
        ind = []
        for i in range(self._n):
            n_slots = len(self.scheduler.valid_slots[i])
            frac = i / max(self._n, 1)
            ind.append(min(int(frac * n_slots), n_slots - 1) if n_slots > 0 else 0)
        return ind

    def run(self, verbose=True):
        # Try seeded starts, pick best
        candidates = []
        for name, method in [("EDF", self.scheduler.edf_slots),
                             ("HERF", self.scheduler.herf_slots),
                             ("WSPT", self.scheduler.wspt_slots)]:
            try:
                candidates.append((name, method()))
            except Exception:
                pass
        candidates.append(("heuristic", self._heuristic_individual()))
        for _ in range(3):
            candidates.append(("random", [self._random_slot(i) for i in range(self._n)]))

        best_start = None
        best_start_fit = float('-inf')
        for name, ind in candidates:
            fit = self._fitness(ind)
            if fit > best_start_fit:
                best_start_fit = fit
                best_start = ind[:]

        current = best_start[:]
        best = best_start[:]
        f_cur = best_start_fit
        f_best = best_start_fit
        self.best_fitness = f_best
        self.best_solution = best[:]

        if verbose:
            print(f"[SlotSA] tasks={self._n} t_max={self.t_max} alpha={self.alpha}")

        t = self.t_max
        evals = 0
        for outer in range(self.max_outer):
            for _ in range(self.steps_per_temp):
                i = random.randrange(self._n)
                n_slots = len(self.scheduler.valid_slots[i])
                if n_slots <= 1:
                    continue
                old_val = current[i]
                # Perturb: small jump (like GA mutation)
                if random.random() < 0.7:
                    delta = random.randint(-5, 5)
                    new_val = max(0, min(n_slots - 1, old_val + delta))
                else:
                    new_val = random.randint(0, n_slots - 1)
                if new_val == old_val:
                    continue
                current[i] = new_val
                f_new = self._fitness(current)
                evals += 1
                delta_f = f_new - f_cur
                if delta_f > 0 or random.random() < math.exp(delta_f / max(t, 1e-9)):
                    f_cur = f_new
                    if f_cur > f_best:
                        f_best = f_cur
                        best = current[:]
                else:
                    current[i] = old_val  # revert

            self.history.append(f_best)
            if f_best > self.best_fitness:
                self.best_fitness = f_best
                self.best_solution = best[:]

            t *= self.alpha
            if t < self.t_min:
                break

            if verbose and outer % max(1, self.max_outer // 5) == 0:
                print(f"  t={t:.4f} | best={f_best:.2f} cur={f_cur:.2f} evals={evals}")

        # Local search polish
        if self.best_solution is not None:
            self._local_search()

        if verbose:
            print(f"[SlotSA DONE] best_fitness={self.best_fitness:.2f} evals={evals}")
        return self.best_solution, self.history

    def _local_search(self):
        best = self.best_solution[:]
        best_fit = self.best_fitness
        improved = True
        iters = 0
        while improved and iters < self.local_steps:
            improved = False
            for i in range(self._n):
                n_slots = len(self.scheduler.valid_slots[i])
                if n_slots <= 1:
                    continue
                orig = best[i]
                for si in range(n_slots):
                    if si == orig:
                        continue
                    best[i] = si
                    fit = self._fitness(best)
                    if fit > best_fit:
                        best_fit = fit
                        orig = si
                        improved = True
                best[i] = orig
            iters += 1
        if best_fit > self.best_fitness:
            self.best_solution = best
            self.best_fitness = best_fit
