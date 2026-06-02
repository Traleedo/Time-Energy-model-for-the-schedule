"""Slot-based GA — evolves direct slot-index assignments.  Standalone, no inheritance."""

import copy
import random
import numpy as np
from .slot_scheduler import SlotScheduler


class SlotGA:
    """Genetic algorithm that directly assigns tasks to time slots.

    Chromosome: list[int] of length N.
        chromosome[i] = index into scheduler.valid_slots[i].
        Task i is placed at the pre-computed valid slot identified by that index.

    No decoder, no best_fit, no first_fit.  The chromosome IS the schedule.
    Conflicts (overlaps, busy-period hits) are penalised in fitness.
    """

    def __init__(self, scheduler, config=None):
        self.scheduler = scheduler
        self.tasks = scheduler.tasks
        self._n = len(self.tasks)

        self.pop_size = scheduler._c('POP_SIZE', 50)
        self.generations = scheduler._c('GENERATIONS', 100)
        self.mut_rate = scheduler._c('MUT_RATE', 0.2)
        self.cross_rate = scheduler._c('CROSS_RATE', 0.8)
        self.tournament_k = scheduler._c('TOURNAMENT_K', 3)
        self.local_steps = scheduler._c('GA_LOCAL_SEARCH_STEPS', 80)

        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []

    # ------------------------------------------------------------------
    # Chromosome helpers
    # ------------------------------------------------------------------

    def _random_slot(self, task_idx):
        n_slots = len(self.scheduler.valid_slots[task_idx])
        return random.randint(0, n_slots - 1) if n_slots > 0 else 0

    def random_individual(self):
        return [self._random_slot(i) for i in range(self._n)]

    def _heuristic_individual(self):
        """Tasks evenly spread across available slot ranges."""
        ind = []
        for i in range(self._n):
            n_slots = len(self.scheduler.valid_slots[i])
            frac = i / max(self._n, 1)
            ind.append(min(int(frac * n_slots), n_slots - 1) if n_slots > 0 else 0)
        return ind

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def evaluate(self, individual):
        return self.scheduler.evaluate_slots(individual)

    def _fitness(self, individual):
        return self.scheduler.fitness(self.scheduler.assign(individual))

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def crossover(self, p1, p2):
        if random.random() > self.cross_rate:
            return p1[:], p2[:]
        c1, c2 = [], []
        for i in range(self._n):
            if random.random() < 0.5:
                c1.append(p1[i]); c2.append(p2[i])
            else:
                c1.append(p2[i]); c2.append(p1[i])
        return c1, c2

    def mutate(self, ind):
        for i in range(self._n):
            if random.random() < self.mut_rate:
                # 70% chance: small perturbation (adjacent slot)
                # 30% chance: complete reassignment
                if random.random() < 0.7:
                    n_slots = len(self.scheduler.valid_slots[i])
                    if n_slots > 1:
                        delta = random.randint(-3, 3)
                        ind[i] = max(0, min(n_slots - 1, ind[i] + delta))
                else:
                    ind[i] = self._random_slot(i)
        return ind

    def select(self, pop, fits):
        selected = []
        for _ in range(len(pop)):
            idxs = random.sample(range(len(pop)), min(self.tournament_k, len(pop)))
            best = max(idxs, key=lambda j: fits[j])
            selected.append(copy.deepcopy(pop[best]))
        return selected

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, verbose=True):
        # Init — seed with diverse heuristics for a strong starting Pareto front
        seeds = []
        try:
            seeds.append(self.scheduler.edf_slots())
        except Exception:
            pass
        try:
            seeds.append(self.scheduler.herf_slots())
        except Exception:
            pass
        try:
            seeds.append(self.scheduler.wspt_slots())
        except Exception:
            pass
        # Remove duplicates
        uniq = []
        for s in seeds:
            if s not in uniq:
                uniq.append(s)
        # Pad with heuristic-spread and random individuals
        pop = uniq[:] + [self._heuristic_individual()] + [self.random_individual() for _ in range(max(0, self.pop_size - len(uniq) - 1))]
        pop = pop[:self.pop_size]

        if verbose:
            print(f"[SlotGA] tasks={self._n} pop={self.pop_size} gens={self.generations}")

        for gen in range(self.generations):
            fits = [self._fitness(ind) for ind in pop]
            best_idx = max(range(len(fits)), key=lambda i: fits[i])

            if fits[best_idx] > self.best_fitness:
                self.best_fitness = fits[best_idx]
                self.best_solution = copy.deepcopy(pop[best_idx])

            self.history.append(self.best_fitness)

            selected = self.select(pop, fits)
            children = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i + 1] if i + 1 < len(selected) else selected[i]
                c1, c2 = self.crossover(p1, p2)
                children.extend([c1, c2])

            pop = [self.mutate(c) for c in children[:max(1, self.pop_size - 1)]]
            pop.append(copy.deepcopy(self.best_solution))

            if verbose and gen % max(1, self.generations // 5) == 0:
                avg = sum(fits) / len(fits)
                print(f"  gen {gen:3d} | max={fits[best_idx]:.2f} best={self.best_fitness:.2f} avg={avg:.2f}")

        # Local search: small slot perturbations
        if self.best_solution is not None:
            self._local_search()

        if verbose:
            print(f"[SlotGA DONE] best_fitness={self.best_fitness:.2f}")
        return self.best_solution, self.history

    def _local_search(self):
        """Coordinate-ascent: for each task, try ALL valid slots, keep the best.
        Repeats until no improvement or max iterations reached."""
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
                # Try every valid slot for task i
                for si in range(n_slots):
                    if si == orig:
                        continue
                    best[i] = si
                    fit = self._fitness(best)
                    if fit > best_fit:
                        best_fit = fit
                        orig = si
                        improved = True
                best[i] = orig  # restore best for this task
            iters += 1
        if best_fit > self.best_fitness:
            self.best_solution = best
            self.best_fitness = best_fit
