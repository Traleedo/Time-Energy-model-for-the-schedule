"""遗传算法求解器"""

import random
import copy
from .schedule_engine import MultiDayScheduleEngine
from .config import CONFIG as DEFAULT_CONFIG


class ScheduleGA(MultiDayScheduleEngine):
    """调度遗传算法求解器 — 支持独立配置注入"""

    def __init__(self, tasks, energy_curve, config=None):
        cfg = config if config is not None else DEFAULT_CONFIG
        super().__init__(tasks, energy_curve, config=cfg)
        if isinstance(cfg, dict):
            self.pop_size = cfg['POP_SIZE']
            self.generations = cfg['GENERATIONS']
        else:
            self.pop_size = cfg.POP_SIZE
            self.generations = cfg.GENERATIONS
        self.best_fitness_history = []
        self.best_solution = None
        self._best_fit_val = float('-inf')

    def create_individual(self):
        ind = list(range(len(self.tasks)))
        random.shuffle(ind)
        return ind

    def _perm_edf(self):
        n = len(self.tasks)
        return sorted(range(n), key=lambda i: (self.tasks[i].deadline, -self.tasks[i].priority))

    def _perm_herf(self):
        n = len(self.tasks)
        return sorted(range(n), key=lambda i: (-self.tasks[i].energy_req, -self.tasks[i].priority))

    def _perm_wspt(self):
        n = len(self.tasks)
        return sorted(range(n), key=lambda i: -(self.tasks[i].priority / self.tasks[i].duration))

    def create_initial_population(self):
        """部分个体用启发式排列初始化"""
        pop = []
        seed_heuristics = self._cfg('GA_SEED_HEURISTICS', True)
        if seed_heuristics:
            for fn in (self._perm_herf, self._perm_edf, self._perm_wspt):
                pop.append(fn())
            herf, edf = self._perm_herf(), self._perm_edf()
            used, hybrid = set(), []
            for g in herf[:max(1, len(herf) // 2)]:
                hybrid.append(g)
                used.add(g)
            for g in edf:
                if g not in used:
                    hybrid.append(g)
                    used.add(g)
            for g in range(len(herf)):
                if g not in used:
                    hybrid.append(g)
            pop.append(hybrid)
        while len(pop) < self.pop_size:
            pop.append(self.create_individual())
        return pop[:self.pop_size]

    def improve_local_search(self, individual, max_steps=None):
        """邻域交换爬山，强化解质量"""
        max_steps = max_steps or self._cfg('GA_LOCAL_SEARCH_STEPS', 80)
        n = len(individual)
        if n < 2:
            return individual
        best = individual[:]
        _, best_fit = self.evaluate_individual(best)
        for _ in range(max_steps):
            i, j = random.sample(range(n), 2)
            trial = best[:]
            trial[i], trial[j] = trial[j], trial[i]
            _, fit = self.evaluate_individual(trial)
            if fit > best_fit:
                best, best_fit = trial, fit
        return best

    def select_tournament(self, pop, fitnesses):
        selected = []
        tournament_k = self._cfg('TOURNAMENT_K', 3)
        for _ in range(len(pop)):
            contestant_indices = random.sample(range(len(pop)), min(tournament_k, len(pop)))
            contestants = [(pop[i], fitnesses[i]) for i in contestant_indices]
            winner = max(contestants, key=lambda x: x[1])[0]
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover_order(self, p1, p2):
        cross_rate = self._cfg('CROSS_RATE', 0.8)
        if random.random() > cross_rate:
            return p1, p2
        size = len(p1)
        start, end = sorted([random.randint(0, size - 1), random.randint(0, size - 1)])
        c1, c2 = [-1] * size, [-1] * size
        c1[start:end + 1], c2[start:end + 1] = p1[start:end + 1], p2[start:end + 1]

        def fill_child(child, parent):
            ptr = 0
            for gene in parent:
                if gene not in child:
                    while child[ptr] != -1:
                        ptr += 1
                    child[ptr] = gene
            return child

        return fill_child(c1, p2), fill_child(c2, p1)

    def mutate_swap(self, ind):
        mut_rate = self._cfg('MUT_RATE', 0.2)
        if random.random() < mut_rate:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def mutate_scramble(self, ind):
        scramble_rate = self._cfg('MUT_SCRAMBLE_RATE', 0.15)
        if random.random() < scramble_rate and len(ind) > 2:
            i, j = sorted(random.sample(range(len(ind)), 2))
            sub = ind[i:j + 1]
            random.shuffle(sub)
            ind[i:j + 1] = sub
        return ind

    def mutate_inversion(self, ind):
        mut_rate = self._cfg('MUT_RATE', 0.2)
        if random.random() < mut_rate * 0.5 and len(ind) > 2:
            i, j = sorted(random.sample(range(len(ind)), 2))
            ind[i:j + 1] = ind[i:j + 1][::-1]
        return ind

    def apply_mutations(self, ind):
        ind = self.mutate_swap(ind)
        ind = self.mutate_scramble(ind)
        ind = self.mutate_inversion(ind)
        return ind

    def run(self, verbose=True):
        """运行遗传算法"""
        population = self.create_initial_population()
        if verbose:
            mode_str = f"Multi-day({self.planning_days} days)" if self._cfg('MULTI_DAY_MODE', True) else "Single-day"
            decode_str = self._cfg('DECODE_STRATEGY', 'best_fit')
            print(f"[START] Evolution | Mode:{mode_str} | Strategy:{decode_str} | Pop:{self.pop_size} | Gens:{self.generations}")

        for gen in range(self.generations):
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            if self.best_solution is None or max_fit > self._best_fit_val:
                self.best_solution = copy.deepcopy(population[best_idx])
                self._best_fit_val = max_fit
            self.best_fitness_history.append(self._best_fit_val)

            selected = self.select_tournament(population, fitnesses)
            children = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i + 1] if i + 1 < len(selected) else selected[i]
                c1, c2 = self.crossover_order(p1, p2)
                children.extend([c1, c2])

            population = [self.apply_mutations(ind) for ind in children[:max(1, self.pop_size - 1)]]
            population.append(copy.deepcopy(self.best_solution))

            if verbose and gen % max(1, self.generations // 5) == 0:
                print(f"  Gen {gen}: Max Fit = {max_fit:.2f} | Best = {self._best_fit_val:.2f}")

        if self.best_solution is not None:
            refined = self.improve_local_search(self.best_solution)
            _, refined_fit = self.evaluate_individual(refined)
            if refined_fit > self._best_fit_val:
                self.best_solution = refined
                self._best_fit_val = refined_fit

        if verbose:
            print("[DONE] Evolution completed")
        return self.best_solution, self.best_fitness_history

    def get_convergence_info(self):
        """获取收敛信息"""
        if not self.best_fitness_history:
            return None
        convergence_gen = 0
        best_fitness = float('-inf')
        for i, fit in enumerate(self.best_fitness_history):
            if fit > best_fitness:
                best_fitness = fit
                convergence_gen = i
        return {
            'final_fitness': self._best_fit_val,
            'convergence_generation': convergence_gen,
            'improvement_rate': len(set(self.best_fitness_history)) / len(self.best_fitness_history) if self.best_fitness_history else 0
        }
