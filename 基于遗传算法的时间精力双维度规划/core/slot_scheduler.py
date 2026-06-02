"""Slot-based scheduler: direct task-to-slot assignment. No decoder needed."""

from datetime import datetime, timedelta
import numpy as np
from .fitness_match_models import compute_energy_match


class SlotScheduler:
    """Direct slot assignment engine.

    Pre-computes all valid (day, start_hour) slots for each task, filtered by
    duration, work hours, and busy periods.  Algorithms pick slot INDICES for
    each task — no permutation, no decoder, no first_fit / best_fit.

    The fitness function penalises conflicts (overlaps, busy-period collisions)
    and rewards energy-match + on-time completion.  Random search produces
    genuinely bad schedules because conflicting slots are penalised, not fixed.
    """

    def __init__(self, tasks, energy_curve, config):
        self.tasks = tasks
        self.energy_curve = energy_curve
        self.config = config

        # Unpack config (handle both dict and object)
        self._c = lambda k, d: config.get(k, d) if isinstance(config, dict) else getattr(config, k, d)

        self.start_date = datetime.strptime(self._c('START_DATE', '2026-04-06'), "%Y-%m-%d")
        self.planning_days = self._c('PLANNING_DAYS', 7) if self._c('MULTI_DAY_MODE', True) else 1
        self.work_start = self._c('DAILY_WORK_START', 8.0)
        self.work_end = self._c('DAILY_WORK_END', 20.0)
        self.slot_step = self._c('SLOT_SCAN_STEP', 15 / 60)
        self.daily_max = self._c('DAILY_MAX_TASKS', 2)
        self.weights = self._c('WEIGHTS', {})
        self.match_mode = self._c('ENERGY_MATCH_MODE', 'linear_l1')

        # Build busy periods (lunch + courses)
        self.busy_periods = self._build_busy_periods()

        # Per-task valid slot list: valid_slots[i] = [(day_offset, hour), ...]
        self.valid_slots = [self._slots_for_task(t) for t in tasks]

    # ------------------------------------------------------------------
    # Busy periods (same logic as MultiDayScheduleEngine)
    # ------------------------------------------------------------------

    def _build_busy_periods(self):
        blocks = []
        c = self._c
        if c('TIME_SEGMENT_MODE', True):
            segments = c('SEGMENTS', {})
            lunch = segments.get('LUNCH', (12.0, 14.0))
            for d in range(self.planning_days):
                day = self.start_date + timedelta(days=d)
                blocks.append({
                    'name': '午休',
                    'start': day.replace(hour=int(lunch[0]), minute=int((lunch[0] % 1) * 60)),
                    'end': day.replace(hour=int(lunch[1]), minute=int((lunch[1] % 1) * 60)),
                })

        if c('ENABLE_COURSE_SCHEDULE', True) and c('MULTI_DAY_MODE', True):
            schedule = c('COURSE_SCHEDULE', {})
            for cid, info in schedule.items():
                target_day = info.get('day')
                if not target_day:
                    continue
                for d in range(self.planning_days):
                    day = self.start_date + timedelta(days=d)
                    if day.strftime('%A') == target_day:
                        s = info['start']
                        e = info['end']
                        blocks.append({
                            'name': info.get('name', cid),
                            'start': day.replace(hour=int(s),
                                                 minute=int((s % 1) * 60)),
                            'end': day.replace(hour=int(e),
                                               minute=int((e % 1) * 60)),
                        })
        blocks.sort(key=lambda x: x['start'])
        return blocks

    # ------------------------------------------------------------------
    # Slot generation
    # ------------------------------------------------------------------

    def _slots_for_task(self, task):
        """All valid (day_offset, hour) slots where *task* can start."""
        slots = []
        duration = task.duration
        for day_idx in range(self.planning_days):
            day = self.start_date + timedelta(days=day_idx)
            t = day.replace(hour=int(self.work_start),
                            minute=int((self.work_start % 1) * 60))
            day_end = day.replace(hour=int(self.work_end),
                                  minute=int((self.work_end % 1) * 60))
            while t + timedelta(hours=duration) <= day_end:
                if not self._overlaps_busy(t, duration):
                    hour_in_day = t.hour + t.minute / 60.0
                    slots.append((day_idx, hour_in_day))
                t += timedelta(hours=self.slot_step)
        return slots

    def _overlaps_busy(self, start_dt, duration):
        end_dt = start_dt + timedelta(hours=duration)
        for b in self.busy_periods:
            if start_dt < b['end'] and end_dt > b['start']:
                return True
        return False

    # ------------------------------------------------------------------
    # Assignment & fitness
    # ------------------------------------------------------------------

    def assign(self, slot_indices):
        """Place each task at slot_indices[i] within valid_slots[i].
        Returns schedule list of {task, start, end}."""
        schedule = []
        for i, idx in enumerate(slot_indices):
            task = self.tasks[i]
            day_off, hour = self.valid_slots[i][idx]
            day = self.start_date + timedelta(days=int(day_off))
            start = day.replace(hour=int(hour), minute=int((hour % 1) * 60))
            end = start + timedelta(hours=task.duration)
            schedule.append({'task': task, 'start': start, 'end': end, 'idx': i})
        return schedule

    def fitness(self, schedule):
        """Compute fitness from a schedule.  Penalises conflicts heavily.

        Energy matches are raised to power 4 before summing — this creates
        non-linear reward that strongly favours excellent matches (0.9⁴=0.66)
        over mediocre ones (0.7⁴=0.24), giving GA a steep gradient to climb.
        """
        w = self.weights
        n = len(schedule)

        # --- conflict penalty ---
        conflict_penalty = 0.0
        for i in range(n):
            si, ei = schedule[i]['start'], schedule[i]['end']
            for j in range(i + 1, n):
                sj, ej = schedule[j]['start'], schedule[j]['end']
                if si < ej and ei > sj:
                    overlap_h = ((min(ei, ej) - max(si, sj)).total_seconds() / 3600)
                    conflict_penalty += overlap_h * 10.0
            for b in self.busy_periods:
                if si < b['end'] and ei > b['start']:
                    overlap_h = ((min(ei, b['end']) - max(si, b['start'])).total_seconds() / 3600)
                    conflict_penalty += overlap_h * 15.0

        # --- energy match (power-4 for steep gradient; evaluation uses gaussian) ---
        energy_score = 0.0
        for item in schedule:
            task = item['task']
            slot_e = self.energy_curve.get_average_energy(item['start'], item['end'])
            m = float(compute_energy_match(task.energy_req, slot_e, self.match_mode))
            energy_score += m ** 2  # power-2: smoother gradient for GA

        # --- deadline: continuous gradient + on-time cliff bonus ---
        on_time_bonus = 0.0
        deadline_penalty = 0.0
        for item in schedule:
            task = item['task']
            if item['end'] <= task.deadline:
                on_time_bonus += 12.0 * task.priority
            else:
                delay_h = (item['end'] - task.deadline).total_seconds() / 3600
                deadline_penalty += delay_h * 8.0 * task.priority

        # --- daily overload ---
        daily_count = {}
        for item in schedule:
            d = item['start'].date()
            daily_count[d] = daily_count.get(d, 0) + 1
        overload_penalty = 0.0
        for d, cnt in daily_count.items():
            if cnt > self.daily_max:
                overload_penalty += (cnt - self.daily_max) ** 2 * w.get('daily_overload_penalty', 3.0)

        # --- overtime ---
        max_end = self.start_date + timedelta(days=self.planning_days)
        overtime_penalty = 0.0
        for item in schedule:
            if item['end'] > max_end:
                ot_h = (item['end'] - max_end).total_seconds() / 3600
                overtime_penalty += ot_h * w.get('overtime_penalty', 5.0)

        # --- day spread bonus ---
        used_days = len(set(item['start'].date() for item in schedule))
        day_bonus = used_days * 2.0

        fitness_val = (energy_score * w.get('energy_match', 2.5)
                       + on_time_bonus
                       - deadline_penalty
                       - overload_penalty
                       - overtime_penalty
                       - conflict_penalty
                       + day_bonus)
        return fitness_val

    # ------------------------------------------------------------------
    # Independent per-task heuristics (no sequential dependency)
    # ------------------------------------------------------------------

    def edf_slots(self):
        """Each task independently picks the slot ending closest to its deadline
        without exceeding it. Tasks with earlier deadlines naturally get earlier slots."""
        indices = []
        for i, task in enumerate(self.tasks):
            best_idx, best_score = 0, -float('inf')
            for si, (day_off, hour) in enumerate(self.valid_slots[i]):
                day = self.start_date + timedelta(days=int(day_off))
                end = day.replace(hour=int(hour), minute=int((hour % 1) * 60)) + timedelta(hours=task.duration)
                if end <= task.deadline:
                    score = (end - self.start_date).total_seconds() / 3600  # later is better (closer to deadline)
                else:
                    score = -((end - task.deadline).total_seconds() / 3600) * 100  # penalty for exceeding
                if score > best_score:
                    best_score, best_idx = score, si
            indices.append(best_idx)
        return indices

    def herf_slots(self):
        """Each task independently picks the slot with best energy match."""
        indices = []
        for i, task in enumerate(self.tasks):
            best_idx, best_match = 0, -float('inf')
            for si, (day_off, hour) in enumerate(self.valid_slots[i]):
                day = self.start_date + timedelta(days=int(day_off))
                start = day.replace(hour=int(hour), minute=int((hour % 1) * 60))
                end = start + timedelta(hours=task.duration)
                slot_e = self.energy_curve.get_average_energy(start, end)
                match = 1.0 - abs(task.energy_req - slot_e)
                if match > best_match:
                    best_match, best_idx = match, si
            indices.append(best_idx)
        return indices

    def wspt_slots(self):
        """Each task independently picks a slot based on WSPT score (priority/duration).
        Higher-priority shorter tasks get earlier slots; longer low-priority tasks later."""
        indices = []
        scores = [t.priority / max(t.duration, 0.01) for t in self.tasks]
        smax, smin = max(scores), min(scores)
        span = smax - smin if smax > smin else 1.0
        for i, task in enumerate(self.tasks):
            n = len(self.valid_slots[i])
            frac = (scores[i] - smin) / span  # 0 (lowest WSPT) → 1 (highest WSPT)
            si = int((1.0 - frac) * (n - 1))  # highest WSPT → earliest slot
            indices.append(si)
        return indices

    def random_slots(self, rng=None):
        """Each task independently picks a random valid slot."""
        rng = rng or np.random.default_rng()
        return [rng.integers(0, len(self.valid_slots[i])) for i in range(len(self.tasks))]

    # ------------------------------------------------------------------
    # Metrics (same interface as evaluate_permutation)
    # ------------------------------------------------------------------

    def evaluate_slots(self, slot_indices):
        import time
        t0 = time.time()
        schedule = self.assign(slot_indices)
        fitness_val = self.fitness(schedule)
        n = len(schedule)

        if n == 0:
            return {'fitness': 0.0, 'energy_match': 0.0, 'on_time_rate': 0.0,
                    'load_std': 0.0, 'makespan': 0.0, 'composite': 0.0,
                    'runtime': time.time() - t0}

        energy_matches, on_time, daily_load = [], 0, {}
        for item in schedule:
            task = item['task']
            s, e = item['start'], item['end']
            daily_load[s.date()] = daily_load.get(s.date(), 0) + 1
            slot_e = self.energy_curve.get_average_energy(s, e)
            energy_matches.append(float(compute_energy_match(task.energy_req, slot_e, self.match_mode)))
            if e <= task.deadline:
                on_time += 1

        makespan = (max(x['end'] for x in schedule) - min(x['start'] for x in schedule)).total_seconds() / 3600
        em = float(np.mean(energy_matches))
        otr = on_time / n
        lstd = float(np.std(list(daily_load.values()))) if len(daily_load) > 1 else 0.0
        composite = (0.40 * em + 0.40 * otr
                     + 0.10 * (1.0 - min(lstd / 4.0, 1.0))
                     + 0.10 * (1.0 - min(makespan / 200.0, 1.0)))
        return {'fitness': float(fitness_val), 'energy_match': em, 'on_time_rate': otr,
                'load_std': lstd, 'makespan': makespan, 'composite': composite,
                'runtime': time.time() - t0}
