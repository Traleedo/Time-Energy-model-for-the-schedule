"""多日调度引擎"""

import numpy as np
import random
from datetime import datetime, timedelta
from .config import CONFIG as DEFAULT_CONFIG
from .fitness_match_models import compute_energy_match


class MultiDayScheduleEngine:
    """多日调度引擎 — 支持独立配置注入（用于 benchmark 等场景）"""

    def __init__(self, tasks, energy_curve, config=None):
        cfg = config if config is not None else DEFAULT_CONFIG
        self.config = cfg
        self.tasks = tasks
        self.energy_curve = energy_curve
        if isinstance(cfg, dict):
            self.start_date = datetime.strptime(cfg['START_DATE'], "%Y-%m-%d")
            self.planning_days = cfg['PLANNING_DAYS'] if cfg.get('MULTI_DAY_MODE', True) else 1
        else:
            self.start_date = datetime.strptime(cfg.START_DATE, "%Y-%m-%d")
            self.planning_days = cfg.PLANNING_DAYS if cfg.MULTI_DAY_MODE else 1
        self.busy_periods = self._build_busy_periods()

    def _cfg(self, key, default=None):
        """统一取值：兼容 dict 和 Config 对象"""
        cfg = self.config
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_busy_periods(self):
        cfg = self.config
        blocks = []
        if isinstance(cfg, dict):
            time_seg = cfg.get('TIME_SEGMENT_MODE', True)
            enable_course = cfg.get('ENABLE_COURSE_SCHEDULE', True)
            multi_day = cfg.get('MULTI_DAY_MODE', True)
        else:
            time_seg = cfg.TIME_SEGMENT_MODE
            enable_course = cfg.ENABLE_COURSE_SCHEDULE
            multi_day = cfg.MULTI_DAY_MODE

        if time_seg:
            for day_idx in range(self.planning_days):
                current_date = self.start_date + timedelta(days=day_idx)
                if isinstance(cfg, dict):
                    ls, le = cfg['SEGMENTS']['LUNCH']
                else:
                    ls, le = cfg.SEGMENTS['LUNCH']
                blocks.append({
                    'start': current_date.replace(hour=int(ls), minute=int((ls % 1) * 60)),
                    'end': current_date.replace(hour=int(le), minute=int((le % 1) * 60)),
                    'type': 'lunch'
                })

        if enable_course:
            course_schedule = cfg['COURSE_SCHEDULE'] if isinstance(cfg, dict) else cfg.COURSE_SCHEDULE
            for cid, info in course_schedule.items():
                target_day = info.get('day')
                if target_day and multi_day:
                    for day_idx in range(self.planning_days):
                        current_date = self.start_date + timedelta(days=day_idx)
                        if current_date.strftime('%A') == target_day:
                            blocks.append({
                                'start': current_date.replace(hour=int((info['start'] - 0.33) // 1),
                                                             minute=int(((info['start'] % 1) * 60) + 40) % 60),
                                'end': current_date.replace(hour=(int(info['end'] + 0.5) // 1),
                                                           minute=int(((info['end'] % 1) * 60 + 30)) % 60),
                                'type': 'course',
                                'id': cid
                            })

        blocks.sort(key=lambda x: x['start'])
        return blocks

    def decode_schedule(self, individual):
        strategy = self._cfg('DECODE_STRATEGY', 'best_fit')
        if strategy == 'first_fit':
            return self._decode_first_fit(individual)
        return self._decode_global_slots(individual, strategy)

    def calculate_fitness(self, individual):
        return self.fitness_from_schedule(self.decode_schedule(individual))

    def evaluate_individual(self, individual):
        """一次解码，返回 (schedule, fitness)"""
        schedule = self.decode_schedule(individual)
        return schedule, self.fitness_from_schedule(schedule)

    def _decode_first_fit(self, individual):
        """First-fit decoder: iterate tasks in permutation order,
        place each at the earliest valid slot across the entire planning window."""
        schedule, occupied, daily_task_count = [], [], {}
        daily_work_start = self._cfg('DAILY_WORK_START', 8.0)
        for idx in individual:
            task = self.tasks[idx]
            req_start = self.start_date.replace(
                hour=int(daily_work_start),
                minute=int((daily_work_start % 1) * 60))
            start = self._find_first_available(req_start, task.duration, occupied, daily_task_count)
            end = start + timedelta(hours=task.duration)
            schedule.append({'task': task, 'start': start, 'end': end})
            occupied.append((start, end))
            daily_task_count[start.date()] = daily_task_count.get(start.date(), 0) + 1
        return schedule

    def _decode_global_slots(self, individual, strategy):
        schedule, occupied, daily_task_count = [], [], {}
        daily_start = self._cfg('DAILY_WORK_START', 8.0)
        for idx in individual:
            task = self.tasks[idx]
            candidates = self._enumerate_candidate_slots(task, occupied, daily_task_count)
            if not candidates:
                start_time = self._find_first_available(
                    self.start_date.replace(hour=int(daily_start), minute=0),
                    task.duration, occupied, daily_task_count)
            else:
                start_time = self._pick_slot(task, candidates, daily_task_count, strategy)
            end_time = start_time + timedelta(hours=task.duration)
            schedule.append({'task': task, 'start': start_time, 'end': end_time})
            occupied.append((start_time, end_time))
            daily_task_count[start_time.date()] = daily_task_count.get(start_time.date(), 0) + 1
        return schedule

    def _slot_conflicts(self, start, duration, occupied):
        end = start + timedelta(hours=duration)
        for occ_start, occ_end in occupied:
            if start < occ_end and end > occ_start:
                return True
        for block in self.busy_periods:
            if start < block['end'] and end > block['start']:
                return True
        return False

    def _is_slot_valid(self, start, duration, occupied, daily_task_count):
        max_end_date = self.start_date + timedelta(days=self.planning_days)
        if start >= max_end_date:
            return False
        end = start + timedelta(hours=duration)
        if end > max_end_date:
            return False
        daily_work_start = self._cfg('DAILY_WORK_START', 8.0)
        daily_work_end = self._cfg('DAILY_WORK_END', 20.0)
        hour_start = start.hour + start.minute / 60.0
        hour_end = end.hour + end.minute / 60.0
        if hour_start < daily_work_start or hour_end > daily_work_end:
            return False
        if start.date() != end.date():
            return False
        daily_max = self._cfg('DAILY_MAX_TASKS', 2)
        if daily_task_count.get(start.date(), 0) >= daily_max:
            return False
        return not self._slot_conflicts(start, duration, occupied)

    def _enumerate_candidate_slots(self, task, occupied, daily_task_count):
        step_h = self._cfg('SLOT_SCAN_STEP', 0.5)
        step = timedelta(hours=step_h)
        candidates = []
        daily_work_start = self._cfg('DAILY_WORK_START', 8.0)
        daily_work_end = self._cfg('DAILY_WORK_END', 20.0)
        for day_idx in range(self.planning_days):
            day = self.start_date + timedelta(days=day_idx)
            t = day.replace(hour=int(daily_work_start), minute=int((daily_work_start % 1) * 60))
            day_end = day.replace(hour=int(daily_work_end), minute=int((daily_work_end % 1) * 60))
            while t + timedelta(hours=task.duration) <= day_end:
                if self._is_slot_valid(t, task.duration, occupied, daily_task_count):
                    candidates.append(t)
                t += step
        return candidates

    def _score_slot(self, task, start, daily_task_count):
        w = self._cfg('WEIGHTS', {})
        slot_energy = self.energy_curve.get_energy_datetime(start)
        mode = self._cfg('ENERGY_MATCH_MODE', 'linear_l1')
        energy_match = float(compute_energy_match(task.energy_req, slot_energy, mode))
        end = start + timedelta(hours=task.duration)
        energy_w = w.get('energy_match', 2.5) * (1.0 + 0.6 * task.energy_req)
        score = energy_match * energy_w
        if end > task.deadline:
            delay_h = (end - task.deadline).total_seconds() / 3600
            score -= delay_h * w.get('deadline_penalty', 8.0) * task.priority
        day_load = daily_task_count.get(start.date(), 0)
        score -= day_load * 0.4
        daily_max = self._cfg('DAILY_MAX_TASKS', 2)
        if day_load >= daily_max:
            score -= (day_load - daily_max + 1) ** 2 * 0.5
        lateness_h = (start - self.start_date).total_seconds() / 3600
        score -= lateness_h * 0.02
        return score

    def _pick_slot(self, task, candidates, daily_task_count, strategy):
        if strategy == 'best_fit':
            scored = [(t, self._score_slot(task, t, daily_task_count)) for t in candidates]
            return max(scored, key=lambda x: x[1])[0]
        scored = [(t, self._score_slot(task, t, daily_task_count)) for t in candidates]
        top_k = min(self._cfg('STOCHASTIC_TOP_K', 8), len(scored))
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        weights = [max(s, 0.01) for _, s in top]
        return random.choices([t for t, _ in top], weights=weights, k=1)[0]

    def _find_first_available(self, req_start, duration, occupied, daily_task_count):
        t = req_start
        max_end_date = self.start_date + timedelta(days=self.planning_days)
        daily_work_end = self._cfg('DAILY_WORK_END', 20.0)
        daily_work_start = self._cfg('DAILY_WORK_START', 8.0)
        hour = t.hour + t.minute / 60.0
        if hour < daily_work_start:
            t = t.replace(hour=int(daily_work_start), minute=int((daily_work_start % 1) * 60))
        while t < max_end_date:
            if self._is_slot_valid(t, duration, occupied, daily_task_count):
                return t
            t += timedelta(minutes=30)
            hour = t.hour + t.minute / 60.0
            if hour >= daily_work_end:
                t = (t + timedelta(days=1)).replace(
                    hour=int(daily_work_start),
                    minute=int((daily_work_start % 1) * 60))
        return t

    def fitness_from_schedule(self, schedule):
        """由已解码日程计算适应度（避免重复 decode）"""
        w = self._cfg('WEIGHTS', {})
        daily_task_count = {}
        used_days = set()
        for item in schedule:
            day_key = item['start'].date()
            daily_task_count[day_key] = daily_task_count.get(day_key, 0) + 1
            used_days.add(day_key)

        energy_score = 0
        penalty = 0
        mode = self._cfg('ENERGY_MATCH_MODE', 'linear_l1')
        for item in schedule:
            task = item['task']
            start, end = item['start'], item['end']
            slot_energy = self.energy_curve.get_energy_datetime(start)
            energy_score += float(compute_energy_match(task.energy_req, slot_energy, mode))
            if end > task.deadline:
                delay_hours = (end - task.deadline).total_seconds() / 3600
                penalty += delay_hours * w.get('deadline_penalty', 8.0) * task.priority
            max_end = self.start_date + timedelta(days=self.planning_days)
            if end > max_end:
                overtime_hours = (end - max_end).total_seconds() / 3600
                penalty += overtime_hours * w.get('overtime_penalty', 5.0)

        overload_penalty = 0
        daily_max = self._cfg('DAILY_MAX_TASKS', 2)
        for day, task_count in daily_task_count.items():
            if task_count > daily_max:
                overload_penalty += (task_count - daily_max) ** 2 * w.get('daily_overload_penalty', 3.0)

        day_spread_bonus = len(used_days) * 2.0
        penalty += overload_penalty
        return energy_score * w.get('energy_match', 2.5) - penalty + day_spread_bonus

    def evaluate_metrics(self, schedule):
        n = len(schedule)
        if n == 0:
            return {}
        matches = []
        deadline_miss = 0
        out_of_range = 0
        max_end = self.start_date + timedelta(days=self.planning_days)
        for item in schedule:
            task = item['task']
            start = item['start']
            end = item['end']
            slot_energy = self.energy_curve.get_energy_datetime(start)
            matches.append(1.0 - abs(task.energy_req - slot_energy))
            if end > task.deadline:
                deadline_miss += 1
            if end > max_end:
                out_of_range += 1
        return {
            'avg_energy_match': np.mean(matches),
            'deadline_miss_rate': deadline_miss / n,
            'out_of_range_rate': out_of_range / n,
            'total_tasks': n,
            'total_days': self.planning_days
        }
