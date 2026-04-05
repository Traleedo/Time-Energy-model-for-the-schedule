import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from datetime import datetime, timedelta
from utils import generate_energy_curve, validate_config

# ================= 配置中心 =================
CONFIG = {
    # === 基础参数 ===
    'POP_SIZE': 50,
    'GENERATIONS': 1000,
    'MUT_RATE': 0.2,
    'CROSS_RATE': 0.8,
    'TOURNAMENT_K': 3,

    # === 🆕 多日模式控制 ===
    'MULTI_DAY_MODE': True,  # 多日模式开关（False=单日，True=多日）
    'PLANNING_DAYS': 3,  # 规划天数（多日模式下有效）
    'START_DATE': '2026-04-05',  # 开始日期
    'DAYS_OF_WEEK': ['Monday', 'Tuesday', 'Wednesday'],  # 规划的星期几

    # === 每日时间窗口 ===
    'DAILY_WORK_START': 8.0,  # 每日工作开始时间
    'DAILY_WORK_END': 20.0,  # 每日工作结束时间
    'SLOT_DURATION': 0.5,

    # === 分时间段控制 ===
    'TIME_SEGMENT_MODE': True,
    'SEGMENTS': {'MORNING': (8.0, 12.0), 'AFTERNOON': (14.0, 20.0), 'LUNCH': (12.0, 14.0)},

    # === 课程表控制 ===
    'ENABLE_COURSE_SCHEDULE': True,
    'COURSE_SCHEDULE': {
        # 格式: 课程ID -> {'start': 小时, 'end': 小时, 'day': '星期几', 'date': '日期(可选)'}
        'math_mon': {'start': 9.0, 'end': 10.5, 'day': 'Monday', 'date': '2026-04-05'},
        'eng_mon': {'start': 14.0, 'end': 15.5, 'day': 'Monday', 'date': '2026-04-05'},
        'phy_tue': {'start': 10.0, 'end': 11.5, 'day': 'Tuesday', 'date': '2026-04-06'},
        'chem_wed': {'start': 15.0, 'end': 16.5, 'day': 'Wednesday', 'date': '2026-04-07'},
    },

    # === 适应度权重 ===
    'WEIGHTS': {
        'energy_match': 1.0,
        'deadline_penalty': 10.0,
        'overtime_penalty': 5.0,
        'priority_reward': 2.0
    }
}


# ================= 任务模型 =================
class Task:
    def __init__(self, tid, duration, energy_req, deadline, priority=1.0):
        """
        任务类（支持单日/多日模式）

        参数:
            tid: 任务ID
            duration: 耗时(小时)
            energy_req: 精力需求(0-1)
            deadline: 截止时间
                - 单日模式: float (如 17.0 表示当天17:00)
                - 多日模式: datetime 或 str ("2026-04-05 17:00")
            priority: 优先级
        """
        self.id = tid
        self.duration = duration
        self.energy_req = energy_req
        self.priority = priority

        # 解析截止时间
        if isinstance(deadline, str):
            self.deadline = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
        elif isinstance(deadline, datetime):
            self.deadline = deadline
        else:
            # 单日模式：float转换为datetime
            base_date = datetime.strptime(CONFIG['START_DATE'], "%Y-%m-%d")
            self.deadline = base_date.replace(hour=int(deadline), minute=int((deadline % 1) * 60))

    def get_deadline_hour(self, base_date=None):
        """获取截止时间的小时数（用于单日模式兼容）"""
        if base_date is None:
            base_date = datetime.strptime(CONFIG['START_DATE'], "%Y-%m-%d")

        if self.deadline.date() == base_date.date():
            return self.deadline.hour + self.deadline.minute / 60.0
        return 99.0  # 非当天任务，返回大值避免惩罚

    def __repr__(self):
        if CONFIG['MULTI_DAY_MODE']:
            dl_str = self.deadline.strftime("%m-%d %H:%M")
        else:
            dl_str = self.deadline.strftime("%H:%M")
        return f"T{self.id}(D:{self.duration}h, E:{self.energy_req}, DL:{dl_str})"


# ================= 精力曲线模型 =================
class EnergyCurve:
    """精力曲线类：模拟人体昼夜节律（每日重复）"""

    def __init__(self):
        self.hours = np.arange(0, 24, 0.1)
        self.values = self._generate_curve()

    def _generate_curve(self):
        # 双正弦函数：上午高峰(10点) + 下午次高峰(16点)
        val = 0.5 + 0.3 * np.sin((self.hours - 6) * np.pi / 12) + 0.1 * np.sin((self.hours - 14) * np.pi / 8)
        return np.clip(val, 0.1, 1.0)

    def get_energy(self, hour):
        # 获取某时刻的精力值 (插值)
        idx = int((hour % 24) * 10)
        if idx >= len(self.values): idx = len(self.values) - 1
        return self.values[idx]

    def get_energy_datetime(self, dt):
        """根据datetime获取精力值"""
        hour = dt.hour + dt.minute / 60.0
        return self.get_energy(hour)


# ================= 多日调度引擎 =================
class MultiDayScheduleEngine:
    """多日调度引擎：支持跨天任务安排"""

    def __init__(self, tasks, energy_curve):
        self.tasks = tasks
        self.energy_curve = energy_curve
        self.start_date = datetime.strptime(CONFIG['START_DATE'], "%Y-%m-%d")
        self.planning_days = CONFIG['PLANNING_DAYS'] if CONFIG['MULTI_DAY_MODE'] else 1
        self.busy_periods = self._build_busy_periods()

    def _build_busy_periods(self):
        """构建不可用时间段（课程、午休等）"""
        blocks = []

        # 1. 午休时间（每天）
        if CONFIG['TIME_SEGMENT_MODE']:
            for day_idx in range(self.planning_days):
                current_date = self.start_date + timedelta(days=day_idx)
                ls, le = CONFIG['SEGMENTS']['LUNCH']
                blocks.append({
                    'start': current_date.replace(hour=int(ls), minute=int((ls % 1) * 60)),
                    'end': current_date.replace(hour=int(le), minute=int((le % 1) * 60)),
                    'type': 'lunch'
                })

        # 2. 课程表
        if CONFIG['ENABLE_COURSE_SCHEDULE']:
            for cid, info in CONFIG['COURSE_SCHEDULE'].items():
                # 查找对应的日期
                target_day = info.get('day')
                target_date_str = info.get('date')

                if target_date_str:
                    # 指定了具体日期
                    course_date = datetime.strptime(target_date_str, "%Y-%m-%d")
                    if course_date >= self.start_date and course_date < self.start_date + timedelta(
                            days=self.planning_days):
                        blocks.append({
                            'start': course_date.replace(hour=int(info['start']), minute=int((info['start'] % 1) * 60)),
                            'end': course_date.replace(hour=int(info['end']), minute=int((info['end'] % 1) * 60)),
                            'type': 'course',
                            'id': cid
                        })
                elif target_day and CONFIG['MULTI_DAY_MODE']:
                    # 指定星期几，在多日规划中查找
                    for day_idx in range(self.planning_days):
                        current_date = self.start_date + timedelta(days=day_idx)
                        if current_date.strftime('%A') == target_day:
                            blocks.append({
                                'start': current_date.replace(hour=int(info['start']),
                                                              minute=int((info['start'] % 1) * 60)),
                                'end': current_date.replace(hour=int(info['end']), minute=int((info['end'] % 1) * 60)),
                                'type': 'course',
                                'id': cid
                            })

        # 按开始时间排序
        blocks.sort(key=lambda x: x['start'])
        return blocks

    def decode_schedule(self, individual):
        """解码：将任务排列转换为多日时间表"""
        schedule = []
        current_time = self.start_date.replace(hour=int(CONFIG['DAILY_WORK_START']), minute=0)

        for idx in individual:
            task = self.tasks[idx]

            # 寻找可用开始时间
            start_time = self._find_available_start(current_time, task.duration)
            end_time = start_time + timedelta(hours=task.duration)

            schedule.append({
                'task': task,
                'start': start_time,
                'end': end_time
            })

            # 更新当前时间
            current_time = end_time

        return schedule

    def _find_available_start(self, req_start, duration):
        """寻找下一个可用开始时间，自动跳过忙时区间和跨天"""
        t = req_start

        # 检查是否超出规划天数
        max_end_date = self.start_date + timedelta(days=self.planning_days)

        while True:
            # 检查是否超出规划范围
            if t >= max_end_date:
                break

            # 检查每日工作时间
            hour = t.hour + t.minute / 60.0
            if hour >= CONFIG['DAILY_WORK_END']:
                # 跳到第二天早上
                next_day = t + timedelta(days=1)
                t = next_day.replace(hour=int(CONFIG['DAILY_WORK_START']), minute=0)
                continue

            # 检查是否与忙时区间冲突
            conflict = False
            for block in self.busy_periods:
                if t < block['end'] and t + timedelta(hours=duration) > block['start']:
                    # 冲突，跳到忙时区间之后
                    t = block['end']
                    conflict = True
                    break

            if not conflict:
                # 检查是否会在今天下班前完成
                end_hour = (t + timedelta(hours=duration)).hour + (t + timedelta(hours=duration)).minute / 60.0
                if end_hour > CONFIG['DAILY_WORK_END'] and t.date() == (t + timedelta(hours=duration)).date():
                    # 今天做不完，跳到明天
                    next_day = t + timedelta(days=1)
                    t = next_day.replace(hour=int(CONFIG['DAILY_WORK_START']), minute=0)
                    continue
                break

        return t

    def calculate_fitness(self, individual):
        """适应度函数（多日版本）"""
        schedule = self.decode_schedule(individual)
        energy_score = 0
        penalty = 0
        w = CONFIG['WEIGHTS']

        for item in schedule:
            task = item['task']
            start = item['start']
            end = item['end']

            # 1. 精力匹配度
            slot_energy = self.energy_curve.get_energy_datetime(start)
            energy_score += (1.0 - abs(task.energy_req - slot_energy))

            # 2. 截止时间惩罚（多日精确比较）
            if end > task.deadline:
                delay_hours = (end - task.deadline).total_seconds() / 3600
                penalty += delay_hours * w['deadline_penalty'] * task.priority

            # 3. 超出规划范围惩罚
            max_end = self.start_date + timedelta(days=self.planning_days)
            if end > max_end:
                overtime_hours = (end - max_end).total_seconds() / 3600
                penalty += overtime_hours * w['overtime_penalty']

        # 优先级奖励
        priority_bonus = sum(t.priority for t in self.tasks)

        return energy_score * w['energy_match'] - penalty + priority_bonus * w['priority_reward']

    def evaluate_metrics(self, schedule):
        """评价指标计算"""
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

            # 精力匹配度
            slot_energy = self.energy_curve.get_energy_datetime(start)
            matches.append(1.0 - abs(task.energy_req - slot_energy))

            # 逾期
            if end > task.deadline:
                deadline_miss += 1

            # 超出规划范围
            if end > max_end:
                out_of_range += 1

        return {
            'avg_energy_match': np.mean(matches),
            'deadline_miss_rate': deadline_miss / n,
            'out_of_range_rate': out_of_range / n,
            'total_tasks': n,
            'total_days': self.planning_days
        }


# ================= 遗传算法求解器 =================
class ScheduleGA(MultiDayScheduleEngine):
    def __init__(self, tasks, energy_curve):
        super().__init__(tasks, energy_curve)
        self.pop_size = CONFIG['POP_SIZE']
        self.generations = CONFIG['GENERATIONS']
        self.best_fitness_history = []
        self.best_solution = None

    def create_individual(self):
        """创建个体：任务排列"""
        ind = list(range(len(self.tasks)))
        random.shuffle(ind)
        return ind

    def select_tournament(self, pop, fitnesses):
        """锦标赛选择"""
        selected = []
        for _ in range(len(pop)):
            contestants = random.sample(list(zip(pop, fitnesses)), CONFIG['TOURNAMENT_K'])
            winner = max(contestants, key=lambda x: x[1])[0]
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover_order(self, p1, p2):
        """顺序交叉"""
        if random.random() > CONFIG['CROSS_RATE']:
            return p1, p2

        size = len(p1)
        start, end = sorted([random.randint(0, size - 1), random.randint(0, size - 1)])

        c1, c2 = [-1] * size, [-1] * size
        c1[start:end + 1], c2[start:end + 1] = p1[start:end + 1], p2[start:end + 1]

        def fill(child, parent):
            ptr = 0
            for gene in parent:
                if gene not in child:
                    while child[ptr] != -1:
                        ptr += 1
                    child[ptr] = gene
            return child

        return fill(c1, p2), fill(c2, p1)

    def mutate_swap(self, ind):
        """交换变异"""
        if random.random() > CONFIG['MUT_RATE']:
            return ind

        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
        return ind

    def run(self):
        """主进化流程"""
        population = [self.create_individual() for _ in range(self.pop_size)]

        mode_str = f"多日({self.planning_days}天)" if CONFIG['MULTI_DAY_MODE'] else "单日"
        print(f"🚀 开始进化 | 模式:{mode_str} | 种群:{self.pop_size} | 代数:{self.generations}")

        for gen in range(self.generations):
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            self.best_fitness_history.append(max_fit)

            if self.best_solution is None or max_fit > self.calculate_fitness(self.best_solution):
                self.best_solution = copy.deepcopy(population[best_idx])

            selected = self.select_tournament(population, fitnesses)

            children = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]
                c1, c2 = self.crossover_order(p1, p2)
                children.extend([c1, c2])

            population = [self.mutate_swap(ind) for ind in children[:self.pop_size]]

            if gen % 100 == 0:
                print(f"  Gen {gen}: Best Fit = {max_fit:.2f}")

        print("✅ 进化完成")
        return self.best_solution, self.best_fitness_history


# ================= 可视化模块 =================
def plot_results(tasks, best_ind, solver, fitness_history):
    """可视化结果（支持多日）"""
    schedule = solver.decode_schedule(best_ind)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 收敛曲线
    ax1.plot(fitness_history, color='#2ca02c', linewidth=2)
    ax1.set_title('算法收敛曲线 (Convergence Curve)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.grid(True, alpha=0.3)

    # 2. 多日甘特图
    ax2.plot(solver.energy_curve.hours, solver.energy_curve.values, 'k--', alpha=0.3, label='Energy Curve')
    ax2.fill_between(solver.energy_curve.hours, solver.energy_curve.values, alpha=0.1, color='yellow')

    # 绘制忙时区间
    for i, block in enumerate(solver.busy_periods):
        day_offset = (block['start'] - solver.start_date).days
        label = 'Busy' if i == 0 else ""
        ax2.axvspan(block['start'].hour, block['end'].hour,
                    ymin=day_offset / solver.planning_days,
                    ymax=(day_offset + 1) / solver.planning_days if solver.planning_days > 1 else 1,
                    color='gray', alpha=0.2, label=label)

    # 绘制任务
    colors = plt.cm.Reds(np.linspace(0.2, 0.9, len(tasks)))
    for i, item in enumerate(schedule):
        task = item['task']
        start_hour = item['start'].hour + item['start'].minute / 60.0
        duration = task.duration

        # 多日模式：按天分层显示
        if CONFIG['MULTI_DAY_MODE']:
            day_idx = (item['start'] - solver.start_date).days
            y_pos = task.id + day_idx * (len(tasks) + 1)
        else:
            y_pos = task.id

        ax2.barh(y=y_pos, width=duration, left=start_hour,
                 color=colors[i], edgecolor='black', label=f'T{task.id}' if i == 0 else "")
        ax2.text(start_hour + duration / 2, y_pos, f'E:{task.energy_req}',
                 ha='center', va='center', fontsize=8, color='white')

    ax2.set_title(f'优化后日程甘特图 (模式:{"多日" if CONFIG["MULTI_DAY_MODE"] else "单日"})')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Task ID (按天分层)')
    ax2.set_xlim(CONFIG['DAILY_WORK_START'], CONFIG['DAILY_WORK_END'])

    if CONFIG['MULTI_DAY_MODE']:
        ax2.set_ylim(-1, len(tasks) * solver.planning_days + 1)
        # 添加天数分隔线
        for day in range(1, solver.planning_days):
            ax2.axhline(y=day * (len(tasks) + 1) - 0.5, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax2.text(CONFIG['DAILY_WORK_START'], day * (len(tasks) + 1) - 1,
                     f'Day {day}', fontsize=10, fontweight='bold')
    else:
        ax2.set_ylim(-1, len(tasks))

    ax2.legend(loc='upper right')
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('schedule_result.png', dpi=150)
    print("📊 结果已保存为 schedule_result.png")
    plt.show()


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 1. 初始化
    energy_model = EnergyCurve()
    validate_config(CONFIG)

    # 2. 任务数据 - 🆕 多日测试样本
    if CONFIG['MULTI_DAY_MODE']:
        # 多日模式测试样本（跨3天）
        base_date = datetime.strptime(CONFIG['START_DATE'], "%Y-%m-%d")
        raw_tasks = [
            # Day 1 任务
            Task(1, 1.5, 0.9, (base_date + timedelta(days=0)).replace(hour=12, minute=0), 2.0),  # 高脑力，Day1中午截止
            Task(2, 1.0, 0.3, (base_date + timedelta(days=1)).replace(hour=18, minute=0), 1.0),  # 低脑力，Day2傍晚截止
            Task(3, 2.0, 0.8, (base_date + timedelta(days=0)).replace(hour=17, minute=0), 1.5),  # 高脑力，Day1下午截止
            Task(4, 0.5, 0.2, (base_date + timedelta(days=2)).replace(hour=20, minute=0), 0.5),  # 琐事，Day2晚上截止

            # Day 2 任务
            Task(5, 1.5, 0.7, (base_date + timedelta(days=1)).replace(hour=15, minute=0), 1.2),  # 中等脑力，Day2下午截止
            Task(6, 1.0, 0.9, (base_date + timedelta(days=0)).replace(hour=11, minute=0), 2.0),  # 紧急高脑力，Day1上午截止
            Task(7, 2.0, 0.4, (base_date + timedelta(days=2)).replace(hour=19, minute=0), 1.0),  # 长任务低脑力，Day2晚上截止

            # Day 3 任务
            Task(8, 0.5, 0.5, (base_date + timedelta(days=2)).replace(hour=20, minute=0), 0.8),  # 琐事，Day3晚上截止
            Task(9, 1.5, 0.8, (base_date + timedelta(days=2)).replace(hour=16, minute=0), 1.5),  # 高脑力，Day3下午截止
            Task(10, 1.0, 0.6, (base_date + timedelta(days=1)).replace(hour=12, minute=0), 1.0),  # 中等，Day2中午截止
        ]
        print(f"📅 多日模式测试 | 规划天数:{CONFIG['PLANNING_DAYS']} | 任务数:{len(raw_tasks)}")
    else:
        # 单日模式测试样本（原有）
        raw_tasks = [
            Task(1, 1.5, 0.9, 12.0, 2.0),
            Task(2, 1.0, 0.3, 18.0, 1.0),
            Task(3, 2.0, 0.8, 15.0, 1.5),
            Task(4, 0.5, 0.2, 20.0, 0.5),
            Task(5, 1.5, 0.7, 17.0, 1.2),
            Task(6, 1.0, 0.9, 11.0, 2.0),
            Task(7, 2.0, 0.4, 19.0, 1.0),
            Task(8, 0.5, 0.5, 20.0, 0.8),
        ]
        print("📅 单日模式测试")

    # 3. 运行GA
    ga_solver = ScheduleGA(raw_tasks, energy_model)
    best_ind, history = ga_solver.run()

    # 4. 评估指标
    ga_metrics = ga_solver.evaluate_metrics(ga_solver.decode_schedule(best_ind))
    print(f"\n📊 优化结果:")
    print(f"  精力匹配度: {ga_metrics['avg_energy_match']:.3f}")
    print(f"  逾期率: {ga_metrics['deadline_miss_rate']:.2%}")
    if CONFIG['MULTI_DAY_MODE']:
        print(f"  规划天数: {ga_metrics['total_days']}")

    # 5. 可视化
    plot_results(raw_tasks, best_ind, ga_solver, history)

    # 6. 详细日程输出
    print("\n📋 最优日程安排详情:")
    for item in ga_solver.decode_schedule(best_ind):
        t = item['task']
        start_str = item['start'].strftime("%Y-%m-%d %H:%M")
        end_str = item['end'].strftime("%Y-%m-%d %H:%M")
        match = 1 - abs(t.energy_req - energy_model.get_energy_datetime(item['start']))
        print(f"任务 {t.id}: {start_str} - {end_str} | 精力需求:{t.energy_req} | 匹配度:{match:.2f}")