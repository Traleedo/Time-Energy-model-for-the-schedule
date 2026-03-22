import numpy as np
import matplotlib.pyplot as plt
import random
import copy

CONFIG = {
    'POP_SIZE': 500,          # 种群大小
    'GENERATIONS': 1000,      # 迭代代数
    'MUT_RATE': 0.2,         # 变异率
    'CROSS_RATE': 0.8,       # 交叉率
    'TOURNAMENT_K': 3,       # 锦标赛选择大小
    'WORK_START': 8,         # 工作开始时间 (8:00)
    'WORK_END': 20,          # 工作结束时间 (20:00)
    'SLOT_DURATION': 0.5,    # 时间片粒度 (小时)
}

class Task:
    def __init__(self,tid,duration,energy_req,deadline_hour,priority=1.0):
        self.id = tid
        self.duration = duration  # 耗时 (小时)
        self.energy_req = energy_req  # 精力需求 (0.0-1.0, 1.0 为高脑力消耗)
        self.deadline = deadline_hour  # 截止时间 (小时，如 17.0 表示 17:00)
        self.priority = priority  # 优先级
    def __repr__(self):
        return f"T{self.id}(D:{self.duration}h, E:{self.energy_req}, DL:{self.deadline})"


class EnergyCurve:
    """精力曲线类：模拟人体昼夜节律"""

    def __init__(self):
        # 使用双正弦函数模拟：上午高峰 + 下午次高峰
        self.hours = np.arange(0, 24, 0.1)
        self.values = self._generate_curve()

    def _generate_curve(self):
        # 主峰 10 点，次峰 16 点，低谷 14 点
        val = 0.5 + 0.3 * np.sin((self.hours - 6) * np.pi / 12) + 0.1 * np.sin((self.hours - 14) * np.pi / 8)
        return np.clip(val, 0.1, 1.0)

    def get_energy(self, hour):
        # 获取某时刻的精力值 (插值)
        idx = int((hour % 24) * 10)
        if idx >= len(self.values): idx = len(self.values) - 1
        return self.values[idx]


class ScheduleGA:
    def __init__(self, tasks, energy_curve):
        self.tasks = tasks
        self.energy_curve = energy_curve
        self.pop_size = CONFIG['POP_SIZE']
        self.generations = CONFIG['GENERATIONS']
        self.best_fitness_history = []
        self.best_solution = None

    def create_individual(self):
        """创建个体：任务执行顺序的排列"""
        indices = list(range(len(self.tasks)))
        random.shuffle(indices)
        return indices

    def decode_schedule(self, individual):
        """解码：将任务顺序转换为具体时间表"""
        schedule = []
        current_time = CONFIG['WORK_START']

        for idx in individual:
            task = self.tasks[idx]
            start_time = current_time
            end_time = start_time + task.duration

            # 简单约束：如果超过下班时间，推到第二天或标记为未完成 (此处简化为惩罚)
            schedule.append({
                'task': task,
                'start': start_time,
                'end': end_time
            })
            current_time = end_time

        return schedule

    def calculate_fitness(self, individual):
        """适应度函数：精力匹配度 - 惩罚项"""
        schedule = self.decode_schedule(individual)
        energy_score = 0
        penalty = 0

        for item in schedule:
            task = item['task']
            start = item['start']
            end = item['end']

            # 1. 精力匹配度计算 (取任务时间段内的平均精力)
            # 简化：取开始时刻的精力值作为代表
            slot_energy = self.energy_curve.get_energy(start)
            match = 1.0 - abs(task.energy_req - slot_energy)
            energy_score += match

            # 2. 截止时间惩罚
            if end > task.deadline:
                penalty += (end - task.deadline) * 10 * task.priority

            # 3. 下班超时惩罚
            if end > CONFIG['WORK_END']:
                penalty += (end - CONFIG['WORK_END']) * 5

        # 归一化适应度 (越大越好)
        fitness = energy_score - penalty
        return fitness

    def select_tournament(self, pop, fitnesses):
        """锦标赛选择"""
        selected = []
        for _ in range(len(pop)):
            contestants = random.sample(list(zip(pop, fitnesses)), CONFIG['TOURNAMENT_K'])
            winner = max(contestants, key=lambda x: x[1])[0]
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover_order(self, parent1, parent2):
        """顺序交叉 (Order Crossover)"""
        if random.random() > CONFIG['CROSS_RATE']:
            return parent1, parent2

        size = len(parent1)
        start, end = sorted([random.randint(0, size - 1), random.randint(0, size - 1)])

        child1 = [-1] * size
        child2 = [-1] * size

        # 保留父段
        child1[start:end + 1] = parent1[start:end + 1]
        child2[start:end + 1] = parent2[start:end + 1]

        # 填充剩余
        def fill_child(child, parent):
            ptr = 0
            for gene in parent:
                if gene not in child:
                    while child[ptr] != -1:
                        ptr += 1
                    child[ptr] = gene
            return child

        return fill_child(child1, parent2), fill_child(child2, parent1)

    def mutate_swap(self, individual):
        """交换变异"""
        if random.random() > CONFIG['MUT_RATE']:
            return individual

        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run(self):
        """主运行流程"""
        # 初始化种群
        population = [self.create_individual() for _ in range(self.pop_size)]

        print(f"🚀 开始进化，种群大小:{self.pop_size}, 代数:{self.generations}")

        for gen in range(self.generations):
            # 评估
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # 记录最优
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)
            self.best_fitness_history.append(max_fit)

            if self.best_solution is None or max_fit > self.calculate_fitness(self.best_solution):
                self.best_solution = copy.deepcopy(population[best_idx])

            # 选择
            selected = self.select_tournament(population, fitnesses)

            # 交叉
            children = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]
                c1, c2 = self.crossover_order(p1, p2)
                children.extend([c1, c2])

            # 变异
            population = [self.mutate_swap(ind) for ind in children[:self.pop_size]]

            if gen % 20 == 0:
                print(f"Gen {gen}: Best Fitness = {max_fit:.2f}")

        print("✅ 进化完成")
        return self.best_solution, self.best_fitness_history


def plot_results(tasks, best_individual, energy_curve, fitness_history):
    ga = ScheduleGA(tasks, energy_curve)
    schedule = ga.decode_schedule(best_individual)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 1. 收敛曲线
    ax1.plot(fitness_history, color='green', linewidth=2)
    ax1.set_title('算法收敛曲线 (Convergence Curve)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.grid(True, alpha=0.3)

    # 2. 甘特图 + 精力曲线背景
    ax2.plot(energy_curve.hours, energy_curve.values, 'k--', alpha=0.3, label='Energy Curve')
    ax2.fill_between(energy_curve.hours, energy_curve.values, alpha=0.1, color='yellow')

    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))
    for i, item in enumerate(schedule):
        task = item['task']
        # 颜色深度代表任务精力需求
        color = plt.cm.Reds(task.energy_req)
        ax2.barh(y=task.id, width=task.duration, left=item['start'],
                 color=color, edgecolor='black', label=f'T{task.id}' if i == 0 else "")
        ax2.text(item['start'] + task.duration / 2, task.id, f'E:{task.energy_req}',
                 ha='center', va='center', fontsize=8, color='white')

    ax2.set_title('优化后日程甘特图 (背景虚线为精力曲线)')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Task ID')
    ax2.set_xlim(CONFIG['WORK_START'], CONFIG['WORK_END'])
    ax2.set_ylim(-1, len(tasks))
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('schedule_result.png')
    print("📊 结果已保存为 schedule_result.png")
    plt.show()


if __name__ == "__main__":
    # 1. 初始化精力模型
    energy_model = EnergyCurve()

    # 2. 生成任务池 (模拟数据)
    # 格式：ID, 耗时，精力需求 (0-1), 截止时间，优先级
    raw_tasks = [
        Task(1, 1.5, 0.9, 12.0),  # 高脑力，需上午做
        Task(2, 1.0, 0.3, 18.0),  # 低脑力，可下午做
        Task(3, 2.0, 0.8, 15.0),  # 高脑力，需午前完成
        Task(4, 0.5, 0.2, 20.0),  # 琐事
        Task(5, 1.5, 0.7, 17.0),  # 中等脑力
        Task(6, 1.0, 0.9, 11.0),  # 紧急高脑力
        Task(7, 2.0, 0.4, 19.0),  # 长任务，低脑力
        Task(8, 0.5, 0.5, 20.0),  # 琐事
    ]

    # 3. 运行遗传算法
    solver = ScheduleGA(raw_tasks, energy_model)
    best_ind, history = solver.run()

    # 4. 可视化结果
    plot_results(raw_tasks, best_ind, energy_model, history)

    # 5. 打印最优 schedule
    print("\n📅 最优日程安排:")
    final_schedule = solver.decode_schedule(best_ind)
    for item in final_schedule:
        t = item['task']
        print(
            f"任务 {t.id}: {item['start']:.2f}:00 - {item['end']:.2f}:00 (精力匹配:{1 - abs(t.energy_req - energy_model.get_energy(item['start'])):.2f})")