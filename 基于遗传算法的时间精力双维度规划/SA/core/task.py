"""任务类"""

from datetime import datetime
import numpy as np
from .config import CONFIG


class Task:
    """任务类 - 精简优化版"""

    # 预定义精力需求映射
    ENERGY_MAP = {
        'review': 0.4,      # 复习/阅读
        'coding': 0.7,      # 编程
        'writing': 0.8,     # 写作/报告
        'meeting': 0.5,     # 会议/讨论
        'admin': 0.3,       # 行政/杂事
        'learning': 0.9,    # 学习新知识
    }

    def __init__(self, name, duration, task_type, deadline, priority=1.0):
        self.name = name
        self.duration = duration
        self.task_type = task_type
        self.priority = priority
        self.energy_req = self.estimate_energy(task_type, duration)

        # 统一处理截止时间
        if isinstance(deadline, str):
            self.deadline = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
        elif isinstance(deadline, datetime):
            self.deadline = deadline
        else:
            # 假设传入的是小时数
            base_date = datetime.strptime(CONFIG.START_DATE, "%Y-%m-%d")
            self.deadline = base_date.replace(hour=int(deadline), minute=int((deadline % 1) * 60))

    def estimate_energy(self, task_type, duration_hours):
        """基于规则的精力估算（可迭代优化）"""
        base = self.ENERGY_MAP.get(task_type, 0.6)

        # 时长越长，单位精力略降（疲劳效应）
        fatigue_factor = 1.0 - 0.05 * min(duration_hours, 4)
        return np.clip(base * fatigue_factor, 0.2, 1.0)

    def get_deadline_hour(self, base_date=None):
        from config import CONFIG
        if base_date is None:
            base_date = datetime.strptime(CONFIG.START_DATE, "%Y-%m-%d")
        if self.deadline.date() == base_date.date():
            return self.deadline.hour + self.deadline.minute / 60.0
        return 99.0

    def __repr__(self):
        from config import CONFIG
        if CONFIG.MULTI_DAY_MODE:
            dl_str = self.deadline.strftime("%m-%d %H:%M")
        else:
            dl_str = self.deadline.strftime("%H:%M")
        return f"T{self.name}(D:{self.duration}h, E:{self.energy_req}, DL:{dl_str})"