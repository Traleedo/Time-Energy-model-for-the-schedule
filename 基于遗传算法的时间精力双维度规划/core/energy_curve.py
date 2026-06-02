"""能量曲线类 — 支持多方案模型（见 energy_curve_models.py）"""

import numpy as np

from .energy_curve_models import compute_energy_values, get_model_meta


class EnergyCurve:
    """日内精力曲线；model_id 可选 default / lark / owl / bimodal / gaussian / flat"""

    def __init__(self, model_id: str = "default", step: float = 0.1):
        self.model_id = model_id
        self.meta = get_model_meta(model_id)
        self.hours = np.arange(0, 24, step)
        self.values = compute_energy_values(self.hours, model_id)

    def get_energy(self, hour):
        """获取指定小时的精力值"""
        hour_in_day = hour % 24
        idx = int(hour_in_day * 10)
        if idx >= len(self.values):
            idx = len(self.values) - 1
        return self.values[idx]

    def get_energy_datetime(self, dt):
        """获取datetime对象对应时间的精力值"""
        hour = dt.hour + dt.minute / 60.0
        return self.get_energy(hour)

    def get_average_energy(self, start_dt, end_dt):
        """获取任务时段内的平均精力值（等距采样）。"""
        duration_h = (end_dt - start_dt).total_seconds() / 3600.0
        if duration_h <= 0:
            return self.get_energy_datetime(start_dt)
        n_samples = max(2, int(duration_h / 0.1))
        total = 0.0
        for i in range(n_samples):
            t = start_dt + (end_dt - start_dt) * i / max(n_samples - 1, 1)
            total += self.get_energy_datetime(t)
        return total / n_samples

    def get_energy_batch(self, hours):
        """批量获取精力值，用于性能优化"""
        hours_in_day = np.mod(hours, 24)
        indices = (hours_in_day * 10).astype(int)
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]