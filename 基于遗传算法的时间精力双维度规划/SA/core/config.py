"""配置管理模块"""

import numpy as np
from datetime import datetime, timedelta


class Config:
    """配置类，集中管理所有配置参数"""

    def __init__(self):
        # === 基础参数 ===
        self.POP_SIZE = 80
        self.GENERATIONS = 250
        self.MUT_RATE = 0.2
        self.CROSS_RATE = 0.8
        self.TOURNAMENT_K = 3

        # === 多日模式控制 ===
        self.MULTI_DAY_MODE = True  # 多日模式开关（False=单日，True=多日）
        self.PLANNING_DAYS = 7  # 规划天数（多日模式下有效）
        self.START_DATE = '2026-04-06'  # 开始日期（比如今天是2026-04-06，周一）
        self.DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # === 每日时间窗口 ===
        self.DAILY_WORK_START = 8.0  # 每日工作开始时间
        self.DAILY_WORK_END = 22.0  # 每日工作结束时间
        self.SLOT_DURATION = 0.5

        # === 分时间段控制 ===
        self.TIME_SEGMENT_MODE = True
        self.SEGMENTS = {'MORNING': (8.0, 13.0), 'AFTERNOON': (14.0, 20.0), 'LUNCH': (13.0, 14.0)}

        # === 课程表控制 ===
        self.ENABLE_COURSE_SCHEDULE = True
        self.COURSE_SCHEDULE = {
            # 周一 (Monday)
            "media_mon_1": {"start": 8.5, "end": 11.0, "day": "Monday"},  # 第1-3节 多媒体技术
            "mathsci_mon_9": {"start": 19.0, "end": 20.5, "day": "Monday"},  # 第9-10节 计算机科学中的数学

            # 周二 (Tuesday)
            "marx_tue_1": {"start": 8.5, "end": 11.0, "day": "Tuesday"},  # 第1-3节 马克思主义基本原理
            "pe_tue_5": {"start": 14.0, "end": 15.5, "day": "Tuesday"},  # 第5-6节 大学体育（4）
            "eng_tue_7": {"start": 15.7, "end": 17.2, "day": "Tuesday"},  # 第7-8节 基础英语（4）
            "cnet_tue_9": {"start": 19.0, "end": 20.5, "day": "Tuesday"},  # 第9-10节 计算机网络

            # 周三 (Wednesday)
            "ci_wed_1": {"start": 8.5, "end": 10.0, "day": "Wednesday"},  # 第1-2节 计算智能
            "mathmod_wed_3": {"start": 10.2, "end": 11.7, "day": "Wednesday"},  # 第3-4节 数学建模方法
            "pe_5": {"start": 14, "end": 15.5, "day": "Wednesday"},  # 第5-6节 体育
            "iot_wed_7": {"start": 15.7, "end": 17.2, "day": "Wednesday"},  # 第7-8节 物联网技术
            "art_wed_9": {"start": 19.0, "end": 20.5, "day": "Wednesday"},  # 第9-10节 创想数字文化艺术空间

            # 周四 (Thursday)
            "cnet_thu_1": {"start": 8.5, "end": 10.0, "day": "Thursday"},  # 第1-2节 计算机网络
            "os_thu_5": {"start": 14.0, "end": 15.5, "day": "Thursday"},  # 第5-6节 操作系统

            # 周五 (Friday)
            "os_fri_1": {"start": 8.5, "end": 10.0, "day": "Friday"},  # 第1-2节 操作系统
            "db_fri_3": {"start": 10.2, "end": 11.7, "day": "Friday"},  # 第3-4节 数据库系统原理
        }
        self.DAILY_MAX_TASKS = 2  # 每天理想的最大自定义任务数，超过就触发惩罚

        # === 解码相关（旧版兼容）===
        self.DECODE_STRATEGY = 'stochastic'
        self.SLOT_SCAN_STEP = 0.5
        self.STOCHASTIC_TOP_K = 8
        self.MUT_SCRAMBLE_RATE = 0.15

        # === 适应度权重 ===
        self.WEIGHTS = {
            'daily_overload_penalty': 7.0,
            'energy_match': 300.0,
            'deadline_penalty': 8.0,
            'overtime_penalty': 5.0,
            'priority_reward': 2.0
        }

        # === 精力匹配与GA增强 ===
        self.ENERGY_MATCH_MODE = 'gaussian'  # 越近奖励越高，远处快速衰减
        self.GA_SEED_HEURISTICS = True
        self.GA_LOCAL_SEARCH_STEPS = 80
        self.SA_OUTER_LOOPS = 80

    def validate(self):
        """配置校验"""
        assert self.DAILY_WORK_START < self.DAILY_WORK_END, "每日工作时间设置错误"

        if self.MULTI_DAY_MODE:
            assert self.PLANNING_DAYS >= 1, "规划天数必须>=1"
            assert len(self.DAYS_OF_WEEK) == self.PLANNING_DAYS, "星期数与规划天数不匹配"
            # 验证开始日期
            try:
                datetime.strptime(self.START_DATE, "%Y-%m-%d")
            except ValueError:
                raise ValueError("START_DATE格式错误，应为YYYY-MM-DD")

        if self.TIME_SEGMENT_MODE:
            ls, le = self.SEGMENTS['LUNCH']
            assert ls < le, "午休时间设置错误"

        if self.ENABLE_COURSE_SCHEDULE:
            for cid, info in self.COURSE_SCHEDULE.items():
                assert 'start' in info and 'end' in info, f"课程{cid}缺少start/end字段"
                assert info['start'] < info['end'], f"课程{cid}时间逻辑错误"

        decode = getattr(self, 'DECODE_STRATEGY', 'best_fit')
        assert decode in ('first_fit', 'best_fit', 'stochastic', 'direct'), "DECODE_STRATEGY 应为 first_fit/best_fit/stochastic/direct"
        assert getattr(self, 'SLOT_SCAN_STEP', 0.5) > 0, "SLOT_SCAN_STEP 必须>0"

    def get(self, key, default=None):
        """获取配置值"""
        return getattr(self, key, default)


# 创建全局配置实例
CONFIG = Config()