import numpy as np
from datetime import datetime


def generate_energy_curve():
    """生成标准精力曲线"""
    hours = np.arange(0, 24, 0.5)
    energy = 0.5 + 0.3 * np.sin((hours - 6) * np.pi / 12) + 0.1 * np.sin((hours - 14) * np.pi / 8)
    energy = np.clip(energy, 0.1, 1.0)
    return dict(zip(hours, energy))


def validate_config(cfg):
    """配置校验"""
    assert cfg['DAILY_WORK_START'] < cfg['DAILY_WORK_END'], "每日工作时间设置错误"

    if cfg['MULTI_DAY_MODE']:
        assert cfg['PLANNING_DAYS'] >= 1, "规划天数必须>=1"
        assert len(cfg['DAYS_OF_WEEK']) == cfg['PLANNING_DAYS'], "星期数与规划天数不匹配"
        # 验证开始日期
        try:
            datetime.strptime(cfg['START_DATE'], "%Y-%m-%d")
        except ValueError:
            raise ValueError("START_DATE格式错误，应为YYYY-MM-DD")

    if cfg['TIME_SEGMENT_MODE']:
        ls, le = cfg['SEGMENTS']['LUNCH']
        assert ls < le, "午休时间设置错误"

    if cfg['ENABLE_COURSE_SCHEDULE']:
        for cid, info in cfg['COURSE_SCHEDULE'].items():
            assert 'start' in info and 'end' in info, f"课程{cid}缺少start/end字段"
            assert info['start'] < info['end'], f"课程{cid}时间逻辑错误"

    print("✅ 配置校验通过")
    print(f"   模式: {'多日' if cfg['MULTI_DAY_MODE'] else '单日'}")
    if cfg['MULTI_DAY_MODE']:
        print(f"   规划期: {cfg['START_DATE']} 起 {cfg['PLANNING_DAYS']}天")