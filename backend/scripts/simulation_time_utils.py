"""模拟时间与"空跑"判定的共享工具（三个 run_*_simulation.py 共用）。

为什么单独成模块：三份脚本各自实现过一遍时段逻辑，而下面这两条规则是
**模拟是否成立**的判据，必须一处定义、一处修改（见 `掩盖台账.md` N-18 / N-20）：

1. 模拟起点不能固定为 00:00——短轮数运行会整场落在 off_peak 时段，
   0 动作却"正常结束"（状态卡 `running`）；
2. 低活跃时段的激活数不能被乘法归零——现实里凌晨也有人发帖，只是少。

另外把"整场 0 动作 = 失败"的判定放在这里：空产物即失败（AGENTS.md §1.1）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_START_HOUR = 9


class EmptySimulationRun(RuntimeError):
    """整场模拟没有产生任何动作（空产物即失败）。"""


def resolve_simulation_start_hour(config: Optional[Dict[str, Any]]) -> int:
    """模拟起点小时：优先取 `time_config.peak_hours` 里最早的一个，否则 9 点。

    这是**语义修正**而不是兜底：起点决定"这 N 轮发生在一天中的什么时候"，
    固定 00:00 会让任何短轮数运行都落在低活跃时段（实测：3 轮 × 30 分钟 =
    00:00–01:30，激活数恒为 0）。
    """

    time_config = (config or {}).get("time_config") or {}
    peak_hours = [
        hour
        for hour in (time_config.get("peak_hours") or [])
        if isinstance(hour, int) and 0 <= hour <= 23
    ]
    if peak_hours:
        return min(peak_hours)
    return DEFAULT_START_HOUR


def simulated_clock(
    round_num: int,
    *,
    start_hour: int,
    minutes_per_round: int,
) -> tuple[int, int]:
    """返回 (第几天, 小时)：从 `start_hour` 起算当前轮对应的模拟时间。"""

    total_minutes = start_hour * 60 + round_num * minutes_per_round
    return total_minutes // (60 * 24) + 1, (total_minutes // 60) % 24


def active_agent_target(
    *,
    base_min: int,
    base_max: int,
    multiplier: float,
    candidate_count: int,
) -> int:
    """本轮目标激活数：低活跃时段不允许被乘法归零（候选非空时至少 1 个）。"""

    import random

    target = int(random.uniform(base_min, base_max) * multiplier)
    if candidate_count and target < 1:
        return 1
    return min(target, candidate_count)


def assert_run_produced_actions(
    total_actions: int,
    *,
    total_rounds: int,
    minutes_per_round: int,
    start_hour: int,
    peak_hours: List[int],
    off_peak_hours: List[int],
) -> None:
    """0 动作 → 抛 `EmptySimulationRun`（写明覆盖时段与活跃时段）。"""

    if total_actions:
        return
    end_minutes = start_hour * 60 + total_rounds * minutes_per_round
    end_hour = (end_minutes // 60) % 24
    raise EmptySimulationRun(
        f"整场模拟 0 个动作：{total_rounds} 轮 × {minutes_per_round} 分钟，"
        f"模拟时间从 {start_hour:02d}:00 起只覆盖到 {end_hour:02d}:{end_minutes % 60:02d}，"
        f"全部落在低活跃时段（off_peak_hours={off_peak_hours}，peak_hours={peak_hours}）。"
        "请提高轮数，或检查 time_config 的 peak_hours——"
        "本轮不会被当作成功（空产物即失败）"
    )
