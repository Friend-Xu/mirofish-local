"""空跑契约测试：整场 0 动作不算"跑完"，必须以失败结束。

背景（L-a 现场实测，`sim_f5e489a20173`：2 agent × 3 轮 × twitter）：
`minutes_per_round=30` 让 3 轮只覆盖模拟时间 00:00–01:30，全部落在
`off_peak_hours=[0..5]`，激活数按 `int(uniform(min,max) * 0.3)` 归零 →
每轮 `active_agents` 为空，循环瞬间走完 → 脚本进入"等待命令模式" →
运行状态永远 `running`、`current_round=0`、动作数 0、**没有任何错误**。

对应规约：`MiroFish/AGENTS.md §1.1`（空产物算成功）；
掩盖台账条目：N-18 / N-19。
"""

import pytest

from scripts.simulation_time_utils import (
    EmptySimulationRun,
    active_agent_target,
    assert_run_produced_actions,
    resolve_simulation_start_hour,
    simulated_clock,
)

OFF_PEAK = [0, 1, 2, 3, 4, 5]
PEAK = [9, 10, 19, 20, 21, 22]


def test_zero_actions_raises_with_covered_hours():
    with pytest.raises(EmptySimulationRun) as error:
        assert_run_produced_actions(
            0, total_rounds=3, minutes_per_round=30, start_hour=0,
            peak_hours=PEAK, off_peak_hours=OFF_PEAK,
        )
    message = str(error.value)
    assert "0 个动作" in message
    assert "00:00" in message and "01:30" in message, "要写清覆盖了哪段时间"
    assert "peak_hours" in message and "off_peak_hours" in message


def test_non_zero_actions_pass():
    assert_run_produced_actions(
        7, total_rounds=3, minutes_per_round=30, start_hour=19,
        peak_hours=PEAK, off_peak_hours=OFF_PEAK,
    )


# ---------------------------------------------------------------- N-20 起点


def test_start_hour_comes_from_first_peak_hour():
    assert resolve_simulation_start_hour({"time_config": {"peak_hours": [19, 20, 21]}}) == 19
    assert resolve_simulation_start_hour({"time_config": {"peak_hours": [9, 20, 21]}}) == 9


def test_start_hour_falls_back_to_nine_without_peak_hours():
    assert resolve_simulation_start_hour({}) == 9
    assert resolve_simulation_start_hour({"time_config": {"peak_hours": []}}) == 9
    assert resolve_simulation_start_hour({"time_config": {"peak_hours": ["19", None]}}) == 9


def test_simulated_clock_starts_at_start_hour_and_wraps_day():
    assert simulated_clock(0, start_hour=19, minutes_per_round=30) == (1, 19)
    assert simulated_clock(3, start_hour=19, minutes_per_round=30) == (1, 20)
    # 19:00 起 + 11 轮 × 30 分钟 = 次日 00:30
    assert simulated_clock(10, start_hour=19, minutes_per_round=30) == (2, 0)


def test_short_run_from_peak_start_stays_in_peak_window():
    """3 轮 × 30 分钟从 19:00 起 → 19:00/19:30/20:00，全部落在活跃时段。"""

    hours = [
        simulated_clock(round_num, start_hour=19, minutes_per_round=30)[1]
        for round_num in range(3)
    ]
    assert all(hour in PEAK for hour in hours), hours


# ---------------------------------------------------------------- N-20 下限


def test_off_peak_target_is_not_multiplied_to_zero():
    """候选非空时至少 1 个：凌晨也有人活动，只是少。"""

    assert active_agent_target(
        base_min=2, base_max=2, multiplier=0.3, candidate_count=2
    ) == 1


def test_no_candidates_still_means_zero():
    assert active_agent_target(
        base_min=2, base_max=2, multiplier=0.3, candidate_count=0
    ) == 0


def test_target_never_exceeds_candidates():
    assert active_agent_target(
        base_min=5, base_max=20, multiplier=1.5, candidate_count=2
    ) == 2


def test_runner_marks_failed_when_child_exits_nonzero():
    """非零退出码必须被 runner 记为失败（否则状态会一直 running）。"""

    import inspect

    from app.services.simulation_runner import SimulationRunner

    source = inspect.getsource(SimulationRunner._monitor_simulation)
    assert "exit_code != 0" in source
    assert "RunnerStatus.FAILED" in source
    assert "PYTHONUNBUFFERED" in inspect.getsource(SimulationRunner.start_simulation)

