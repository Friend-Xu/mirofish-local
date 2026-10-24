"""运行完成度契约（N-26）：跑完 ≠ 卡住，终态后账目仍在，报告不必先 stop。

背景（掩盖台账 N-26）：产品设计里模拟跑完后环境继续存活等采访命令，`runner_status`
仍是 `running`——于是"跑完等命令"与"卡住"在 API 上完全一样（实测卡 611 秒）；
而为了拿到终态必须 `stop`，stop 之后报告的 `interview_agents` 又没人应答。
本文件锁三件事：

1. `simulation_finished` / `ingestion_drained` 语义（含 `None` = 没开回写，不许当"已完成"）；
2. 回写账目在更新器注销后仍可查（终态快照持久化到 run_state.json）；
3. 报告门槛只有一处判据 `_report_readiness`，且"跑完+回写无待处理"即可开始。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api.report import _report_readiness  # noqa: E402
from app.services import simulation_runner as runner_module  # noqa: E402
from app.services.simulation_runner import (  # noqa: E402
    RunnerStatus,
    SimulationRunner,
    SimulationRunState,
)


@pytest.fixture
def sim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    return tmp_path


def _add_action_log(sim_dir: Path, simulation_id: str, platform: str) -> None:
    target = sim_dir / simulation_id / platform
    target.mkdir(parents=True, exist_ok=True)
    (target / "actions.jsonl").write_text("{}\n", encoding="utf-8")


# ------------------------------------------------------ 跑完没有（simulation_finished）


def test_simulation_finished_requires_every_enabled_platform(sim_dir):
    state = SimulationRunState(simulation_id="sim_tw")
    _add_action_log(sim_dir, "sim_tw", "twitter")
    assert SimulationRunner.has_simulation_finished(state) is False

    state.twitter_completed = True
    assert SimulationRunner.has_simulation_finished(state) is True

    # 双平台：只完成 twitter 不算跑完
    _add_action_log(sim_dir, "sim_both", "twitter")
    _add_action_log(sim_dir, "sim_both", "reddit")
    both = SimulationRunState(simulation_id="sim_both")
    both.twitter_completed = True
    assert SimulationRunner.has_simulation_finished(both) is False
    both.reddit_completed = True
    assert SimulationRunner.has_simulation_finished(both) is True


def test_simulation_finished_is_false_without_action_logs(sim_dir):
    """没有 actions.jsonl = 没有平台被启用 → 不能说"跑完了"。"""

    assert SimulationRunner.has_simulation_finished(SimulationRunState(simulation_id="sim_none")) is False


# ------------------------------------------------------ 回写排空（ingestion_drained）


def test_writeback_drained_is_none_when_graph_memory_disabled(sim_dir, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "_graph_memory_enabled", {}, raising=False)
    state = SimulationRunState(simulation_id="sim_nomem")
    assert SimulationRunner.writeback_drained("sim_nomem", state) is None


def test_writeback_drained_reads_live_updater(sim_dir, monkeypatch):
    simulation_id = "sim_live"
    monkeypatch.setitem(SimulationRunner._graph_memory_enabled, simulation_id, True)
    stats = {
        "undelivered_items": 0,
        "buffered_items": 0,
        "queue_size": 0,
        "worker_error": None,
        "pending_episode_count": 3,   # 已送出等确认，不算"未排空"
    }
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda cls, sid: SimpleNamespace(get_stats=lambda: dict(stats))),
    )
    assert SimulationRunner.writeback_drained(simulation_id) is True

    stats["buffered_items"] = 2
    assert SimulationRunner.writeback_drained(simulation_id) is False

    stats["buffered_items"] = 0
    stats["worker_error"] = "boom"
    assert SimulationRunner.writeback_drained(simulation_id) is False


def test_writeback_drained_falls_back_to_final_snapshot(sim_dir, monkeypatch):
    """更新器注销后仍要能回答"排空没有"——用终态快照。"""

    simulation_id = "sim_snapshot"
    monkeypatch.setitem(SimulationRunner._graph_memory_enabled, simulation_id, False)
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda cls, sid: None),
    )
    state = SimulationRunState(
        simulation_id=simulation_id,
        graph_memory_enabled=True,
        graph_memory_summary={"undelivered_items": 1, "buffered_items": 0, "queue_size": 0},
    )
    assert SimulationRunner.writeback_drained(simulation_id, state) is False

    state.graph_memory_summary["undelivered_items"] = 0
    assert SimulationRunner.writeback_drained(simulation_id, state) is True


# ------------------------------------------------------ 报告门槛（唯一判据）


def test_readiness_allows_running_but_finished_and_drained():
    """N-26 的核心：跑完 + 回写无待处理 → 允许，环境还活着（采访可用）。"""

    allowed, code, _ = _report_readiness(
        RunnerStatus.RUNNING,
        simulation_finished=True,
        ingestion_drained=True,
        pending_items=0,
    )
    assert allowed is True and code == "ready"


def test_readiness_allows_terminal_statuses():
    for status in (RunnerStatus.COMPLETED, RunnerStatus.STOPPED):
        allowed, code, _ = _report_readiness(
            status,
            simulation_finished=True,
            ingestion_drained=True,
                pending_items=0,
        )
        assert allowed is True and code == "ready"


def test_readiness_rejects_running_not_finished():
    allowed, code, message = _report_readiness(
        RunnerStatus.RUNNING,
        simulation_finished=False,
        ingestion_drained=True,
        pending_items=0,
    )
    assert allowed is False and code == "simulation_not_finished"
    assert "simulation_end" in message


def test_readiness_rejects_when_ingestion_has_backlog():
    allowed, code, message = _report_readiness(
        RunnerStatus.RUNNING,
        simulation_finished=True,
        ingestion_drained=False,
        pending_items=7,
    )
    assert allowed is False and code == "ingestion_pending"
    assert "7" in message


def test_readiness_rejects_failed_run():
    allowed, code, _ = _report_readiness(
        RunnerStatus.FAILED,
        simulation_finished=True,
        ingestion_drained=True,
        pending_items=0,
    )
    assert allowed is False and code == "simulation_failed"


def test_readiness_allows_running_without_graph_memory_when_finished():
    """没开回写时 `ingestion_drained` 是 None：不许当成"未排空"，也不许当成"已排空"以外的意思。"""

    allowed, code, _ = _report_readiness(
        RunnerStatus.RUNNING,
        simulation_finished=True,
        ingestion_drained=None,
        pending_items=0,
    )
    assert allowed is True and code == "ready"


# ------------------------------------------------------ 终态账目仍可查


def test_run_state_round_trips_graph_memory_summary(sim_dir):
    simulation_id = "sim_roundtrip"
    state = SimulationRunState(
        simulation_id=simulation_id,
        runner_status=RunnerStatus.STOPPED,
        graph_memory_enabled=True,
        graph_memory_summary={"items_sent": 25, "expected_items": 25, "delivery_ratio": 1.0},
    )
    SimulationRunner._save_run_state(state)

    on_disk = json.loads(
        (sim_dir / simulation_id / "run_state.json").read_text(encoding="utf-8")
    )
    assert on_disk["graph_memory_enabled"] is True
    assert on_disk["graph_memory_summary"]["items_sent"] == 25

    reloaded = SimulationRunner._load_run_state(simulation_id)
    assert reloaded.graph_memory_enabled is True
    assert reloaded.graph_memory_summary["delivery_ratio"] == 1.0
