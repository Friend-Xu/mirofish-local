"""终态回写账目契约（掩盖台账 N-49）：**排空之后的数字才算数**。

F22 现场（`sim_273040f9ce8f`）：运行终态 `stopped`、`error=null`、`failed_count=0`，
图库副本 Episodic 93→112 证明 93 条活动全部写入；但 `/run-status/detail` 的回写账目
仍停在**排空之前**的 `items_sent=10 / expected=93`、`undelivered_items=83`、
`delivery_ratio=0.1075`、`ingestion_drained=false`——把"已交付"读成"丢了 83 条"。

真因：`simulation_runner.py` 的 finalize 在 `stop_updater()`（排空）**之前**拍快照，
之后不再刷新；同步 stop 路径干脆不拍快照。

本文件锁：

1. 排空成功后，账目必须刷新为 delivered（`undelivered=0`、`ratio=1.0`），且
   `writeback_drained()` 为 True；
2. 排空失败时账目仍刷新（显示还剩多少）、运行被判失败、原始异常保留为 `cause`，
   更新器留在管理器里供重试；
3. 顺序：STOPPING 快照（排空前数字）→ 排空 → 刷新（排空后数字），先保存的是 STOPPING；
4. 同步 `stop_simulation` 路径也把排空后的 summary 持久化进终态；
5. 没开回写时不产生任何账目（`None` 是显式第三态，不能被读成"已排空"）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

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


def _fake_updater(stats: dict) -> SimpleNamespace:
    return SimpleNamespace(get_stats=lambda: dict(stats))


def _patch_updater(monkeypatch, *, updater, on_stop):
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _sid: updater),
    )

    def stop_updater(_cls, _sid):
        on_stop()

    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "stop_updater",
        classmethod(stop_updater),
    )


def _register(simulation_id: str) -> None:
    SimulationRunner._graph_memory_enabled[simulation_id] = True


def _cleanup(simulation_id: str) -> None:
    SimulationRunner._graph_memory_enabled.pop(simulation_id, None)
    SimulationRunner._manual_stop_requests.discard(simulation_id)


# ---------------------------------------------------------------- 1. 成功


def test_drain_refreshes_snapshot_to_delivered(sim_dir, monkeypatch):
    simulation_id = "sim-n49-ok"
    stats = {
        "items_sent": 10,
        "expected_items": 93,
        "undelivered_items": 83,
        "buffered_items": 0,
        "queue_size": 78,
        "worker_error": None,
    }
    updater = _fake_updater(stats)

    def drain():
        stats.update(items_sent=93, undelivered_items=0, queue_size=0,
                     delivery_ratio=1.0)

    _patch_updater(monkeypatch, updater=updater, on_stop=drain)
    monkeypatch.setattr(SimulationRunner, "_save_run_state",
                        classmethod(lambda _cls, _state: None))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status",
                        classmethod(lambda *_args, **_kwargs: None))
    _register(simulation_id)
    state = SimulationRunState(simulation_id=simulation_id,
                               runner_status=RunnerStatus.RUNNING)

    try:
        ok, error, cause = SimulationRunner._drain_writeback(
            simulation_id, state, publish_stopping=False
        )
        assert (ok, error, cause) == (True, None, None)
        assert state.graph_memory_summary["items_sent"] == 93
        assert state.graph_memory_summary["undelivered_items"] == 0
        assert state.graph_memory_summary["delivery_ratio"] == 1.0

        # 更新器注销后靠终态快照回答"排空没有"——现在是 True，而不是 F22 的 False
        monkeypatch.setattr(
            runner_module.ZepGraphMemoryManager, "get_updater",
            classmethod(lambda _cls, _sid: None),
        )
        assert SimulationRunner.writeback_drained(simulation_id, state) is True
    finally:
        _cleanup(simulation_id)


# ---------------------------------------------------------------- 2. 失败可见


def test_drain_failure_keeps_undelivered_visible_and_cause(sim_dir, monkeypatch):
    simulation_id = "sim-n49-fail"
    stats = {
        "items_sent": 10,
        "expected_items": 93,
        "undelivered_items": 83,
        "buffered_items": 5,
        "queue_size": 68,
        "worker_error": None,
    }
    updater = _fake_updater(stats)

    def timeout():
        raise TimeoutError("drain timeout after 600s")

    _patch_updater(monkeypatch, updater=updater, on_stop=timeout)
    _register(simulation_id)
    state = SimulationRunState(simulation_id=simulation_id,
                               runner_status=RunnerStatus.RUNNING)

    try:
        ok, error, cause = SimulationRunner._drain_writeback(
            simulation_id, state, publish_stopping=False
        )
        assert ok is False
        assert "Zep图谱写入未完整完成" in error
        assert isinstance(cause, TimeoutError)
        assert state.graph_memory_summary["undelivered_items"] == 83
        # 失败时更新器必须还留在管理器里，才能重试排空
        assert simulation_id in SimulationRunner._graph_memory_enabled
    finally:
        _cleanup(simulation_id)


# ---------------------------------------------------------------- 3. 顺序


def test_stopping_snapshot_is_taken_before_drain(sim_dir, monkeypatch):
    simulation_id = "sim-n49-order"
    stats = {"items_sent": 10, "expected_items": 93,
             "undelivered_items": 83, "buffered_items": 0,
             "queue_size": 78, "worker_error": None}
    updater = _fake_updater(stats)
    events = []

    def drain():
        events.append("drain")
        stats.update(items_sent=93, undelivered_items=0, queue_size=0)

    def save(_cls, state):
        events.append(
            ("save", state.runner_status,
             (state.graph_memory_summary or {}).get("items_sent"))
        )

    _patch_updater(monkeypatch, updater=updater, on_stop=drain)
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(save))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status",
                        classmethod(lambda *_args, **_kwargs: None))
    _register(simulation_id)
    state = SimulationRunState(simulation_id=simulation_id,
                               runner_status=RunnerStatus.RUNNING)

    try:
        SimulationRunner._drain_writeback(simulation_id, state,
                                          publish_stopping=True)
        assert events[0] == ("save", RunnerStatus.STOPPING, 10), (
            "STOPPING 快照必须发生在排空之前（N-26 的可见性）"
        )
        assert events[1] == "drain"
        assert state.graph_memory_summary["items_sent"] == 93
    finally:
        _cleanup(simulation_id)


# ---------------------------------------------------------------- 4. 同步 stop 路径


def test_synchronous_stop_persists_post_drain_summary(sim_dir, monkeypatch):
    simulation_id = "sim-n49-stop"
    stats = {"items_sent": 10, "expected_items": 93,
             "undelivered_items": 83, "buffered_items": 0,
             "queue_size": 78, "worker_error": None,
             "failed_count": 0}
    updater = _fake_updater(stats)

    def drain():
        stats.update(items_sent=93, undelivered_items=0, queue_size=0,
                     delivery_ratio=1.0)

    _patch_updater(monkeypatch, updater=updater, on_stop=drain)
    state = SimulationRunState(simulation_id=simulation_id,
                               runner_status=RunnerStatus.RUNNING)
    saved = []
    monkeypatch.setattr(SimulationRunner, "get_run_state",
                        classmethod(lambda _cls, _sid: state))
    monkeypatch.setattr(SimulationRunner, "_save_run_state",
                        classmethod(lambda _cls, value: saved.append(value)))
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status",
                        classmethod(lambda *_args, **_kwargs: None))
    SimulationRunner._processes.pop(simulation_id, None)
    SimulationRunner._monitor_threads.pop(simulation_id, None)
    _register(simulation_id)

    try:
        result = SimulationRunner.stop_simulation(simulation_id)
        assert result.runner_status == RunnerStatus.STOPPED
        assert result.error is None
        assert result.graph_memory_summary["items_sent"] == 93
        assert result.graph_memory_summary["undelivered_items"] == 0
        assert result.graph_memory_summary["delivery_ratio"] == 1.0
        assert simulation_id not in SimulationRunner._graph_memory_enabled
        assert saved[-1].graph_memory_summary["items_sent"] == 93
    finally:
        _cleanup(simulation_id)


# ---------------------------------------------------------------- 5. 没开回写


def test_disabled_writeback_produces_no_numbers(sim_dir, monkeypatch):
    simulation_id = "sim-n49-off"
    state = SimulationRunState(simulation_id=simulation_id,
                               runner_status=RunnerStatus.RUNNING)

    ok, error, cause = SimulationRunner._drain_writeback(
        simulation_id, state, publish_stopping=False
    )
    assert (ok, error, cause) == (True, None, None)
    assert state.graph_memory_summary is None
