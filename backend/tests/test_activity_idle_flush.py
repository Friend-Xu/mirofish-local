"""L0 离线单测：回写缓冲的"空闲刷新"。

背景（真机事故）：把批大小从 5 提到 20 后，某平台只产生 11 条活动，
缓冲区永远到不了 20 条的触发阈值，updater 永不发送 →
SimulationRunner 一直等"图谱写入完成"，模拟状态永久 running。

本文件用一个假 SDK 客户端 + 极小 IDLE_FLUSH_SECONDS 验证：
不足一批的活动也能在静默后被发送。
"""

import threading
import time
from types import SimpleNamespace

from app.services import zep_graph_memory_updater as updater_module
from app.services.zep_graph_memory_updater import AgentActivity, ZepGraphMemoryUpdater
from graph_store_fakes import cloud_store


def _activity(index: int) -> AgentActivity:
    return AgentActivity(
        platform="twitter",
        agent_id=index,
        agent_name=f"Agent {index}",
        action_type="CREATE_POST",
        action_args={"content": f"post {index}"},
        round_num=1,
        timestamp="2026-10-10T07:34:44+08:00",
    )


def _updater(monkeypatch, writes, *, batch_size: int, idle_seconds: float):
    client = cloud_store(
        SimpleNamespace(
            graph=SimpleNamespace(
                add=lambda **kwargs: writes.append(kwargs) or SimpleNamespace(uuid_="ep-1"),
                episode=SimpleNamespace(get=lambda **_kwargs: SimpleNamespace(processed=True)),
            )
        )
    )
    monkeypatch.setattr(updater_module, "get_graph_store", lambda _key=None: client)

    updater = ZepGraphMemoryUpdater("graph-1", api_key="test-key", simulation_id="sim-1")
    updater.BATCH_SIZE = batch_size
    updater.IDLE_FLUSH_SECONDS = idle_seconds
    updater.SEND_INTERVAL = 0
    return updater


def test_idle_flush_sends_a_partial_buffer(monkeypatch):
    """活动数 < BATCH_SIZE 时，静默后仍应发送。"""

    writes = []
    updater = _updater(monkeypatch, writes, batch_size=20, idle_seconds=0.3)

    updater.start()
    try:
        for index in range(3):
            updater.add_activity(_activity(index))

        deadline = time.time() + 8
        while not writes and time.time() < deadline:
            time.sleep(0.1)
    finally:
        updater.stop()

    assert writes, "空闲刷新没有把不足一批的活动发出去"
    assert len(writes) >= 1


def test_full_buffer_still_flushes_immediately(monkeypatch):
    """达到批大小时应立即发送，不必等空闲超时。"""

    writes = []
    updater = _updater(monkeypatch, writes, batch_size=2, idle_seconds=60)

    updater.start()
    try:
        updater.add_activity(_activity(1))
        updater.add_activity(_activity(2))

        deadline = time.time() + 5
        while not writes and time.time() < deadline:
            time.sleep(0.05)
    finally:
        updater.stop()

    assert writes, "满批没有立即发送"
