"""回写可观测性契约测试：交付率、落盘积压、停滞告警。

对应规约：`MiroFish/AGENTS.md §1.1`（超时后丢弃未写出的数据）；
掩盖台账条目：M-02 / M-13 / N-02 / N-03。
"""

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

from app.services import zep_graph_memory_updater as updater_module
from app.services.zep_graph_memory_updater import pending_backlog


# --------------------------------------------------------- 落盘积压可读


def test_pending_backlog_is_zero_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(updater_module, "PENDING_DIR", str(tmp_path))
    backlog = pending_backlog("graph_none")
    assert backlog["pending_items"] == 0
    assert backlog["path"] is None


def test_pending_backlog_counts_by_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(updater_module, "PENDING_DIR", str(tmp_path))
    path = tmp_path / "graph_x.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"unsent_reason": "drain deadline elapsed", "timestamp": "T1"}),
                json.dumps({"unsent_reason": "drain deadline elapsed", "timestamp": "T2"}),
                json.dumps({"unsent_reason": "send failed", "timestamp": "T3"}),
                "{不是合法 JSON",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    backlog = pending_backlog("graph_x")
    assert backlog["pending_items"] == 4
    assert backlog["reasons"]["drain deadline elapsed"] == 2
    assert backlog["reasons"]["send failed"] == 1
    assert backlog["reasons"]["(无法解析的行)"] == 1
    assert backlog["first_at"] == "T1" and backlog["last_at"] == "T3"


# --------------------------------------------------------- 交付率


def _stats_only_updater():
    """只构造 get_stats 需要的字段（不启动工作线程）。"""

    updater = updater_module.ZepGraphMemoryUpdater.__new__(
        updater_module.ZepGraphMemoryUpdater
    )
    updater.graph_id = "graph_stats"
    updater.BATCH_SIZE = 5
    updater._buffer_lock = threading.Lock()
    updater._platform_buffers = {"twitter": [1, 2]}
    updater._total_activities = 10
    updater._skipped_count = 2
    updater._total_sent = 1
    updater._total_items_sent = 5
    updater._failed_count = 1
    updater._persisted_count = 3
    updater._pending_episode_uuids = []
    updater._activity_queue = SimpleNamespace(qsize=lambda: 1)
    updater._running = False
    updater._worker_error = None
    updater._truncated_count = 0
    updater._last_flush_at = 0.0
    updater._last_activity_at = 0.0
    updater._stall_reported_at = 0.0
    return updater


def test_stats_expose_delivery_ratio_and_persisted_items():
    stats = _stats_only_updater().get_stats()
    # 应入库条数 = 被接收进队列的活动数（_total_activities）；_skipped_count 是
    # 在累加之前就被拦下的 DO_NOTHING，不能再从 expected 里扣一次（N-27）
    assert stats["expected_items"] == 10
    assert stats["items_sent"] == 5
    assert stats["delivery_ratio"] == 0.5
    assert stats["undelivered_items"] == 5
    assert stats["persisted_count"] == 3
    assert stats["buffered_items"] == 2


def test_delivery_ratio_never_exceeds_one_with_skipped_activities():
    """回归 N-27：5 条活动里有 1 条 DO_NOTHING 时，旧口径算出 1.25 与 undelivered=-1。

    这个错口径会**高估交付**：真的只送到 4 条时它也报 1.0（"全部送达"）。
    """

    updater = _stats_only_updater()
    updater._total_activities = 5   # 4 条真实活动 + 已经是接收前被拦下的 1 条 DO_NOTHING 不计入
    updater._skipped_count = 1
    updater._total_items_sent = 5
    stats = updater.get_stats()
    assert stats["expected_items"] == 5
    assert stats["delivery_ratio"] == 1.0
    assert stats["undelivered_items"] == 0

    updater._total_items_sent = 4
    stats = updater.get_stats()
    assert stats["delivery_ratio"] == 0.8, "少送一条就必须掉下来，不能仍报 1.0"
    assert stats["undelivered_items"] == 1


def test_delivery_ratio_is_none_without_expected_items():
    updater = _stats_only_updater()
    updater._total_activities = 0
    updater._total_items_sent = 0
    assert updater.get_stats()["delivery_ratio"] is None


# --------------------------------------------------------- 停滞告警


def _capture(records):
    handler = logging.Handler()
    handler.emit = records.append
    target = logging.getLogger("mirofish.zep_graph_memory_updater")
    target.addHandler(handler)
    return target, handler


def test_stall_is_reported_when_buffer_never_drains():
    """缓冲有内容且长时间没送出去 → 必须告警（历史上它是静默的）。"""

    import time

    records = []
    target, handler = _capture(records)
    try:
        updater = _stats_only_updater()
        updater._last_flush_at = 0.0
        updater._last_activity_at = time.time() - 10_000
        updater._stall_reported_at = 0.0
        updater._report_stall_if_idle(time.time())
    finally:
        target.removeHandler(handler)

    assert any("回写停滞告警" in record.getMessage() for record in records)


def test_no_stall_report_when_buffer_is_empty():
    records = []
    target, handler = _capture(records)
    try:
        updater = _stats_only_updater()
        updater._platform_buffers = {}
        updater._report_stall_if_idle(1_000_000.0)
    finally:
        target.removeHandler(handler)
    assert not records
