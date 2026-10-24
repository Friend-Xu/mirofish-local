"""Kuzu 执行并发契约：本项目对 Kuzu 的使用是**单连接 + 单线程**。

对应掩盖台账 N-24：回写写图谱期间 `kuzu.Connection.execute` 以原生 access violation
杀死整个后端进程（faulthandler dump 实测），于是 `state.json` 永远停在 `running`、
子进程变孤儿、验证跑不下去。graphiti 的 `KuzuDriver` 只暴露 `max_concurrent_queries`，
`max_threads_per_query` 固定为 0（查询内部不限制并行），所以必须由本项目显式收紧。

证据等级：本文件是**单测**（配置默认值 + 强制逻辑）。"收紧后崩溃不再发生"由 L-a 实跑证明，
单测不能替代它。
"""

import importlib
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Config  # noqa: E402
from app.services.zep_adapter import GraphStoreUnavailableError  # noqa: E402
from app.services.zep_graphiti_impl import _enforce_kuzu_single_threaded  # noqa: E402


class _FakeConnection:
    def __init__(self):
        self.threads = []

    def set_max_threads_for_exec(self, num_threads):
        self.threads.append(num_threads)


class _FakeClient:
    def __init__(self, count):
        self.connections = [_FakeConnection() for _ in range(count)]


class _FakeDriver:
    def __init__(self, count):
        self.client = _FakeClient(count) if count is not None else object()


def test_defaults_are_single_threaded():
    assert Config.GRAPHITI_MAX_CONCURRENT_QUERIES == 1
    assert Config.GRAPHITI_MAX_THREADS_PER_QUERY == 1


def test_dead_config_knobs_are_gone():
    """声明了却从没被读过的配置项会让人以为"改了配置就改了行为"（N-28）。"""

    assert not hasattr(Config, "GRAPHITI_USE_NATIVE_QUERIES")


def test_enforce_sets_threads_on_every_connection():
    driver = _FakeDriver(2)
    assert _enforce_kuzu_single_threaded(driver) == 2
    assert [conn.threads for conn in driver.client.connections] == [[1], [1]]


def test_enforce_raises_when_connections_unavailable():
    """拿不到连接句柄就是"单线程"这个前提没建立：必须失败，不能跳过。"""

    with pytest.raises(GraphStoreUnavailableError) as error:
        _enforce_kuzu_single_threaded(_FakeDriver(None))
    assert "N-24" in str(error.value)


@pytest.mark.parametrize(
    "name, value",
    [
        ("GRAPHITI_MAX_CONCURRENT_QUERIES", "0"),
        ("GRAPHITI_MAX_THREADS_PER_QUERY", "0"),
    ],
)
def test_config_rejects_zero(monkeypatch, name, value):
    """0 在 Kuzu 里表示"不限制"，正是崩溃的触发条件：配置层就挡住。"""

    config_module = importlib.import_module("app.config")
    monkeypatch.setenv(name, value)
    try:
        with pytest.raises(ValueError) as error:
            importlib.reload(config_module)
        assert name in str(error.value)
    finally:
        monkeypatch.delenv(name, raising=False)
        importlib.reload(config_module)
