"""禁止兜底（No fallbacks）契约测试。

这些测试锁死的是一类行为，而不是某个实现细节：失败必须在本层暴露，
未写出的数据必须落盘，缺失的字段/时间戳不得用默认值补齐。

对应规约：MiroFish/AGENTS.md §1。
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import graphiti_patch, zep_graph_memory_updater as updater_module
from app.services.graphiti_patch import (
    KuzuFulltextUnavailableError,
    rebuild_kuzu_fulltext_indexes_async,
)
from app.services.zep_graph_memory_updater import (
    AgentActivity,
    ZepGraphMemoryUpdater,
)
from app.services.zep_graphiti_impl import GraphitiClient
from app.services.zep_adapter import GraphStoreError, GraphStoreUnavailableError
from graph_store_fakes import cloud_store

BACKEND_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BACKEND_DIR / "app" / "config.py"


# ---------------------------------------------------------------- 配置默认值


def _load_config_module(monkeypatch, env: dict[str, str]):
    """在受控环境变量下加载一份独立的 config 模块（不污染全局 Config）。"""

    for key in ("GRAPHITI_MIN_SCORE", "GRAPHITI_DB_PATH", "ZEP_BACKEND",
                "LLM_API_KEY", "GRAPH_MEMORY_BATCH_SIZE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    spec = importlib.util.spec_from_file_location("_no_fallback_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_min_score_is_unset_by_default_so_upstream_guard_applies(monkeypatch):
    module = _load_config_module(monkeypatch, {})
    assert module.Config.GRAPHITI_MIN_SCORE is None, (
        "未配置时必须让 graphiti 上游默认阈值（0.6）生效，"
        "禁止用 0 关闭相关性过滤"
    )


def test_min_score_zero_is_rejected_by_validation(monkeypatch):
    module = _load_config_module(
        monkeypatch,
        {
            "LLM_API_KEY": "test-key",
            "ZEP_BACKEND": "graphiti",
            "GRAPHITI_DB_PATH": "x.kuzu",
            "GRAPHITI_MIN_SCORE": "0",
        },
    )
    errors = module.Config.validate()
    assert any("GRAPHITI_MIN_SCORE" in error for error in errors)


def test_explicit_min_score_is_honoured(monkeypatch):
    module = _load_config_module(monkeypatch, {"GRAPHITI_MIN_SCORE": "0.6"})
    assert module.Config.GRAPHITI_MIN_SCORE == pytest.approx(0.6)


def test_writeback_batch_size_keeps_upstream_default(monkeypatch):
    module = _load_config_module(monkeypatch, {})
    assert module.Config.GRAPH_MEMORY_BATCH_SIZE == 5


# ---------------------------------------------------------------- 全文索引


class _FakeResult:
    """SHOW_INDEXES 的最小返回：支持 has_next / get_next。"""

    def __init__(self, rows):
        self._rows = [list(row) for row in rows]
        self._index = 0

    def has_next(self):
        return self._index < len(self._rows)

    def get_next(self):
        row = self._rows[self._index]
        self._index += 1
        return row


class _FailingFtsClient:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if statement.startswith("CALL SHOW_INDEXES"):
            return _FakeResult([])
        if statement.startswith("CALL CREATE_FTS_INDEX"):
            raise RuntimeError("fts extension unavailable")


class _ExistingIndexClient:
    """库里已经有两个 FTS 索引：必须 DROP 再 CREATE（旧索引会让写回崩进程，N-24）。"""

    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if statement.startswith("CALL SHOW_INDEXES"):
            return _FakeResult(
                [
                    ["Entity", "node_name_and_summary", "FTS"],
                    ["RelatesToNode_", "edge_name_and_fact", "FTS"],
                ]
            )
        return None


@pytest.mark.asyncio
async def test_fulltext_index_failure_raises_instead_of_degrading():
    driver = SimpleNamespace(client=_FailingFtsClient())
    with pytest.raises(KuzuFulltextUnavailableError) as error:
        await rebuild_kuzu_fulltext_indexes_async(driver)
    assert "fts extension unavailable" in str(error.value)


@pytest.mark.asyncio
async def test_existing_fulltext_indexes_are_dropped_and_recreated():
    client = _ExistingIndexClient()
    driver = SimpleNamespace(client=client)
    assert await rebuild_kuzu_fulltext_indexes_async(driver) is True

    drops = [s for s in client.statements if s.startswith("CALL DROP_FTS_INDEX")]
    creates = [s for s in client.statements if s.startswith("CALL CREATE_FTS_INDEX")]
    assert len(drops) == 2 and len(creates) == 2
    assert "node_name_and_summary" in drops[0] and "edge_name_and_fact" in drops[1]
    # 每个索引都是先 drop 后 create
    assert client.statements.index(drops[0]) < client.statements.index(creates[0])
    assert client.statements.index(drops[1]) < client.statements.index(creates[1])


@pytest.mark.asyncio
async def test_index_listing_failure_is_reported():
    """拿不到索引清单就不能假定"库里没有索引"，否则会漏掉重建。"""

    class _NoListingClient:
        async def execute(self, statement):
            if statement.startswith("CALL SHOW_INDEXES"):
                raise RuntimeError("SHOW_INDEXES unavailable")
            return None

    with pytest.raises(KuzuFulltextUnavailableError) as error:
        await rebuild_kuzu_fulltext_indexes_async(SimpleNamespace(client=_NoListingClient()))
    assert "SHOW_INDEXES unavailable" in str(error.value)


# ---------------------------------------------------------------- 检索路径


class _SearchStub:
    """最小 graphiti 替身：只提供 search_ 方法与可配置的返回。"""

    def __init__(self, edges, scores=None):
        self._result = SimpleNamespace(
            edges=edges,
            nodes=[],
            edge_reranker_scores=scores or [],
            node_reranker_scores=[],
        )
        self.calls = []

    async def search_(self, query, config, group_ids):
        self.calls.append(config)
        return self._result


class _Edge:
    def __init__(self, index):
        self.uuid = f"edge-{index}"
        self.name = "RELATES_TO"
        self.fact = f"fact {index}"
        self.source_node_uuid = "a"
        self.target_node_uuid = "b"
        self.episodes = []


def _search_client(monkeypatch, edges, scores=None):
    client = GraphitiClient()
    client._ensure_initialized = lambda: None  # 不碰数据库
    client._graphiti = _SearchStub(edges, scores)
    return client


def test_search_uses_bm25_and_cosine_in_one_path(monkeypatch):
    client = _search_client(monkeypatch, [_Edge(1)])
    client.search("graph-1", "query", limit=5)

    config = client._graphiti.calls[0]
    edge_methods = {str(m) for m in config.edge_config.search_methods}
    assert any("bm25" in m for m in edge_methods)
    assert any("cosine" in m for m in edge_methods)


def test_search_propagates_failure_without_second_path(monkeypatch):
    calls = []

    client = GraphitiClient()
    client._ensure_initialized = lambda: None
    client._graphiti = _SearchStub([])

    def boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("kuzu exploded")

    client._execute_search = boom
    with pytest.raises(RuntimeError, match="kuzu exploded"):
        client.search("graph-1", "query", limit=5)
    assert len(calls) == 1, "检索失败后不得再走第二条检索路径"


def test_saturated_search_result_is_reported(monkeypatch):
    """返回条数顶到上限时必须告警：否则"有结果"会被误读成"结果相关"。"""

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    target = logging.getLogger("mirofish.zep_graphiti")
    target.addHandler(handler)
    try:
        client = _search_client(monkeypatch, [_Edge(i) for i in range(5)])
        client.search("graph-1", "query", limit=5)
    finally:
        target.removeHandler(handler)
    assert any("检索饱和" in record.getMessage() for record in records)


def test_threshold_is_only_applied_when_explicitly_configured(monkeypatch):
    client = _search_client(monkeypatch, [_Edge(1)])
    monkeypatch.setattr(
        "app.services.zep_graphiti_impl.Config.GRAPHITI_MIN_SCORE", None, raising=False
    )
    client.search("graph-1", "query", limit=5)
    assert "sim_min_score" not in client._graphiti.calls[0].edge_config.model_fields_set

    monkeypatch.setattr(
        "app.services.zep_graphiti_impl.Config.GRAPHITI_MIN_SCORE", 0.6, raising=False
    )
    client.search("graph-1", "query", limit=5)
    config = client._graphiti.calls[1]
    assert config.edge_config.sim_min_score == pytest.approx(0.6)


def test_conflicting_reranker_request_is_rejected(monkeypatch):
    client = GraphitiClient()
    monkeypatch.setattr(
        "app.services.zep_graphiti_impl.Config.GRAPHITI_RERANKER", "rrf", raising=False
    )
    with pytest.raises(GraphStoreError, match="不一致"):
        client._resolve_reranker("cross_encoder")


@pytest.mark.parametrize("value", ["cross_encoder", "none"])
def test_reranker_values_this_backend_cannot_use_are_rejected(monkeypatch, value):
    """配置值必须真的能生效：映射不到的取值在配置阶段就拒绝。

    历史：`cross_encoder` 先是被静默改成 `rrf`，修完校验后 `_execute_search()`
    仍硬编码 `rrf` —— 于是"配置 cross_encoder"变成校验通过但实际不生效。
    """

    client = GraphitiClient()
    monkeypatch.setattr(
        "app.services.zep_graphiti_impl.Config.GRAPHITI_RERANKER", value, raising=False
    )
    with pytest.raises(GraphStoreError, match="在本后端不可用"):
        client._resolve_reranker(None)


def test_reranker_without_enum_mapping_is_rejected():
    """没有枚举映射的名字一律拒绝，避免"硬编码回 rrf"再次发生。"""

    from app.services.zep_graphiti_impl import _reranker_enums

    with pytest.raises(GraphStoreError, match="没有对应的 graphiti 枚举"):
        _reranker_enums("cross_encoder")


def test_resolved_reranker_is_passed_into_execute_search(monkeypatch):
    """解析结果必须一路传到 `_execute_search`，不能只用于校验。"""

    client = GraphitiClient()
    client._ensure_initialized = lambda: None
    client._graphiti = _SearchStub([_Edge(1)])
    seen = {}

    def spy(query, graph_id, scope, limit, reranker="rrf"):
        seen["reranker"] = reranker
        return [], [], None

    client._execute_search = spy
    client.search("graph-1", "query", limit=5)
    assert seen["reranker"] == "rrf"


def test_reranker_value_reaches_graphiti_search_config(monkeypatch):
    client = _search_client(monkeypatch, [_Edge(1)])
    client.search("graph-1", "query", limit=5, scope="both")
    config = client._graphiti.calls[0]
    from graphiti_core.search.search_config import EdgeReranker, NodeReranker

    assert config.edge_config.reranker == EdgeReranker.rrf
    assert config.node_config.reranker == NodeReranker.rrf


def test_patch_failure_fails_initialization(monkeypatch):
    """补丁没打上就不是"少一点优化"：写入会整批失败，因此必须初始化失败。"""

    from app.services import graphiti_patch

    monkeypatch.setattr(graphiti_patch, "apply_patch", lambda: False)
    client = GraphitiClient()
    with pytest.raises(GraphStoreUnavailableError, match="属性清洗补丁未应用"):
        client._ensure_initialized()


# ---------------------------------------------------------------- 数据不丢


def _activity(index=1):
    return AgentActivity(
        platform="twitter",
        agent_id=index,
        agent_name=f"Agent {index}",
        action_type="CREATE_POST",
        action_args={"content": "hello"},
        round_num=index,
        timestamp="2026-07-22T12:00:00+08:00",
    )


def _updater(monkeypatch, tmp_path, add=None):
    client = SimpleNamespace(
        graph=SimpleNamespace(
            add=add or (lambda **kwargs: SimpleNamespace(uuid_="episode-1")),
            episode=SimpleNamespace(get=lambda **kwargs: SimpleNamespace(processed=True)),
        )
    )
    monkeypatch.setattr(
        updater_module, "get_graph_store", lambda _key=None: cloud_store(client)
    )
    monkeypatch.setattr(updater_module, "PENDING_DIR", str(tmp_path), raising=True)
    updater = ZepGraphMemoryUpdater("graph-1", api_key="test-key", simulation_id="sim-1")
    updater.SEND_INTERVAL = 0
    return updater


def test_unsent_activities_are_persisted_not_dropped(monkeypatch, tmp_path):
    updater = _updater(monkeypatch, tmp_path)
    activities = [_activity(1), _activity(2)]

    path = updater._persist_unsent("twitter", activities, reason="drain deadline elapsed")

    assert path is not None
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert [row["agent_id"] for row in rows] == [1, 2]
    assert all(row["unsent_reason"] == "drain deadline elapsed" for row in rows)


def test_drain_deadline_persists_buffer_before_raising(monkeypatch, tmp_path):
    updater = _updater(monkeypatch, tmp_path)
    updater._platform_buffers["twitter"] = [_activity(1)]

    with pytest.raises(TimeoutError):
        updater._flush_remaining(deadline=0)

    pending = list(Path(tmp_path).glob("*.jsonl"))
    assert pending, "drain 超时前必须先落盘，不能静默丢弃"


def test_send_failure_persists_the_batch(monkeypatch, tmp_path):
    def explode(**_kwargs):
        raise RuntimeError("graph store unavailable")

    updater = _updater(monkeypatch, tmp_path, add=explode)
    processed = updater._send_batch_activities([_activity(1)], "twitter")

    assert processed == 1
    pending = list(Path(tmp_path).glob("*.jsonl"))
    assert pending, "发送失败的批次也必须落盘"
    assert "graph store unavailable" in Path(pending[0]).read_text(encoding="utf-8")


def test_worker_loop_error_persists_batch_and_stops(monkeypatch, tmp_path):
    """工作线程异常时：批次必须落盘，且不能继续假装正常（stop 必须报错）。"""

    updater = _updater(monkeypatch, tmp_path)
    updater.BATCH_SIZE = 1

    def boom(platform, batch, **kwargs):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(updater, "_send_batch_activities", boom)
    updater.start()
    updater.add_activity(_activity(1))
    updater._worker_thread.join(timeout=5)

    assert updater.get_stats()["worker_error"], "故障必须出现在统计里"
    pending = list(Path(tmp_path).glob("*.jsonl"))
    assert pending, "工作线程异常前已切走的批次必须落盘"
    with pytest.raises(RuntimeError, match="工作线程异常"):
        updater.stop()


# ---------------------------------------------------------------- 不发明默认值


def test_missing_action_fields_are_rejected(monkeypatch, tmp_path):
    updater = _updater(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="缺少字段"):
        updater.add_activity_from_dict(
            {"agent_id": 1, "agent_name": "A", "action_type": "CREATE_POST"},
            "twitter",
        )


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError, match="无法解析"):
        ZepGraphMemoryUpdater._to_rfc3339("not-a-timestamp")


def test_parse_time_rejects_missing_value():
    with pytest.raises(GraphStoreError, match="缺少时间戳"):
        GraphitiClient._parse_time(None)


def test_parse_time_rejects_invalid_value():
    with pytest.raises(GraphStoreError, match="无法解析"):
        GraphitiClient._parse_time("2026-13-45")


def test_unknown_episode_type_is_rejected():
    class _EpisodeType:
        text = "text"
        message = "message"
        json = "json"

    with pytest.raises(GraphStoreError, match="不支持的 episode 类型"):
        GraphitiClient._episode_type("binary", _EpisodeType)
