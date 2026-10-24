"""L0 离线单测：Graphiti 补丁的纯函数与幂等性。

不联网、不碰数据库、不调用 LLM；`app.services.graphiti_patch` 内部的 graphiti
导入都是惰性的，因此本文件在未安装 graphiti-core 时也能收集运行。
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from app.services import graphiti_patch as patch


class _EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(default_factory=list)
    contradicted_facts: list[int] = Field(default_factory=list)


class _Nested(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)


# ---------------- _schema_example ----------------


def test_schema_example_fills_object_fields():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {"type": "object", "properties": {"k": {"type": "string"}}},
        },
    }

    example = patch._schema_example(schema)

    assert example["name"] == ""
    assert example["count"] == 0
    assert example["ratio"] == 0.0
    assert example["enabled"] is False
    assert example["tags"] == []
    assert example["meta"] == {"k": ""}


def test_schema_example_prefers_non_null_anyof_branch():
    schema = {
        "type": "object",
        "properties": {
            "description": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }

    assert patch._schema_example(schema) == {"description": ""}


def test_schema_example_resolves_refs_from_defs():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/_Nested"}}},
        "$defs": {
            "_Nested": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            }
        },
    }

    example = patch._schema_example(schema, schema["$defs"])

    # 对象数组给"一个元素"的示例，而不是空数组
    assert example == {"items": [{"name": "", "tags": []}]}


def test_schema_example_handles_primitive_array_and_depth_guard():
    assert patch._schema_example({"type": "array", "items": {"type": "integer"}}) == []
    assert patch._schema_example({"type": "string"}, depth=7) is None
    assert patch._schema_example("not-a-schema") is None


# ---------------- _looks_like_schema_echo ----------------


def test_detects_schema_echo_for_edge_duplicate():
    echoed = {
        "properties": {
            "duplicate_facts": {"items": {"type": "integer"}, "type": "array"},
            "contradicted_facts": {"items": {"type": "integer"}, "type": "array"},
        },
        "required": ["duplicate_facts", "contradicted_facts"],
        "type": "object",
    }

    assert patch._looks_like_schema_echo(echoed, _EdgeDuplicate) is True


def test_accepts_a_real_instance():
    assert patch._looks_like_schema_echo({"duplicate_facts": [], "contradicted_facts": [1]}, _EdgeDuplicate) is False


def test_partial_instance_is_not_treated_as_echo():
    # 只要出现任一期望字段就认为模型在回答实例（哪怕不全），交给 Pydantic 报错
    assert patch._looks_like_schema_echo({"duplicate_facts": []}, _EdgeDuplicate) is False


def test_non_dict_payload_is_not_echo():
    assert patch._looks_like_schema_echo([1, 2], _EdgeDuplicate) is False
    assert patch._looks_like_schema_echo(None, _EdgeDuplicate) is False


# ---------------- 属性清洗 ----------------


def test_sanitize_for_graph_keeps_primitives_and_flat_arrays():
    assert patch.sanitize_for_graph(None) is None
    assert patch.sanitize_for_graph("x") == "x"
    assert patch.sanitize_for_graph(3) == 3
    assert patch.sanitize_for_graph(True) is True
    assert patch.sanitize_for_graph(["a", 1, None]) == ["a", 1, None]


def test_sanitize_for_graph_serializes_nested_structures():
    nested_dict = {"a": {"b": 1}}
    nested_list = [{"a": 1}, "x"]

    serialized_dict = patch.sanitize_for_graph(nested_dict)
    serialized_list = patch.sanitize_for_graph(nested_list)

    assert isinstance(serialized_dict, str) and '"a"' in serialized_dict
    assert isinstance(serialized_list, str) and "a" in serialized_list


def test_sanitize_attributes_handles_empty_and_mixed():
    assert patch.sanitize_attributes({}) == {}
    cleaned = patch.sanitize_attributes({"ok": 1, "bad": {"nested": True}})
    assert cleaned["ok"] == 1
    assert isinstance(cleaned["bad"], str)


# ---------------- 驱动兼容 ----------------


class _FakeDriver:
    pass


def test_driver_compat_only_touches_kuzu_driver():
    fake = _FakeDriver()
    patch.apply_driver_compat(fake)
    assert not hasattr(fake, "_database")


def test_driver_compat_sets_database_once():
    kuzu_driver_module = pytest.importorskip("graphiti_core.driver.kuzu_driver")
    driver = kuzu_driver_module.KuzuDriver(db=":memory:")

    patch.apply_driver_compat(driver)
    assert driver._database == ""

    driver._database = "kept"
    patch.apply_driver_compat(driver)
    assert driver._database == "kept"


# ---------------- 幂等性 ----------------


def test_bulk_utils_patch_is_idempotent():
    bulk_utils = pytest.importorskip("graphiti_core.utils.bulk_utils")

    assert patch.apply_patch() is True
    first = bulk_utils.add_nodes_and_edges_bulk_tx
    assert patch.apply_patch() is True
    assert bulk_utils.add_nodes_and_edges_bulk_tx is first


def test_llm_prompt_patch_is_idempotent(monkeypatch):
    client_module = pytest.importorskip("graphiti_core.llm_client.openai_generic_client")

    async def fake_generate_response(self, messages, response_model=None, *args, **kwargs):
        return {}

    monkeypatch.setattr(client_module.OpenAIGenericClient, "generate_response", fake_generate_response)
    monkeypatch.setattr(client_module.OpenAIGenericClient, "_mirofish_prompt_patched", False, raising=False)

    assert patch.apply_llm_prompt_patch() is True
    first = client_module.OpenAIGenericClient.generate_response
    assert patch.apply_llm_prompt_patch() is True
    assert client_module.OpenAIGenericClient.generate_response is first


def _patched_client(monkeypatch, generate_response):
    """把补丁应用到类上，并返回一个最小可用的假客户端。"""

    client_module = pytest.importorskip("graphiti_core.llm_client.openai_generic_client")
    monkeypatch.setattr(client_module.OpenAIGenericClient, "generate_response", generate_response)
    monkeypatch.setattr(client_module.OpenAIGenericClient, "_mirofish_prompt_patched", False, raising=False)
    patch.apply_llm_prompt_patch()

    calls = []

    async def fake_internal(messages, response_model=None, max_tokens=None, model_size=None):
        calls.append(getattr(messages[-1], "content", ""))
        return generate_response(len(calls))

    client = SimpleNamespace(
        structured_output_mode="json_object",
        max_tokens=1024,
        _generate_response=fake_internal,
    )
    return client_module, client, calls


def test_llm_prompt_patch_injects_example_instead_of_schema(monkeypatch):
    client_module, client, calls = _patched_client(monkeypatch, lambda _n: {"duplicate_facts": [], "contradicted_facts": []})
    messages = [SimpleNamespace(content="extract duplicates")]

    payload = asyncio.run(client_module.OpenAIGenericClient.generate_response(client, messages, _EdgeDuplicate))

    assert payload == {"duplicate_facts": [], "contradicted_facts": []}
    assert len(calls) == 1
    # 提示词里应当是示例骨架，而不是 graphiti 默认注入的 schema 原文
    assert '{"duplicate_facts": [], "contradicted_facts": []}' in calls[0]
    # schema 原文才有 required/title/$defs；示例骨架没有
    assert '"required"' not in calls[0]
    assert '"$defs"' not in calls[0]
    assert "Do NOT return the schema itself" in calls[0]


def test_llm_prompt_patch_retries_once_on_schema_echo(monkeypatch):
    def responder(call_index):
        if call_index == 1:
            return {"properties": {"duplicate_facts": {"type": "array"}}}
        return {"duplicate_facts": [], "contradicted_facts": []}

    client_module, client, calls = _patched_client(monkeypatch, responder)
    messages = [SimpleNamespace(content="extract duplicates")]

    payload = asyncio.run(client_module.OpenAIGenericClient.generate_response(client, messages, _EdgeDuplicate))

    assert payload == {"duplicate_facts": [], "contradicted_facts": []}
    assert len(calls) == 2
    assert "was the schema definition, not an instance" in calls[1]


def test_llm_prompt_patch_delegates_for_non_json_object_mode(monkeypatch):
    seen = {}

    async def fake_generate_response(self, messages, response_model=None, *args, **kwargs):
        seen["delegated"] = True
        return {"ok": True}

    client_module = pytest.importorskip("graphiti_core.llm_client.openai_generic_client")
    monkeypatch.setattr(client_module.OpenAIGenericClient, "generate_response", fake_generate_response)
    monkeypatch.setattr(client_module.OpenAIGenericClient, "_mirofish_prompt_patched", False, raising=False)
    patch.apply_llm_prompt_patch()

    client = SimpleNamespace(structured_output_mode="json_schema")
    payload = asyncio.run(
        client_module.OpenAIGenericClient.generate_response(
            client, [SimpleNamespace(content="x")], _EdgeDuplicate
        )
    )

    assert seen.get("delegated") is True
    assert payload == {"ok": True}


# ---------------- 节点类型标签的写边界守卫（掩盖台账 N-34） ----------------


class _FakeKuzuDriver:
    """只回答"库里这个节点现在有哪些类型"的最小驱动。"""

    def __init__(self, labels_by_uuid):
        driver_module = pytest.importorskip("graphiti_core.driver.driver")
        self.provider = driver_module.GraphProvider.KUZU
        self.labels_by_uuid = labels_by_uuid
        self.queries = []
        # bulk_utils 会先看这个（真实驱动上有；Kuzu 走 tx.run 分支）
        self.graph_operations_interface = None

    async def execute_query(self, cypher, **kwargs):
        self.queries.append((cypher, kwargs))
        uuid = kwargs.get("uuid")
        if uuid in self.labels_by_uuid:
            return ([{"labels": self.labels_by_uuid[uuid]}], None, None)
        return ([], None, None)


class _FakeSession:
    def __init__(self):
        self.runs = []

    async def run(self, query, **kwargs):
        self.runs.append((query, kwargs))
        return None


class _FakeEmbedder:
    async def create(self, input_data):
        return [0.0] * 4

    async def create_batch(self, input_data):
        return [[0.0] * 4 for _ in input_data]


def _entity_node(uuid, name, labels, group_id="mirofish_probe"):
    from datetime import datetime, timezone

    nodes_module = pytest.importorskip("graphiti_core.nodes")
    return nodes_module.EntityNode(
        uuid=uuid,
        name=name,
        group_id=group_id,
        labels=list(labels),
        created_at=datetime.now(timezone.utc),
        summary="",
        attributes={},
    )


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    from app.config import Config

    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    path = tmp_path / "_diagnostics" / "node-label-mutations.jsonl"
    return path


def test_retain_labels_keeps_the_type_when_incoming_has_none(ledger):
    """实测形状：抽到同名实体但没给类型 → 旧实现会把 Engineer 抹成只剩 Entity。"""

    node = _entity_node("uuid-1", "某工程师", ["Entity"])
    driver = _FakeKuzuDriver({"uuid-1": ["Entity", "Engineer"]})

    repaired = asyncio.run(patch.retain_node_labels(driver, [node]))

    assert repaired == 1
    assert set(node.labels) >= {"Entity", "Engineer"}
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(line)
    assert record["reason"] == "incoming_has_no_specific_type"
    assert record["labels_in_db"] == ["Engineer"]
    assert record["labels_written"] == ["Engineer"]


def test_retain_labels_unions_a_different_incoming_type(ledger):
    """同一节点被抽成别的具体类型时不许二选一：类型只累积。"""

    node = _entity_node("uuid-2", "某分析师", ["Person"])
    driver = _FakeKuzuDriver({"uuid-2": ["Entity", "IndustryAnalyst"]})

    repaired = asyncio.run(patch.retain_node_labels(driver, [node]))

    assert repaired == 1
    assert set(node.labels) >= {"Person", "IndustryAnalyst"}
    record = json.loads(ledger.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["reason"] == "incoming_has_a_different_type"
    assert record["labels_incoming"] == ["Person"]


def test_retain_labels_leaves_matching_types_alone(ledger):
    node = _entity_node("uuid-3", "某主管", ["Entity", "CorporateManager"])
    driver = _FakeKuzuDriver({"uuid-3": ["Entity", "CorporateManager"]})

    assert asyncio.run(patch.retain_node_labels(driver, [node])) == 0
    assert node.labels == ["Entity", "CorporateManager"]
    assert not ledger.exists()


def test_retain_labels_ignores_brand_new_nodes(ledger):
    node = _entity_node("uuid-4", "新人", ["Entity", "Engineer"])
    driver = _FakeKuzuDriver({})

    assert asyncio.run(patch.retain_node_labels(driver, [node])) == 0
    assert node.labels == ["Entity", "Engineer"]
    assert not ledger.exists()


def test_retain_labels_is_a_noop_for_other_providers(ledger):
    node = _entity_node("uuid-5", "某工程师", ["Entity"])
    driver = SimpleNamespace(provider=SimpleNamespace(name="NEO4J"))
    driver.execute_query = None  # 非 Kuzu 时连查询都不该发

    assert asyncio.run(patch.retain_node_labels(driver, [node])) == 0
    assert node.labels == ["Entity"]


def test_label_ledger_path_follows_upload_folder(ledger):
    """台账路径必须跟着 Config.UPLOAD_FOLDER（N-42 的教训：不许是导入期常量）。"""

    assert patch.label_ledger_path().startswith(str(ledger.parents[0]))


def test_installed_bulk_patch_writes_the_retained_labels(ledger):
    """端到端：走真实的 `add_nodes_and_edges_bulk_tx`，写库参数里必须带住旧类型。"""

    bulk_utils = pytest.importorskip("graphiti_core.utils.bulk_utils")
    assert patch.apply_patch() is True

    node = _entity_node("uuid-6", "某工程师", ["Entity"])
    driver = _FakeKuzuDriver({"uuid-6": ["Entity", "Engineer"]})
    session = _FakeSession()

    asyncio.run(
        bulk_utils.add_nodes_and_edges_bulk_tx(
            session, [], [], [node], [], _FakeEmbedder(), driver
        )
    )

    written = [kwargs for _query, kwargs in session.runs if kwargs.get("uuid") == "uuid-6"]
    assert written, "没有向库里写这个节点"
    assert set(written[0]["labels"]) >= {"Entity", "Engineer"}, written[0]["labels"]


def test_apply_patch_also_wraps_entity_node_save():
    """单节点保存是同一条能力（整列覆盖 n.labels），必须一起守。"""

    nodes_module = pytest.importorskip("graphiti_core.nodes")
    assert patch.apply_patch() is True

    wrapped = getattr(nodes_module.EntityNode.save, "__wrapped__", None)
    assert wrapped is not None, "EntityNode.save 没有被包装：单节点保存仍会抹掉类型"

