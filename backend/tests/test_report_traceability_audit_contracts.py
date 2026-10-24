"""报告可追溯抽检工具的 N-50 契约：候选支撑必须同时覆盖边与节点摘要。

背景（`掩盖台账.md` N-50）：`_support()` 固定 `scope="edges"`，而 R1 的判定口径
允许“边事实或节点摘要”。第二轮 `S5-5` 的引语只在 `HR / 招聘负责人` 节点摘要里，
工作表却显示 0 条对应候选，R1 被低估。这里锁住两件事：

1. 工具必须查两次（edges / nodes），不能只搜边；
2. claim 命中的实体摘要必须被附上（语义检索不一定召回它，但实体名命中是确定的）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "report_traceability_audit.py"
)
SPEC = importlib.util.spec_from_file_location("report_traceability_audit", TOOL_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class _FakeService:
    def __init__(self, edges, nodes):
        self.edges = edges
        self.nodes = nodes
        self.scopes = []

    def search_graph(self, graph_id, query, limit, scope):  # noqa: ANN001
        self.scopes.append(scope)
        return SimpleNamespace(
            edges=list(self.edges) if scope == "edges" else [],
            nodes=list(self.nodes) if scope == "nodes" else [],
        )


def _edge(uuid, fact):
    return {
        "uuid": uuid,
        "fact": fact,
        "source_node_uuid": "a",
        "target_node_uuid": "b",
        "name": "RELATES_TO",
    }


def _node(uuid, name, summary):
    return {"uuid": uuid, "name": name, "summary": summary, "labels": ["Entity"]}


def test_support_searches_edges_and_nodes():
    service = _FakeService(
        edges=[_edge("e1", "边事实：HR 提到署名负责。")],
        nodes=[_node("n1", "HR / 招聘负责人", "节点摘要：会被定格成支援型工程师。")],
    )
    entities = [
        {
            "uuid": "n1",
            "name": "HR / 招聘负责人",
            "labels": ["Entity"],
            "summary": "节点摘要：会被定格成支援型工程师。",
        }
    ]
    support = AUDIT._support(
        service,
        "graph",
        "HR 说会被定格成支援型工程师，后面翻盘成本翻倍。",
        {"a": "甲", "b": "乙"},
        entities,
        mentioned_entities=[
            {"name": "HR / 招聘负责人", "type": "", "uuid": "n1"}
        ],
    )

    assert service.scopes == ["edges", "nodes"], "必须同时查边与节点，不能只搜边"
    kinds = {item["kind"] for item in support}
    assert kinds == {"edge", "node"}
    node_facts = [item["fact"] for item in support if item["kind"] == "node"]
    assert "节点摘要：会被定格成支援型工程师。" in node_facts


def test_mentioned_entity_summary_is_added_even_if_node_search_misses_it():
    service = _FakeService(edges=[], nodes=[])  # 语义检索没召回节点
    entities = [
        {
            "uuid": "hr",
            "name": "HR / 招聘负责人",
            "labels": ["Entity"],
            "summary": "24个月内拿不到署名负责，会被定格成支援型工程师，后面翻盘成本翻倍。",
        }
    ]
    support = AUDIT._support(
        service,
        "graph",
        "HR 说会被定格成支援型工程师。",
        {},
        entities,
        mentioned_entities=[{"name": "HR / 招聘负责人", "type": "", "uuid": "hr"}],
    )

    assert [item["kind"] for item in support] == ["node"]
    assert "支援型工程师" in support[0]["fact"]


def test_support_does_not_fabricate_when_no_candidates():
    service = _FakeService(edges=[], nodes=[])
    support = AUDIT._support(
        service,
        "graph",
        "查不到任何东西的陈述。",
        {},
        [],
        mentioned_entities=[],
    )
    assert support == []


def test_quoted_claim_finds_node_summary_even_when_entity_name_is_not_in_claim():
    """S5-5 场景：claim 只写 "HR"，引语只存在于 `HR / 招聘负责人` 摘要。"""

    summary = (
        "晋升看核心型号负责权，24个月内拿不到任何本体级模块的署名负责，"
        "会被定格成「支援型工程师」，后面翻盘成本翻倍。"
    )
    service = _FakeService(edges=[], nodes=[])  # 语义节点检索没召回
    entities = [
        {
            "uuid": "boss",
            "name": "HR / 招聘负责人",
            "labels": ["Entity"],
            "summary": summary,
        }
    ]
    support = AUDIT._support(
        service,
        "graph",
        'HR说"拿不到任何本体级模块的署名负责，会被定格成支援型工程师，后面翻盘成本翻倍"。',
        {},
        entities,
        mentioned_entities=[],  # claim 里的 "HR" 指向另一个节点，不是这个
    )
    assert [item["kind"] for item in support] == ["node"]
    assert "支援型工程师" in support[0]["fact"]


def test_split_claims_tracks_fact_and_inference_blocks():
    text = (
        f"{AUDIT.FACT_LABEL}\n\n"
        "> \"这是一条足够长的模拟事实陈述，来自工具结果。\"\n\n"
        f"{AUDIT.INFERENCE_LABEL}\n\n"
        f"{AUDIT.INFERENCE_PREFIX} 这是一条足够长的作者推断，不是模拟事实。"
    )
    records = AUDIT.split_claims(text)
    assert [record["claim_kind"] for record in records] == ["fact", "inference"]


def test_split_claims_without_labels_is_unlabeled():
    records = AUDIT.split_claims("这是一条足够长的普通陈述，没有块标签。")
    assert records and records[0]["claim_kind"] == "unlabeled"


def test_split_claims_skips_lead_in_lines_ending_with_colon():
    records = AUDIT.split_claims(
        "三条信号如下：\n\n这是一条足够长的普通陈述，没有块标签。"
    )
    assert len(records) == 1
    assert "三条信号如下" not in records[0]["text"]


def test_split_claims_skips_bold_subheadings_but_keeps_labeled_inference():
    text = (
        f"{AUDIT.FACT_LABEL}\n\n"
        "**一、小标题不是结论**\n\n"
        "> \"这是一条足够长的模拟事实陈述，来自工具结果。\"\n\n"
        f"{AUDIT.INFERENCE_LABEL}\n\n"
        "**[推断] 这是一条足够长的作者推断，必须保留。**"
    )
    records = AUDIT.split_claims(text)
    assert len(records) == 2
    assert records[0]["claim_kind"] == "fact"
    assert records[1]["claim_kind"] == "inference"
    assert records[1]["text"].startswith("[推断]")


def test_summarize_reports_r1_all_and_r1_fact(capsys):
    worksheet = {
        "claims": [
            {"claim_kind": "fact", "verdict": "supported", "conflict_with_scope": False},
            {"claim_kind": "fact", "verdict": "partial", "conflict_with_scope": False},
            {"claim_kind": "inference", "verdict": "partial", "conflict_with_scope": False},
            {"claim_kind": "unlabeled", "verdict": "supported", "conflict_with_scope": False},
        ],
        "grounding": {"sections": [{"final_violations": []}]},
    }
    assert AUDIT.summarize(worksheet) == 0
    output = capsys.readouterr().out
    assert "R1_all（全部条目）  : 50.0%" in output
    assert "R1_fact（事实块）   : 50.0%（1/2）" in output
    assert "unlabeled 条数      : 1" in output


def test_quote_violations_reads_final_gate_violations():
    worksheet = {
        "grounding": {
            "sections": [
                {
                    "final_violations": [
                        {"code": "quote_not_verbatim"},
                        {"code": "unlabeled_strength"},
                    ]
                }
            ]
        }
    }
    assert AUDIT._quote_violations(worksheet) == 1


def test_support_deduplicates_same_node_from_search_and_mention():
    summary = "同一条节点摘要。"
    service = _FakeService(
        edges=[],
        nodes=[_node("n1", "HR / 招聘负责人", summary)],
    )
    entities = [
        {
            "uuid": "n1",
            "name": "HR / 招聘负责人",
            "labels": ["Entity"],
            "summary": summary,
        }
    ]
    support = AUDIT._support(
        service,
        "graph",
        "HR 的摘要。",
        {},
        entities,
        mentioned_entities=[{"name": "HR / 招聘负责人", "type": "", "uuid": "n1"}],
    )
    assert len(support) == 1
    assert support[0]["kind"] == "node"
