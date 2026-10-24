"""采访契约测试：没回答就不是成功；选择失败不许伪造理由。

对应规约：`MiroFish/AGENTS.md §1.1`；
掩盖台账条目：A4 / A6 / N-08 / N-09 / N-14。
"""

from types import SimpleNamespace

import pytest

from app.config import Config
from app.services.zep_adapter import GraphEdge
from app.services.zep_graphiti_impl import GraphitiClient, _optional_str
from app.services.zep_tools import (
    AgentInterview,
    InterviewResult,
    InterviewSelectionError,
    ZepToolsService,
)


# ------------------------------------------------- 时间字段（N-08 的真因 N-14）


def test_optional_str_maps_none_like_values_to_none():
    assert _optional_str(None) is None
    assert _optional_str("") is None
    assert _optional_str("   ") is None
    assert _optional_str("None") is None
    assert _optional_str("null") is None
    assert _optional_str("2026-10-09 22:15:28") == "2026-10-09 22:15:28"


def test_graph_edge_temporals_are_none_not_string_none():
    """`str(None)` == "None" 是真值 → 会让每条边都被判为历史（实测 0 有效/68 历史）。"""

    edge = SimpleNamespace(
        uuid="e1",
        name="RELATES_TO",
        fact="某事实",
        source_node_uuid="a",
        target_node_uuid="b",
        attributes={},
        created_at=None,
        valid_at=None,
        invalid_at=None,
        expired_at=None,
        episodes=[],
        fact_type=None,
    )
    converted = GraphitiClient._to_graph_edge(edge)
    assert converted.invalid_at is None
    assert converted.expired_at is None
    assert converted.valid_at is None
    assert converted.created_at is None

    # EdgeInfo 的 is_invalid / is_expired 就是 `is not None`
    info = SimpleNamespace(invalid_at=converted.invalid_at, expired_at=converted.expired_at)
    assert not (info.invalid_at is not None or info.expired_at is not None)


def test_graph_edge_keeps_real_timestamps():
    edge = SimpleNamespace(
        uuid="e2", name="RELATES_TO", fact="某事实",
        source_node_uuid="a", target_node_uuid="b", attributes={},
        created_at="2026-01-01 00:00:00",
        valid_at="2026-01-01 00:00:00",
        invalid_at="2026-02-01 00:00:00",
        expired_at=None, episodes=[], fact_type=None,
    )
    converted = GraphitiClient._to_graph_edge(edge)
    assert converted.invalid_at == "2026-02-01 00:00:00"
    assert converted.expired_at is None


# ------------------------------------------------------------ A4 采访计数


def test_missing_answers_are_not_counted_as_interviews():
    """采访结果文本必须能区分"回答了"和"没回答"，且不写占位文案冒充内容。"""

    result = InterviewResult(interview_topic="话题", interview_questions=["问题1"])
    result.total_agents = 3
    result.interviews = [
        AgentInterview(
            agent_name="甲", agent_role="工程师", agent_bio="简介", question="问题1",
            response="【Twitter平台回答】\n我的看法……\n\n【Reddit平台无回答】",
        )
    ]
    result.interviewed_count = len(result.interviews)
    result.failed_agents = ["乙", "丙"]

    text = result.to_text()
    assert "获得回答:** 1 / 3" in text
    assert "未获得回答:** 2 位" in text
    assert "乙" in text and "丙" in text
    assert "该平台未获得回复" not in text, "占位文案不允许出现在采访证据里"
    assert result.to_dict()["failed_agents"] == ["乙", "丙"]


# ------------------------------------------------------------ A6 选择失败


def _selector(llm_result=None, error=None):
    service = ZepToolsService.__new__(ZepToolsService)

    class _LLM:
        def chat_json(self, messages=None, temperature=None):  # noqa: ANN001
            if error:
                raise error
            return llm_result

    service._llm_client = _LLM()
    return service


_PROFILES = [{"realname": f"甲{i}", "profession": "工程师"} for i in range(5)]


def test_agent_selection_failure_raises_instead_of_picking_the_first_n():
    service = _selector(error=RuntimeError("llm down"))
    with pytest.raises(InterviewSelectionError) as error:
        service._select_agents_for_interview(_PROFILES, "需求", "背景", 3)
    assert "llm down" in str(error.value)
    assert error.value.__cause__ is not None


def test_agent_selection_without_reasoning_is_rejected():
    service = _selector(llm_result={"selected_indices": [0, 1]})
    with pytest.raises(InterviewSelectionError, match="选择理由"):
        service._select_agents_for_interview(_PROFILES, "需求", "背景", 3)


def test_agent_selection_with_empty_indices_is_rejected():
    service = _selector(llm_result={"selected_indices": [], "reasoning": "理由"})
    with pytest.raises(InterviewSelectionError, match="没有选出"):
        service._select_agents_for_interview(_PROFILES, "需求", "背景", 3)


def test_agent_selection_returns_real_reasoning():
    service = _selector(llm_result={"selected_indices": [0, 2], "reasoning": "前者是当事人，后者是HR"})
    agents, indices, reasoning = service._select_agents_for_interview(
        _PROFILES, "需求", "背景", 3
    )
    assert indices == [0, 2]
    assert reasoning == "前者是当事人，后者是HR"
    assert [a["realname"] for a in agents] == ["甲0", "甲2"]


# ------------------------------------------------------------ N-09 等待预算


def test_interview_timeout_budget_scales_with_workload():
    """10 次调用（5 项 × 2 平台）不该被固定 180 秒卡死。"""

    assert Config.INTERVIEW_TIMEOUT_SECONDS >= 600
    assert Config.INTERVIEW_TIMEOUT_PER_CALL_SECONDS >= 60

    from app.services.simulation_runner import SimulationRunner
    import inspect

    source = inspect.signature(SimulationRunner.interview_agents_batch).parameters["timeout"]
    assert source.default is None, "缺省必须按工作量计算，不能再用魔法常数"
