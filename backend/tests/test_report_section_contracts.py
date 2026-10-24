"""报告可信链契约测试：空章节不允许落盘，报告不得以成功收尾。

背景（实测证据）：`backend/uploads/reports/report_3ca0c1d57c5f/section_05.md`
只有 85 字节（一行标题），`meta.json` 里该章 `content` 是空串，而报告
`status: "completed"`、`error: null`。根因是 LLM 的空响应被当成章节正文采纳
（`report_agent.py` 的"情况 3"分支），下游没有任何非空校验。

对应规约：`MiroFish/AGENTS.md §1.1`（空产物算成功）；
掩盖台账条目：N-01 / M-03 / N-04。
"""

from types import MethodType, SimpleNamespace

import pytest

from app.config import Config
from app.services.report_agent import (
    OutlinePlanningError,
    ReportAgent,
    ReportManager,
    ReportOutline,
    ReportSection,
    SectionGenerationError,
    _require_section_content,
)

TOOL_CALL = (
    '<tool_call>{"name": "quick_search", "parameters": {"query": "测试"}}</tool_call>'
)
TOOL_RESULT = "(工具返回的模拟事实)"


class _ScriptedLLM:
    """按脚本依次返回响应；脚本只剩一项时重复返回它。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.max_tokens_seen = []

    def chat(self, messages=None, temperature=None, max_tokens=None):  # noqa: ANN001
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]


def _fake_agent(llm):
    """够 `_generate_section_react` 跑完一轮循环的最小对象。"""

    agent = SimpleNamespace(
        llm=llm,
        simulation_requirement="测试需求",
        MAX_TOOL_CALLS_PER_SECTION=5,
        report_logger=None,
        _execute_tool=lambda name, params, report_context=None: TOOL_RESULT,
        _get_tools_description=lambda: "- quick_search: 简易检索",
    )
    agent.VALID_TOOL_NAMES = ReportAgent.VALID_TOOL_NAMES
    agent._parse_tool_calls = MethodType(ReportAgent._parse_tool_calls, agent)
    agent._parse_dsml_tool_calls = MethodType(ReportAgent._parse_dsml_tool_calls, agent)
    agent._is_valid_tool_call = MethodType(ReportAgent._is_valid_tool_call, agent)
    agent._parse_dsml_parameter = ReportAgent._parse_dsml_parameter
    return agent


def _generate(script):
    return _generate_with(_ScriptedLLM(script))


def _generate_with(llm):
    section = ReportSection(title="测试章节")
    outline = ReportOutline(title="测试报告", summary="摘要", sections=[section])
    agent = _fake_agent(llm)
    return ReportAgent._generate_section_react(
        agent,
        section=section,
        outline=outline,
        previous_sections=[],
        progress_callback=None,
        section_index=1,
    )


# --------------------------------------------------------- 校验点（单元）


def test_empty_content_is_rejected_by_the_single_validator():
    with pytest.raises(SectionGenerationError) as error:
        _require_section_content(
            "",
            section_title="空章节",
            section_index=3,
            reason="测试用原因",
            raw_response="",
            tool_calls_count=2,
        )
    message = str(error.value)
    assert "空章节" in message and "index=3" in message
    assert "测试用原因" in message


def test_whitespace_only_content_is_rejected():
    with pytest.raises(SectionGenerationError):
        _require_section_content(
            "   \n\t ",
            section_title="空白章节",
            section_index=1,
            reason="测试用原因",
            raw_response="   \n\t ",
            tool_calls_count=0,
        )


def test_non_empty_content_passes_through():
    assert (
        _require_section_content(
            "正文内容",
            section_title="章节",
            section_index=1,
            reason="测试用原因",
            raw_response="Final Answer: 正文内容",
            tool_calls_count=3,
        )
        == "正文内容"
    )


# ------------------------------------------------- 空响应不得被当成正文（真实循环）


def test_empty_llm_response_is_not_adopted_as_section_content():
    """复现 report_3ca0c1d57c5f 的现场：工具调用够了，第 4 轮返回空串。"""

    with pytest.raises(SectionGenerationError) as error:
        _generate([TOOL_CALL, TOOL_CALL, TOOL_CALL, ""])
    message = str(error.value)
    assert "测试章节" in message
    assert "输出为空" in message
    assert "工具调用=3" in message


def test_final_answer_without_body_is_rejected():
    with pytest.raises(SectionGenerationError) as error:
        _generate([TOOL_CALL, TOOL_CALL, TOOL_CALL, "Final Answer:"])
    assert "Final Answer:" in str(error.value)


def test_force_finish_returning_none_raises_instead_of_placeholder():
    """强制收尾返回 None 时，占位文案只允许进日志，不允许当章节内容。"""

    with pytest.raises(SectionGenerationError) as error:
        _generate([None, None, None, None, None])
    message = str(error.value)
    assert "强制收尾" in message
    assert "测试章节" in message


def test_non_empty_final_answer_is_returned():
    content = _generate(
        [TOOL_CALL, TOOL_CALL, TOOL_CALL, "Final Answer:\n\n## 小节\n\n这是正文。"]
    )
    assert "这是正文。" in content
    assert content.strip()


# ------------------------------------------------- 章节输出上限来自配置（N-30 后果）


def test_section_react_uses_the_configured_token_cap():
    """章节生成的 `max_tokens` 必须来自 `Config.REPORT_SECTION_MAX_TOKENS`。

    4096 会把推理 token 与正文一起卡断（实测第 3 章只剩 122 字，N-30），
    所以这里的字面量本身就是缺陷。
    """

    from app.config import Config

    llm = _ScriptedLLM(
        [TOOL_CALL, TOOL_CALL, TOOL_CALL, "Final Answer:\n\n## 小节\n\n这是正文。"]
    )
    _generate_with(llm)

    assert llm.max_tokens_seen, "没有向 LLM 传过 max_tokens"
    assert set(llm.max_tokens_seen) == {Config.REPORT_SECTION_MAX_TOKENS}


def test_force_finish_uses_the_configured_token_cap():
    """强制收尾是第二个调用点，同样不允许写死 4096。"""

    from app.config import Config

    llm = _ScriptedLLM([None])
    with pytest.raises(SectionGenerationError):
        _generate_with(llm)

    assert llm.max_tokens_seen[-1] == Config.REPORT_SECTION_MAX_TOKENS


# --------------------------------------------------------- 落盘边界


def test_save_section_refuses_empty_content(monkeypatch, tmp_path):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    section = ReportSection(title="空章节", content="")
    with pytest.raises(SectionGenerationError) as error:
        ReportManager.save_section("report_empty", 1, section)
    assert "空章节" in str(error.value)
    assert not (tmp_path / "report_empty" / "section_01.md").exists(), (
        "拒绝落盘时不允许留下任何文件（历史上会留下只有标题的 85 字节文件）"
    )


def test_save_section_writes_non_empty_content(monkeypatch, tmp_path):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    section = ReportSection(title="正常章节", content="正文内容")
    path = ReportManager.save_section("report_ok", 2, section)
    written = (tmp_path / "report_ok" / "section_02.md").read_text(encoding="utf-8")
    assert "## 正常章节" in written
    assert "正文内容" in written
    assert path.endswith("section_02.md")


# --------------------------------------------------------- 大纲失败

_CONTEXT = {
    "graph_statistics": {"total_nodes": 1, "total_edges": 2, "entity_types": {}},
    "total_entities": 1,
    "related_facts": [],
}


def _fake_agent_for_outline(error):
    class _LLM:
        def chat_json(self, messages=None, temperature=None, max_tokens=None):  # noqa: ANN001
            raise error

    return SimpleNamespace(
        llm=_LLM(),
        graph_id="graph_test",
        simulation_requirement="测试需求",
        zep_tools=SimpleNamespace(get_simulation_context=lambda **_: dict(_CONTEXT)),
    )


def test_plan_outline_failure_raises_instead_of_hardcoded_template():
    cause = RuntimeError("endpoint down")
    agent = _fake_agent_for_outline(cause)
    with pytest.raises(OutlinePlanningError) as error:
        ReportAgent.plan_outline(agent, progress_callback=None)
    message = str(error.value)
    assert "endpoint down" in message, "原始失败必须出现在异常里"
    assert error.value.__cause__ is cause, "必须保留 cause"
    assert "未来预测报告" not in message, "不允许再退回硬编码三章模板"


def test_plan_outline_uses_explicit_output_budget():
    """大纲也是 JSON 输出；不能再用 chat_json 默认 4096（reasoning 会先吃满）。"""

    seen = {}

    class _LLM:
        def chat_json(self, messages=None, temperature=None, max_tokens=None):  # noqa: ANN001
            seen["max_tokens"] = max_tokens
            return {
                "title": "测试报告",
                "summary": "摘要",
                "sections": [{"title": "第一章"}],
            }

    agent = SimpleNamespace(
        llm=_LLM(),
        graph_id="graph_test",
        simulation_requirement="测试需求",
        zep_tools=SimpleNamespace(get_simulation_context=lambda **_: dict(_CONTEXT)),
    )
    outline = ReportAgent.plan_outline(agent, progress_callback=None)
    assert outline.title == "测试报告"
    assert seen["max_tokens"] == Config.REPORT_SECTION_MAX_TOKENS
