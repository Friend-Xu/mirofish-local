"""报告正文不得残留工具调用协议标记（掩盖台账 N-40）。

实测证据：`backend/uploads/reports/report_69379f275a5f/section_03.md`（587 字节）
正文只有两句，后面直接跟着一段模型自己拼装的工具调用标记；报告仍 `completed`。
真因是解析器只认提示词教的 `<tool_call>{...}</tool_call>`，而 DeepSeek 端点
原生吐的是自己的特殊标记（全角竖线包裹的 `DSML`），于是这次工具调用连
"存在过"都看不见，整段被 `split("Final Answer:")[-1]` 切下来当正文落盘。

同一份 `agent_log.jsonl` 里 24 条 `llm_response` 有 23 条带这类标记——
它不是偶发，是这条链路一直在发生。

对应规约：`MiroFish/AGENTS.md §1.1`（空产物算成功；失败必须是失败）。
"""

from types import MethodType, SimpleNamespace

import pytest

from app.services.report_agent import (
    ReportAgent,
    ReportManager,
    ReportOutline,
    ReportSection,
    SectionGenerationError,
    _first_tool_call_end,
    _protocol_markers_in,
    _require_section_content,
)

# 实测方言的分隔符是全角竖线 U+FF5C。这里显式拼出来，保证测试文件本身
# 读起来就知道它对准的是哪个字符。
BAR = "｜｜"
DSML = f"<{BAR}DSML{BAR} "

TOOL_CALL = (
    '<tool_call>{"name": "quick_search", "parameters": {"query": "测试"}}</tool_call>'
)
TOOL_RESULT = "(工具返回的模拟事实)"

# 实测原文（report_69379f275a5f/section_03.md 去掉首行标题后的正文）
LEAKED_SECTION_BODY = (
    "我已有丰富的一手采访数据。让我再做一次全景检索，确认外部三方"
    "（猎头池、招聘官、舆论场）的分歧全貌与演变。\n\n"
    f'{DSML}calls>\n'
    f'{DSML}invoke name="panorama_search">\n'
    f'{DSML}parameter name="parameters": '
    '{"query": "猎头可推荐池 头部设备厂招聘官 职场舆论场", "include_expired": true}}\n'
    "</tool_call>"
)

LEAKED_RESPONSE = "Final Answer:\n\n" + LEAKED_SECTION_BODY


class _ScriptedLLM:
    """按脚本依次返回响应；脚本只剩一项时重复返回它。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.messages_seen = []

    def chat(self, messages=None, temperature=None, max_tokens=None):  # noqa: ANN001
        self.calls += 1
        self.messages_seen.append(list(messages or []))
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]


def _fake_agent(llm):
    agent = SimpleNamespace(
        llm=llm,
        simulation_requirement="测试需求",
        MAX_TOOL_CALLS_PER_SECTION=5,
        VALID_TOOL_NAMES=ReportAgent.VALID_TOOL_NAMES,
        report_logger=None,
        _execute_tool=lambda name, params, report_context=None: TOOL_RESULT,
        _get_tools_description=lambda: "- quick_search: 简易检索",
    )
    for name in (
        "_parse_tool_calls",
        "_parse_dsml_tool_calls",
        "_is_valid_tool_call",
    ):
        setattr(agent, name, MethodType(getattr(ReportAgent, name), agent))
    agent._parse_dsml_parameter = ReportAgent._parse_dsml_parameter
    return agent


def _generate(script):
    llm = _ScriptedLLM(script)
    section = ReportSection(title="测试章节")
    outline = ReportOutline(title="测试报告", summary="摘要", sections=[section])
    agent = _fake_agent(llm)
    content = ReportAgent._generate_section_react(
        agent,
        section=section,
        outline=outline,
        previous_sections=[],
        progress_callback=None,
        section_index=1,
    )
    return content, llm


# ------------------------------------------------------------- 标记检测


@pytest.mark.parametrize(
    "marker",
    [
        "<tool_call>",
        "</tool_call>",
        "<tool_result>",
        "</tool_result>",
        f"{DSML}calls>",
        f"{DSML}invoke",
        # 半角退化形式
        "<||DSML|| calls>",
        # 单枚竖线的退化形式
        "<｜DSML｜ invoke",
    ],
)
def test_protocol_markers_are_detected(marker):
    assert _protocol_markers_in(f"正文在前。\n\n{marker} name=\"x\">") != []


@pytest.mark.parametrize(
    "text",
    [
        "这是一段正常正文，讨论了两平台的分歧与演变。",
        "正文里可以提到 tool call 这个词，也可以写 < 和 > 符号。",
        "- 列表项一\n- 列表项二\n\n> 引用一段话",
    ],
)
def test_normal_prose_has_no_markers(text):
    assert _protocol_markers_in(text) == []


# --------------------------------------------------------- 唯一校验点（单元）


def test_validator_rejects_residual_markers():
    """残骸进正文即失败——不清洗、不放行。"""

    with pytest.raises(SectionGenerationError) as error:
        _require_section_content(
            LEAKED_SECTION_BODY,
            section_title="行业外部视角",
            section_index=3,
            reason="测试用原因",
            raw_response=LEAKED_RESPONSE,
            tool_calls_count=3,
        )
    message = str(error.value)
    assert "行业外部视角" in message and "index=3" in message
    assert "工具调用标记" in message
    assert "DSML" in message, "异常里必须写明命中的是哪种标记"


def test_validator_still_rejects_empty_content():
    with pytest.raises(SectionGenerationError):
        _require_section_content(
            "   \n\t ",
            section_title="空章节",
            section_index=1,
            reason="测试用原因",
            raw_response="   ",
            tool_calls_count=0,
        )


def test_validator_passes_clean_content():
    assert (
        _require_section_content(
            "## 小节\n\n这是正常正文。",
            section_title="章节",
            section_index=1,
            reason="测试用原因",
            raw_response="Final Answer: 正文",
            tool_calls_count=3,
        )
        == "## 小节\n\n这是正常正文。"
    )


# ----------------------------------------------------- 方言解析（单元）


def test_dsml_attribute_form_is_parsed():
    """实测形态：`parameter name="parameters": {json}`。"""

    response = (
        f'{DSML}calls>\n'
        f'{DSML}invoke name="panorama_search">\n'
        f'{DSML}parameter name="parameters": '
        '{"query": "猎头池", "include_expired": true}}\n'
        "</tool_call>"
    )
    calls = _fake_agent(_ScriptedLLM([""]))._parse_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "panorama_search"
    # 提示词的 parameters 包装层必须被展开，而不是套两层
    assert calls[0]["parameters"] == {"query": "猎头池", "include_expired": True}


def test_dsml_element_form_is_parsed():
    """实测形态：`parameter name="query" string="true">值</parameter>`。"""

    response = (
        f'{DSML}calls>\n'
        f'{DSML}invoke name="quick_search">\n'
        f'{DSML}parameter name="query" string="true">广川科技组织阶段｜晋升名额\n'
        f'{DSML}parameter name="limit" string="false">12\n'
        f'{DSML}invoke>\n'
        f'{DSML}calls>'
    )
    calls = _fake_agent(_ScriptedLLM([""]))._parse_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "quick_search"
    assert calls[0]["parameters"]["query"].startswith("广川科技组织阶段")
    assert calls[0]["parameters"]["limit"] == "12"


def test_dsml_with_unknown_tool_name_yields_nothing():
    """认不出来就返回空——让残留标记走"正文不得含标记"的失败路径。"""

    response = f'{DSML}invoke name="not_a_tool">\n{DSML}parameter name="q": "x"}}\n</tool_call>'
    assert _fake_agent(_ScriptedLLM([""]))._parse_tool_calls(response) == []


def test_tool_call_block_with_stray_closing_tag_is_parsed():
    """实测形态：JSON 后面多一行 `</parameter>`，旧正则因此整块失配。"""

    response = (
        "<tool_call>\n"
        '{"name": "insight_forge", "parameters": {"query": "错配点"}}\n'
        "</parameter>\n"
        "</tool_call>"
    )
    calls = _fake_agent(_ScriptedLLM([""]))._parse_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "insight_forge"


def test_xml_form_still_works():
    calls = _fake_agent(_ScriptedLLM([""]))._parse_tool_calls(TOOL_CALL)
    assert calls == [{"name": "quick_search", "parameters": {"query": "测试"}}]


# --------------------------------------------- 块结束位置（冲突降级用）


@pytest.mark.parametrize(
    "response",
    [
        "思考\n" + TOOL_CALL + "\nFinal Answer:\n\n正文",
        f'{DSML}invoke name="quick_search">\n{DSML}parameter name="q": "x"}}\n'
        f"{DSML}invoke>\n{DSML}calls>\nFinal Answer:\n\n正文",
    ],
)
def test_first_tool_call_end_covers_both_dialects(response):
    """旧实现写死 `</tool_call>`，方言形式一个都找不到。"""

    end = _first_tool_call_end(response)
    assert end != -1
    kept = response[:end]
    assert "Final Answer:" not in kept


def test_first_tool_call_end_returns_minus_one_when_absent():
    assert _first_tool_call_end("只有正文，没有工具调用。") == -1


# ------------------------------------------------- 真实循环（集成）


def test_leaked_response_is_never_adopted_as_section_content():
    """复现 report_69379f275a5f：模型写了正文又接一段方言调用。

    修复前：`split("Final Answer:")[-1]` 把残骸切下来当正文返回。
    修复后：这一轮被识别为"工具调用 + Final Answer 冲突"，残骸没有机会
    成为正文；反复冲突到底则整章失败，报告按失败处理。
    """

    with pytest.raises(SectionGenerationError) as error:
        _generate([LEAKED_RESPONSE])
    message = str(error.value)
    assert "测试章节" in message
    assert "工具调用标记" in message


def test_leaked_response_after_enough_tool_calls_is_a_conflict_not_content():
    """真实现场的顺序：工具调用已凑够，下一轮才是"正文 + 方言调用"。

    这一步很关键：只有 `tool_calls_count >= min_tool_calls` 时，"Final Answer"
    才会被直接采纳。实测 section_03 正是在这个位置上泄露的。
    变异验证（撤掉方言解析）时，这条会退化成"残骸被当成正文返回"。
    """

    content, llm = _generate(
        [
            TOOL_CALL,
            TOOL_CALL,
            TOOL_CALL,
            LEAKED_RESPONSE,
            "Final Answer:\n\n## 小节\n\n这是干净的正文。",
        ]
    )
    assert "这是干净的正文。" in content
    assert _protocol_markers_in(content) == []

    conflict_notice = "同时包含了工具调用和 Final Answer"
    seen = [
        str(message.get("content", ""))
        for turn in llm.messages_seen
        for message in turn
    ]
    assert any(conflict_notice in text for text in seen), (
        "第 4 轮应被识别为工具调用 + Final Answer 冲突并重问，"
        "而不是被当成正常收尾"
    )


def test_conflict_response_is_not_echoed_into_content():
    """模型先冲突一次，再正常回答：正文必须是后者，且不含任何标记。"""

    content, llm = _generate(
        [
            LEAKED_RESPONSE,
            TOOL_CALL,
            TOOL_CALL,
            TOOL_CALL,
            "Final Answer:\n\n## 小节\n\n这是干净的正文。",
        ]
    )
    assert "这是干净的正文。" in content
    assert _protocol_markers_in(content) == []
    assert llm.calls > 4, "第一轮应被当作冲突处理而不是直接采纳"


def test_dsml_tool_call_is_executed_instead_of_ignored():
    """方言调用现在会被真的执行，工具计数因此前进。"""

    content, llm = _generate(
        [
            f'{DSML}invoke name="quick_search">\n'
            f'{DSML}parameter name="query" string="true">测试查询\n',
            TOOL_CALL,
            TOOL_CALL,
            "Final Answer:\n\n## 小节\n\n正文。",
        ]
    )
    assert "正文。" in content
    joined = "\n".join(
        str(message.get("content", ""))
        for call in llm.messages_seen[-1]
        for message in [call]
    )
    assert "quick_search" in joined or "测试查询" in joined


# --------------------------------------------------------- 落盘边界


def test_save_section_refuses_residual_markers(monkeypatch, tmp_path):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    section = ReportSection(title="脏章节", content=LEAKED_SECTION_BODY)
    with pytest.raises(SectionGenerationError) as error:
        ReportManager.save_section("report_dirty", 3, section)
    assert "脏章节" in str(error.value)
    assert not (tmp_path / "report_dirty" / "section_03.md").exists(), (
        "拒绝落盘时不允许留下文件——实测那份 587 字节的残骸就是这么留下的"
    )
