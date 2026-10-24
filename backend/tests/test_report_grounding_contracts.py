"""N-44/N-45 报告章节证据校验契约。

锁住四件事：
1. 事实块/推断块的结构与标签；
2. 事实块引号外的程度词、量化词、推断词必须失败；
3. 引号内必须是工具结果里的逐字内容；译文/转述有显式出口；
4. 校验器只返回违规，不做清洗/截断；修复由生成侧一次纠正完成。
"""

from types import MethodType, SimpleNamespace

import pytest

from app.services.report_agent import (
    SECTION_SYSTEM_PROMPT_TEMPLATE,
    GroundingValidationError,
    Report,
    ReportAgent,
    ReportStatus,
)
from app.services.report_grounding import (
    FACT_LABEL,
    INFERENCE_LABEL,
    INFERENCE_PREFIX,
    GroundingViolation,
    build_correction_prompt,
    validate_section_grounding,
)

TOOL = "工具返回：该工程师在 24 个月内拿到 EFEM 子模块负责权，颗粒数达标。"


def _valid_section() -> str:
    return (
        f"{FACT_LABEL}\n\n"
        "> \"该工程师在 24 个月内拿到 EFEM 子模块负责权，颗粒数达标。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这说明他的负责权已经进入可验收阶段。"
    )


def test_valid_section_passes():
    assert validate_section_grounding(_valid_section(), [TOOL]) == []


def test_missing_fact_block_fails():
    content = f"{INFERENCE_LABEL}\n\n{INFERENCE_PREFIX} 只有推断。"
    codes = {item.code for item in validate_section_grounding(content, [TOOL])}
    assert "missing_fact_block" in codes


def test_prose_outside_blocks_fails():
    content = "这段正文没有放进任何块。\n\n" + _valid_section()
    codes = {item.code for item in validate_section_grounding(content, [TOOL])}
    assert "unlabeled_prose" in codes


def test_strength_word_outside_quotes_in_fact_block_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "这是最整齐的一次交付。\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "unlabeled_strength" for item in violations)


def test_inference_marker_outside_quotes_in_fact_block_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "颗粒数达标，这意味着工艺已经锁死。\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "unlabeled_inference" for item in violations)


def test_inference_line_without_prefix_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "> \"该工程师在 24 个月内拿到 EFEM 子模块负责权，颗粒数达标。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        "这说明他的负责权已经进入可验收阶段。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "inference_missing_prefix" for item in violations)


def test_quote_not_in_tool_results_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "> \"这是一句工具结果里根本没有的话。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "quote_not_verbatim" for item in violations)


def test_translation_marker_skips_verbatim_check():
    content = (
        f"{FACT_LABEL}\n\n"
        "> [译文] \"This is a translation not present in the tool result.\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    # 译文明确标注，允许与工具结果不同；不得再报 quote_not_verbatim。
    assert validate_section_grounding(content, [TOOL]) == []


def test_paraphrase_marker_must_not_use_quotes():
    content = (
        f"{FACT_LABEL}\n\n"
        "> [转述] \"这是概括，不是原话。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "paraphrase_in_quotes" for item in violations)


def test_lead_in_sentence_in_fact_block_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "三条关键信号如下：\n\n"
        "> \"该工程师在 24 个月内拿到 EFEM 子模块负责权，颗粒数达标。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "lead_in_in_fact_block" for item in violations)


def test_heading_in_section_fails():
    content = (
        f"{FACT_LABEL}\n\n"
        "## 小标题\n\n"
        "> \"该工程师在 24 个月内拿到 EFEM 子模块负责权，颗粒数达标。\"\n"
    )
    violations = validate_section_grounding(content, [TOOL])
    assert any(item.code == "heading_not_allowed" for item in violations)


def test_ellipsis_quote_checks_each_segment():
    tool = "原文：第一段完整事实内容。第二段完整事实内容。"
    content = (
        f"{FACT_LABEL}\n\n"
        "> \"第一段完整事实内容……第二段被改写了内容。\"\n\n"
        f"{INFERENCE_LABEL}\n\n"
        f"{INFERENCE_PREFIX} 这是判断。"
    )
    violations = validate_section_grounding(content, [tool])
    assert any(item.code == "quote_not_verbatim" for item in violations)


def test_violation_is_typed_and_correction_prompt_contains_rules():
    violations = validate_section_grounding("没有块的正文。", [])
    assert violations and isinstance(violations[0], GroundingViolation)
    prompt = build_correction_prompt(violations)
    assert "模拟事实" in prompt
    assert "推断" in prompt
    assert "引号" in prompt


# ── 生成流程：一次纠正后通过 / 仍失败即异常 ──────────────────────

INVALID = "这段正文没有放进任何块。"


def _grounded_agent(first: str, revised: str):
    agent = SimpleNamespace(
        report_logger=None,
        simulation_requirement="测试需求",
        _get_tools_description=lambda: "- quick_search",
    )
    calls = {"revise": 0}

    def fake_generate(**kwargs):  # noqa: ANN003
        kwargs["tool_results_sink"].append(TOOL)
        return first

    def fake_revise(**kwargs):  # noqa: ANN003
        calls["revise"] += 1
        return revised

    agent._generate_section_react = fake_generate
    agent._revise_section_grounding = fake_revise
    agent._generate_grounded_section = MethodType(
        ReportAgent._generate_grounded_section, agent
    )
    return agent, calls


def _section():
    from app.services.report_agent import ReportOutline, ReportSection

    section = ReportSection(title="测试章节")
    outline = ReportOutline(title="测试报告", summary="摘要", sections=[section])
    return section, outline


def test_grounded_section_passes_without_correction():
    agent, calls = _grounded_agent(_valid_section(), _valid_section())
    section, outline = _section()
    content, stats = agent._generate_grounded_section(
        section=section, outline=outline, previous_sections=[], section_index=1
    )
    assert content == _valid_section()
    assert stats["attempts"] == 1
    assert stats["corrections"] == 0
    assert stats["final_ok"] is True
    assert calls["revise"] == 0


def test_grounded_section_corrects_once_then_passes():
    agent, calls = _grounded_agent(INVALID, _valid_section())
    section, outline = _section()
    content, stats = agent._generate_grounded_section(
        section=section, outline=outline, previous_sections=[], section_index=1
    )
    assert content == _valid_section()
    assert stats["attempts"] == 2
    assert stats["corrections"] == 1
    assert stats["final_ok"] is True
    assert calls["revise"] == 1
    assert stats["first_violations"]
    assert stats["final_violations"] == []


def test_grounded_section_raises_after_two_failed_corrections():
    agent, calls = _grounded_agent(INVALID, INVALID)
    section, outline = _section()
    with pytest.raises(GroundingValidationError) as exc:
        agent._generate_grounded_section(
            section=section, outline=outline, previous_sections=[], section_index=1
        )
    assert calls["revise"] == 2, "最多两次有界纠正，之后必须失败"
    assert exc.value.violations, "异常必须带最终违规项，不能只给一句消息"
    assert exc.value.stats["corrections"] == 2


def test_grounded_section_second_correction_can_pass():
    agent, calls = _grounded_agent(INVALID, INVALID)
    sequence = [INVALID, _valid_section()]

    def fake_revise(**kwargs):  # noqa: ANN003
        calls["revise"] += 1
        return sequence[min(calls["revise"] - 1, len(sequence) - 1)]

    agent._revise_section_grounding = fake_revise
    section, outline = _section()
    content, stats = agent._generate_grounded_section(
        section=section, outline=outline, previous_sections=[], section_index=1
    )
    assert content == _valid_section()
    assert calls["revise"] == 2
    assert stats["corrections"] == 2
    assert stats["final_ok"] is True


def test_prompt_contains_fact_inference_and_quote_rules():
    assert FACT_LABEL in SECTION_SYSTEM_PROMPT_TEMPLATE
    assert INFERENCE_LABEL in SECTION_SYSTEM_PROMPT_TEMPLATE
    assert INFERENCE_PREFIX in SECTION_SYSTEM_PROMPT_TEMPLATE
    assert "[译文]" in SECTION_SYSTEM_PROMPT_TEMPLATE
    assert "[转述]" in SECTION_SYSTEM_PROMPT_TEMPLATE


def test_report_to_dict_includes_grounding():
    report = Report(
        report_id="report_x",
        simulation_id="sim_x",
        graph_id="graph_x",
        simulation_requirement="需求",
        status=ReportStatus.COMPLETED,
        grounding={"final_ok": True, "corrections": 0, "sections": []},
    )
    payload = report.to_dict()
    assert payload["grounding"]["final_ok"] is True
