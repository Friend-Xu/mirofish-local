"""LLM 文本输出契约：被 token 上限截断的半截文本不许当内容。

对应掩盖台账 N-30：报告某一章生成时命中 `max_tokens`，返回 122 字的半截文本，
`_require_section_content` 只查"非空"所以放行，报告仍然 `completed`、`empty_sections=[]`——
用户拿到的是一份缺了一段却声称完整的报告。`chat_json` 早就在 `finish_reason=length`
时报错，文本路径漏了同一条规则。

后半部分锁"上限该多大"这件事：本端点（deepseek-flash）返回 `reasoning_content`，
推理 token 计入 `completion_tokens`（实测一句「只回答两个字」烧掉 54 个），
所以 4096 会把"推理 + 正文"一起卡断（N-30 的后果）。上限来自
`Config.REPORT_SECTION_MAX_TOKENS`，且截断错误必须带上用量——调多大要是数字，不是猜。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Config  # noqa: E402
from app.utils.llm_client import LLMClient, LLMResponseError  # noqa: E402


def _client_with_response(monkeypatch, *, content: str, finish_reason, usage=None):
    client = LLMClient.__new__(LLMClient)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=usage,
    )
    monkeypatch.setattr(client, "_create_completion", lambda **_kwargs: response)
    return client


def test_truncated_text_raises_instead_of_returning_half_content(monkeypatch):
    client = _client_with_response(
        monkeypatch, content="## 第三章\n\n这一章讲到一半就断了", finish_reason="length"
    )
    with pytest.raises(LLMResponseError) as error:
        client.chat(messages=[{"role": "user", "content": "写第三章"}], max_tokens=4096)
    assert error.value.finish_reason == "length"
    assert "截断" in str(error.value)


@pytest.mark.parametrize("finish_reason", ["content_filter", "tool_calls"])
def test_unexpected_stop_reason_raises(monkeypatch, finish_reason):
    client = _client_with_response(monkeypatch, content="文本", finish_reason=finish_reason)
    with pytest.raises(LLMResponseError):
        client.chat(messages=[{"role": "user", "content": "x"}])


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_normal_completion_returns_text(monkeypatch, finish_reason):
    client = _client_with_response(
        monkeypatch, content="  完成的正文  ", finish_reason=finish_reason
    )
    assert client.chat(messages=[{"role": "user", "content": "x"}]) == "完成的正文"


# --------------------------------------------------- 截断时的用量必须可见（N-30 后果）


def test_truncation_error_reports_usage_so_the_cap_can_be_sized(monkeypatch):
    """截断错误里要有 completion/reasoning token：否则"该调多大"只能靠猜。"""

    usage = SimpleNamespace(
        completion_tokens=4096,
        prompt_tokens=900,
        total_tokens=4996,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3100),
    )
    client = _client_with_response(
        monkeypatch, content="半截正文", finish_reason="length", usage=usage
    )

    with pytest.raises(LLMResponseError) as error:
        client.chat(messages=[{"role": "user", "content": "写第三章"}], max_tokens=4096)

    message = str(error.value)
    assert "completion_tokens=4096" in message
    assert "reasoning_tokens=3100" in message, "推理 token 计入上限，必须单独报出来"


def test_truncation_error_survives_a_missing_usage_block(monkeypatch):
    client = _client_with_response(
        monkeypatch, content="半截正文", finish_reason="length", usage=None
    )
    with pytest.raises(LLMResponseError) as error:
        client.chat(messages=[{"role": "user", "content": "x"}], max_tokens=4096)
    assert "usage=未知" in str(error.value)


# --------------------------------------------------- 章节上限来自配置（N-30 后果）


def test_section_token_cap_is_configured_above_the_old_truncating_value():
    assert Config.REPORT_SECTION_MAX_TOKENS > 4096, (
        "4096 会把推理 token 与正文一起卡断（实测第 3 章只剩 122 字）"
    )


def test_zero_section_token_cap_is_rejected_by_validation(monkeypatch):
    monkeypatch.setattr(Config, "REPORT_SECTION_MAX_TOKENS", 0)
    errors = Config.validate()
    assert any("REPORT_SECTION_MAX_TOKENS" in error for error in errors)
