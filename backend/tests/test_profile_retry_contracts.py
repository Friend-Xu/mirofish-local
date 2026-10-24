"""人设必需字段缺失必须重试、而不是让整场 prepare 报废（掩盖台账 N-39 / F12 现场）。

实测现场：35 个实体里有一个（行业分析师）返回的人设缺 `age/mbti/country`，
写盘边界（`_save_reddit_json`）按规矩拒绝补默认值 → **整场准备在 90 秒后失败**。
缺字段是"模型没按要求返回"，属于可重试的失败；补默认值则会让"没生成"看起来像
"生成了"（A 族缺陷）。正确做法是：同一条路径重试（有界、记录原因、把缺的字段写回提示），
三次仍缺才失败。

证据等级：本文件是**单测**（假 LLM 按脚本返回）。真实重试效果由 F12 复跑证明。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import oasis_profile_generator as module  # noqa: E402
from app.services.oasis_profile_generator import (  # noqa: E402
    OasisProfileGenerator,
    ProfileGenerationError,
)

COMPLETE = {
    "bio": "半导体设备机械工程师",
    "persona": "在上海广川科技做晶圆传输设备的机械结构设计。",
    "age": 31,
    "gender": "male",
    "mbti": "ISTJ",
    "country": "中国",
    "karma": 1200,
    "friend_count": 120,
    "follower_count": 340,
    "statuses_count": 260,
}


def _response(payload: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=payload),
            )
        ]
    )


def _generator(monkeypatch, payloads):
    import json

    calls = []

    def fake_completion(client, **kwargs):  # noqa: ANN001 - 对应真实签名的第一个位置参数
        calls.append(kwargs)
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return _response(json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(module, "create_chat_completion", fake_completion)
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    generator.client = object()
    generator.model_name = "test-model"
    return generator, calls


def _run(generator):
    return generator._generate_profile_with_llm(
        entity_name="行业分析师",
        entity_type="IndustryAnalyst",
        entity_summary="关注半导体设备行业的分析师",
        entity_attributes={},
        context="",
    )


def test_missing_fields_are_retried_then_succeed(monkeypatch):
    incomplete = {k: v for k, v in COMPLETE.items() if k not in ("age", "mbti", "country")}
    generator, calls = _generator(monkeypatch, [incomplete, COMPLETE])

    result = _run(generator)

    assert result == COMPLETE
    assert len(calls) == 2, "第一次缺字段应当重试，而不是直接失败"


def test_retry_prompt_names_the_missing_fields(monkeypatch):
    incomplete = {k: v for k, v in COMPLETE.items() if k in ("bio", "persona")}
    generator, calls = _generator(monkeypatch, [incomplete, COMPLETE])

    _run(generator)

    second_prompt = calls[1]["messages"][-1]["content"]
    assert "age" in second_prompt and "mbti" in second_prompt and "country" in second_prompt
    assert "上一次返回被拒绝" in second_prompt


def test_exhausted_retries_still_fail_with_the_missing_fields(monkeypatch):
    """重试不是无限兜底：三次都缺就必须失败，并写明缺了什么。"""

    incomplete = {k: v for k, v in COMPLETE.items() if k != "mbti"}
    generator, calls = _generator(monkeypatch, [incomplete])

    with pytest.raises(ProfileGenerationError) as error:
        _run(generator)

    assert len(calls) == 3
    assert "mbti" in str(error.value)


def test_required_fields_helper_covers_the_write_boundary_set():
    """重试判据必须覆盖写盘边界要求的字段，否则"重试成功"仍会在保存时失败。"""

    assert set(OasisProfileGenerator.REQUIRED_PROFILE_FIELDS) >= {
        "bio", "persona", "age", "gender", "mbti", "country",
    }
    missing = OasisProfileGenerator._missing_required_profile_fields(
        {"bio": "x", "persona": "y", "age": None, "gender": "male", "mbti": "", "country": "中国"}
    )
    assert missing == ["age", "mbti"]


def test_synthetic_metrics_are_not_part_of_the_required_set():
    """社交影响力字段不在必需集合里：它们由 `_resolve_oasis_required_fields` 确定性合成。"""

    assert not set(OasisProfileGenerator.OASIS_SYNTHETIC_METRICS) & set(
        OasisProfileGenerator.REQUIRED_PROFILE_FIELDS
    )
