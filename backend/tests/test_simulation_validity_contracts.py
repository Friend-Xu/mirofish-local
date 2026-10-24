"""模拟有效性门禁契约测试：人设、事件配置、采访计数。

对应规约：`MiroFish/AGENTS.md §1.1`；
掩盖台账条目：A2 / A8 / A5（N 族与 M-06 相关）。

这三条决定的是"这场模拟是否成立"：
- 人设是编的 → Agent 的发帖内容与传播权重没有依据；
- 没有种子事件 → 模拟自己凭空启动，报告仍在解释它；
- 采访计数按请求数统计 → "没回答"被算作"已采访"，采访证据是假的。
"""

from types import SimpleNamespace

import pytest

from app.services.oasis_profile_generator import (
    OasisProfileGenerator,
    ProfileGenerationError,
)
from app.services.simulation_config_generator import (
    AgentActivityConfig,
    ConfigGenerationError,
    EventConfig,
    SimulationConfigGenerator,
)


# ------------------------------------------------------------------ A2 人设


class _FakeCompletions:
    def __init__(self, payload, finish_reason="stop"):
        self.payload = payload
        self.finish_reason = finish_reason

    def create(self, **kwargs):  # noqa: ANN003
        message = SimpleNamespace(content=self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)]
        )


def _generator(payload, finish_reason="stop") -> OasisProfileGenerator:
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(
        payload, finish_reason
    )))
    generator.client = client
    generator.model_name = "test-model"
    generator.zep_client = None
    generator.graph_id = None
    return generator


def test_profile_generation_failure_raises_instead_of_rule_fallback():
    """三次都拿不到人设就是失败：不允许换成规则生成（更不允许 random 现编）。"""

    generator = _generator("not json at all")
    with pytest.raises(ProfileGenerationError) as error:
        generator._generate_profile_with_llm(
            entity_name="张三",
            entity_type="Person",
            entity_summary="摘要",
            entity_attributes={},
            context="",
        )
    assert "张三" in str(error.value)


def test_truncated_profile_output_is_not_repaired():
    """截断不是运气问题：不允许盲补括号把截断当成功。"""

    generator = _generator('{"bio": "被截断的一', finish_reason="length")
    with pytest.raises(ProfileGenerationError) as error:
        generator._generate_profile_with_llm(
            entity_name="李四",
            entity_type="Person",
            entity_summary="摘要",
            entity_attributes={},
            context="",
        )
    assert "截断" in str(error.value)


def test_profile_missing_required_text_fields_is_rejected():
    generator = _generator('{"bio": "只有简介"}')
    with pytest.raises(ProfileGenerationError) as error:
        generator._generate_profile_with_llm(
            entity_name="王五",
            entity_type="Person",
            entity_summary="摘要",
            entity_attributes={},
            context="",
        )
    assert "persona" in str(error.value)


def test_rule_based_profile_generator_is_gone():
    """旧名字必须不存在：留着就会被重新接到失败路径上。"""

    assert not hasattr(OasisProfileGenerator, "_generate_profile_rule_based")
    assert not hasattr(OasisProfileGenerator, "_try_fix_json")
    assert not hasattr(OasisProfileGenerator, "_fix_truncated_json")


def test_without_llm_profile_is_deterministic():
    """显式无 LLM 模式允许存在，但必须是确定性的（可复现）。"""

    generator = _generator("{}")
    first = generator._generate_profile_without_llm("某工程师", "Person", "摘要", {})
    second = generator._generate_profile_without_llm("某工程师", "Person", "摘要", {})
    assert first == second
    assert first["age"] and first["gender"] and first["mbti"] and first["country"]


def test_batch_generation_fails_instead_of_fabricating_a_spare_profile():
    """批量生成里也不许"造一个备用人设凑数"。"""

    generator = _generator("{}")

    def boom(entity, user_id, use_llm=True):  # noqa: ANN001
        raise ProfileGenerationError("模拟失败")

    generator.generate_profile_from_entity = boom
    entity = SimpleNamespace(name="某实体", uuid="u-1", summary="摘要", get_entity_type=lambda: "Person")
    with pytest.raises(ProfileGenerationError):
        generator.generate_profiles_from_entities(
            entities=[entity], use_llm=True, parallel_count=1
        )


# -------------------------------------------------------------- A8 事件配置


def _config_generator() -> SimulationConfigGenerator:
    return SimulationConfigGenerator.__new__(SimulationConfigGenerator)


def test_empty_initial_posts_is_rejected():
    generator = _config_generator()
    with pytest.raises(ConfigGenerationError, match="没有种子事件"):
        generator._parse_event_config({"initial_posts": [], "hot_topics": []})


def test_post_without_content_or_poster_type_is_rejected():
    generator = _config_generator()
    with pytest.raises(ConfigGenerationError, match="没有内容"):
        generator._parse_event_config({"initial_posts": [{"poster_type": "Person"}]})
    with pytest.raises(ConfigGenerationError, match="poster_type"):
        generator._parse_event_config({"initial_posts": [{"content": "内容"}]})


def test_valid_event_config_passes():
    generator = _config_generator()
    config = generator._parse_event_config({
        "initial_posts": [{"content": "内容", "poster_type": "Person"}],
        "hot_topics": ["话题"],
        "narrative_direction": "方向",
    })
    assert len(config.initial_posts) == 1
    assert config.hot_topics == ["话题"]


def _agent(agent_id: int, entity_type: str, influence: float = 1.0):
    return AgentActivityConfig(
        agent_id=agent_id,
        entity_uuid=f"uuid-{agent_id}",
        entity_name="某人",
        entity_type=entity_type,
        activity_level=0.5,
        posts_per_hour=1.0,
        comments_per_hour=1.0,
        active_hours=[9],
        response_delay_min=1,
        response_delay_max=5,
        influence_weight=influence,
    )


def test_unmatched_poster_type_is_rejected_instead_of_using_top_agent():
    """poster_type 匹配不上时不允许换成"影响力最高的 Agent"顶替。"""

    generator = _config_generator()
    event_config = EventConfig(
        initial_posts=[{"content": "内容", "poster_type": "Alien"}],
    )
    with pytest.raises(ConfigGenerationError) as error:
        generator._assign_initial_post_agents(event_config, [_agent(1, "Person", 3.0)])
    assert "alien" in str(error.value).lower()
    assert "person" in str(error.value).lower(), "错误里要给出可用类型"


def test_matching_poster_type_is_assigned():
    generator = _config_generator()
    event_config = EventConfig(
        initial_posts=[{"content": "内容", "poster_type": "Person"}],
    )
    updated = generator._assign_initial_post_agents(event_config, [_agent(7, "Person")])
    assert updated.initial_posts[0]["poster_agent_id"] == 7


# -------------------------------------------------------------- A5 采访计数


class _TraceDB:
    """最小 sqlite 替身：只回答 trace 查询。"""

    def __init__(self, row):
        self.row = row

    def cursor(self):  # noqa: ANN201
        return self

    def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def fetchone(self):  # noqa: ANN201
        return self.row

    def close(self):  # noqa: ANN201
        return None


def _runner_for_interview(tmp_path, row):
    """构造一个只带 simulation_dir 的 IPC handler（不启动任何环境）。"""

    from scripts.run_parallel_simulation import ParallelIPCHandler

    runner = ParallelIPCHandler.__new__(ParallelIPCHandler)
    runner.simulation_dir = str(tmp_path)
    (tmp_path / "twitter_simulation.db").write_bytes(b"")
    return runner


def test_interview_without_response_is_none_not_success(tmp_path, monkeypatch):
    """trace 里有记录但没有 response 字段时，不能拿 info 当回答。"""

    import sqlite3

    runner = _runner_for_interview(tmp_path, None)

    def fake_connect(_path):
        return _TraceDB((1, '{"prompt": "问了什么"}', "2026-01-01"))

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    result = runner._get_interview_result(7, "twitter")
    assert result["response"] is None, "没有 response 就是没有回答"


def test_interview_with_response_is_returned(tmp_path, monkeypatch):
    import sqlite3

    runner = _runner_for_interview(tmp_path, None)

    def fake_connect(_path):
        return _TraceDB((1, '{"response": "我的看法是……"}', "2026-01-01"))

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    result = runner._get_interview_result(7, "twitter")
    assert result["response"] == "我的看法是……"
