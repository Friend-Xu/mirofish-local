"""L0 离线单测：Graphiti 客户端在结构化输出偶发失败时的有限重试。

不联网、不建库：用假 graphiti 对象驱动 `_add_episode_with_retry`。
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services import zep_graphiti_impl as impl


class ValidationError(Exception):
    """名字与 pydantic.ValidationError 一致，用于命中重试白名单。"""


class _FakeGraphiti:
    def __init__(self, failures: int, error: Exception | None = None):
        self.failures = failures
        self.calls = 0
        self.error = error or ValidationError("Field required")

    async def add_episode(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"ep-{self.calls}"))


def _client(graphiti) -> impl.GraphitiClient:
    client = object.__new__(impl.GraphitiClient)
    client._graphiti = graphiti
    client._initialized = True
    client._ontology_cache = {}
    client._graph_cache = {}
    client._last_ontology = {}
    client._ontology_cache["g"] = {
        "entity_types": None,
        "edge_types": None,
        "edge_type_map": None,
    }
    return client


def _call(client):
    return client._add_episode_with_retry(
        name="episode_x",
        data="文本",
        reference_time=datetime.now(timezone.utc),
        episode_type="text",
        graph_id="g",
        kwargs=client._ontology_kwargs("g"),
        source_description="test",
    )


def test_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr(impl.time, "sleep", lambda _seconds: None)
    fake = _FakeGraphiti(failures=1)

    result = _call(_client(fake))

    assert fake.calls == 2
    assert result.episode.uuid == "ep-2"


def test_gives_up_after_bounded_attempts(monkeypatch):
    monkeypatch.setattr(impl.time, "sleep", lambda _seconds: None)
    fake = _FakeGraphiti(failures=99)

    with pytest.raises(impl.GraphStoreError) as captured:
        _call(_client(fake))

    assert fake.calls == impl.GraphitiClient.ADD_EPISODE_ATTEMPTS
    assert "连续 3 次失败" in str(captured.value)


def test_non_retryable_error_is_not_replayed(monkeypatch):
    monkeypatch.setattr(impl.time, "sleep", lambda _seconds: None)
    fake = _FakeGraphiti(failures=99, error=RuntimeError("connection reset"))

    with pytest.raises(impl.GraphStoreError):
        _call(_client(fake))

    assert fake.calls == 1
