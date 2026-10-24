"""批大小守卫契约（掩盖台账 N-48）：**OASIS 写死的 1000 在 8 GB 卡上装不下**。

实测背景（`.tools/measure-recsys-round-cost.py`，4035 条语料、批内最长 267 token）：

* batch 30 / 32 / 64 / 100 → 峰值 1648 / 1684 / 2277 / 2945 MB（安全）；
* batch 256 / 512 → 把卡打成 `GPU is lost`（只能重启恢复）；
* **batch 1000（OASIS 生产值）→ `torch.OutOfMemoryError`：`13.69 GiB allocated`**，
  是 8 GB 卡的 1.7 倍。

OASIS 的 `recsys.py:533-536` 把 `batch_size=1000` 写死，我们**不改 vendored 包**：
在注入模型时把 `oasis.social_platform.recsys.generate_post_vector` 包一层——
本轮定的策略是 **B 打底（按可用显存算）+ A 兜底（算不出时用实测安全档 64）**。

本文件锁这些判据（全部 fake，不加载真模型、不碰 GPU）：

1. 算得出 → `cuda-adaptive`，且数字符合实测公式；
2. **算不出（连 1 条都放不下）→ 抛 `RecsysModelUnavailable`，整轮不启动**；
3. 查不到/读不到 → 用实测安全档（A 兜底），而不是放行 1000；
4. 守卫只会**把批变小**，不会放大；OASIS 请求 1000 时被压到安全值；
5. 守卫装不上（OASIS 没有这个函数）→ 拒绝启动；
6. 守卫可重复调用、不会套娃；
7. **vendored 的 `recsys.py` 一个字节都没改**（补丁是运行期的）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BACKEND_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import twitter_recsys_model as model_module  # noqa: E402
from twitter_recsys_model import (  # noqa: E402
    FALLBACK_SAFE_BATCH,
    OASIS_REQUESTED_BATCH,
    PEAK_OVER_ATTENTION_MATRIX,
    VRAM_RESERVE_MB,
    RecsysModelUnavailable,
    _batch_unit_mb,
    _install_batch_guard,
    _longest_token_bound,
    _safe_batch_now,
    resolve_batch_policy,
)

HEADS = 12
WORST_SEQ = 512


class _Config:
    num_attention_heads = HEADS
    max_position_embeddings = WORST_SEQ


class _Model:
    config = _Config()


class _TokenlessModel:
    """没有 `config` 的假模型：模拟"读不到结构 → 算不出 → 走兜底"。"""


def _fake_torch(monkeypatch, *, available: bool, free_mb: int | None = 7000):
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def mem_get_info():
            if free_mb is None:
                raise RuntimeError("查不到显存")
            return free_mb * 1024 * 1024, 8188 * 1024 * 1024

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_oasis(monkeypatch, *, with_generate_post_vector: bool = True):
    recsys = types.ModuleType("oasis.social_platform.recsys")
    recsys.twhin_tokenizer = None
    recsys.twhin_model = None
    if with_generate_post_vector:
        recsys.generate_post_vector = (
            lambda model, tokenizer, texts, batch_size=None: {"batch_size": batch_size}
        )
    oasis_package = types.ModuleType("oasis")
    social_platform = types.ModuleType("oasis.social_platform")
    social_platform.recsys = recsys
    oasis_package.social_platform = social_platform
    monkeypatch.setitem(sys.modules, "oasis", oasis_package)
    monkeypatch.setitem(sys.modules, "oasis.social_platform", social_platform)
    monkeypatch.setitem(sys.modules, "oasis.social_platform.recsys", recsys)
    return recsys


def _cuda_policy(**overrides) -> dict:
    policy = {
        "mode": "cuda-adaptive", "device": "cuda", "heads": HEADS, "worst_seq": WORST_SEQ,
        "free_mb": 7000, "preflight_batch": 75,
    }
    policy.update(overrides)
    return policy


# --- 1. 算得出 → 按可用显存算 -------------------------------------------------


def test_policy_computes_batch_from_available_vram(monkeypatch):
    """B：数字来自公式 + 实测系数，不是拍脑袋的常量。"""

    _fake_torch(monkeypatch, available=True, free_mb=7000)

    policy = resolve_batch_policy(_Model(), "cuda")

    expected_unit = 5.7 * HEADS * WORST_SEQ * WORST_SEQ * 4 / (1024 * 1024)
    expected_safe = int((7000 - VRAM_RESERVE_MB) / expected_unit)
    assert policy["mode"] == "cuda-adaptive"
    assert policy["preflight_batch"] == expected_safe
    assert policy["preflight_batch"] > FALLBACK_SAFE_BATCH, (
        "7000 MB 可用时应该能比兜底档放更多条"
    )
    assert policy["unit_mb"] == pytest.approx(round(expected_unit, 2))


def test_batch_unit_scales_with_sequence_length_squared():
    """每批单位的显存随序列长度平方增长（注意力矩阵），这是公式的依据。"""

    assert _batch_unit_mb(512, HEADS) == pytest.approx(_batch_unit_mb(256, HEADS) * 4)
    # 实测锚点：seq=267 时每批单位约 18.5 MB（batch 30→64 的实测斜率）
    assert _batch_unit_mb(267, HEADS) == pytest.approx(18.5, rel=0.08)
    assert PEAK_OVER_ATTENTION_MATRIX > 1, "其它中间量必须计入，否则会低估 5 倍"


# --- 2. 算不出（连 1 条都放不下）→ 不启动 ------------------------------------


def test_policy_refuses_to_start_when_no_safe_batch_exists(monkeypatch):
    """B 的硬判据：可用显存连 1 条都放不下 → 抛错，整轮不启动。"""

    _fake_torch(monkeypatch, available=True, free_mb=800)  # 减预留后只剩 32 MB

    with pytest.raises(RecsysModelUnavailable) as error:
        resolve_batch_policy(_Model(), "cuda")

    message = str(error.value)
    assert "算不出安全批大小就不启动" in message
    assert "N-48" in message


# --- 3. 算不出 → A 兜底（而不是放行 1000） -----------------------------------


def test_policy_falls_back_to_measured_cap_on_cpu(monkeypatch):
    """CPU 上没有显存可算：用实测安全档 64（1000 曾把机器吃死）。"""

    _fake_torch(monkeypatch, available=False)
    policy = resolve_batch_policy(_Model(), "cpu")

    assert policy["mode"] == "cpu-fallback"
    assert policy["preflight_batch"] == FALLBACK_SAFE_BATCH
    assert policy["preflight_batch"] < OASIS_REQUESTED_BATCH


def test_policy_falls_back_when_model_shape_is_unreadable(monkeypatch):
    _fake_torch(monkeypatch, available=True, free_mb=7000)

    policy = resolve_batch_policy(_TokenlessModel(), "cuda")

    assert policy["mode"] == "estimate-unavailable"
    assert policy["preflight_batch"] == FALLBACK_SAFE_BATCH


def test_policy_falls_back_when_vram_query_fails(monkeypatch):
    _fake_torch(monkeypatch, available=True, free_mb=None)

    policy = resolve_batch_policy(_Model(), "cuda")

    assert policy["mode"] == "estimate-unavailable"
    assert policy["preflight_batch"] == FALLBACK_SAFE_BATCH


# --- 4. 守卫只会把批变小 ------------------------------------------------------


def _set_policy(monkeypatch, policy: dict) -> None:
    monkeypatch.setattr(model_module, "_batch_policy", policy)
    monkeypatch.setattr(model_module, "_last_batch_notice", {})


def test_guard_downsizes_oasis_batch_1000(monkeypatch):
    """OASIS 请求 1000 → 实际传给它自己的 `generate_post_vector` 的 batch 必须更小。"""

    _fake_torch(monkeypatch, available=True, free_mb=7000)
    recsys = _fake_oasis(monkeypatch)
    _set_policy(monkeypatch, _cuda_policy())

    original = recsys.generate_post_vector
    calls = []

    def recorder(model, tokenizer, texts, batch_size=None):
        calls.append(batch_size)
        return original(model, tokenizer, texts, batch_size=batch_size)

    recsys.generate_post_vector = recorder
    _install_batch_guard(object(), _Model())

    guarded = recsys.generate_post_vector
    guarded(object(), object(), ["短文本"] * 10, batch_size=OASIS_REQUESTED_BATCH)

    assert calls, "守卫必须真的调用原函数"
    assert calls[0] != OASIS_REQUESTED_BATCH
    assert calls[0] <= 10, "批不许超过语料条数"


def test_guard_never_increases_smaller_request(monkeypatch):
    """OASIS 请求 8 条时不许被"放大"成安全上限。"""

    _fake_torch(monkeypatch, available=True, free_mb=7000)
    recsys = _fake_oasis(monkeypatch)
    _set_policy(monkeypatch, _cuda_policy())
    _install_batch_guard(object(), _Model())

    result = recsys.generate_post_vector(
        object(), object(), ["短文本"] * 10, batch_size=8
    )

    assert result["batch_size"] == 8


def test_guard_shrinks_when_free_vram_drops(monkeypatch):
    """自适应：同样的语料，可用显存变少 → 批更小。"""

    texts = ["字" * 267] * 1000  # 批内最长上界 267 token（与本机真实语料同量级）

    _fake_torch(monkeypatch, available=True, free_mb=7000)
    _set_policy(monkeypatch, _cuda_policy())
    roomy, _ = _safe_batch_now(object(), _Model(), texts, 1000)

    _fake_torch(monkeypatch, available=True, free_mb=2000)
    tight, _ = _safe_batch_now(object(), _Model(), texts, 1000)

    assert tight < roomy
    assert tight >= 1


def test_guard_uses_longest_text_as_bound(monkeypatch):
    """长文本占的显存更多 → 批要更小（约束来自批内最长序列）。"""

    _fake_torch(monkeypatch, available=True, free_mb=7000)
    _set_policy(monkeypatch, _cuda_policy())
    short, _ = _safe_batch_now(object(), _Model(), ["短"] * 200, 1000)
    long, _ = _safe_batch_now(object(), _Model(), ["长" * 512] * 200, 1000)

    assert long < short


def test_longest_token_bound_never_exceeds_model_limit():
    texts = ["字" * 900, "短"]
    assert _longest_token_bound(object(), texts) == model_module.MODEL_MAX_LENGTH


# --- 5. 守卫装不上 → 拒绝启动 ------------------------------------------------


def test_missing_generate_post_vector_refuses_to_start(monkeypatch):
    """OASIS 的 API 变了（找不到这个函数）→ 守卫装不上 → 不许继续跑。"""

    _fake_oasis(monkeypatch, with_generate_post_vector=False)

    with pytest.raises(RecsysModelUnavailable) as error:
        _install_batch_guard(object(), _Model())

    assert "批大小守卫装不上" in str(error.value)


# --- 6. 可重复调用、不套娃 ----------------------------------------------------


def test_guard_is_idempotent(monkeypatch):
    _fake_torch(monkeypatch, available=True, free_mb=7000)
    recsys = _fake_oasis(monkeypatch)
    _set_policy(monkeypatch, _cuda_policy())

    _install_batch_guard(object(), _Model())
    first = recsys.generate_post_vector
    _install_batch_guard(object(), _Model())

    assert recsys.generate_post_vector is first, "重复注入不许再包一层"


# --- 7. vendored 的 OASIS 一个字节都没改 -------------------------------------


def test_vendored_oasis_is_not_modified():
    """补丁是运行期的：`recsys.py` 里那个写死的 1000 还在（我们不碰第三方源码）。"""

    recsys_py = (
        BACKEND_DIR / ".venv" / "Lib" / "site-packages"
        / "oasis" / "social_platform" / "recsys.py"
    )
    if not recsys_py.exists():  # 依赖没装时跳过，而不是假装通过
        pytest.skip("camel-oasis 未安装")
    source = recsys_py.read_text(encoding="utf-8", errors="replace")
    assert "generate_post_vector(twhin_model," in source
    assert "batch_size=1000" in source, (
        "vendored 源码被改过：本项目的约定是只在运行期包一层，不改第三方文件"
    )


def test_callers_log_the_batch_policy():
    """两个运行脚本都要把"这一轮批大小是多少、依据是什么"写进运行日志。

    没有这行，产物里就看不出 OASIS 的 1000 有没有被压下来——那是 N-48 的唯一现场证据。
    """

    for filename in ("run_twitter_simulation.py", "run_parallel_simulation.py"):
        source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
        assert "batch_policy()" in source, f"{filename} 没有读取批大小策略"
        assert "推荐表批大小策略" in source, f"{filename} 没有把策略打进日志"
        assert "policy['preflight_batch']" in source, f"{filename} 没写批大小数值"

