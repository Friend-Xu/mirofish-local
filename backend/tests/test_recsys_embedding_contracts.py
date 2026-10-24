"""句向量契约（掩盖台账 N-47）：**推荐表的余弦不能是随机投影**。

实测背景（读代码 + 实测）：

* OASIS 的 `process_recsys_posts.process_batch` 返回 `outputs.pooler_output`（`process_batch:33`）；
* twhin 的 `model.safetensors` 里**没有任何 `pooler.*` 张量**（215 个张量逐个清点），
  `BertModel` 会建一层 `BertPooler` 并**随机初始化**（加载日志自己会警告
  `pooler.dense.bias / pooler.dense.weight are newly initialized`）；
* 于是"相似度"是随机投影：同一条文本两次加载余弦 **0.0379**、推荐表只有 **18%** 用户一致（实测）；
* 而 `recsys.py:583` 的 `cosine_similarities * scores[...]` **在** `if enable_like_score:`
  外面——相似度是**真的参与**排序的（余弦 × 新旧分数），所以随机向量会把推荐表搅乱，
  而且每次运行都不一样（可复现性）。

修法：把 `process_batch` 换成**确定性均值池化**（`last_hidden_state` 按 attention mask 平均），
不含可训练参数、padding 被排除、同样输入永远同样向量。仍然**不改 vendored 源码**。

本文件锁这些判据（全部 fake，不加载真模型、不碰 GPU）：

1. 均值池化算法正确（padding 不参与平均）；
2. 确定性：同样输入两次调用逐位相同；
3. padding 不变性：同一条文本单独跑 == 混在批里跑；
4. 补丁装上之后 OASIS 的 `process_batch` 就是我们的函数，且重复注入不套娃；
5. OASIS 没有这个函数时**拒绝启动**（不许继续用随机 pooler）；
6. 策略可见、两个运行脚本都把它打进日志；
7. **vendored 的 `process_recsys_posts.py` 一个字节都没改**。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BACKEND_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import twitter_recsys_model as model_module  # noqa: E402
from twitter_recsys_model import (  # noqa: E402
    RecsysModelUnavailable,
    _install_embedding_patch,
    _mean_pooled_process_batch,
    embedding_policy,
)

HIDDEN = 4


class _CharTokenizer:
    """字符级假 tokenizer：`字` 的 id 就是位置序号，方便手算均值。"""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        lengths = [len(text) for text in texts]
        longest = max(lengths)
        ids = torch.zeros((len(texts), longest), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1, dtype=torch.long)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _MeanModel:
    """返回可手算的隐藏态：第 t 个位置是 `token_id × [1,2,3,4]`。"""

    def __init__(self):
        self.to_calls = []

    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def __call__(self, **kwargs):
        ids = kwargs["input_ids"].float().unsqueeze(-1)
        scales = torch.arange(1, HIDDEN + 1, dtype=torch.float)
        return types.SimpleNamespace(last_hidden_state=ids * scales)


def _fake_oasis(monkeypatch, *, with_process_batch: bool = True):
    posts = types.ModuleType("oasis.social_platform.process_recsys_posts")
    if with_process_batch:
        posts.process_batch = lambda model, tokenizer, batch_texts: torch.zeros(1, HIDDEN)
    social_platform = types.ModuleType("oasis.social_platform")
    social_platform.process_recsys_posts = posts
    oasis_package = types.ModuleType("oasis")
    oasis_package.social_platform = social_platform
    monkeypatch.setitem(sys.modules, "oasis", oasis_package)
    monkeypatch.setitem(sys.modules, "oasis.social_platform", social_platform)
    monkeypatch.setitem(sys.modules, "oasis.social_platform.process_recsys_posts", posts)
    return posts


def _expected_vector(text: str) -> torch.Tensor:
    """手算：位置 t 的 token id = t，隐藏态 = id × [1..H]；均值 = mean(1..n) × [1..H]。"""

    ids = torch.arange(1, len(text) + 1, dtype=torch.float)
    mean_id = ids.mean()
    scales = torch.arange(1, HIDDEN + 1, dtype=torch.float)
    return mean_id * scales


# --- 1. 均值池化算法正确（padding 不参与平均） --------------------------------


def test_mean_pool_excludes_padding():
    texts = ["abc", "a"]  # 批内 pad 到 3
    vectors = _mean_pooled_process_batch(_MeanModel(), _CharTokenizer(), texts)

    assert vectors.shape == (2, HIDDEN)
    assert torch.allclose(vectors[0], _expected_vector("abc")), "第一条被 padding 污染了"
    assert torch.allclose(vectors[1], _expected_vector("a")), "短文本不应被补零平均"
    assert not vectors.requires_grad, "推理期不应带梯度"


# --- 2. 确定性：同样输入两次逐位相同 -----------------------------------------


def test_embedding_is_deterministic():
    tokenizer = _CharTokenizer()
    first = _mean_pooled_process_batch(_MeanModel(), tokenizer, ["甲乙丙", "丁"])
    second = _mean_pooled_process_batch(_MeanModel(), tokenizer, ["甲乙丙", "丁"])

    assert torch.equal(first, second), "确定性池化两次结果必须逐位相同"


# --- 3. padding 不变性：单独跑 == 混在批里跑 ----------------------------------


def test_padding_does_not_change_a_text_vector():
    tokenizer = _CharTokenizer()
    model = _MeanModel()
    alone = _mean_pooled_process_batch(model, tokenizer, ["abc"])
    batched = _mean_pooled_process_batch(model, tokenizer, ["abc", "abcdefgh"])

    assert torch.allclose(alone[0], batched[0], atol=1e-6), (
        "同一条文本在批里的向量必须与单独跑一致，否则推荐结果会随同批内容漂移"
    )


# --- 4. 补丁装到 OASIS 上、可重复调用 ----------------------------------------


def test_patch_replaces_oasis_process_batch(monkeypatch):
    posts = _fake_oasis(monkeypatch)
    original = posts.process_batch

    _install_embedding_patch()

    assert posts.process_batch is _mean_pooled_process_batch
    assert posts.process_batch is not original
    assert posts.process_batch._mirofish_original is original, "原始函数必须留档"


def test_patch_is_idempotent(monkeypatch):
    posts = _fake_oasis(monkeypatch)
    _install_embedding_patch()
    first = posts.process_batch
    _install_embedding_patch()

    assert posts.process_batch is first, "重复注入不许再包一层"


# --- 5. 装不上就拒绝启动 ------------------------------------------------------


def test_missing_process_batch_refuses_to_start(monkeypatch):
    _fake_oasis(monkeypatch, with_process_batch=False)

    with pytest.raises(RecsysModelUnavailable) as error:
        _install_embedding_patch()

    assert "确定性池化装不上" in str(error.value)
    assert "0.0379" in str(error.value), "错误信息要带上实测数字，让人知道代价"


# --- 6. 策略可见 + 运行脚本写进日志 ------------------------------------------


def test_embedding_policy_is_explicit():
    policy = embedding_policy()
    assert policy["pooling"] == "mean_last_hidden_state"
    assert policy["deterministic"] is True
    assert "process_batch" in policy["replaces"]


def test_load_and_inject_installs_the_embedding_patch():
    """接缝判据：补丁必须由 `load_and_inject()` 装上，而不是只存在于模块里。"""

    source = (SCRIPTS_DIR / "twitter_recsys_model.py").read_text(encoding="utf-8")
    body = source[source.index("def load_and_inject("):]
    body = body[:body.index("def describe(")]
    assert "_install_embedding_patch()" in body, (
        "load_and_inject() 没有装确定性句向量补丁：推荐表会退回随机 pooler"
    )


def test_callers_log_the_embedding_policy():
    for filename in ("run_twitter_simulation.py", "run_parallel_simulation.py"):
        source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
        assert "embedding_policy()" in source, f"{filename} 没有读取句向量策略"
        assert "推荐表句向量" in source, f"{filename} 没有把句向量策略打进日志"


# --- 7. vendored 源码没被改 ---------------------------------------------------


def test_vendored_process_recsys_posts_is_not_modified():
    path = (
        BACKEND_DIR / ".venv" / "Lib" / "site-packages"
        / "oasis" / "social_platform" / "process_recsys_posts.py"
    )
    if not path.exists():
        pytest.skip("camel-oasis 未安装")
    source = path.read_text(encoding="utf-8", errors="replace")
    assert "return outputs.pooler_output" in source, (
        "vendored 源码被改过：约定是运行期替换 process_batch，不改第三方文件"
    )
