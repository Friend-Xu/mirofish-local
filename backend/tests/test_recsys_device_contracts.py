"""推荐模型的**设备契约**：这一轮到底在 GPU 还是 CPU 上算（掩盖台账 N-47 / F19）。

实测背景：装了 CUDA 版 torch 也**不会自动**把模型搬上卡。OASIS 自己的加载器有
`.to(device)`（`oasis/social_platform/recsys.py:78-79`），但它只在 `twhin_model is None`
时才自己加载；我们的 `load_and_inject()` 已经先把模型注入了，于是那条 `.to(device)`
**永远不会执行**——实测模型参数留在 CPU：92 条语料 9.09 s、400 条 41.40 s、
4000 条约 6.9 min，168 轮累计小时级，并且把机器吃满过一次。

本文件锁五件事（都**不加载真模型**，全部 fake/monkeypatch）：

1. 设备解析的判据与 OASIS 一致（`cuda.is_available()`），且**没有环境变量开关**；
2. `load_and_inject()` 必须把模型搬到解析出的设备，且**发生在注入之前**；
3. 设备描述要进返回值（调用方要把它打进日志，产物里必须能看出在哪算的）；
4. 显存/加载失败 → `RecsysModelUnavailable`，且保留原始异常（不悄悄退回 CPU 跑）；
5. 测量脚本**不许再吃满 CPU/内存**（默认线程数小于逻辑核、有 padded-token 上限）。

真实的 GPU 端到端由实跑证明（组件级重放工具 `--arm project-module`：
整轮 `update_rec_table()` 6.46 s → **0.94 s**），静态断言不替代它。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BACKEND_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import local_models as local_module  # noqa: E402 - 依赖上面的 sys.path 设置
import twitter_recsys_model as model_module  # noqa: E402
from local_models import MANIFEST_NAME, sha256_file  # noqa: E402
from twitter_recsys_model import (  # noqa: E402
    RecsysModelUnavailable,
    load_and_inject,
    resolve_device,
)

MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")

#: 调用方脚本：都要把设备打进日志
RUN_SCRIPTS = ("run_twitter_simulation.py", "run_parallel_simulation.py")


def _write_model(root: Path, name: str = model_module.MODEL_NAME) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename in MODEL_FILES:
        (directory / filename).write_bytes(f"内容-{filename}".encode())
    manifest = {
        "name": name,
        "repo": "Twitter/twhin-bert-base",
        "revision": "deadbeef",
        "source": {"endpoint": "https://example.invalid", "method": "测试"},
        "manifest_written_at": "2026-10-11T00:00:00",
        "files": {
            filename: {
                "bytes": (directory / filename).stat().st_size,
                "sha256": sha256_file(directory / filename),
            }
            for filename in MODEL_FILES
        },
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


@pytest.fixture
def models_root(monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(local_module, "MODELS_DIR", root)
    return root


def _fake_torch(monkeypatch, *, available: bool, name: str = "假显卡",
                free_mb: int = 7000):
    """假 torch：只提供 `resolve_device()`、`.to()` 与批大小策略用得到的那几个入口。"""

    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def get_device_name(index):
            return name

        @staticmethod
        def mem_get_info():
            # 返回 (free, total) 字节数
            return free_mb * 1024 * 1024, 8188 * 1024 * 1024

    torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _fake_transformers(monkeypatch, calls):
    class _FakeModel:
        def __init__(self):
            self.to_calls = []
            self.training = True

        def to(self, device):
            self.to_calls.append(str(device))
            calls.append(("to", str(device)))
            return self

        def eval(self):
            self.training = False
            calls.append(("eval", None))
            return self

        def parameters(self):
            return iter([types.SimpleNamespace(device="fake")])

    class _Loader:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, path, **kwargs):
            calls.append(("load", self.kind))
            return _FakeModel() if self.kind == "model" else object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Loader("tokenizer")
    transformers.AutoModel = _Loader("model")
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _fake_oasis(monkeypatch):
    recsys = types.ModuleType("oasis.social_platform.recsys")
    recsys.twhin_tokenizer = None
    recsys.twhin_model = None
    # 批大小守卫（N-48）在注入时要包装它；没有它守卫装不上 = 拒绝启动
    recsys.generate_post_vector = lambda model, tokenizer, texts, batch_size=None: []
    # N-47 的确定性池化补丁要替换 process_recsys_posts.process_batch
    posts = types.ModuleType("oasis.social_platform.process_recsys_posts")
    posts.process_batch = lambda model, tokenizer, batch_texts: []
    oasis_package = types.ModuleType("oasis")
    social_platform = types.ModuleType("oasis.social_platform")
    social_platform.recsys = recsys
    social_platform.process_recsys_posts = posts
    oasis_package.social_platform = social_platform
    monkeypatch.setitem(sys.modules, "oasis", oasis_package)
    monkeypatch.setitem(sys.modules, "oasis.social_platform", social_platform)
    monkeypatch.setitem(sys.modules, "oasis.social_platform.recsys", recsys)
    monkeypatch.setitem(sys.modules, "oasis.social_platform.process_recsys_posts", posts)
    return recsys


# --- 1. 设备解析：判据与 OASIS 一致，且没有环境变量开关 -----------------------


def test_resolve_device_uses_cuda_when_available(monkeypatch):
    _fake_torch(monkeypatch, available=True, name="NVIDIA GeForce RTX 4060 Laptop GPU")

    device, description = resolve_device()

    assert device == "cuda"
    assert "RTX 4060" in description


def test_resolve_device_falls_back_to_cpu_when_torch_has_no_cuda(monkeypatch):
    """CPU-only 构建（本机曾长期如此）：如实报 cpu，而不是假装有卡。"""

    _fake_torch(monkeypatch, available=False)

    assert resolve_device() == ("cpu", "cpu")


def test_device_choice_has_no_environment_switch(monkeypatch):
    """不许出现"用环境变量选设备"的第二条路径：要 CPU 就设 `CUDA_VISIBLE_DEVICES=`。"""

    _fake_torch(monkeypatch, available=False)
    monkeypatch.setenv("MIROFISH_RECSYS_DEVICE", "cuda")
    monkeypatch.setenv("RECSYS_DEVICE", "cuda")

    assert resolve_device() == ("cpu", "cpu")


# --- 2. 模型必须被搬到设备上（OASIS 那条 `.to(device)` 不会替我们执行） -------


def test_load_moves_the_model_to_the_resolved_device(models_root, monkeypatch):
    directory = _write_model(models_root)
    _fake_torch(monkeypatch, available=True, name="假显卡 A")
    calls = []
    _fake_transformers(monkeypatch, calls)
    recsys = _fake_oasis(monkeypatch)

    loaded, _seconds, device = load_and_inject()

    assert loaded == directory
    assert device == "cuda:0 / 假显卡 A"
    # `.to()` 必须在 `.eval()` 之前、且在注入之后模型已经就位
    assert ("to", "cuda") in calls, f"模型没有被搬到 cuda：{calls}"
    assert [kind for kind, _ in calls].index("to") < [kind for kind, _ in calls].index("eval")
    assert recsys.twhin_model.to_calls == ["cuda"]
    assert recsys.twhin_model.training is False, "推理期必须 eval()（关掉 dropout）"


def test_load_on_cpu_build_still_moves_to_cpu_explicitly(models_root, monkeypatch):
    """CPU 构建下也要走同一条 `.to("cpu")`：设备永远是显式决定的，不靠"反正在 cpu"。"""

    _write_model(models_root)
    _fake_torch(monkeypatch, available=False)
    calls = []
    _fake_transformers(monkeypatch, calls)
    recsys = _fake_oasis(monkeypatch)

    _loaded, _seconds, device = load_and_inject()

    assert device == "cpu"
    assert recsys.twhin_model.to_calls == ["cpu"]


# --- 3. 设备描述要进返回值，调用方要打进日志 ---------------------------------


def test_device_description_is_returned_and_logged_by_callers():
    for filename in RUN_SCRIPTS:
        source = (SCRIPTS_DIR / filename).read_text(encoding="utf-8")
        assert "load_and_inject()" in source, f"{filename} 没有走产品加载路径"
        assert "model_device" in source, (
            f"{filename} 没有接住设备描述：产物里就看不出来这一轮在哪算的"
        )
        assert "设备 {model_device}" in source, (
            f"{filename} 没有把设备打进日志"
        )


# --- 4. 失败即失败：显存/加载出错不许悄悄退回 CPU ----------------------------


def test_out_of_memory_becomes_the_boundary_error(models_root, monkeypatch):
    _write_model(models_root)
    _fake_torch(monkeypatch, available=True)

    class _BoomModel:
        def to(self, device):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    class _BoomLoader:
        def from_pretrained(self, path, **kwargs):
            return _BoomModel()

    class _TokenizerLoader:
        def from_pretrained(self, path, **kwargs):
            return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _TokenizerLoader()
    transformers.AutoModel = _BoomLoader()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _fake_oasis(monkeypatch)

    with pytest.raises(RecsysModelUnavailable) as error:
        load_and_inject()

    assert "out of memory" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError), "原始异常必须保留为 __cause__"


def test_load_failure_does_not_inject_a_half_built_model(models_root, monkeypatch):
    _write_model(models_root)
    _fake_torch(monkeypatch, available=True)

    class _BoomLoader:
        def from_pretrained(self, path, **kwargs):
            raise OSError("假加载失败")

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _BoomLoader()
    transformers.AutoModel = _BoomLoader()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    recsys = _fake_oasis(monkeypatch)

    with pytest.raises(RecsysModelUnavailable):
        load_and_inject()

    assert recsys.twhin_model is None and recsys.twhin_tokenizer is None, (
        "加载失败时不许留下半个模型：OASIS 会以为已经就绪"
    )


# --- 5. 测量脚本不许再吃满 CPU / 内存（那次卡死的直接原因） -------------------


def test_cost_tool_keeps_its_safety_limits():
    """静态守卫：默认批大小要小于生产值、要有 padded-token / 显存 / 线程三道上限。

    实测教训：① 一次 4035 条 × 267 token 的 CPU attention 张量到 GB 级
    （`not enough memory: 1,380,725,352 bytes`），线程取满 20 个逻辑核把桌面饿死；
    ② 同一规模在 GPU 上把 8 GB 的卡打成 `GPU is lost`（只能重启恢复，发生两次）。
    """

    source = (BACKEND_DIR.parent / "scripts" / "measure-recsys-round-cost.py").read_text(
        encoding="utf-8"
    )
    assert "--max-padded-tokens" in source
    assert "--cpu-threads" in source
    assert "SAFE_BATCH_SIZE = 100" in source, "默认批大小必须小于生产值 1000"
    assert "MAX_PADDED_TOKENS" in source
    # 显存两道闸：实测系数粗筛（跑之前）+ 实测峰值（跑之中）
    assert "--vram-budget-mb" in source and "PEAK_MB_PER_BATCH_UNIT" in source
    assert "--vram-ceiling-mb" in source and "VRAM_CEILING_MB" in source
    # 池化探针的语料硬上限：掉卡都出在它身上
    assert "--pooler-max-items" in source
    # **不许有跳过粗筛的开关**：试过一次就把卡打成 GPU is lost
    assert "def assert_vram_budget(label: str, batch_size: int, longest_tokens: int,\n" \
           "                       budget_mb: int, free_mb: float | None,\n" \
           "                       actual_batch: int | None = None) -> dict:" in source, (
        "assert_vram_budget 的签名里不许再出现 skip 之类的绕过参数"
    )
    assert "skip: bool" not in source
    # 线程不许取满逻辑核
    assert "min(8, os.cpu_count()" in source, "默认线程数必须是 min(8, 逻辑核数)"
    assert "torch.set_num_threads(os.cpu_count() or 1)" not in source


def test_cost_tool_chunks_the_corpus_like_production():
    """曲线必须走生产的**分块**入口 `generate_post_vector(batch_size=...)`。

    实测缺陷：直接调 `process_batch(整份语料)` 会把 400 条一次 pad 到 267 token，
    `--batch-size` **完全不起作用**——峰值 8509 MB（超 8 GB 卡）、同一份输入从
    1.93 s 变成 23.55 s。这条断言锁住"分块那一层不许被绕过"。
    """

    source = (BACKEND_DIR.parent / "scripts" / "measure-recsys-round-cost.py").read_text(
        encoding="utf-8"
    )
    assert "generate_post_vector(model, tokenizer, texts," in source
    assert "batch_size=args.batch_size" in source
    assert "process_batch(model, tokenizer, texts)" not in source, (
        "曲线不许再把整份语料交给 process_batch：--batch-size 会失效"
    )
