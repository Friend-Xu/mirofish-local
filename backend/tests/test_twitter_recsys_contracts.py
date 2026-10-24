"""Twitter 推荐时间线契约测试（掩盖台账 N-46 / F17 / F18）。

实测背景（开发机组件级重放，证据在 `backend/uploads/_diagnostics/`）：

* OASIS 原先用**仓库名**加载 `Twitter/twhin-bert-base`；transformers 只要
  `vocab_size > 100000`（本模型是 250002）且入参不是本地目录，就会调 `model_info()`
  去请求 Hub → 离线开关下抛 `OfflineModeIsEnabled` → OASIS 把它吞在
  `update_rec_table()` 里 → `rec` 表恒空 → agent 看不到别人的帖子，只剩广播；
* 实测：18 个模拟产物 twitter 的 `rec` 全是 0；改成按目录加载后同一份库 `rec` 0 → 70 行。

本文件锁四件事：**加载走项目自有目录**（不碰 Hub、不看环境变量）、
**清单不符即失败**、**OASIS 的错误不许被静音**、**没有时间线就是失败**。
真实端到端由 L-a 实跑证明（`sim_3a3e1b64c774`：`rec`=80 行、互动边 37），
静态断言不替代它。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import local_models as local_module  # noqa: E402 - 依赖上面的 sys.path 设置
import twitter_recsys_model as model_module  # noqa: E402
from action_recorder import ActionRecorder, TimelineNotBuilt  # noqa: E402
from local_models import MANIFEST_NAME, sha256_file  # noqa: E402
from twitter_recsys_model import (  # noqa: E402
    RecsysModelUnavailable,
    load_and_inject,
    pin_offline_runtime,
)

AGENT_CONFIG = {"agent_configs": [{"agent_id": 0, "entity_name": "甲"}]}

MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")


def _write_model(root: Path, name: str = model_module.MODEL_NAME) -> Path:
    """造一个合法的模型目录：文件 + 真哈希清单（与 local_models 同一个实现）。"""

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


def _fake_transformers(monkeypatch, calls):
    class _FakeModel:
        """够用的假模型：`load_and_inject()` 会调 `.to(device)` 与 `.eval()`（推理期）。"""

        def __init__(self):
            self.device = "cpu"
            self.training = True

        def to(self, device):
            self.device = str(device)
            return self

        def eval(self):
            self.training = False
            return self

    class _Loader:
        def __init__(self, name):
            self.name = name

        def from_pretrained(self, path, **kwargs):
            calls.append((self.name, path, kwargs))
            return _FakeModel() if self.name == "model" else object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Loader("tokenizer")
    transformers.AutoModel = _Loader("model")
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _fake_oasis(monkeypatch):
    """假 OASIS 包：本文件只验证"入参是本地目录 + 结果被注入"，
    不要为了它去 import 真的 oasis（那条链会拖起 torch / sentence_transformers）。

    必须带 `generate_post_vector`：N-48 的批大小守卫在注入时会包装它——
    OASIS 没有这个函数就说明守卫装不上，产品会选择**拒绝启动**。
    """

    recsys = types.ModuleType("oasis.social_platform.recsys")
    recsys.twhin_tokenizer = None
    recsys.twhin_model = None
    recsys.generate_post_vector = lambda model, tokenizer, texts, batch_size=None: []
    # N-47 的确定性池化补丁会替换 `process_recsys_posts.process_batch`：
    # 假 OASIS 必须同时提供这个模块，否则"装不上就拒绝启动"会让加载失败
    posts = types.ModuleType("oasis.social_platform.process_recsys_posts")
    posts.process_batch = lambda model, tokenizer, batch_texts: []
    posts.generate_post_vector = lambda model, tokenizer, texts, batch_size=None: []
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


def _platform_db(sim_dir: Path, *, posts: int, recs: int) -> Path:
    sim_dir.mkdir(parents=True, exist_ok=True)
    db_path = sim_dir / "twitter_simulation.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE post (post_id INTEGER, user_id INTEGER, content TEXT);
        CREATE TABLE rec (user_id INTEGER, post_id INTEGER);
        """
    )
    for index in range(posts):
        connection.execute("INSERT INTO post VALUES (?, 0, '帖子')", (index + 1,))
    for index in range(recs):
        connection.execute("INSERT INTO rec VALUES (0, ?)", (index + 1,))
    connection.commit()
    connection.close()
    return db_path


# --- 1. 运行期只钉"不联网"，不碰 HF_HOME --------------------------------------


def test_pin_offline_runtime_sets_only_the_offline_flags(monkeypatch):
    """模型按路径加载，所以环境里的 `HF_HOME` 不再是变量——我们也不该去改它。"""

    monkeypatch.setenv("HF_HOME", r"C:\somewhere\else")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    pin_offline_runtime()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    # 不设、也不改写 HF_HOME：加载位置由项目目录决定，不由环境决定
    assert os.environ["HF_HOME"] == r"C:\somewhere\else"


def test_load_ignores_ambient_hf_home(models_root, monkeypatch):
    """环境里 `HF_HOME` 指向别处/不存在，也照样从 `<项目>/models/` 加载。"""

    directory = _write_model(models_root)
    monkeypatch.setenv("HF_HOME", str(models_root.parent / "not-a-cache"))
    calls = []
    _fake_transformers(monkeypatch, calls)
    recsys = _fake_oasis(monkeypatch)

    loaded_dir, _, _device = load_and_inject()

    assert loaded_dir == directory
    assert [call[1] for call in calls] == [str(directory), str(directory)]
    assert recsys.twhin_model is not None


# --- 2. 清单不符即失败，不下载、不回退 ----------------------------------------


def test_load_raises_when_the_model_directory_is_missing(models_root):
    with pytest.raises(RecsysModelUnavailable, match="模型目录不存在"):
        load_and_inject()


def test_load_raises_when_a_file_was_tampered(models_root, monkeypatch):
    """清单里某个文件被改一个字节 → 加载前失败（而不是加载到别的东西）。"""

    directory = _write_model(models_root)
    original = (directory / "model.safetensors").read_bytes()
    (directory / "model.safetensors").write_bytes(bytes(reversed(original)))
    _fake_transformers(monkeypatch, [])
    _fake_oasis(monkeypatch)

    with pytest.raises(RecsysModelUnavailable, match="sha256 不符"):
        load_and_inject()


# --- 3. 加载必须传本地目录（这条是 Hub 请求的开关） ---------------------------


def test_load_and_inject_uses_a_local_directory(models_root, monkeypatch):
    """入参必须是**存在的目录**：`os.path.isdir()` 为真，transformers 才走 `_is_local`
    分支、才不会去调 `model_info()`。传仓库名在离线模式下必然失败（实测）。"""

    directory = _write_model(models_root)
    calls = []
    _fake_transformers(monkeypatch, calls)
    recsys = _fake_oasis(monkeypatch)

    loaded_dir, seconds, _device = load_and_inject()

    assert loaded_dir == directory
    assert seconds >= 0
    assert [call[1] for call in calls] == [str(directory), str(directory)]
    for _, path, _ in calls:
        assert path != "Twitter/twhin-bert-base"
        assert Path(path).is_dir(), "传的不是目录 → transformers 会去请求 Hub"
    assert calls[0][2].get("model_max_length") == 512
    # 注入：OASIS 只在全局为空时才自己加载，所以这里必须已经填好
    assert recsys.twhin_tokenizer is not None
    assert recsys.twhin_model is not None


# --- 4. OASIS 的错误不许被静音 ------------------------------------------------


def test_oasis_error_loggers_stay_visible():
    """`social.rec` / `social.twitter` 的 ERROR 是"推荐表为什么没建起来"的唯一证据。"""

    import importlib.util

    import run_parallel_simulation as parallel  # noqa: PLC0415 - 需要 scripts 在 sys.path

    previous = os.getcwd()
    os.chdir(SCRIPTS_DIR)
    try:
        parallel.disable_oasis_logging()
    finally:
        os.chdir(previous)

    for name in ("social.rec", "social.twitter"):
        logger = logging.getLogger(name)
        assert logger.level == logging.ERROR, f"{name} 被压到 {logger.level}：失败会静音"
        assert logger.propagate is True, f"{name} 不传播：simulation.log 里看不到"

    assert logging.getLogger("social.agent").level == logging.CRITICAL
    assert importlib.util.find_spec("run_parallel_simulation") is not None


# --- 5. 有帖子就必须有推荐时间线 ----------------------------------------------


def test_require_timeline_raises_without_a_recommendation_table(tmp_path):
    sim_dir = tmp_path / "sim_x"
    _platform_db(sim_dir, posts=3, recs=0)
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)

    with pytest.raises(TimelineNotBuilt, match="推荐表为空"):
        recorder.require_timeline(2)


def test_require_timeline_passes_with_a_recommendation_table(tmp_path):
    sim_dir = tmp_path / "sim_x"
    _platform_db(sim_dir, posts=3, recs=6)
    ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG).require_timeline(2)


def test_require_timeline_skips_a_world_without_posts(tmp_path):
    """没有帖子就没什么可推荐的：这不是失败，失败由"0 动作"那条判据负责。"""

    sim_dir = tmp_path / "sim_x"
    _platform_db(sim_dir, posts=0, recs=0)
    ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG).require_timeline(1)


def test_require_timeline_raises_when_the_database_is_missing(tmp_path):
    recorder = ActionRecorder("twitter", str(tmp_path / "nope"), AGENT_CONFIG)
    with pytest.raises(TimelineNotBuilt):
        recorder.require_timeline(1)


# --- 6. 接线还在（源级检查；真实行为由 L-a 实跑证明） -------------------------


def test_parallel_script_pins_offline_before_importing_oasis():
    source = (SCRIPTS_DIR / "run_parallel_simulation.py").read_text(encoding="utf-8")
    assert source.index("pin_offline_runtime()") < source.index("    import oasis")
    assert source.index("load_and_inject()") < source.index(
        "platform=oasis.DefaultPlatformType.TWITTER"
    )
    assert source.count("recorder.require_timeline(") == 2  # 两个平台各自一轮一次
    # 加载不再依赖 HF 缓存位置：脚本里不该再出现 HF_HOME
    assert "HF_HOME" not in source


def test_single_platform_scripts_guard_and_preload():
    twitter = (SCRIPTS_DIR / "run_twitter_simulation.py").read_text(encoding="utf-8")
    assert twitter.index("pin_offline_runtime()") < twitter.index("    import oasis")
    assert twitter.index("load_and_inject()") < twitter.index(
        "platform=oasis.DefaultPlatformType.TWITTER"
    )
    assert "recorder.require_timeline(" in twitter
    assert "HF_HOME" not in twitter

    reddit = (SCRIPTS_DIR / "run_reddit_simulation.py").read_text(encoding="utf-8")
    assert "recorder.require_timeline(" in reddit
    # reddit 不用这个模型，就不该引入它（一个能力一条路径）
    assert "twitter_recsys_model" not in reddit


# --- 7. 三个模拟脚本里不许有"用了没导入"的模块名 ------------------------------

ALLOWED_IMPLICIT = {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__builtins__"}


def _undefined_module_names(path: Path) -> list:
    """模块里"当作全局用、但既没导入也没定义、也不是内置"的名字。

    为什么要有：L-a 现场撞到过 `run_twitter_simulation.py:650 NameError: name 'time'
    is not defined`——源码级断言（顺序、字符串）全绿，测试也全绿，只有真跑才炸。
    实现要点：名字是在**函数体**里用的，模块作用域本身不会记它，
    所以要递归所有子符号表、只取 `is_global()` 的引用（闭包变量是 `is_free()`，不算）。
    """

    import builtins
    import symtable

    module_table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    defined = {
        symbol.get_name()
        for symbol in module_table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    known = set(dir(builtins)) | ALLOWED_IMPLICIT
    missing = set()

    def walk(table) -> None:
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and name not in defined
                and name not in known
            ):
                missing.add(name)
        for child in table.get_children():
            walk(child)

    walk(module_table)
    return sorted(missing)


def test_simulation_scripts_reference_only_defined_module_names():
    for name in (
        "run_parallel_simulation.py",
        "run_twitter_simulation.py",
        "run_reddit_simulation.py",
        "action_recorder.py",
        "twitter_recsys_model.py",
        "local_models.py",
    ):
        assert _undefined_module_names(SCRIPTS_DIR / name) == [], (
            f"{name} 用了没导入/没定义的模块级名字——真跑时会 NameError"
        )
