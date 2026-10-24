"""项目自有模型目录的契约测试（掩盖台账 N-46 / F18）。

要锁的规则只有一条：**清单是唯一权威，任何不符都不许继续**。
`<项目>/models/<name>/model.json` 固定 repo、revision、每个文件的字节数与 sha256；
缺目录、缺清单、缺键、缺文件、字节数不符、sha256 不符、出现清单外的第二份权重，
一律抛 `ModelFilesUnavailable`——不下载、不回退、不给默认值。

真实载荷（1.08 GB 的 twhin + 92 MB 的 bge）由
`.tools/provision-models.py --verify` 在闸门里校验，测试不碰它：
测试绑死大文件会让"改一行代码"变成"读 1 GB 磁盘"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import local_models as module  # noqa: E402 - 依赖上面的 sys.path 设置
from local_models import (  # noqa: E402
    MANIFEST_NAME,
    ModelFilesUnavailable,
    read_manifest,
    sha256_file,
    verify_model,
)

DECLARED = ("config.json", "tokenizer.json", "model.safetensors")


def _write_model(root: Path, name: str = "demo", files=DECLARED):
    """造一个合法的模型目录：文件 + 清单（哈希是真的，用同一个实现算）。"""

    directory = root / name
    directory.mkdir(parents=True)
    contents = {}
    for filename in files:
        payload = f"内容-{filename}".encode()
        (directory / filename).write_bytes(payload)
        contents[filename] = payload
    manifest = {
        "name": name,
        "repo": "example/demo",
        "revision": "deadbeef",
        "source": {"endpoint": "https://example.invalid", "method": "测试"},
        "manifest_written_at": "2026-10-11T00:00:00",
        "files": {
            filename: {"bytes": len(payload), "sha256": sha256_file(directory / filename)}
            for filename, payload in contents.items()
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
    monkeypatch.setattr(module, "MODELS_DIR", root)
    return root


def test_verify_returns_a_manifest_summary(models_root):
    _write_model(models_root)

    info = verify_model("demo", check_hashes=True)

    assert info["name"] == "demo"
    assert info["repo"] == "example/demo"
    assert info["revision"] == "deadbeef"
    assert info["hashes_checked"] is True
    assert set(info["files"]) == set(DECLARED)
    assert all(entry["sha256"] for entry in info["files"].values())


def test_missing_directory_raises(models_root):
    with pytest.raises(ModelFilesUnavailable, match="模型目录不存在"):
        verify_model("nope")


def test_missing_manifest_raises(models_root):
    (models_root / "demo").mkdir()
    with pytest.raises(ModelFilesUnavailable, match="缺少模型清单"):
        verify_model("demo")


def test_broken_manifest_raises(models_root):
    directory = _write_model(models_root)
    (directory / MANIFEST_NAME).write_text("{ 不是 JSON", encoding="utf-8")
    with pytest.raises(ModelFilesUnavailable, match="读不动或不是 JSON"):
        verify_model("demo")


def test_manifest_missing_required_keys_raises(models_root):
    directory = _write_model(models_root)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    del manifest["revision"]
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelFilesUnavailable, match="缺字段"):
        read_manifest("demo")


def test_missing_file_raises(models_root):
    directory = _write_model(models_root)
    (directory / "tokenizer.json").unlink()

    with pytest.raises(ModelFilesUnavailable, match="缺文件 tokenizer.json"):
        verify_model("demo")


def test_byte_count_mismatch_raises(models_root):
    directory = _write_model(models_root)
    (directory / "config.json").write_bytes(b"short")

    with pytest.raises(ModelFilesUnavailable, match="字节数不符"):
        verify_model("demo")


def test_hash_mismatch_raises_even_when_size_matches(models_root):
    """内容被换掉（字节数相同）必须被抓住——这正是"只看文件在不在"抓不到的。"""

    directory = _write_model(models_root)
    original = (directory / "tokenizer.json").read_bytes()
    (directory / "tokenizer.json").write_bytes(bytes(reversed(original)))

    with pytest.raises(ModelFilesUnavailable, match="sha256 不符"):
        verify_model("demo")


def test_undeclared_second_weights_file_raises(models_root):
    """目录里多出一份权重 → 加载的可能是它，必须在加载前失败。"""

    directory = _write_model(models_root)
    (directory / "pytorch_model.bin").write_bytes(b"another-weights-file")

    with pytest.raises(ModelFilesUnavailable, match="清单未声明的权重"):
        verify_model("demo")


def test_check_hashes_false_skips_hashing(models_root, monkeypatch):
    """探针路径（只查存在性与字节数）不许偷偷算哈希：1 GB 读一遍不是免费的。"""

    _write_model(models_root)
    calls = []

    def _explode(path, *args, **kwargs):
        calls.append(path)
        raise AssertionError("check_hashes=False 时不该算哈希")

    monkeypatch.setattr(module, "sha256_file", _explode)

    info = verify_model("demo", check_hashes=False)

    assert calls == []
    assert info["hashes_checked"] is False
    assert all("sha256" not in entry for entry in info["files"].values())
