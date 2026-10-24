"""项目自有模型的供应、记录与校验。`--write-manifest` / `--verify` / `--fetch`。

为什么单独一个工具（掩盖台账 N-46 / F18）：

模型文件**不进版本库**（1.12 GB + 92 MB，`.gitignore` 里有 `models/`），所以
"从哪里、哪个修订、哪些文件"必须由一个可重跑的脚本 + 一份清单固定住，
而不是靠"上次是谁下载的"。这里的 `MODELS` 表就是版本锁：

* `repo` + `revision` + `files`：这三个字段是"这个模型是什么"的唯一定义；
* `--write-manifest`：算每个文件的字节数与 sha256，写 `<项目>/models/<name>/model.json`；
* `--verify`：调 `local_models.verify_model()` 逐文件校验（1.12 GB 实测 1.3 秒）；
* `--fetch`：文件不在时按表抓取；默认走 hf-mirror（中国网络前提），可用
  `--endpoint` 显式换源。抓取失败就失败，不回退到别的模型或静默联网。

`local_models` 是唯一的校验实现，本工具不复制一份判据。

用法（在项目根目录执行）：

    backend\\.venv\\Scripts\\python.exe scripts\\provision-models.py --verify
    backend\\.venv\\Scripts\\python.exe scripts\\provision-models.py --write-manifest twhin-bert-base
    backend\\.venv\\Scripts\\python.exe scripts\\provision-models.py --fetch bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

PROJECT = pathlib.Path(__file__).resolve().parent.parent
BACKEND_SCRIPTS = PROJECT / "backend" / "scripts"
if str(BACKEND_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BACKEND_SCRIPTS))

from local_models import (  # noqa: E402 - 依赖上面的 sys.path 设置
    MANIFEST_NAME,
    MODELS_DIR,
    ModelFilesUnavailable,
    sha256_file,  # 同一个实现算哈希，避免两处口径漂移
    verify_model,
)

DEFAULT_ENDPOINT = "https://hf-mirror.com"

# 版本锁：repo + revision + 文件清单。改这里等于换模型，必须同时重跑 --write-manifest
# 与 --verify，并把新数字写进 模拟质量基线.md。
MODELS = {
    "twhin-bert-base": {
        "repo": "Twitter/twhin-bert-base",
        "revision": "82ac392ce81f94560c391311ee2ddd024c5ac1fc",
        "files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
        "source": {
            "endpoint": DEFAULT_ENDPOINT,
            "method": "huggingface_hub.snapshot_download（HuggingFace hub 缓存布局，已由 F18 迁出）",
            "obtained_at": "2026-10-08",
        },
    },
    "bge-small-zh-v1.5": {
        "repo": "BAAI/bge-small-zh-v1.5",
        "revision": "7999e1d3359715c523056ef9478215996d62a620",
        "files": [
            "config.json",
            "modules.json",
            "sentence_bert_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "model.safetensors",
            "1_Pooling/config.json",
        ],
        "source": {
            "endpoint": DEFAULT_ENDPOINT,
            "method": "scripts/provision-models.py --fetch（huggingface_hub.snapshot_download）",
            "obtained_at": "2026-10-09",
        },
    },
}


def _model_names(requested: list) -> list:
    if not requested or requested == ["all"]:
        return list(MODELS)
    unknown = [name for name in requested if name not in MODELS]
    if unknown:
        raise SystemExit(f"未知模型 {unknown}；已知：{list(MODELS)}")
    return requested


def write_manifest(name: str, *, method: str = None, obtained_at: str = None) -> pathlib.Path:
    """按版本锁写 `<模型目录>/model.json`（纯本地计算，不联网）。"""

    spec = MODELS[name]
    directory = MODELS_DIR / name
    if not directory.is_dir():
        raise SystemExit(f"模型目录不存在：{directory}（先用 --fetch {name}）")

    files = {}
    for filename in spec["files"]:
        target = directory / filename
        if not target.is_file():
            raise SystemExit(f"缺文件 {target}：先把模型取全再写清单")
        files[filename] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }

    source = dict(spec["source"])
    if method:
        source["method"] = method
    if obtained_at:
        source["obtained_at"] = obtained_at

    manifest = {
        "name": name,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "source": source,
        "manifest_written_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    path = directory / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(entry["bytes"] for entry in files.values())
    print(f"写入清单 {path}：{len(files)} 个文件、{total / 1048576:.1f} MB")
    for filename, entry in files.items():
        print(f"  {filename:32} {entry['bytes']:>12,} B  sha256={entry['sha256'][:12]}…")
    return path


def fetch(name: str, endpoint: str) -> None:
    """按版本锁抓取缺失文件；抓完写清单并校验。"""

    spec = MODELS[name]
    directory = MODELS_DIR / name
    directory.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download

    print(f"抓取 {spec['repo']}@{spec['revision'][:12]}… → {directory}")
    print(f"  端点 {endpoint}（默认 hf-mirror；失败可就显式换 --endpoint）")
    snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        local_dir=str(directory),
        allow_patterns=spec["files"],
        max_workers=4,
    )
    write_manifest(
        name,
        method="huggingface_hub.snapshot_download(local_dir=…)",
        obtained_at=datetime.date.today().isoformat(),
    )
    verify_model(name, check_hashes=True)
    print(f"OK {name}")


def verify(names: list) -> int:
    exit_code = 0
    for name in names:
        try:
            info = verify_model(name, check_hashes=True)
        except ModelFilesUnavailable as error:
            print(f"FAIL {name}: {error}")
            exit_code = 1
            continue
        total = sum(entry["bytes"] for entry in info["files"].values())
        print(
            f"OK   {name}: {len(info['files'])} 个文件、{total / 1048576:.1f} MB、"
            f"repo={info['repo']}、revision={info['revision'][:12]}…"
        )
        for filename, entry in sorted(info["files"].items()):
            print(
                f"       {filename:32} {entry['bytes']:>12,} B  "
                f"sha256={entry['sha256'][:12]}…"
            )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="逐文件校验（默认）")
    group.add_argument("--write-manifest", action="store_true", help="按版本锁写清单")
    group.add_argument("--fetch", action="store_true", help="按版本锁抓取缺失文件")
    parser.add_argument("names", nargs="*", help="模型名；省略即全部/或 all")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="HF 端点（默认 hf-mirror）")
    args = parser.parse_args()

    names = _model_names(args.names)
    if args.write_manifest:
        for name in names:
            write_manifest(name)
        return 0
    if args.fetch:
        for name in names:
            fetch(name, args.endpoint)
        return 0
    return verify(names)


if __name__ == "__main__":
    raise SystemExit(main())
