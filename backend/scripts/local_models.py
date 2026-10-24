"""项目自有模型目录：位置、清单、校验。

为什么有这个模块（掩盖台账 N-46 / 批次 F18）：

twitter 的推荐模型（`Twitter/twhin-bert-base`）原先躺在 HuggingFace 的 **hub 缓存布局**
里（`<项目>/.hf-cache/hub/models--Twitter--twhin-bert-base/{blobs,snapshots,refs}/…`）。
那个布局是 huggingface_hub 的内部约定，不是我们的：加载要读 `refs/main` 才知道快照
哈希；同一个 1.12 GB 权重在 hub 里存了两份（`blobs/` + `snapshots/`，实测 2.16 GB）；
还混着 `.no_exist/`、`.locks/`、下到一半的 `tmp*` 目录。图谱的 embedding 模型
（`bge-small-zh-v1.5`）本来就是普通目录，twhin 是唯一的例外。

现在两者统一成：**项目内的普通目录 + 一份清单**。

    <项目>/models/<name>/          ← 模型文件，直接喂给 transformers / sentence-transformers
    <项目>/models/<name>/model.json ← 清单：repo、revision、每个文件的字节数与 sha256

规则（这是边界校验，不是兜底）：

* 目录是**项目内的常量**，不读环境变量——加载位置不能取决于谁启动的后端；
* 清单是唯一权威：缺目录、缺清单、缺键、文件缺失、字节数不符、sha256 不符，
  **一律抛 `ModelFilesUnavailable`**，并在信息里写清缺什么、期望值与实际值；
* 目录里出现清单未声明的第二份权重文件（`*.safetensors` / `*.bin`）也抛：
  那意味着加载的可能不是清单里那份；
* 本模块**不下载、不回退**。供应由 `.tools/provision-models.py` 负责，
  它同时是版本锁（repo + revision + 文件清单）。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict

# `<项目>/` = 本文件（`<项目>/backend/scripts/`）往上三层
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: 项目自有模型根目录（不读 `HF_HOME`，也不存在 hub 布局）
MODELS_DIR = PROJECT_ROOT / "models"

#: 清单文件名
MANIFEST_NAME = "model.json"

#: 清单里必须有、且不能为空的键
REQUIRED_MANIFEST_KEYS = ("name", "repo", "revision", "files")

#: 权重文件后缀：清单之外出现这些文件就说明目录里有第二份权重
WEIGHT_SUFFIXES = (".safetensors", ".bin")

#: 校验失败时给出的供应入口（写在这里，保证每条错误信息都能直接照做）
PROVISION_HINT = (
    "供应/校验入口：\n"
    "  backend\\.venv\\Scripts\\python.exe .tools\\provision-models.py --verify\n"
    "  backend\\.venv\\Scripts\\python.exe .tools\\provision-models.py --fetch <name>\n"
    "见 LOCAL-DEPLOY.md「本地模型」一节。项目按离线运行，不会在运行时下载。"
)


class ModelFilesUnavailable(RuntimeError):
    """项目内的模型文件不可用：位置、清单、缺失/不符的文件都写在这条信息里。

    这是**边界**错误，不是可以继续跑的状态：模型不完整或内容不符时，
    继续跑只会把"加载到的不是清单里那份"变成更难查的现场。
    """


def model_dir(name: str) -> pathlib.Path:
    """项目内的模型目录；不是目录即抛错。"""

    if not name:
        raise ModelFilesUnavailable("模型名不能为空")
    directory = MODELS_DIR / name
    if not directory.is_dir():
        raise ModelFilesUnavailable(
            f"模型目录不存在：{directory}（应有 {MANIFEST_NAME} 与模型文件）\n"
            f"{PROVISION_HINT}"
        )
    return directory


def manifest_path(name: str) -> pathlib.Path:
    return model_dir(name) / MANIFEST_NAME


def read_manifest(name: str) -> Dict[str, Any]:
    """读并结构化校验清单；任何一项不合法即抛错（不给默认值）。"""

    path = manifest_path(name)
    if not path.is_file():
        raise ModelFilesUnavailable(
            f"缺少模型清单：{path}\n{PROVISION_HINT}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ModelFilesUnavailable(
            f"模型清单读不动或不是 JSON：{path}（{type(error).__name__}: {error}）"
        ) from error
    if not isinstance(payload, dict):
        raise ModelFilesUnavailable(f"模型清单不是对象：{path}")

    missing = [
        key for key in REQUIRED_MANIFEST_KEYS if payload.get(key) in (None, "", {}, [])
    ]
    if missing:
        raise ModelFilesUnavailable(f"模型清单 {path} 缺字段 {missing}")

    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ModelFilesUnavailable(f"模型清单 {path} 的 files 必须是非空对象")
    for filename, meta in files.items():
        if not isinstance(meta, dict) or "bytes" not in meta or "sha256" not in meta:
            raise ModelFilesUnavailable(
                f"模型清单 {path} 的 files[{filename!r}] 必须含 bytes 与 sha256，"
                f"实际是 {meta!r}"
            )
    return payload


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """文件的 sha256（本模块与供应工具共用同一个实现，避免口径漂移）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reject_undeclared_weights(directory: pathlib.Path, declared: set) -> None:
    """清单之外的第二份权重：抛错（否则加载的可能是另一份）。"""

    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        relative = candidate.relative_to(directory).as_posix()
        if relative in declared:
            continue
        raise ModelFilesUnavailable(
            f"模型目录 {directory} 里有清单未声明的权重文件：{relative}\n"
            f"清单声明的权重：{sorted(declared)}——加载的可能不是清单里那份。"
        )


def verify_model(name: str, *, check_hashes: bool = True) -> Dict[str, Any]:
    """按清单校验模型文件，返回可供打印/记录的摘要。

    Args:
        name: 模型目录名（如 `twhin-bert-base`）。
        check_hashes: True 时逐文件算 sha256（1.12 GB 实测 1.3 秒）；
            False 只查存在性与字节数——给"只想快速看一眼"的探针用。

    Raises:
        ModelFilesUnavailable: 目录/清单/文件/字节数/哈希/额外权重任一不符。
    """

    directory = model_dir(name)
    manifest = read_manifest(name)
    files: Dict[str, Any] = manifest["files"]

    problems = []
    summary: Dict[str, Any] = {}
    for filename, meta in files.items():
        target = directory / filename
        expected_bytes = int(meta["bytes"])
        if not target.is_file():
            problems.append(f"缺文件 {filename}（期望 {expected_bytes} 字节）")
            continue
        actual_bytes = target.stat().st_size
        if actual_bytes != expected_bytes:
            problems.append(
                f"{filename} 字节数不符：期望 {expected_bytes}，实际 {actual_bytes}"
            )
            continue
        entry = {"bytes": actual_bytes}
        if check_hashes:
            expected_hash = str(meta["sha256"]).lower()
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                problems.append(
                    f"{filename} sha256 不符：期望 {expected_hash[:12]}…，"
                    f"实际 {actual_hash[:12]}…"
                )
                continue
            entry["sha256"] = actual_hash
        summary[filename] = entry

    if problems:
        raise ModelFilesUnavailable(
            f"模型 {name} 的校验未通过（目录 {directory}，清单 {MANIFEST_NAME}）：\n  - "
            + "\n  - ".join(problems)
            + f"\n{PROVISION_HINT}"
        )

    _reject_undeclared_weights(directory, set(files))

    return {
        "name": name,
        "dir": str(directory),
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "files": summary,
        "hashes_checked": bool(check_hashes),
    }


def _main() -> int:
    """手动核对：逐个模型打印清单摘要与校验结果。"""

    import sys

    names = sys.argv[1:] or ["twhin-bert-base", "bge-small-zh-v1.5"]
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
            f"OK   {name}: {len(info['files'])} 个文件、{total / 1048576:.1f} MB，"
            f"repo={info['repo']} revision={info['revision'][:12]}…"
        )
        for filename, entry in sorted(info["files"].items()):
            print(
                f"       {filename:32} {entry['bytes']:>12,} B  "
                f"sha256={entry['sha256'][:12]}…"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
