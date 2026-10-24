"""项目自有推理引擎（GPU 版 PyTorch）的供应与校验。`--install` / `--verify`。

为什么需要这个文件（AGENTS.md §2：引擎必须由项目自己拥有、版本钉死、只入库供应步骤）：

* 项目 venv 里的 torch 是**从清华镜像解析来的 CPU 版**（`uv.lock` 里 torch 只是
  `sentence-transformers` 的传递依赖）。CPU 版里没有 CUDA 支持代码，`torch.cuda.is_available()`
  **恒为 False**，于是：
  ① OASIS 的 twitter 推荐表（`recsys.py:46/85` 用 `torch.device("cuda" if ... else "cpu")`）
     永远走 CPU；
  ② 每轮全量语料（画像 + 全部帖子，最多 4000 条）在 CPU 上编码：实测 92 条 9.1 s、
     400 条 41.4 s、4000 条约 6.9 min，168 轮累计小时级，并且会把机器吃满（实测卡死一次）。
* 模型本身只有 1.08 GB、279M 参数，是 GPU 最擅长的活。

**这个文件不下载任何东西**：wheel 由人准备好（官方直链或浏览器），脚本只做三件事——
按 SHA256 校验、装进**项目自己的 venv**、断言 GPU 真的可用。
安装时使用一个**显式指定**的 uv：默认找项目内 `.tools/bin/uv.exe`，否则必须传
`--uv <path>` 或设 `MIROFISH_UV`（不从 PATH 静默取一个同名程序）。uv 建的 venv
里没有 pip：实测 `python -m pip` 报 `No module named pip`。

用法（项目自己的解释器，不要用系统 python）：

    # 1) 校验（先看 wheel 在不在、哈希对不对、GPU 通路通不通）
    backend\\.venv\\Scripts\\python.exe scripts\\provision-torch.py --verify

    # 2) 离线安装（wheel 放在哪都行，用 --wheel 指路径；默认会去几个常见位置找）
    backend\\.venv\\Scripts\\python.exe scripts\\provision-torch.py --install --wheel <wheel 目录> --uv <uv.exe>

    # 3) 再看一次
    backend\\.venv\\Scripts\\python.exe scripts\\provision-torch.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

PROJECT = pathlib.Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT / "backend" / ".venv" / "Scripts" / "python.exe"
#: 默认的项目内 uv；公开 clone 没有它时，用户必须显式传 --uv 或设 MIROFISH_UV
DEFAULT_UV = PROJECT / ".tools" / "bin" / "uv.exe"
#: 项目内首选位置（字节不进版本库，目录已在 .gitignore 里）
ARTIFACTS = PROJECT / ".tools" / "artifacts"

#: 版本锁：改这里等于换引擎，必须同时改 SHA256 与文档里的数字
WHEEL = {
    "filename": "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl",
    "torch_version": "2.9.1",
    "cuda": "cu128",
    "tag": "cp312-cp312-win_amd64",
    "sha256": "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0",
    "url": "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl",
    "bytes": 2862033275,  # 实测 2729.4 MiB（HEAD 的 Content-Length 报 2862211264，以文件实际长度为准）
}
#: 默认搜索位置：手动下载时常放的目录（按顺序找第一个命中的）
SEARCH_DIRS = (
    ARTIFACTS,
    pathlib.Path.cwd(),
    pathlib.Path.home() / "Downloads",
)

INSTALL_HINT = (
    "wheel 不在默认位置。请二选一：\n"
    f"  · 用 --wheel <路径> 指定它（哈希会先校验）\n"
    f"  · 或用浏览器/下载工具取这个直链后再跑 --install：\n"
    f"      {WHEEL['url']}\n"
    f"    期望 SHA256: {WHEEL['sha256']}\n"
    "安装还需要显式指定 uv：--uv <uv.exe> 或设置 MIROFISH_UV。"
)


def find_wheel(explicit: str | None) -> pathlib.Path:
    """按 `--wheel` → 默认目录的顺序找 wheel；找不到就报错并给出直链。"""

    if explicit:
        candidate = pathlib.Path(explicit)
        if candidate.is_dir():
            candidate = candidate / WHEEL["filename"]
        if not candidate.exists():
            raise SystemExit(f"--wheel 指向的文件不存在：{candidate}\n{INSTALL_HINT}")
        return candidate
    for directory in SEARCH_DIRS:
        candidate = directory / WHEEL["filename"]
        if candidate.exists():
            return candidate
    raise SystemExit(INSTALL_HINT)


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheel(path: pathlib.Path) -> dict:
    """校验文件身份：文件名、字节数、SHA256 三者都要对。"""

    size = path.stat().st_size
    if path.name != WHEEL["filename"]:
        raise SystemExit(f"文件名不是锁定的那个：{path.name}（期望 {WHEEL['filename']}）")
    if size != WHEEL["bytes"]:
        raise SystemExit(
            f"字节数不符：{size}（期望 {WHEEL['bytes']}）——下载不完整或被换过"
        )
    print(f"计算 SHA256（{size / 1024 / 1024:.0f} MB，约十几秒）…", flush=True)
    digest = sha256_of(path)
    if digest != WHEEL["sha256"]:
        raise SystemExit(
            f"SHA256 不符：\n  实际 {digest}\n  期望 {WHEEL['sha256']}\n"
            f"文件被换过或下载损坏，拒绝安装"
        )
    print(f"wheel 校验通过：{path}")
    print(f"  {WHEEL['filename']}  {size} 字节  sha256={digest}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def report_runtime() -> dict:
    """把当前 venv 的 GPU 能力读一遍（不加载任何模型）。"""

    import torch

    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_mb": None,
        "threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["total_vram_mb"] = round(properties.total_memory / 1024 / 1024)
        info["capability"] = f"sm_{properties.major}{properties.minor}"
    return info


def assert_cuda(info: dict) -> None:
    """判据是**行为**而不是版本字符串：`torch.version.cuda` 必须非空、`is_available()` 必须真。

    为什么不查 `"+cu128" in torch.__version__`：torch 允许不写本地版本标记，
    "能跑 CUDA"才是这件事的判据。
    """

    problems = []
    if info["cuda_build"] is None:
        problems.append("torch.version.cuda 为 None → 这是 CPU-only 构建")
    if not info["cuda_available"]:
        problems.append("torch.cuda.is_available() 为 False → 有卡也用不上")
    if info["device_count"] < 1:
        problems.append("torch.cuda.device_count() = 0")
    if problems:
        raise SystemExit(
            "GPU 不可用：\n  · " + "\n  · ".join(problems)
            + f"\n当前 torch: {info['torch']}"
            + "\n修法：用 --install 装入 GPU 版 wheel（命令见文件头用法；"
              "本项目 venv 没有 pip，必须用一个显式指定的 uv）"
        )


def do_verify() -> int:
    info = report_runtime()
    print("运行期能力：")
    for key in ("torch", "cuda_build", "cuda_available", "device_count", "device_name",
                "total_vram_mb", "capability", "threads"):
        if key in info:
            print(f"  {key:<16} {info[key]}")
    assert_cuda(info)
    print("GPU 可用。注意：模型是否真的在 GPU 上跑，还要看 "
          "`twitter_recsys_model.load_and_inject()` 的日志里那行设备信息。")
    return 0


def resolve_uv(explicit: str | None) -> pathlib.Path:
    """显式解析 uv：`--uv`/`MIROFISH_UV` → 项目内默认；否则失败，不静默走 PATH。"""

    value = explicit or os.environ.get("MIROFISH_UV")
    if value:
        candidate = pathlib.Path(value).expanduser()
        if candidate.exists():
            return candidate
        found = shutil.which(value)
        if found:
            return pathlib.Path(found)
        raise SystemExit(
            f"找不到 uv：{value}\n"
            "请传 --uv <uv.exe 的绝对路径>，或设置 MIROFISH_UV；"
            "不要依赖 PATH 里恰好有另一个 uv。"
        )
    if DEFAULT_UV.exists():
        return DEFAULT_UV
    raise SystemExit(
        f"项目内默认 uv 不存在：{DEFAULT_UV}\n"
        "请传 --uv <uv.exe 的绝对路径> 或设置 MIROFISH_UV；"
        "安装 uv 见 DEPLOYMENT.md 的『工具链』一节。"
    )


def do_install(wheel: pathlib.Path, uv: pathlib.Path) -> int:
    verify_wheel(wheel)
    if not VENV_PYTHON.exists():
        raise SystemExit(
            f"项目 venv 不存在：{VENV_PYTHON}\n"
            f"先在 backend 目录跑 `uv sync --frozen --extra graphiti`"
        )
    # venv 由 uv 建，里面**没有 pip**；用显式 uv 装，--python 钉住目标 venv
    command = [
        str(uv), "pip", "install",
        "--python", str(VENV_PYTHON),
        "--no-deps",      # torch 的依赖（filelock/fsspec/jinja2/networkx/sympy/typing-extensions）venv 里已有
        "--no-index",     # 绝不联网：就用这一个本地文件
        "--reinstall",    # 现有的是 CPU 版，版本号相同，必须强制覆盖
        str(wheel),
    ]
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(PROJECT / ".uv-cache")
    # uv 解包 2.7 GB 会用临时目录：默认落在用户 %TEMP%，实测**权限被拒**
    # （`os error 5 at C:\Users\0\AppData\Local\Temp\.tmpUG9kwR`）——虽然包已经装好了，
    # 但清理失败会让退出码变 2。把 temp 也钉进项目目录，任何机器上行为一致（AGENTS.md §2）。
    scratch = PROJECT / ".uv-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TEMP", "TMP"):
        environment[name] = str(scratch)
    print("安装命令：\n  " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        # 退出码不为 0 时**不信退出码，信 import**：uv 在清理阶段失败时包其实已经装好，
        # 直接判失败会让人重装一遍 2.7 GB。真正的判据是下面那个新进程自检。
        print(f"注意：uv 退出码 {completed.returncode}（可能是清理临时目录失败），"
              f"继续用新进程自检确认到底装上了没有", flush=True)
    # 装完立刻用**新进程**读一次：同一个进程里的 torch 还是旧的
    probe = subprocess.run(
        [str(VENV_PYTHON), "-c",
         "import torch;print(torch.__version__);print(torch.version.cuda);"
         "print(torch.cuda.is_available());"
         "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"],
        capture_output=True, text=True, check=False,
    )
    print("装后自检：")
    print(probe.stdout.strip() or probe.stderr.strip())
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if probe.returncode != 0 or len(lines) < 4 or lines[2] != "True":
        raise SystemExit("装完 CUDA 仍不可用——看上面的自检输出（第 3 行必须恰好是 True）")
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--install", action="store_true", help="校验 wheel 并离线装进项目 venv")
    parser.add_argument("--verify", action="store_true", help="断言当前 venv 的 GPU 可用")
    parser.add_argument("--wheel", default=None,
                        help="wheel 路径或所在目录（默认在 .tools/artifacts、"
                             "当前目录、Downloads 里找）")
    parser.add_argument("--uv", default=None,
                        help="uv 可执行文件路径；也可用 MIROFISH_UV 指定。"
                             "默认使用项目内 .tools/bin/uv.exe（若存在）。")
    args = parser.parse_args(argv)

    if args.install:
        wheel = find_wheel(args.wheel)
        uv = resolve_uv(args.uv)
        if wheel.parent != ARTIFACTS:
            print(f"提示：wheel 在项目外（{wheel.parent}）。字节不进版本库，"
                  f"放在哪都能装；要有项目内的副本可以手动复制到 {ARTIFACTS}")
        return do_install(wheel, uv)
    if args.verify:
        return do_verify()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
