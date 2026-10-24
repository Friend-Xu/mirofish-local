# 工具链与锁（维护者参考）

本文件说明**项目自己的依赖树、模型和推理引擎**如何被固定；面向使用者的完整部署步骤
在 [DEPLOYMENT.md](DEPLOYMENT.md)。不要把本文件当成快速开始。

## 1. 原则

- 依赖装在项目目录：`backend/.venv`、`frontend/node_modules`。
- 版本由锁文件决定：`backend/uv.lock`、`package-lock.json`；不提交字节。
- `uv` / Python 是**供应工具**，可以用户级安装；安装出来的依赖必须落在项目内。
- 模型权重（约 1.2 GB）和 CUDA torch（约 2.7 GB wheel）不进版本库，由 `scripts/` 校验和供应。

## 2. 语言运行时与依赖

| 组件 | 落点 | 说明 |
|---|---|---|
| Python | 3.11–3.12，推荐 3.12 | 3.13 会触发 PyO3/tiktoken 兼容问题 |
| `uv` | 用户级或项目本地均可 | 只用于安装/供应，不在启动路径上 |
| Python 依赖 | `backend/.venv` | 由 `backend/uv.lock` 锁定 |
| Node 依赖 | `root/node_modules`、`frontend/node_modules` | 由两个 lock 文件锁定 |

```powershell
# 后端（在 backend 下执行；本地图谱 extra 必装）
uv sync --frozen --extra graphiti

# 前端（根目录和 frontend 各一次）
npm ci
cd frontend
npm ci

# 测试：必须用项目 venv
cd ..\backend
.\.venv\Scripts\python.exe -m pytest -q
```

如果希望工具链也留在项目内，可显式设置：

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
uv python install 3.12
```

## 3. 模型（Graphiti embedding + twitter 推荐）

两个模型都在 `models/<name>/`，各带 `model.json` 清单（repo、revision、文件字节数、sha256）：

| 模型 | 目录 | 大小 | 用途 |
|---|---|---|---|
| `BAAI/bge-small-zh-v1.5` | `models/bge-small-zh-v1.5` | 92 MB | Graphiti 检索 embedding（dim=512） |
| `Twitter/twhin-bert-base` | `models/twhin-bert-base` | 1.08 GB | OASIS twitter 推荐时间线 |

```powershell
backend\.venv\Scripts\python.exe scripts\provision-models.py --verify
backend\.venv\Scripts\python.exe scripts\provision-models.py --fetch
backend\.venv\Scripts\python.exe scripts\provision-models.py --write-manifest
```

- 默认抓取端点是 `hf-mirror.com`；可用 `--endpoint` 显式换源。
- 文件缺失、字节数不符、sha256 不符、清单外多出权重 → 抛错；不联网回退。
- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 只在模型已就绪后设置。

## 4. 推理引擎（GPU 版 PyTorch）

`backend/pyproject.toml` 固定 `torch==2.9.1`，并使用 `pytorch-cu128` 显式索引：

```powershell
# 校验：不仅看版本，还断言 cuda.is_available()
backend\.venv\Scripts\python.exe scripts\provision-torch.py --verify

# 如果默认索引仍给出 CPU 版，按文件头准备 wheel 后离线安装：
backend\.venv\Scripts\python.exe scripts\provision-torch.py --install --wheel <wheel 目录> --uv <uv.exe>
```

- wheel：`torch-2.9.1+cu128-cp312-cp312-win_amd64.whl`（2729.4 MiB）；
- sha256：`3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0`；
- 不需要 CUDA Toolkit，只需要 NVIDIA 驱动；
- 没有 GPU 的机器用 `MIROFISH_ALLOW_CPU_RECSYS=1` 显式放行，但推荐时间线会慢很多；
- 8 GB 显存是硬边界；生产批大小守卫在 `load_and_inject()` 中，**不要绕过**。

## 5. 图数据库

- 引擎：`kuzu`（嵌入式，装在 `backend/.venv`；Windows 有 cp312 轮子）。
- 文件：`backend/uploads/graphs/graphiti.kuzu`（+ `.wal`）。
- 本体缓存：`backend/uploads/graphs/ontology/<graph_id>.json`；删掉会让回写退化为无本体抽取。
- **单进程文件锁**：后端运行时，离线脚本不要直接打开同一库；先停后端或走 API。

## 6. 后端选择

- `ZEP_BACKEND=graphiti`（本分支默认）：本地 Graphiti + Kuzu，不需要 `ZEP_API_KEY`。
- `ZEP_BACKEND=cloud`：上游 Zep Cloud 行为，需要 `ZEP_API_KEY`。

模型、torch、图库的排错和完整命令见 [DEPLOYMENT.md](DEPLOYMENT.md)。
