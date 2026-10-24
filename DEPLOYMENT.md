# 部署手册（Windows 11 · 源码方式）

本手册面向 **MiroFish Local**：使用本地 Graphiti + 嵌入式 Kuzu、本地模型，不依赖 Zep Cloud。
它记录的是实际跑通过的步骤和踩过的坑，不是"照 upstream README 抄一遍"。

> **范围**：Windows 11 x64、Python 3.12、Node.js 18+、源码方式。
> Docker、Linux、macOS **没有验证**；现有 Dockerfile / docker-compose 是上游 cloud 镜像，
> 不能用来跑本地图谱模式。详见 §10。

## 1. 前置检查

| 项 | 要求 | 检查 |
|---|---|---|
| Windows | Windows 11 x64 | `winver` |
| Python | 3.11–3.12，推荐 3.12 | `python --version` / `uv python list` |
| Node.js | 18+（含 npm） | `node -v` / `npm -v` |
| uv | 最新版，用于安装依赖 | `uv --version` |
| GPU | 推荐 NVIDIA，8 GB+ 显存 | `nvidia-smi`（可选） |
| 磁盘 | 预留约 6–8 GB（venv + CUDA torch + 模型 + 缓存） | 资源管理器 |
| 网络 | 首次安装依赖和模型需要；之后可离线 | 代理/镜像按你的网络配置 |

没有 NVIDIA GPU 也能启动，但 twitter 推荐时间线会走 CPU，可能非常慢；必须显式设置
`MIROFISH_ALLOW_CPU_RECSYS=1`，不要把它当默认路径。

## 2. 获取代码与工具链

```powershell
git clone https://github.com/Friend-Xu/mirofish-local.git
cd mirofish-local
```

如果本机还没有 uv，先按 uv 官方方式安装（它只是供应工具，项目依赖仍会装进
`backend/.venv`）。希望工具链也留在项目内时，可以显式指定缓存/解释器目录：

```powershell
$env:UV_CACHE_DIR        = "$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
uv python install 3.12
```

安装后端（**必须带 `--extra graphiti`**）：

```powershell
cd backend
uv sync --frozen --extra graphiti
cd ..
```

安装前端（根目录和 `frontend` 各一次；用 lock 文件，别用 `npm install` 改锁）：

```powershell
npm ci
cd frontend
npm ci
cd ..
```

> 如果在受限沙箱/受限终端里遇到 `spawn EPERM` 或原生模块安装失败，换普通终端重试；
> 详见 §8 的 npm 条目。

## 3. 配置 `.env`

```powershell
copy .env.example .env
```

编辑 `.env`，至少填：

```env
LLM_API_KEY=你的key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus

ZEP_BACKEND=graphiti
```

本地模式**不需要** `ZEP_API_KEY`。如果你要回到上游的 Zep Cloud 行为，再改成
`ZEP_BACKEND=cloud` 并填 `ZEP_API_KEY`；本手册不覆盖 cloud 的配额/网络问题。

`.env` 的三个常见坑：

- 等号两边不要有空格；
- 值不要加引号（引号会被当成值的一部分）；
- 不要把配置行并到注释行后面；整行以 `#` 开头就等于没配。

`GRAPHITI_MIN_SCORE` **留空（不设置）** 即使用上游默认 0.6；如果显式设置，必须在
`(0, 1]` 内。**不要写 0**：`Config.validate()` 会直接拒绝，0 还会关闭相关性过滤。
`GRAPHITI_EMBEDDING_MODEL_PATH` 应指向 `models/bge-small-zh-v1.5`，不要再用旧的
`.hf-cache/models/...`。

## 4. 模型与推理引擎

模型权重不进版本库，必须用脚本按清单抓取并校验：

```powershell
backend\.venv\Scripts\python.exe scripts\provision-models.py --fetch
backend\.venv\Scripts\python.exe scripts\provision-models.py --verify
```

默认抓取端点是 hf-mirror；如果你的网络能直连 HuggingFace，可显式换端点：

```powershell
backend\.venv\Scripts\python.exe scripts\provision-models.py --fetch --endpoint https://huggingface.co
```

两个模型：

- `models/bge-small-zh-v1.5`（约 92 MB，Graphiti embedding）；
- `models/twhin-bert-base`（约 1.08 GB，OASIS twitter 推荐）。

文件缺失、字节数不符、sha256 不符、清单外多出权重都会失败；**不会联网偷偷回退**。
抓取完成后再设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，启动器会替你设置。

GPU 版 PyTorch：

```powershell
backend\.venv\Scripts\python.exe scripts\provision-torch.py --verify
```

如果校验失败，按脚本提示准备固定 wheel，再离线安装：

```powershell
backend\.venv\Scripts\python.exe scripts\provision-torch.py --install --wheel <wheel 目录> --uv <uv.exe>
backend\.venv\Scripts\python.exe scripts\provision-torch.py --verify
```

- wheel：`torch-2.9.1+cu128-cp312-cp312-win_amd64.whl`（约 2.7 GB）；
- sha256：`3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0`；
- 只需要 NVIDIA 驱动，**不需要** CUDA Toolkit；
- 没有 GPU：`set MIROFISH_ALLOW_CPU_RECSYS=1`，并接受较慢的推荐时间线；
- 8 GB 显存是硬边界；生产批大小守卫已内置，**不要绕过**。

## 5. 图库与数据

| 路径 | 内容 |
|---|---|
| `backend/uploads/graphs/graphiti.kuzu`（+ `.wal`） | 本地图库主文件 |
| `backend/uploads/graphs/ontology/<graph_id>.json` | 本体缓存；删掉会让回写退化为无本体抽取 |
| `backend/uploads/graphs/pending/` | 回写未送达时的落盘 |
| `backend/uploads/simulations/` | 模拟产物 |
| `backend/uploads/reports/` | 报告与 `meta.json` |

Kuzu 是**单进程文件锁**：

- 后端运行时，不要用另一个进程直接打开同一个 `.kuzu`；
- 要离线检查、复制、审计，先停后端；
- 备份主文件时同时考虑 `.wal`（最稳妥是停后端后整目录复制）。

重置本地图库（会丢图，不会自动删报告/上传文件）：

```powershell
# 先停后端
Remove-Item backend\uploads\graphs\graphiti.kuzu, backend\uploads\graphs\graphiti.kuzu.wal -ErrorAction SilentlyContinue
Remove-Item backend\uploads\graphs\ontology\* -Recurse -Force -ErrorAction SilentlyContinue
```

## 6. 启动与自检

```powershell
scripts\start-local.cmd
```

它会检查 `.venv`、`frontend/node_modules`、`.env`、两个模型和 GPU torch；
缺任何一项都会给出修复命令并退出，**不会自动下载**。只检查、不启动可以用：

```powershell
scripts\start-local.cmd --check
```

启动的是两个可见窗口：

- 后端：`http://localhost:5001/health`
- 前端：`http://localhost:3000`

手动启动（等价）：

```powershell
# 终端 1
cd backend
.\.venv\Scripts\python.exe run.py

# 终端 2
cd frontend
npm run dev
```

后端需要约 10–30 秒；Vite 把 `/api` 代理到 `:5001`，所以后端要先起来。

自检：

```powershell
Invoke-WebRequest http://localhost:5001/health -UseBasicParsing | Select-Object -Expand Content
# 期望 {"service":"MiroFish Backend","status":"ok"}

Invoke-WebRequest http://localhost:3000/ -UseBasicParsing | Select-Object StatusCode
# 期望 200
```

停止：关闭两个窗口；或按端口结束进程：

```powershell
Get-NetTCPConnection -LocalPort 3000,5001 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## 7. 第一次只跑小样

不要一上来跑默认 10 轮。先用小规模（例如 3 个 agent、3 轮、单平台 twitter）走通：

1. 图谱构建有产物；
2. 人设/环境准备完成；
3. 模拟有动作，且 `information_flow` 不是空；
4. 回写排空（`ingestion_drained=true`，`undelivered_items=0`）；
5. 报告 `status=completed`，章节非空；
6. 报告 `meta.grounding.final_ok=true`，最终违规 0；
7. 抽检/人工看几条事实与推断。

当前代码在 `sim_20808935737b` 上的实测：3 轮 / 99 条动作 / 4 章报告 / 352.2 秒 /
`R1_fact=100%`、`R2=0`、`unlabeled=0`、`quote_violations=0`。
`R1_all=40%` 是本地 legacy 指标，继续记录，**当前不作为 N-44 门槛**。

## 8. 故障排查（按实际踩坑）

| 症状 | 原因 | 处理 |
|---|---|---|
| `uv` 不存在 | 没装供应工具 | 安装 uv；或把 `uv.exe` 放到项目内并显式使用 |
| Python 3.13 下 PyO3/tiktoken 报错 | 版本超出 `>=3.11,<3.13` | `uv python install 3.12`，重新 `uv sync --frozen --extra graphiti` |
| 用 `uv run` 启动时重建环境/下载 torch | `uv run` 会按锁重建 | 启动只用 `backend\.venv\Scripts\python.exe run.py`；uv 只负责安装 |
| `torch.cuda.is_available()` 为 False | 默认索引给了 CPU-only torch | `scripts\provision-torch.py --verify`；必要时按固定 wheel 离线安装 |
| `GPU is lost` / CUDA OOM | 8 GB 卡在长帖/大语料下到边界 | 不要绕过批大小守卫；关闭占显存程序；必要时用 CPU 放行 |
| 模型缺失 / hash 不符 | 未抓取或文件被替换 | `provision-models.py --fetch`，再 `--verify`；不要提前设 `HF_HUB_OFFLINE` |
| `GRAPHITI_MIN_SCORE=0` 配置校验失败 | 0 会关闭相关性过滤 | 留空（上游默认 0.6）或设 `(0,1]` |
| 模型路径指向 `.hf-cache` | 旧配置 | 改为 `models/bge-small-zh-v1.5` |
| `langdetect` 构建失败 / PyMuPDF 下载超时 | 受限终端/大文件超时 | 换普通终端；`UV_HTTP_TIMEOUT=600` 后重试 |
| npm `spawn EPERM` / esbuild 失败 | 受限沙箱或 postinstall | 换普通终端；必要时 `npm ci --ignore-scripts` 并确认原生 optional deps |
| `npm run build` 报 `Access is denied` 删除 `%TEMP%\esbuild-*` | 系统临时目录权限/清理失败 | 把 TEMP/TMP 指到项目内再 build：`$env:TEMP="$PWD\.tmp\esbuild-temp"; $env:TMP=$env:TEMP; npm run build` |
| `LLM_API_KEY 未配置` | `.env` 行被注释、等号空格、引号 | 修 `.env`；重启后端 |
| 端口 5001/3000 占用 | 旧进程还在 | 先停旧后端/前端；同一 Kuzu 不要双后端 |
| 离线脚本打不开图库 | Kuzu 单进程锁 | 先停后端，或走 API |
| tiktoken 首次运行卡住/退出 | 需要下载 `o200k_base.tiktoken` | 给网络或预置缓存；不要把网络失败当成模拟逻辑错误 |
| PowerShell `.ps1` 被执行策略/杀软拦截 | 未签名脚本/启发式 | 用 `.cmd` 或手动可见窗口；本项目不依赖隐藏窗口/重定向 |
| `docker compose up` 跑不起来本地模式 | compose 是上游 cloud 镜像 | Docker 不支持本地 Graphiti；按本手册源码部署 |
| 长时间模拟中断只能重来 | 当前无断点续跑 | 先小样；长跑前算预算 |

## 9. 升级、回退、卸载

升级：

```powershell
git pull
cd backend
uv sync --frozen --extra graphiti
cd ..
npm ci
cd frontend
npm ci
cd ..
```

升级后先 `scripts\start-local.cmd` 自检；图库/报告数据在 `backend/uploads/`，升级不会自动清理，
但**升级前建议停后端并备份图库**。

回退：`git checkout <commit>` 后重复安装命令；模型/图库是否兼容要看变更说明。

卸载：删除整个项目目录即可（依赖、模型、缓存都在项目内；`.env` 也只在项目里）。
如果用了用户级 uv/Python，它们不属于项目，按需自行清理。

## 10. 不支持/未验证的边界

- **Docker**：`Dockerfile` / `docker-compose.yml` 是上游 cloud 镜像，未包含本地模型、
  CUDA torch、Graphiti extra；本地模式不支持。
- **Linux/macOS**：没有实跑；固定 wheel 是 `win_amd64`，启动脚本是 `.cmd`。
- **Zep Cloud**：本分支可切回，但本手册不覆盖其配额/网络问题。
- **默认 10 轮 L-c / L-d、前端 Step1–5 全流程**：尚未在本分支完成连续实测。
- **R1_all**：legacy 指标，40% 继续记录；是否作为 N-44 门槛待研究，当前不作门槛。

## 11. 上游与许可证

- 上游：<https://github.com/666ghj/MiroFish>
- 本分支基于 upstream commit `7657031ac01184afe2cb220f5ee3545573b5e843`。
- 许可证：**AGPL-3.0**（见 [LICENSE](LICENSE)）。这是社区/本地化分支，不是上游官方仓库。
- 修改与分发请保留上游版权与许可证声明；网络服务场景请自行确认 AGPL 义务。
