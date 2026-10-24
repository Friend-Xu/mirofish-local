# MiroFish Local（本地图谱版）

**MiroFish 的本地优先分支：Graphiti + 嵌入式 Kuzu、本地模型、不依赖 Zep Cloud，报告区分事实与推断并可核对引用。**

[English](README.md) · [文档索引](docs/README.md) · [部署手册](docs/DEPLOYMENT.md) · [上游 MiroFish](https://github.com/666ghj/MiroFish)

![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-lightgrey)
![Graph: local Kuzu](https://img.shields.io/badge/graph-Graphiti%20%2B%20Kuzu-2f6f4e)

> **这不是 MiroFish 官方仓库。** 本仓库是基于上游 commit
> `7657031ac01184afe2cb220f5ee3545573b5e843` 的社区/本地化分支；上游项目与团队不对本分支负责。
> 许可证：AGPL-3.0。

## 这个分支改了什么

- **本地图谱后端**：`ZEP_BACKEND=graphiti` 使用 Graphiti + 嵌入式 Kuzu，本地模式不需要
  Zep Cloud 账号或 `ZEP_API_KEY`。
- **离线模型**：Graphiti embedding 与 twitter 推荐模型放在 `models/`，逐文件校验字节数和
  SHA-256；不启动时偷偷联网下载。
- **报告 grounding**：正文强制区分 `**模拟事实（可溯源）**` 与 `**作者推断（非模拟事实）**`，
  推断行用 `[推断]`，引号必须逐字；翻译用 `[译文]`、概括用 `[转述]` 且不加引号。
  生成侧有最多两次有界纠正，仍失败就报告 `failed`，不清洗、不截断。
- **小样实测**：当前报告链只在一个 3 轮/99 动作的模拟上完成实测；不代表所有部署模式都已
  生产就绪。

## 当前实测状态

| 项 | 结果 |
|---|---|
| 模拟 | `sim_20808935737b`：40 agent × 3 轮 × twitter，99 条动作 |
| 回写 | `items_sent=99/99`、`undelivered=0`、`delivery_ratio=1.0`、`ingestion_drained=true` |
| 报告 | `report_e3a7f1a1c8bc`：`completed`、4/4 章、352.2 秒 |
| grounding gate | 最终违规 0、`unlabeled=0`、`quote_violations=0` |
| 事实可追溯 | **R1_fact = 8/8 = 100%** |
| 编造检查 | **R2 = 0/20** |
| 旧指标 | `R1_all = 40%`（12 条标注推断按旧口径计 partial）。**legacy：继续记录、与 52%/48% 对照；是否作为 N-44 门槛待研究，当前不作门槛。** |

**尚未验证**：L-c（默认 10 轮）、L-d（全轮数）、Zep Cloud 模式、前端 Step1–5 浏览器全流程、
Linux/macOS、本地模式 Docker。

## 快速开始（Windows 11，源码）

### 前置要求

- Windows 11 x64
- Python 3.11–3.12（推荐 3.12）与 [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- 可选但强烈推荐：NVIDIA GPU，8 GB+ 显存
- 预留约 6–8 GB 磁盘（venv + CUDA torch + 本地模型）

### 安装与启动

```powershell
git clone https://github.com/Friend-Xu/mirofish-local.git
cd mirofish-local

# 后端（本地图谱 extra 必须装）
cd backend
uv sync --frozen --extra graphiti
cd ..

# 前端
npm ci
cd frontend
npm ci
cd ..

# 本地模型与 GPU 版 torch
backend\.venv\Scripts\python.exe scripts\provision-models.py --fetch
backend\.venv\Scripts\python.exe scripts\provision-models.py --verify
backend\.venv\Scripts\python.exe scripts\provision-torch.py --verify

# 本地配置
copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY；ZEP_BACKEND=graphiti 已是默认。
# 本地模式不需要 ZEP_API_KEY。

# 启动两个可见窗口（后端 :5001，前端 :3000）
scripts\start-local.cmd
```

没有 NVIDIA GPU 时，显式允许 CPU（推荐时间线会很慢）：

```powershell
set MIROFISH_ALLOW_CPU_RECSYS=1
scripts\start-local.cmd
```

自检：

- <http://localhost:5001/health> → `{"service":"MiroFish Backend","status":"ok"}`
- <http://localhost:3000/> → 200

完整的配置、备份/重置和故障排查见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。
工具链/锁/模型哈希的维护者说明见 [docs/PROJECT_SETUP.md](docs/PROJECT_SETUP.md)。

## 部署时最容易踩的坑

- **不要用 `uv run` 启动后端**：它可能按锁重建环境并下载 torch。启动只用
  `backend\.venv\Scripts\python.exe run.py`；`uv` 只负责安装。
- **不要把 `GRAPHITI_MIN_SCORE` 写成 0**：留空即用上游默认 0.6；显式值必须在 `(0,1]`。
  0 会关闭相关性过滤，并被 `Config.validate()` 拒绝。
- **模型路径不要再指向 `.hf-cache`**：应为 `models/bge-small-zh-v1.5`。
- **Kuzu 是单进程文件锁**：后端运行时不要用另一个进程直接打开 `.kuzu`；离线检查/审计先停后端。
- **Docker 不支持本地模式**：现有 Dockerfile / docker-compose 是上游 cloud 镜像；请按源码部署。
- **不要提交 `.env`**：它已被忽略；从 `.env.example` 复制。

## 数据与重置

运行数据都在 `backend/uploads/`：

- `graphs/graphiti.kuzu`（+ `.wal`）：本地图库；
- `graphs/ontology/<graph_id>.json`：本体缓存；
- `simulations/`：模拟产物；
- `reports/`：报告与 `meta.json`。

复制或删除图库前先停后端。删除整个项目目录即可卸载；依赖、模型、缓存都在项目内。

## 许可证与上游

- 上游：<https://github.com/666ghj/MiroFish>
- 本分支基于上游 commit `7657031ac01184afe2cb220f5ee3545573b5e843`。
- 许可证：**AGPL-3.0**，见 [LICENSE](LICENSE)。分发时保留上游版权与许可证声明；
  网络服务场景请自行确认 AGPL 义务。
- 仿真引擎由 [OASIS](https://github.com/camel-ai/oasis) 驱动，感谢 CAMEL-AI 团队。
