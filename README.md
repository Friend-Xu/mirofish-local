# MiroFish Local

**A local-first fork of MiroFish: Graphiti + embedded Kuzu, local models, no Zep Cloud, and traceable reports.**

[中文完整说明](README-ZH.md) · [Deployment guide](DEPLOYMENT.md) · [Upstream MiroFish](https://github.com/666ghj/MiroFish)

![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Platform: Windows 11](https://img.shields.io/badge/platform-Windows%2011-lightgrey)
![Graph: local Kuzu](https://img.shields.io/badge/graph-Graphiti%20%2B%20Kuzu-2f6f4e)

> **Not the official MiroFish repository.** This is a community/local-first fork based on
> upstream commit `7657031ac01184afe2cb220f5ee3545573b5e843`. The upstream project and team
> are not responsible for this fork. License: AGPL-3.0.

## What this fork changes

- **Local graph backend**: `ZEP_BACKEND=graphiti` uses Graphiti + embedded Kuzu; no Zep Cloud
  account or API key is required in local mode.
- **Offline models**: Graphiti embedding and the twitter recommendation model live under
  `models/`, verified with per-file byte counts and SHA-256.
- **Report grounding**: generated reports separate `**模拟事实（可溯源）**` from
  `**作者推断（非模拟事实）**`, require `[推断]` markers, and check quotes verbatim
  (`[译文]` / `[转述]` are explicit alternatives). A bounded correction pass runs before a
  report is allowed to complete.
- **Small-sample validation**: the current report chain has a measured 3-round run; see
  “Status” below. This is not a claim that every deployment mode is production-ready.

## Status (measured, not aspirational)

| Item | Result |
|---|---|
| Simulation used | `sim_20808935737b`: 40 agents × 3 rounds × twitter, 99 actions |
| Write-back | `items_sent=99/99`, `undelivered=0`, `delivery_ratio=1.0`, `ingestion_drained=true` |
| Report | `report_e3a7f1a1c8bc`: `completed`, 4/4 sections, 352.2 s |
| Grounding gate | final violations 0, `unlabeled=0`, `quote_violations=0` |
| Fact traceability | **R1_fact = 8/8 = 100%** |
| Fabrication check | **R2 = 0/20** |
| Legacy metric | `R1_all = 40%` (12 labeled inferences counted as partial). **Legacy; kept for comparison; whether it is an N-44 gate is deferred; not a current gate.** |

**Not validated yet**: L-c (default 10 rounds), L-d (full rounds), Zep Cloud mode,
frontend Step1–5 browser flow, Linux/macOS, Docker for local mode.

## Quick start (Windows 11, source)

### Prerequisites

- Windows 11 x64
- Python 3.11–3.12 (3.12 recommended) and [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- Optional but strongly recommended: NVIDIA GPU with 8 GB+ VRAM
- ~6–8 GB free disk for the venv, CUDA torch, and local models

### Install and start

```powershell
git clone https://github.com/Friend-Xu/mirofish-local.git
cd mirofish-local

# Backend (local graph extra is required)
cd backend
uv sync --frozen --extra graphiti
cd ..

# Frontend
npm ci
cd frontend
npm ci
cd ..

# Models and GPU torch
backend\.venv\Scripts\python.exe scripts\provision-models.py --fetch
backend\.venv\Scripts\python.exe scripts\provision-models.py --verify
backend\.venv\Scripts\python.exe scripts\provision-torch.py --verify

# Local-first config
copy .env.example .env
# Edit .env: set LLM_API_KEY; ZEP_BACKEND=graphiti is already the default.
# Local mode does not need ZEP_API_KEY.

# Start two visible windows (backend :5001, frontend :3000)
scripts\start-local.cmd
```

If the machine has no NVIDIA GPU, explicitly allow CPU for the recommendation timeline
(it will be slow):

```powershell
set MIROFISH_ALLOW_CPU_RECSYS=1
scripts\start-local.cmd
```

Self-check:

- <http://localhost:5001/health> → `{"service":"MiroFish Backend","status":"ok"}`
- <http://localhost:3000/> → 200

Full steps, configuration details, backup/reset, and the troubleshooting table are in
[DEPLOYMENT.md](DEPLOYMENT.md). Maintainer notes for the toolchain and locks are in
[PROJECT_SETUP.md](PROJECT_SETUP.md).

## Important deployment notes

- **Do not use `uv run` to start the backend.** It can rebuild the environment and download
  torch. Use `backend\.venv\Scripts\python.exe run.py`; `uv` is only for provisioning.
- **Do not set `GRAPHITI_MIN_SCORE=0`.** Leave it unset (upstream default 0.6) or use a value
  in `(0, 1]`; 0 disables relevance filtering and is rejected by config validation.
- **Do not point `GRAPHITI_EMBEDDING_MODEL_PATH` at `.hf-cache/`.** It is
  `models/bge-small-zh-v1.5`.
- **Kuzu is single-process.** Stop the backend before opening the `.kuzu` file in another
  process; otherwise the graph write-back can stall.
- **Docker is not supported for local mode.** The existing Dockerfile/compose are upstream
  cloud-mode artifacts; use the source deployment above.
- **Never commit `.env`.** It is ignored; use `.env.example`.

## Data and reset

Runtime data stays under `backend/uploads/`:

- `graphs/graphiti.kuzu` (+ `.wal`) — local graph
- `graphs/ontology/<graph_id>.json` — ontology cache
- `simulations/` — simulation products
- `reports/` — reports and `meta.json`

Stop the backend before copying or deleting the graph. Delete the whole project directory to
uninstall; dependencies, models, and caches are project-local.

## License and upstream

- Upstream: <https://github.com/666ghj/MiroFish>
- This fork is based on upstream commit `7657031ac01184afe2cb220f5ee3545573b5e843`.
- License: **AGPL-3.0**; see [LICENSE](LICENSE). Keep upstream copyright/license notices when
  redistributing. For network services, check AGPL obligations with your own counsel.
- The simulation engine is powered by [OASIS](https://github.com/camel-ai/oasis); thanks to the
  CAMEL-AI team.
