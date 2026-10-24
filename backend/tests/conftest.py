"""测试期运行目录隔离：任何一次 pytest 都不许写真实的运行目录（掩盖台账 N-42）。

实测证据：`backend/uploads/graphs/pending/graph-1.jsonl`（1,214,526 字节，10-11 04:30）
里全是**测试**的待处理活动——`graph-1` 是测试 fixture 的 group_id，真实图谱是
`mirofish_2a5aa6e3eb04467c`。也就是说一次 pytest 写进了正在使用的运行目录。

为什么会漏：这一组路径都是**导入期常量**，而且分散在四个所有者手里——
`Config.UPLOAD_FOLDER`/`OASIS_SIMULATION_DATA_DIR`、`ReportManager.REPORTS_DIR`、
`SimulationManager.SIMULATION_DATA_DIR`、`SimulationRunner.RUN_STATE_DIR`、
`zep_graph_memory_updater.PENDING_DIR`——各自的测试各自打补丁，漏一个就写真实目录
（实测 `PENDING_DIR` 只在 2 个测试文件里被隔离过）。

同一类问题还有图谱库：`Config.GRAPHITI_DB_PATH` 默认指向真实的
`backend/uploads/graphs/graphiti.kuzu`，测试与后端共用同一份嵌入式库
（N-24/N-34 的现场风险）。

做法：在**会话级**把这一组路径统一钉到项目内的临时根目录。单个测试仍然可以
monkeypatch 成自己的 `tmp_path`（例如"报告写不出内容时怎么办"这类用例）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

REAL_UPLOADS = (BACKEND_DIR / "uploads").resolve()


@pytest.fixture(scope="session", autouse=True)
def isolated_run_root(tmp_path_factory):
    """把全部运行目录钉到临时根，返回该根目录（会话级）。"""

    from app.config import Config
    from app.services import zep_graph_memory_updater as updater_module
    from app.services.report_agent import ReportManager
    from app.services.simulation_manager import SimulationManager
    from app.services.simulation_runner import SimulationRunner

    root = tmp_path_factory.mktemp("runroot")

    targets = [
        (Config, "UPLOAD_FOLDER", str(root)),
        (Config, "OASIS_SIMULATION_DATA_DIR", str(root / "simulations")),
        (Config, "GRAPHITI_DB_PATH", str(root / "graphs" / "graphiti.kuzu")),
        (ReportManager, "REPORTS_DIR", str(root / "reports")),
        (SimulationManager, "SIMULATION_DATA_DIR", str(root / "simulations")),
        (SimulationRunner, "RUN_STATE_DIR", str(root / "simulations")),
        (updater_module, "PENDING_DIR", str(root / "graphs" / "pending")),
    ]
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in targets]
    for owner, name, value in targets:
        setattr(owner, name, value)

    try:
        yield root
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)
