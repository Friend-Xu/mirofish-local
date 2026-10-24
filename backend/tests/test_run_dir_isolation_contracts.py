"""运行目录隔离契约：测试不许写真实运行目录（掩盖台账 N-42）。

实测现场：`backend/uploads/graphs/pending/graph-1.jsonl`（1,214,526 字节）
是测试写进真实运行目录的待处理活动（`graph-1` 是 fixture 的 group_id）。
同一类风险还有 `Config.GRAPHITI_DB_PATH` 默认指向真实的嵌入式图谱库——
测试与后端共用同一份状态，正是 N-24/N-34 的现场形状。

本文件锁住"隔离真的生效"：所有运行目录都必须解析到会话临时根目录，
并且服务在构造时不会去碰真实的 `backend/uploads/`。
"""

import os
import sys
import threading
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Config  # noqa: E402
from app.services import zep_graph_memory_updater as updater_module  # noqa: E402
from app.services.graph_inventory import inventory_dir  # noqa: E402
from app.services.report_agent import ReportManager  # noqa: E402
from app.services.simulation_manager import SimulationManager  # noqa: E402
from app.services.simulation_runner import SimulationRunner  # noqa: E402
from app.services.simulation_scope import simulation_dir  # noqa: E402
from app.services.zep_graph_memory_updater import AgentActivity  # noqa: E402

REAL_UPLOADS = (BACKEND_DIR / "uploads").resolve()
REAL_PENDING_GRAPH_1 = REAL_UPLOADS / "graphs" / "pending" / "graph-1.jsonl"


def _snapshot(path: Path):
    """(存在, 大小, mtime) —— 用来证明"这次测试没有写那个真实文件"。"""

    if not path.exists():
        return (False, None, None)
    stat = path.stat()
    return (True, stat.st_size, stat.st_mtime)


def _run_dirs():
    return {
        "Config.UPLOAD_FOLDER": Config.UPLOAD_FOLDER,
        "Config.OASIS_SIMULATION_DATA_DIR": Config.OASIS_SIMULATION_DATA_DIR,
        "Config.GRAPHITI_DB_PATH": Config.GRAPHITI_DB_PATH,
        "updater.PENDING_DIR": updater_module.PENDING_DIR,
        "ReportManager.REPORTS_DIR": ReportManager.REPORTS_DIR,
        "SimulationManager.SIMULATION_DATA_DIR": SimulationManager.SIMULATION_DATA_DIR,
        "SimulationRunner.RUN_STATE_DIR": SimulationRunner.RUN_STATE_DIR,
        "inventory_dir()": inventory_dir(),
        "simulation_dir()": simulation_dir("sim_isolation_probe"),
    }


def test_every_run_dir_is_outside_the_real_uploads(isolated_run_root):
    for name, value in _run_dirs().items():
        resolved = Path(value).resolve()
        assert resolved != REAL_UPLOADS, f"{name} 就是真实运行目录：{resolved}"
        assert REAL_UPLOADS not in resolved.parents, (
            f"{name} 落在真实运行目录里：{resolved}"
        )
        assert resolved.is_relative_to(Path(isolated_run_root).resolve()), (
            f"{name} 没被隔离到临时根：{resolved}"
        )


def test_report_folder_resolves_under_the_isolated_root(isolated_run_root):
    ReportManager._ensure_reports_dir()
    folder = Path(ReportManager._get_report_folder("report_isolation_probe")).resolve()
    assert folder.is_relative_to(Path(isolated_run_root).resolve())
    assert folder.parent.exists(), "报告根目录没有被创建在隔离根里"


def test_services_do_not_create_real_run_dirs():
    """构造服务不会在真实运行目录里造目录（真实 uploads 之外才会被创建）。"""

    SimulationManager()  # 构造函数里有 os.makedirs(SIMULATION_DATA_DIR)
    ReportManager._ensure_reports_dir()
    for path in (SimulationManager.SIMULATION_DATA_DIR, ReportManager.REPORTS_DIR):
        assert not Path(path).resolve().is_relative_to(REAL_UPLOADS)


def test_pending_backlog_defaults_to_the_isolated_dir(isolated_run_root):
    """积压文件路径必须跟着隔离根走（实测漏过：它只被 2 个测试文件打过补丁）。"""

    backlog = updater_module.pending_backlog("graph-1")
    assert backlog["pending_items"] == 0
    assert backlog["path"] is None, "隔离前提不该存在积压文件"

    before = _snapshot(REAL_PENDING_GRAPH_1)

    updater = updater_module.ZepGraphMemoryUpdater.__new__(
        updater_module.ZepGraphMemoryUpdater
    )
    updater.graph_id = "graph-1"
    updater.simulation_id = "sim_isolation_probe"
    updater._buffer_lock = threading.Lock()
    updater._persisted_count = 0
    updater._persist_unsent(
        "twitter",
        [
            AgentActivity(
                platform="twitter",
                agent_id=1,
                agent_name="隔离探针",
                action_type="CREATE_POST",
                action_args={"content": "隔离契约测试"},
                round_num=1,
                timestamp="2026-10-11T04:00:00",
            )
        ],
        reason="隔离契约测试",
    )

    written = Path(updater_module.PENDING_DIR) / "graph-1.jsonl"
    assert written.exists(), "积压文件没有落盘"
    assert written.resolve().is_relative_to(Path(isolated_run_root).resolve())
    # 真实运行目录里的那个文件不是这次测试写的
    assert _snapshot(REAL_PENDING_GRAPH_1) == before


def test_graph_database_path_is_not_the_production_file():
    production = REAL_UPLOADS / "graphs" / "graphiti.kuzu"
    assert Path(Config.GRAPHITI_DB_PATH).resolve() != production.resolve()


@pytest.mark.parametrize("name", ["PENDING_DIR", "REPORTS_DIR", "RUN_STATE_DIR"])
def test_constants_are_string_paths(name):
    owner = {
        "PENDING_DIR": updater_module,
        "REPORTS_DIR": ReportManager,
        "RUN_STATE_DIR": SimulationRunner,
    }[name]
    value = getattr(owner, name)
    assert isinstance(value, str) and os.path.isabs(value)
