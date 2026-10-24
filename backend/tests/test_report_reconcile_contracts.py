"""报告任务的启动和解契约：后端死在报告生成期间，状态必须说出来。

对应掩盖台账 N-32：报告生成到一半时后端被重启，`report_7594e1e6716d` 的
`meta.json` 停在 `status=planning`、`progress.json` 停在 `{"status": "generating",
"progress": 48}`——之后任何查询都会看到"还在生成"。报告任务（`TaskManager`）只在内存里，
进程一死就没了，能收尾的只有磁盘上的状态文件。同族缺陷见 N-25（模拟运行的 `run_state.json`）。

对应 N-33：同一个模拟会有多份报告目录（重新生成 + 半路死掉的），历史实现取
`os.listdir` 里第一个匹配的，实测返回了那份停在 `planning` 的失败报告，
而两份 `completed` 就在旁边。

证据等级：本文件是**单测**（写坏报告目录后调用和解 / 伪造多份报告后查询）。
真实现场是 `report_7594e1e6716d`（planning/48%）与 `report_5b1a806e9dbc`（planning/34%），
它们确实是重启后由启动和解判成 failed 的——那一步由启动日志证明，单测不能替代它。
"""

import json
import logging
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.report_agent import ReportManager, ReportStatus  # noqa: E402


def _write_report(
    reports_dir: Path,
    report_id: str,
    simulation_id: str = "sim_test",
    *,
    status: str,
    created_at: str = "2026-10-11T01:00:00",
    progress_status=None,
    error=None,
) -> Path:
    folder = reports_dir / report_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "report_id": report_id,
                "simulation_id": simulation_id,
                "graph_id": "graph_test",
                "simulation_requirement": "测试需求",
                "status": status,
                "outline": None,
                "markdown_content": "",
                "created_at": created_at,
                "completed_at": "",
                "error": error,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if progress_status is not None:
        (folder / "progress.json").write_text(
            json.dumps(
                {"status": progress_status, "progress": 48, "message": "正在生成章节"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return folder


@pytest.fixture
def reports_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def log_records():
    """`mirofish.*` 的 logger 设了 propagate=False，caplog 抓不到，直接挂 handler。"""

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("mirofish.report_agent")
    handler = _Collect(level=logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# ------------------------------------------------------ 启动和解（N-32）


def test_stale_report_is_reconciled_to_failed(reports_env):
    _write_report(
        reports_env,
        "report_stale",
        status=ReportStatus.PLANNING.value,
        progress_status="generating",
    )

    reconciled = ReportManager.reconcile_stale_reports()

    assert [item["report_id"] for item in reconciled] == ["report_stale"]
    assert reconciled[0]["previous_status"] == "planning"
    assert reconciled[0]["progress_status"] == "generating"

    meta = json.loads((reports_env / "report_stale" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert "后端进程在报告生成期间退出" in meta["error"]
    assert "planning" in meta["error"] and "generating" in meta["error"]

    progress = json.loads(
        (reports_env / "report_stale" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "failed"
    assert progress["progress"] == -1
    assert "后端进程在报告生成期间退出" in progress["message"]


@pytest.mark.parametrize(
    "status,progress_status",
    [
        (ReportStatus.PENDING.value, "pending"),
        (ReportStatus.GENERATING.value, "generating"),
    ],
)
def test_every_non_terminal_status_is_reconciled(reports_env, status, progress_status):
    _write_report(reports_env, "report_mid", status=status, progress_status=progress_status)
    assert len(ReportManager.reconcile_stale_reports()) == 1
    meta = json.loads((reports_env / "report_mid" / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"


def test_terminal_reports_are_left_alone(reports_env):
    _write_report(reports_env, "report_done", status=ReportStatus.COMPLETED.value,
                  progress_status="completed")
    _write_report(reports_env, "report_dead", status=ReportStatus.FAILED.value,
                  progress_status="failed", error="原始失败原因")

    assert ReportManager.reconcile_stale_reports() == []

    done = json.loads((reports_env / "report_done" / "meta.json").read_text(encoding="utf-8"))
    assert done["status"] == "completed"
    dead = json.loads((reports_env / "report_dead" / "meta.json").read_text(encoding="utf-8"))
    assert dead["error"] == "原始失败原因", "已失败的报告不允许被改写（要保留原始原因）"


def test_reconcile_is_idempotent(reports_env):
    _write_report(reports_env, "report_stale", status=ReportStatus.PLANNING.value,
                  progress_status="generating")
    assert len(ReportManager.reconcile_stale_reports()) == 1
    assert ReportManager.reconcile_stale_reports() == []


def test_folder_without_meta_is_skipped_without_inventing_state(reports_env, log_records):
    """崩在"建目录"与"写 meta"之间：没有状态可和解，不许猜一个。"""

    (reports_env / "report_orphan").mkdir(parents=True)

    assert ReportManager.reconcile_stale_reports() == []
    assert not (reports_env / "report_orphan" / "meta.json").exists()
    assert any("缺少 meta.json" in record.getMessage() for record in log_records)


def test_unparseable_meta_is_skipped_and_left_as_is(reports_env, log_records):
    folder = reports_env / "report_broken"
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text('{"status": "planni', encoding="utf-8")

    assert ReportManager.reconcile_stale_reports() == []
    assert (folder / "meta.json").read_text(encoding="utf-8") == '{"status": "planni'
    assert any("meta.json 无法解析" in record.getMessage() for record in log_records)


def test_reconcile_is_wired_into_the_entrypoint_only():
    """和解挂在入口 `run.py`，**不能**挂在 Flask 应用工厂上（理由同 N-25：pytest 调 create_app）。"""

    run_py = (BACKEND_DIR / "run.py").read_text(encoding="utf-8")
    assert "ReportManager.reconcile_stale_reports()" in run_py, "run.py 没有调用报告和解"

    app_init = (BACKEND_DIR / "app" / "__init__.py").read_text(encoding="utf-8")
    assert "reconcile_stale_reports" not in app_init, (
        "应用工厂不允许调用和解：pytest 会调用 create_app()，会污染真实报告目录"
    )


# ------------------------------------------------------ 原子写（N-32 的前置）


def test_atomic_write_json_leaves_no_temp_file(reports_env):
    path = reports_env / "report_x" / "meta.json"
    ReportManager._atomic_write_json(str(path), {"status": "planning"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "planning"}
    leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == [], "原子写不允许留下临时文件"


def test_atomic_write_failure_keeps_previous_state(reports_env):
    path = reports_env / "report_y" / "meta.json"
    ReportManager._atomic_write_json(str(path), {"status": "generating"})

    class _Unserializable:
        pass

    with pytest.raises(TypeError):
        ReportManager._atomic_write_json(str(path), {"status": _Unserializable()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "generating"}, (
        "写失败时旧状态必须原样保留：半截 JSON 会让启动和解读不出上一份状态"
    )
    leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


# ------------------------------------------------- 多份报告的选择（N-33）


def test_completed_report_wins_over_a_stale_one(reports_env):
    """实测现场：报告目录按名字排序时，停在 planning 的那份排在 completed 前面。"""

    _write_report(reports_env, "report_5b1a806e9dbc", status=ReportStatus.PLANNING.value,
                  created_at="2026-10-11T01:45:00", progress_status="generating")
    _write_report(reports_env, "report_baef0b8c686e", status=ReportStatus.COMPLETED.value,
                  created_at="2026-10-11T01:49:14")

    selected = ReportManager.get_report_by_simulation("sim_test")
    assert selected.report_id == "report_baef0b8c686e"
    assert selected.status == ReportStatus.COMPLETED


def test_newest_completed_report_is_selected(reports_env):
    _write_report(reports_env, "report_old", status=ReportStatus.COMPLETED.value,
                  created_at="2026-10-11T01:09:35")
    _write_report(reports_env, "report_new", status=ReportStatus.COMPLETED.value,
                  created_at="2026-10-11T01:49:14")

    assert ReportManager.get_report_by_simulation("sim_test").report_id == "report_new"


def test_failed_reports_stay_visible_when_nothing_completed(reports_env):
    _write_report(reports_env, "report_a", status=ReportStatus.FAILED.value,
                  created_at="2026-10-11T01:00:00", error="第一次失败")
    _write_report(reports_env, "report_b", status=ReportStatus.FAILED.value,
                  created_at="2026-10-11T02:00:00", error="第二次失败")

    selected = ReportManager.get_report_by_simulation("sim_test")
    assert selected.report_id == "report_b"
    assert selected.status == ReportStatus.FAILED, "没有完成报告时必须照实返回失败状态"


def test_reconciled_stale_report_does_not_hide_the_completed_one(reports_env):
    """N-32 与 N-33 的组合：和解把陈旧报告判失败后，查询仍要拿到已完成的那份。"""

    _write_report(reports_env, "report_stale", status=ReportStatus.PLANNING.value,
                  created_at="2026-10-11T01:45:00", progress_status="generating")
    _write_report(reports_env, "report_done", status=ReportStatus.COMPLETED.value,
                  created_at="2026-10-11T01:49:14")

    ReportManager.reconcile_stale_reports()

    selected = ReportManager.get_report_by_simulation("sim_test")
    assert selected.report_id == "report_done"
    assert selected.status == ReportStatus.COMPLETED
