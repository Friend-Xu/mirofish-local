"""崩溃后和解契约：后端进程死过一次，状态必须说出来，不能停在 `running`。

对应掩盖台账 N-24（Kuzu 原生崩溃杀死后端进程，没有异常、没有 atexit）与 M-05
（历史上把"长跑中途死"归因成"工具会话回收宿主进程"，失败就此消失）。

证据等级：本文件是**单测**（写坏 run_state.json 后调用和解）。
"真实崩溃后重启后端会判 failed"由 L-a 现场的转储 + 启动日志证明，单测不能替代它。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import simulation_ipc  # noqa: E402
from app.services.simulation_runner import SimulationRunner, _pid_alive  # noqa: E402


def _write_run_state(sim_dir: Path, status: str, pid):
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "run_state.json").write_text(
        json.dumps(
            {
                "simulation_id": sim_dir.name,
                "runner_status": status,
                "current_round": 2,
                "process_pid": pid,
                "updated_at": "2026-10-10T23:29:59",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _dead_pid() -> int:
    """拿到一个确定已经退出的进程号。"""

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=30)
    return process.pid


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    synced = []
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(
            lambda cls, simulation_id, status, error=None: synced.append(
                (simulation_id, status, error)
            )
        ),
    )
    return tmp_path, synced


def test_dead_process_is_reconciled_to_failed(runner_env):
    tmp_path, synced = runner_env
    _write_run_state(tmp_path / "sim_dead", "running", _dead_pid())

    reconciled = SimulationRunner.reconcile_stale_runs()

    assert [item["simulation_id"] for item in reconciled] == ["sim_dead"]
    assert reconciled[0]["process_alive"] is False

    persisted = json.loads((tmp_path / "sim_dead" / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["runner_status"] == "failed"
    assert "后端进程在模拟未完成时退出" in persisted["error"]
    assert persisted["completed_at"]
    assert synced and synced[0][0] == "sim_dead" and synced[0][1].value == "failed"


def test_terminal_runs_are_left_alone(runner_env):
    tmp_path, synced = runner_env
    _write_run_state(tmp_path / "sim_done", "completed", 12345)
    _write_run_state(tmp_path / "sim_idle", "idle", None)

    assert SimulationRunner.reconcile_stale_runs() == []
    assert synced == []
    persisted = json.loads((tmp_path / "sim_done" / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["runner_status"] == "completed"


def test_live_orphan_is_asked_to_close_and_marked_failed(runner_env, monkeypatch):
    """进程还活着说明它已经收不到命令（宿主没了）：用 IPC 请求关闭，同样判失败。"""

    tmp_path, synced = runner_env
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    calls = []

    class _FakeIPC:
        def __init__(self, simulation_dir):
            self.simulation_dir = simulation_dir

        def send_close_env(self, timeout=30.0):
            calls.append((self.simulation_dir, timeout))
            return None

    monkeypatch.setattr(simulation_ipc, "SimulationIPCClient", _FakeIPC)
    try:
        _write_run_state(tmp_path / "sim_orphan", "running", child.pid)
        reconciled = SimulationRunner.reconcile_stale_runs()
    finally:
        child.terminate()
        child.wait(timeout=30)

    assert reconciled and reconciled[0]["process_alive"] is True
    assert calls and calls[0][0] == str(tmp_path / "sim_orphan")
    persisted = json.loads((tmp_path / "sim_orphan" / "run_state.json").read_text(encoding="utf-8"))
    assert persisted["runner_status"] == "failed"
    assert "残留模拟进程" in persisted["error"]


def test_pid_alive_helper():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(_dead_pid()) is False
    assert _pid_alive(None) is False
    assert _pid_alive(0) is False
    assert _pid_alive("123") is False


# ------------------------------------------- 跑完但没人 stop 的运行（N-38）


def _write_actions_log(sim_dir: Path, platform: str, *, simulation_end: bool):
    platform_dir = sim_dir / platform
    platform_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"timestamp": "2026-10-11T02:15:00", "event_type": "simulation_start",
             "platform": platform, "total_rounds": 10, "agents_count": 3},
            ensure_ascii=False,
        )
    ]
    if simulation_end:
        lines.append(
            json.dumps(
                {"timestamp": "2026-10-11T02:17:00", "event_type": "simulation_end",
                 "platform": platform, "total_rounds": 10, "total_actions": 7},
                ensure_ascii=False,
            )
        )
    (platform_dir / "actions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_finished_run_is_reconciled_to_stopped_not_failed(runner_env):
    """实测现场（`sim_583b25e0abca`）：模拟 10 轮全跑完，没人 stop，后端被重启——
    旧和解把这场**已经跑完**的模拟判成 failed，报告从此被拒（`simulation_failed`）。"""

    tmp_path, synced = runner_env
    _write_run_state(tmp_path / "sim_finished", "running", _dead_pid())
    _write_actions_log(tmp_path / "sim_finished", "twitter", simulation_end=True)
    _write_actions_log(tmp_path / "sim_finished", "reddit", simulation_end=True)

    reconciled = SimulationRunner.reconcile_stale_runs()

    assert reconciled and reconciled[0]["simulation_finished"] is True
    assert reconciled[0]["terminal_status"] == "stopped"
    persisted = json.loads(
        (tmp_path / "sim_finished" / "run_state.json").read_text(encoding="utf-8")
    )
    assert persisted["runner_status"] == "stopped"
    assert "跑完后退出" in persisted["error"]
    assert "判为已停止" in persisted["error"]
    assert synced and synced[0][1].value == "stopped"


def test_partially_finished_run_stays_failed(runner_env):
    """只有一个平台写了 simulation_end：这是真的没跑完，仍然判 failed。"""

    tmp_path, synced = runner_env
    _write_run_state(tmp_path / "sim_half", "running", _dead_pid())
    _write_actions_log(tmp_path / "sim_half", "twitter", simulation_end=True)
    _write_actions_log(tmp_path / "sim_half", "reddit", simulation_end=False)

    reconciled = SimulationRunner.reconcile_stale_runs()

    assert reconciled and reconciled[0]["simulation_finished"] is False
    assert reconciled[0]["terminal_status"] == "failed"
    persisted = json.loads(
        (tmp_path / "sim_half" / "run_state.json").read_text(encoding="utf-8")
    )
    assert persisted["runner_status"] == "failed"
    assert "模拟结果不完整" in persisted["error"]


def test_disk_judgement_sees_completion_the_memory_flags_missed(runner_env):
    """run_state 里的 completed 标志还没落盘时，磁盘判据仍然能看到"跑完了"。

    这正是 N-38 的前提：启动和解没有内存状态可用，只能以 `<platform>/actions.jsonl` 为准；
    两条判据在标志齐全时必须一致，否则运行期与启动期会给出不同答案。
    """

    tmp_path, _synced = runner_env
    sim_dir = tmp_path / "sim_disk"
    _write_actions_log(sim_dir, "twitter", simulation_end=True)
    _write_actions_log(sim_dir, "reddit", simulation_end=True)

    from app.services.simulation_runner import SimulationRunState

    state = SimulationRunState(simulation_id="sim_disk")
    assert (state.twitter_completed, state.reddit_completed) == (False, False)
    assert SimulationRunner.has_simulation_finished(state) is False, "标志没落盘时内存判据说没跑完"
    assert SimulationRunner._simulation_end_recorded_on_disk("sim_disk") is True

    state.twitter_completed = True
    state.reddit_completed = True
    assert SimulationRunner.has_simulation_finished(state) is True, "两条判据必须一致"

    _write_actions_log(tmp_path / "sim_half", "twitter", simulation_end=True)
    _write_actions_log(tmp_path / "sim_half", "reddit", simulation_end=False)
    assert SimulationRunner._simulation_end_recorded_on_disk("sim_half") is False


def test_reconcile_is_wired_into_the_entrypoint_only():
    """启动和解必须挂在入口 `run.py`，且**不能**挂在 Flask 应用工厂上。

    理由：测试会反复调用 `create_app()`。和解会写真实运行目录——挂在工厂上意味着
    一次 pytest 就可能把正在跑的模拟判成失败（本轮实测发生过一次）。
    这是静态断言：它锁"挂在哪"，行为本身由上面 5 条单测 + 真实崩溃现场证明。
    """

    import app as app_package

    run_py = (BACKEND_DIR / "run.py").read_text(encoding="utf-8")
    assert "reconcile_stale_runs()" in run_py, "run.py 没有调用启动和解"
    assert "def reconcile_previous_run" in run_py

    app_init = (BACKEND_DIR / "app" / "__init__.py").read_text(encoding="utf-8")
    assert "reconcile_stale_runs" not in app_init, (
        "应用工厂不允许调用启动和解：pytest 会调用 create_app()，会污染真实运行目录"
    )

    # 工厂本身仍然可用（不触发和解）
    monkeypatched = []
    import app.services.simulation_runner as runner_module

    original = runner_module.SimulationRunner.reconcile_stale_runs
    runner_module.SimulationRunner.reconcile_stale_runs = classmethod(
        lambda cls: monkeypatched.append(1) or []
    )
    try:
        app_package.create_app()
    finally:
        runner_module.SimulationRunner.reconcile_stale_runs = original
    assert monkeypatched == []
