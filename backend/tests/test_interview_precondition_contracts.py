"""采访前置条件契约（N-31）：环境已经停了就不许白等预算。

实测（掩盖台账 N-31）：报告的 ReACT 每章都调 `interview_agents`，而模拟环境在出报告前
通常已经 stop；工具无条件发 IPC 命令再等满预算——每次 600 秒（3 Agent × 2 平台），
三章就浪费掉半小时，最后拿到的还是"没有回答"。正确做法是**先确认环境在等命令模式**，
不在就立刻失败并把原因写进结果。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import zep_tools as zep_tools_module  # noqa: E402
from app.services.simulation_runner import (  # noqa: E402
    RunnerStatus,
    SimulationRunner,
    SimulationRunState,
    _pid_alive,
)
from app.services.zep_tools import ZepToolsService  # noqa: E402


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=30)
    return process.pid


def _service(monkeypatch):
    service = ZepToolsService.__new__(ZepToolsService)
    service.graph_id = "graph-1"
    monkeypatch.setattr(
        service,
        "_load_agent_profiles",
        lambda simulation_id: [
            {"agent_id": 0, "realname": "同事A", "username": "a", "profession": "机械", "bio": "b"},
            {"agent_id": 1, "realname": "同事B", "username": "b", "profession": "机械", "bio": "b"},
        ],
    )
    monkeypatch.setattr(
        service,
        "_select_agents_for_interview",
        lambda **kwargs: (
            kwargs["profiles"],
            [0, 1],
            "选择理由",
        ),
    )
    monkeypatch.setattr(
        service,
        "_generate_interview_questions",
        lambda **kwargs: ["问题一", "问题二"],
    )
    return service


def test_interview_fails_fast_when_environment_is_stopped(monkeypatch):
    service = _service(monkeypatch)
    calls = []
    monkeypatch.setattr(
        SimulationRunner,
        "env_accepts_interviews",
        classmethod(lambda cls, simulation_id: (False, "模拟运行已收尾（runner_status=stopped）")),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "interview_agents_batch",
        classmethod(lambda cls, **kwargs: calls.append(kwargs) or {"success": True}),
    )

    result = service.interview_agents(
        simulation_id="sim-stopped",
        interview_requirement="问问同事怎么看",
    )

    assert calls == [], "环境已停止时不允许再发采访命令（会白等满预算）"
    assert "运行已收尾" in result.summary
    assert result.interviews == []


def test_env_accepts_interviews_rejects_terminal_run_even_if_status_file_says_alive(
    tmp_path, monkeypatch
):
    """回归 N-31：强杀后 `env_status.json` 仍写 alive，但运行已收尾 → 不能采访。"""

    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    sim_dir = tmp_path / "sim_terminal"
    sim_dir.mkdir()
    (sim_dir / "env_status.json").write_text(
        json.dumps({"status": "alive"}), encoding="utf-8"
    )
    state = SimulationRunState(
        simulation_id="sim_terminal",
        runner_status=RunnerStatus.STOPPED,
        process_pid=os.getpid(),
    )
    monkeypatch.setattr(
        SimulationRunner, "get_run_state", classmethod(lambda cls, simulation_id: state)
    )

    accepted, reason = SimulationRunner.env_accepts_interviews("sim_terminal")
    assert accepted is False
    assert "已收尾" in reason


def test_env_accepts_interviews_requires_live_process(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    sim_dir = tmp_path / "sim_dead"
    sim_dir.mkdir()
    (sim_dir / "env_status.json").write_text(
        json.dumps({"status": "alive"}), encoding="utf-8"
    )
    state = SimulationRunState(
        simulation_id="sim_dead",
        runner_status=RunnerStatus.RUNNING,
        process_pid=_dead_pid(),
    )
    monkeypatch.setattr(
        SimulationRunner, "get_run_state", classmethod(lambda cls, simulation_id: state)
    )

    accepted, reason = SimulationRunner.env_accepts_interviews("sim_dead")
    assert accepted is False
    assert "进程已不存在" in reason


def test_interview_proceeds_when_environment_is_alive(monkeypatch):
    service = _service(monkeypatch)
    calls = []
    monkeypatch.setattr(
        SimulationRunner,
        "env_accepts_interviews",
        classmethod(lambda cls, simulation_id: (True, "ok")),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "interview_agents_batch",
        classmethod(
            lambda cls, **kwargs: calls.append(kwargs)
            or {
                "success": True,
                "interviews_count": 0,
                "result": {"results": {}},
            }
        ),
    )
    monkeypatch.setattr(
        zep_tools_module,
        "Config",
        SimpleNamespace(
            INTERVIEW_TIMEOUT_SECONDS=600,
            INTERVIEW_TIMEOUT_PER_CALL_SECONDS=90,
        ),
    )

    service.interview_agents(simulation_id="sim-alive", interview_requirement="问问同事怎么看")

    assert calls, "环境存活时必须真的发采访命令"
