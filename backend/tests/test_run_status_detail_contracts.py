"""`run-status/detail` 的账目契约：跑到终态后仍然要能看见最终账目。

对应掩盖台账 N-29：F7 给这个接口加回写账目时写了 `run_state.graph_id`，
而 `SimulationRunState` 根本没有这个字段——于是**回写更新器一注销，接口必然 500**，
恰好把"模拟结束后的最终账目"挡在门外（L-a 实测：终态后第一次查询就 ERROR）。
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as app_package  # noqa: E402
from app.api import simulation as simulation_api  # noqa: E402
from app.services.simulation_runner import RunnerStatus, SimulationRunState  # noqa: E402


def test_run_state_has_no_graph_id():
    """回归锁：运行态对象没有 graph_id，接口不允许读它。"""

    assert not hasattr(SimulationRunState(simulation_id="sim_x"), "graph_id")


@pytest.fixture
def client_with_finished_run(monkeypatch):
    app = app_package.create_app()
    client = app.test_client()

    state = SimulationRunState(simulation_id="sim_finished")
    state.runner_status = RunnerStatus.STOPPED
    state.current_round = 3

    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "get_run_state",
        classmethod(lambda cls, simulation_id: state),
    )
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "get_all_actions",
        classmethod(lambda cls, **kwargs: []),
    )
    monkeypatch.setattr(
        simulation_api.ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda cls, simulation_id: None),
    )
    monkeypatch.setattr(
        simulation_api, "_simulation_graph_id", lambda simulation_id: "graph_from_metadata"
    )
    return client


def test_detail_reports_graph_memory_after_updater_unregistered(client_with_finished_run):
    response = client_with_finished_run.get("/api/simulation/sim_finished/run-status/detail")

    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    payload = response.get_json()
    assert payload["success"] is True
    graph_memory = payload["data"]["graph_memory"]
    assert graph_memory["graph_id"] == "graph_from_metadata"
    assert graph_memory["pending_backlog"]["graph_id"] == "graph_from_metadata"
    assert "已注销" in graph_memory["note"]


def test_detail_says_so_when_graph_id_is_unknown(client_with_finished_run, monkeypatch):
    """取不到 graph_id 时必须在响应里写明，不能编一个 ID 顶上。"""

    monkeypatch.setattr(simulation_api, "_simulation_graph_id", lambda simulation_id: None)
    response = client_with_finished_run.get("/api/simulation/sim_finished/run-status/detail")

    assert response.status_code == 200
    graph_memory = response.get_json()["data"]["graph_memory"]
    assert graph_memory["graph_id"] is None
    assert "取不到 graph_id" in graph_memory["note"]
