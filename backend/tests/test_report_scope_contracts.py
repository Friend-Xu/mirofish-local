"""报告范围声明契约：报告必须写明"这次模拟跑了什么"（验收标准 R3 / 掩盖台账 N-41）。

实测现场：`report_69379f275a5f` 的正文与 `meta.json` 都没有"本次模拟有多少 agent、
多少条动作、哪个平台根本没有传播"。twitter 那半场只有广播没有传播
（`information_flow.has_information_flow=false`），报告里读不出来；而 N-34 期间
整场模拟只带 2–3 个角色跑完，报告照样交付。两件事都只能靠"产物自己声明范围"消除。

证据等级：本文件是**单测**（造模拟产物 / 直接调用采集与组装）。
"真实产物里的数字对不对"由 `sim_c98c1c01c802` 的实跑报告证明。
"""

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Config  # noqa: E402
from app.services import simulation_scope  # noqa: E402
from app.services.report_agent import (  # noqa: E402
    Report,
    ReportManager,
    ReportOutline,
    ReportStatus,
    ReportSection,
)
from app.services.simulation_scope import (  # noqa: E402
    SimulationScopeUnavailable,
    collect_simulation_scope,
    render_scope_markdown,
)

SIM_ID = "sim_scope_test"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _information_flow(*, has_flow: bool, rec: int = 560, social_edges: int = 205):
    return {
        "db_path": "x_simulation.db",
        "exists": True,
        "counts": {
            "post": 16, "comment": 40, "like": 144, "follow": 21, "rec": rec, "trace": 355,
        },
        "trace_by_action": {"create_post": 16, "like_post": 144},
        "has_timeline": has_flow,
        "social_edges": social_edges,
        "has_information_flow": has_flow,
    }


def _write_actions(
    sim_dir: Path,
    platform: str,
    action_types,
    flow,
    *,
    agents_count: int = 35,
) -> None:
    rows = [{
        "timestamp": "2026-10-11T03:19:01",
        "event_type": "simulation_start",
        "platform": platform,
        "total_rounds": 5,
        "agents_count": agents_count,
    }]
    for name in action_types:
        rows.append({
            "timestamp": "2026-10-11T03:19:10",
            "platform": platform,
            "agent_id": 1,
            "agent_name": "某工程师",
            "action_type": name,
            "action_args": {},
            "success": True,
        })
    rows.append({
        "timestamp": "2026-10-11T03:25:00",
        "event_type": "simulation_end",
        "platform": platform,
        "total_rounds": 5,
        "information_flow": flow,
    })
    path = sim_dir / platform / "actions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    """一套完整的模拟产物：twitter 无传播、reddit 有传播。"""

    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    sim_dir = tmp_path / "simulations" / SIM_ID
    _write_json(sim_dir / "state.json", {
        "simulation_id": SIM_ID,
        "graph_id": "graph_scope",
        "enable_twitter": True,
        "enable_reddit": True,
        "status": "running",
        "entities_count": 35,
        "profiles_count": 35,
        "entity_types": ["Engineer", "Person"],
    })
    _write_json(sim_dir / "run_state.json", {
        "simulation_id": SIM_ID,
        "total_rounds": 5,
        "current_round": 5,
    })
    _write_actions(
        sim_dir, "twitter", ["CREATE_POST"] * 3,
        _information_flow(has_flow=False, rec=0, social_edges=0),
    )
    _write_actions(
        sim_dir, "reddit", ["LIKE_POST"] * 4 + ["CREATE_COMMENT"] * 2,
        _information_flow(has_flow=True),
    )
    return sim_dir


def test_scope_collects_agents_actions_and_propagation(sim_env):
    scope = collect_simulation_scope(SIM_ID)

    assert scope["graph_id"] == "graph_scope"
    assert scope["agents"]["profiles_count"] == 35
    assert scope["agents"]["entity_types"] == ["Engineer", "Person"]
    assert scope["rounds"] == {"configured": 5, "executed": 5}
    assert scope["total_actions"] == 9
    assert scope["platforms"]["twitter"]["actions"] == 3
    assert scope["platforms"]["twitter"]["information_flow"]["has_information_flow"] is False
    assert scope["platforms"]["reddit"]["actions"] == 6
    assert scope["platforms"]["reddit"]["information_flow"]["has_information_flow"] is True
    assert scope["platforms"]["reddit"]["information_flow"]["counts"]["rec"] == 560


def test_scope_markdown_states_the_numbers_and_the_missing_propagation(sim_env):
    markdown = render_scope_markdown(collect_simulation_scope(SIM_ID))

    assert "## 模拟范围与传播声明" in markdown
    assert "35 个 agent" in markdown
    assert "动作总数 | 303 条" not in markdown  # 数字按本次产物给，不许沿用别人的
    assert "9 条" in markdown
    assert "**不构成传播**" in markdown
    # 验收标准 §3-2：无传播的平台必须在产物里被显式声明，而不是沉默地让读者以为它参与了
    assert "涉及传播的结论不适用于 twitter" in markdown
    assert "twitter" in markdown.split("涉及传播的结论不适用于", 1)[1].split("**", 1)[0]


def test_missing_state_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    with pytest.raises(SimulationScopeUnavailable, match="state.json"):
        collect_simulation_scope("sim_missing")


def test_missing_entities_count_raises(sim_env):
    state = json.loads((sim_env / "state.json").read_text(encoding="utf-8"))
    del state["entities_count"]
    _write_json(sim_env / "state.json", state)

    with pytest.raises(SimulationScopeUnavailable, match="entities_count"):
        collect_simulation_scope(SIM_ID)


def test_missing_run_state_raises(sim_env):
    (sim_env / "run_state.json").unlink()
    with pytest.raises(SimulationScopeUnavailable, match="run_state.json"):
        collect_simulation_scope(SIM_ID)


def test_missing_run_state_round_fields_raise(sim_env):
    _write_json(sim_env / "run_state.json", {"simulation_id": SIM_ID, "total_rounds": 5})
    with pytest.raises(SimulationScopeUnavailable, match="current_round"):
        collect_simulation_scope(SIM_ID)


def test_enabled_platform_without_actions_raises(sim_env):
    (sim_env / "reddit" / "actions.jsonl").unlink()
    with pytest.raises(SimulationScopeUnavailable, match="reddit"):
        collect_simulation_scope(SIM_ID)


def test_enabled_platform_without_simulation_end_raises(sim_env):
    path = sim_env / "reddit" / "actions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("event_type") != "simulation_end"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SimulationScopeUnavailable, match="simulation_end"):
        collect_simulation_scope(SIM_ID)


def test_simulation_end_without_information_flow_raises(sim_env):
    """F12 之前的模拟产物没有 information_flow：声明不出来就必须失败。"""

    path = sim_env / "twitter" / "actions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("event_type") == "simulation_end":
            del row["information_flow"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SimulationScopeUnavailable, match="information_flow"):
        collect_simulation_scope(SIM_ID)


def test_information_flow_missing_counts_raises(sim_env):
    path = sim_env / "reddit" / "actions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("event_type") == "simulation_end":
            del row["information_flow"]["counts"]["rec"]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SimulationScopeUnavailable, match="counts"):
        collect_simulation_scope(SIM_ID)


def test_platform_db_missing_raises(sim_env):
    path = sim_env / "reddit" / "actions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row.get("event_type") == "simulation_end":
            row["information_flow"] = {
                "db_path": "reddit_simulation.db",
                "exists": False,
                "note": "平台库不存在：无法判断本轮有没有信息流",
            }
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SimulationScopeUnavailable, match="平台库不存在"):
        collect_simulation_scope(SIM_ID)


def test_participant_mismatch_raises(sim_env):
    """实测 sim_4267208ff665：prepare 生成 21 份人设，实际只跑了 17 个 agent。"""

    _write_actions(
        sim_env, "reddit", ["LIKE_POST"],
        _information_flow(has_flow=True),
        agents_count=17,
    )
    with pytest.raises(SimulationScopeUnavailable, match="参与者数在两份产物里矛盾"):
        collect_simulation_scope(SIM_ID)


def test_disabled_platform_is_declared_not_silent(sim_env):
    state = json.loads((sim_env / "state.json").read_text(encoding="utf-8"))
    state["enable_reddit"] = False
    _write_json(sim_env / "state.json", state)
    (sim_env / "reddit" / "actions.jsonl").unlink()

    scope = collect_simulation_scope(SIM_ID)
    assert scope["platforms"]["reddit"] == {"enabled": False}
    assert scope["total_actions"] == 3

    markdown = render_scope_markdown(scope)
    assert "**未启用**" in markdown
    assert "reddit 本次未启用" in markdown


def test_assemble_full_report_requires_a_scope_block(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(reports_dir))
    report_id = "report_scope_unit"
    folder = reports_dir / report_id
    folder.mkdir(parents=True)
    (folder / "section_01.md").write_text("## 第一章\n\n正文。\n", encoding="utf-8")

    outline = ReportOutline(
        title="测试报告", summary="摘要",
        sections=[ReportSection(title="第一章", content="")],
    )

    with pytest.raises(TypeError):
        ReportManager.assemble_full_report(report_id, outline)  # 缺 scope_markdown
    with pytest.raises(ValueError, match="scope_markdown"):
        ReportManager.assemble_full_report(report_id, outline, scope_markdown="   ")


def test_scope_block_survives_post_processing_and_leads_the_report(tmp_path, monkeypatch, sim_env):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(reports_dir))
    report_id = "report_scope_unit2"
    folder = reports_dir / report_id
    folder.mkdir(parents=True)
    (folder / "section_01.md").write_text("## 第一章\n\n正文。\n", encoding="utf-8")

    outline = ReportOutline(
        title="测试报告", summary="摘要",
        sections=[ReportSection(title="第一章", content="")],
    )
    scope = collect_simulation_scope(SIM_ID)
    content = ReportManager.assemble_full_report(
        report_id, outline, scope_markdown=render_scope_markdown(scope)
    )

    assert "## 模拟范围与传播声明" in content
    # 声明必须在正文之前出现，读者先看到"跑了什么"
    assert content.index("## 模拟范围与传播声明") < content.index("## 第一章")
    saved = (folder / "full_report.md").read_text(encoding="utf-8")
    assert saved == content


def test_assembly_fails_when_the_scope_slot_is_lost(tmp_path, monkeypatch, sim_env):
    """占位被吃掉时必须失败：声明不能悄悄从成稿里消失。"""

    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(reports_dir))
    report_id = "report_scope_unit3"
    folder = reports_dir / report_id
    folder.mkdir(parents=True)
    (folder / "section_01.md").write_text("## 第一章\n\n正文。\n", encoding="utf-8")

    monkeypatch.setattr(
        ReportManager, "_post_process_report",
        classmethod(lambda cls, content, outline: content.replace("<!-- simulation-scope -->", "")),
    )

    outline = ReportOutline(
        title="测试报告", summary="摘要",
        sections=[ReportSection(title="第一章", content="")],
    )
    with pytest.raises(ValueError, match="范围声明占位不唯一"):
        ReportManager.assemble_full_report(
            report_id, outline, scope_markdown="## 模拟范围与传播声明\n\n| a | b |\n"
        )


def test_report_to_dict_carries_the_scope():
    report = Report(
        report_id="report_x", simulation_id="sim_x", graph_id="graph_x",
        simulation_requirement="要求", status=ReportStatus.COMPLETED,
        simulation_scope={"simulation_id": "sim_x"},
    )
    assert report.to_dict()["simulation_scope"] == {"simulation_id": "sim_x"}


def test_scope_collection_is_wired_into_report_completion():
    """静态断言：组装报告前必须取范围声明，且组装调用带上了它。"""

    source = (BACKEND_DIR / "app" / "services" / "report_agent.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def generate_report", 1)[1].split("def chat", 1)[0]
    assert "collect_simulation_scope(self.simulation_id)" in body
    assert "scope_markdown=render_scope_markdown(simulation_scope)" in body
    # 取声明必须发生在组装之前
    assert body.index("collect_simulation_scope") < body.index("assemble_full_report")
    # 报告模块不允许绕过采集器自己拼数字
    assert "uploads/simulations" not in source
